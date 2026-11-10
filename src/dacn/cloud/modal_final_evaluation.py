"""Four-arm validation/final evaluation on the shared T4; never trains a model.

Prepare captures a NEW runtime/source inventory, not the historical Colab one.
Validation must reproduce the historical maps/metrics before seal permits test.
"""

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import modal

PROJECT = Path(__file__).resolve().parents[3]
REMOTE = Path("/opt/dacn-final")
app = modal.App("dacn-four-arm-final", include_source=False)
data_volume = modal.Volume.from_name("dacn-bmad-datasets")
run_volume = modal.Volume.from_name("dacn-bmad-runs")
ROOT = Path("/mnt/runs/final-evaluation")
DATASET = "dacn-bmad-team/bmad-flair-patient-disjoint-v1"
REVISION = "350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7"
MODEL = "dacn-bmad-team/dacn-runs"
MODEL_REVISION = "fae9b50a4b78114fb51d87e803c77d019668f523"
PREFIX = "evaluations/autoddpm-ddim2030-full83-t4-20261106T074200Z/comparison/"
METADATA_SHA = "9c60fac0ff2a2d6274e82db466e5e338303d0dc24986b3bab4b32b6e710af218"
REFERENCE_SHA = "22656e1f63a0047d3dbe32f7833eea8e303988414e5942dec1ba5a1d9ed1ef19"

# Capture exact installed versions in prepare, then enforce the same image and
# source bytes for validation and test. This is a new acceptance environment.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.13.0",
        "torchvision==0.28.0",
        "lpips==0.1.4",
        "numpy==2.4.6",
        "scipy",
        "scikit-learn",
        "PyYAML",
        "numba",
        "huggingface_hub",
        "matplotlib",
    )
    .run_commands("python -c \"import lpips; lpips.LPIPS(net='squeeze', spatial=True)\"")
    .env({"PYTHONPATH": str(REMOTE / "src"), "MPLBACKEND": "Agg"})
    .add_local_dir(PROJECT / "src", remote_path=str(REMOTE / "src"), ignore=["**/__pycache__/**"])
    .add_local_dir(
        PROJECT / "scripts", remote_path=str(REMOTE / "scripts"), ignore=["**/__pycache__/**"]
    )
    .add_local_dir(PROJECT / "configs", remote_path=str(REMOTE / "configs"))
    .add_local_dir(PROJECT / "docs/evidence", remote_path=str(REMOTE / "evidence"))
)
SETTINGS = {
    "image": image,
    "volumes": {"/mnt/datasets": data_volume, "/mnt/runs": run_volume},
    "secrets": [modal.Secret.from_name("huggingface-secret-2")],
    "timeout": 3600,
}


def _sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _root(protocol_id):
    import re

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", protocol_id):
        raise ValueError("Invalid protocol ID")
    return ROOT / protocol_id


def _fetch(repo, filename, revision, destination, kind="model"):
    import shutil

    from huggingface_hub import hf_hub_download

    source = hf_hub_download(
        repo, filename, revision=revision, repo_type=kind, token=os.environ["HF_TOKEN"]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def _dataset(root, split):
    """Only called after all input/runtime/final gates; never implicitly reads test."""
    from dacn.data.shards import ShardDataset

    metadata_path = root / "dataset_metadata.json"
    if _sha(metadata_path) != METADATA_SHA:
        raise ValueError("Dataset metadata differs from immutable provenance")
    metadata = json.loads(metadata_path.read_bytes())
    if metadata["split_policy"] != "patient-disjoint-v1":
        raise ValueError("Unreviewed split")
    for item in [metadata["splits"][split]["index"], *metadata["splits"][split]["shards"]]:
        relative = Path(item["path"])
        if (
            relative.parts[:2] != ("data", split)
            or len(relative.parts) != 3
            or ".." in relative.parts
        ):
            raise ValueError("Invalid split shard path")
        destination = root / relative
        if not destination.exists():
            _fetch(DATASET, relative.as_posix(), REVISION, destination, "dataset")
        if _sha(destination) != item["sha256"]:
            raise ValueError("Dataset content checksum differs")
    data = ShardDataset(root / "data" / split, split, include_masks=True)
    if len(data) != {"val": 83, "test": 3665}[split]:
        raise ValueError("Dataset sample count differs")
    return data


def _paths(root):
    return {
        name: root / "artifacts" / name
        for name in (
            "source_snapshot",
            "config",
            "checkpoint",
            "lpips_backbone",
            "lpips_calibration",
            "dependency_lock",
            "validation_index",
            "validation_partition",
            "mask_parameters",
        )
    }


def _runtime(captured):
    import dacn
    from dacn.evaluation.runtime_binding import verify_runtime_inputs

    if Path(dacn.__file__).resolve() != REMOTE / "src/dacn/__init__.py":
        raise ValueError("Import tree is outside pinned source")
    freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--exclude-editable"], text=True
    )
    return verify_runtime_inputs(captured, REMOTE, freeze)


@app.function(**SETTINGS, cpu=2, memory=8192)
def prepare(protocol_id: str):
    import csv
    import io

    import lpips
    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from sklearn.model_selection import StratifiedGroupKFold

    from dacn.evaluation.autoddpm_lpips import perceptual_provenance
    from dacn.evaluation.dependency_snapshot import version_snapshot
    from dacn.evaluation.four_arm import (
        atomic_json,
        digest,
        json_bytes,
        restore_pinned_text,
        verify_controls,
    )

    root = _root(protocol_id)
    root.mkdir(parents=True, exist_ok=False)
    paths = _paths(root)
    (root / "artifacts").mkdir()
    # Public aggregate start record; prepare does not train/evaluate a model.
    atomic_json(root / "prepare.json", {"status": "preparing", "test_accessed": False})
    run_volume.commit()
    api = HfApi(token=os.environ["HF_TOKEN"])
    if not api.repo_info(MODEL).private or not api.repo_info(DATASET, repo_type="dataset").private:
        raise ValueError("Private dataset and artifact repositories required")
    _fetch(DATASET, "dataset_metadata.json", REVISION, root / "dataset_metadata.json", "dataset")
    data = _dataset(root, "val")
    paths["validation_index"].write_bytes((root / "data/val/index.csv").read_bytes())
    _fetch(
        MODEL,
        "runs/ddpm-partial-t4-b8-e30-cont-20261102T034644Z/ddpm_epoch_030.pt",
        MODEL_REVISION,
        paths["checkpoint"],
    )
    _fetch(MODEL, PREFIX + "mask_parameters.json", MODEL_REVISION, paths["mask_parameters"])
    _fetch(MODEL, PREFIX + "private_ablation.npz", MODEL_REVISION, root / "private_reference.npz")
    if _sha(root / "private_reference.npz") != REFERENCE_SHA:
        raise ValueError("Historical validation cache checksum differs")
    candidate = json.loads((REMOTE / "configs/final_evaluation_candidate.json").read_bytes())
    paths["config"].write_bytes(
        restore_pinned_text(
            (REMOTE / "configs/autoddpm_ddim20_30.json").read_bytes(),
            candidate["artifacts"]["config"],
        )
    )
    for key in ("checkpoint", "config", "validation_index", "mask_parameters"):
        if _sha(paths[key]) != candidate["artifacts"][key]:
            raise ValueError("Scientific artifact differs: " + key)
    rows = list(csv.DictReader(io.StringIO(paths["validation_index"].read_text())))
    groups = np.array([r["patient_group_id"] for r in rows])
    cal, hold = next(
        StratifiedGroupKFold(2, shuffle=True, random_state=17).split(
            np.zeros((83, 1)), data.labels, groups
        )
    )
    # Recovered membership is accepted ONLY if all historical four-arm metrics
    # reproduce. This does not assert byte identity with the old private JSON.
    partition = {
        "schema_version": 1,
        "index_sha256": _sha(paths["validation_index"]),
        "calibration_positions": cal.tolist(),
        "holdout_positions": hold.tolist(),
    }
    atomic_json(paths["validation_partition"], partition)
    with np.load(root / "private_reference.npz", allow_pickle=False) as reference:
        for i in range(83):
            im, mask, label = data[i]
            if not (
                np.array_equal(im[0].numpy(), reference["images"][i])
                and np.array_equal(mask.numpy(), reference["masks"][i])
                and int(label) == int(reference["labels"][i])
            ):
                raise ValueError("Historical validation alignment differs")
    network = lpips.LPIPS(net="squeeze", spatial=True, version="0.1").eval()
    if perceptual_provenance(network)["state_sha256"] != (
        "18fed4d1fc5c04f2918f64df3a2463344108576d1888f0d75538ca6d5c4dffe6"
    ):
        raise ValueError("Historical native LPIPS state differs")
    torch.save(network.net.state_dict(), paths["lpips_backbone"])
    torch.save(
        {k: v for k, v in network.state_dict().items() if not k.startswith("net.")},
        paths["lpips_calibration"],
    )
    freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--exclude-editable"], text=True
    )
    paths["dependency_lock"].write_text(version_snapshot(freeze))
    with zipfile.ZipFile(paths["source_snapshot"], "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in ("src", "scripts"):
            for path in sorted((REMOTE / folder).rglob("*.py")):
                entry = zipfile.ZipInfo(path.relative_to(REMOTE).as_posix(), (1980, 1, 1, 0, 0, 0))
                archive.writestr(entry, path.read_bytes())
    controls = json.loads((REMOTE / "configs/final_control_candidates.json").read_bytes())
    controls.update(
        status="frozen",
        pending=[],
        freeze_authority="User instruction 2026-11-10",
        scope="Fixed four-arm thresholds; no retuning after validation or test",
    )
    candidate.update(
        status="candidate",
        artifacts={k: _sha(p) for k, p in paths.items()},
        pending=[],
        controls_sha256=digest(json_bytes(controls)),
        dataset_metadata_sha256=METADATA_SHA,
        environment_scope="New Modal T4 acceptance; captured actual source/dependency inventory",
        source_archive_commit=None,
        scope="Method unchanged; new launcher/runtime require validation before final seal",
        freeze_authority="User instruction 2026-11-10; CI local-evidence exception recorded",
    )
    verify_controls(candidate, controls)
    atomic_json(root / "manifest.json", candidate)
    atomic_json(root / "controls.json", controls)
    result = {
        "status": "prepared",
        "protocol_id": protocol_id,
        "test_accessed": False,
        "validation_samples": len(data),
        "partition_recovered_requires_metric_acceptance": True,
        "artifacts": candidate["artifacts"],
        "controls_sha256": candidate["controls_sha256"],
    }
    atomic_json(root / "prepare.json", result)
    run_volume.commit()
    return result


@app.function(**{**SETTINGS, "timeout": 4 * 3600}, gpu="T4", cpu=4, memory=32768)
def evaluate(protocol_id: str, attempt_id: str, stage: str = "smoke"):
    import numpy as np
    import torch
    from huggingface_hub import HfApi

    from dacn.evaluation.four_arm import atomic_json, digest, json_bytes, run_four_arm
    from dacn.evaluation.frozen_ddim import FrozenDDIMAdapter
    from dacn.evaluation.registered_partition import registered_partition

    if stage not in {"smoke", "validation", "test"}:
        raise ValueError("Invalid stage")
    root = _root(protocol_id)
    output = root / "attempts" / _root(attempt_id).name
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_bytes())
    controls = json.loads((root / "controls.json").read_bytes())
    paths = _paths(root)
    run_manifest_path = output / "run_manifest.json"
    record = {
        "status": "started",
        "stage": stage,
        "attempt_id": attempt_id,
        "protocol_id": protocol_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset_revision": REVISION,
        "frozen_inputs": manifest,
        "held_out_test_used": False,
        "gpu": torch.cuda.get_device_name(0),
    }
    if run_manifest_path.exists():
        previous = json.loads(run_manifest_path.read_bytes())
        if any(
            previous[k] != record[k]
            for k in ("stage", "attempt_id", "protocol_id", "frozen_inputs")
        ):
            raise ValueError("Attempt resume provenance differs")
        record["started_at"] = previous["started_at"]
    atomic_json(run_manifest_path, record)
    run_volume.commit()
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()

    def samples():
        if stage == "test":
            record["held_out_test_used"] = True
            atomic_json(run_manifest_path, record)
            run_volume.commit()
        data = _dataset(root, "test" if stage == "test" else "val")
        if stage == "smoke":
            indices = [int(np.flatnonzero(data.labels == target)[0]) for target in (0, 1)]
        elif stage == "validation":
            import csv

            rows = list(csv.DictReader((root / "data/val/index.csv").open()))
            groups = np.array([r["patient_group_id"] for r in rows])
            _, indices = registered_partition(
                paths["validation_partition"].read_bytes(),
                paths["validation_index"].read_bytes(),
                groups,
            )
        else:
            indices = range(len(data))
        for i in indices:
            image_value, mask, _ = data[int(i)]
            yield (
                data.sample_ids[i],
                {
                    "image": image_value[0].numpy(),
                    "mask": mask.numpy(),
                    "sample_id": data.sample_ids[i],
                },
            )

    try:
        if stage == "validation":
            smoke = json.loads((root / "smoke_acceptance.json").read_bytes())
            from dacn.evaluation.four_arm import method_contract

            if smoke.get("status") != "completed" or smoke.get(
                "method_contract"
            ) != method_contract(manifest):
                raise ValueError("Matching bounded smoke required before full validation")
        report = run_four_arm(
            manifest=manifest,
            controls=controls,
            artifact_paths=paths,
            samples=samples,
            infer=FrozenDDIMAdapter("cuda", include_controls=True),
            output_dir=output,
            expected_samples={"smoke": 2, "validation": 43, "test": 3665}[stage],
            split="test" if stage == "test" else "validation",
            runtime_check=_runtime,
            validation_receipt=root / "validation_acceptance.json" if stage == "test" else None,
            on_commit=run_volume.commit,
        )
        report.update(
            gpu=torch.cuda.get_device_name(0),
            cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        )
        if stage == "validation":
            partition = json.loads(paths["validation_partition"].read_bytes())
            hold = partition["holdout_positions"]
            with np.load(root / "private_reference.npz", allow_pickle=False) as reference:
                errors = []
                for i, position in enumerate(hold):
                    with np.load(output / f"shard-{i:06d}.npz", allow_pickle=False) as current:
                        errors.append(
                            float(
                                np.max(
                                    np.abs(
                                        current["final_lpips"]
                                        - reference["autoddpm_lpips"][position]
                                    )
                                )
                            )
                        )
            expected = json.loads(
                (REMOTE / "evidence/2026-11-08-colab-registered-acceptance.json").read_bytes()
            )
            metric_error = 0.0
            for arm in controls["arms"]:
                for level in ("pixel", "image"):
                    for field in ("auroc", "auprc", "precision", "recall", "f1", "threshold"):
                        metric_error = max(
                            metric_error,
                            abs(
                                report["arms"][arm][level][field]
                                - expected["arms"][arm]["metrics"][level][field]
                            ),
                        )
            report.update(
                max_abs_final_map_error=max(errors),
                max_abs_reference_metric_error=metric_error,
                reference_maps_verified=max(errors) <= 1e-6 and metric_error <= 1e-6,
            )
            if not report["reference_maps_verified"]:
                atomic_json(output / "aggregate.json", report)
                raise ValueError("New runtime did not reproduce historical validation")
        atomic_json(output / "aggregate.json", report)
        if stage == "smoke":
            atomic_json(root / "smoke_acceptance.json", report)
        if stage == "validation":
            atomic_json(root / "validation_acceptance.json", report)
            manifest.update(
                status="frozen",
                validation_acceptance_sha256=digest(json_bytes(report)),
                frozen_at=datetime.now(timezone.utc).isoformat(),
                scope="Final four-arm evaluation; fixed method accepted on validation",
            )
            atomic_json(root / "manifest.json", manifest)
        record.update(
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            aggregate_sha256=_sha(output / "aggregate.json"),
        )
        atomic_json(run_manifest_path, record)
        run_volume.commit()
        api = HfApi(token=os.environ["HF_TOKEN"])
        if not api.repo_info(MODEL).private:
            raise ValueError("Private output repository required")
        # Private shards stay on durable Volume. Upload safe aggregate/provenance,
        # curves and frozen artifact bundle; never create a public repository.
        upload = output / "handoff"
        upload.mkdir(exist_ok=True)
        import shutil

        for filename in ("aggregate.json", "curves.json", "run_manifest.json"):
            shutil.copyfile(output / filename, upload / filename)
        shutil.copyfile(root / "manifest.json", upload / "frozen_manifest.json")
        shutil.copyfile(root / "controls.json", upload / "controls.json")
        commit = api.upload_folder(
            repo_id=MODEL, folder_path=str(upload), path_in_repo=f"evaluations/{attempt_id}"
        )
        record.update(artifact_sync_status="uploaded", model_revision=commit.oid)
        atomic_json(run_manifest_path, record)
        run_volume.commit()
        return {
            "status": "completed",
            "stage": stage,
            "attempt_id": attempt_id,
            "revision": commit.oid,
            "aggregate": report,
        }
    except BaseException as exc:
        record.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            exception_type=type(exc).__name__,
        )
        atomic_json(run_manifest_path, record)
        run_volume.commit()
        raise


@app.function(**SETTINGS, cpu=1, memory=1024)
def inspect_run(protocol_id: str, attempt_id: str = ""):
    """Return only safe records; no token, sample identity, image or score array."""
    root = _root(protocol_id)
    if not attempt_id:
        return {
            "prepare": json.loads((root / "prepare.json").read_bytes()),
            "manifest": json.loads((root / "manifest.json").read_bytes()),
            "controls": json.loads((root / "controls.json").read_bytes()),
        }
    output = root / "attempts" / _root(attempt_id).name
    result = {}
    for filename in ("run_manifest.json", "aggregate.json", "curves.json"):
        if (output / filename).exists():
            result[filename] = json.loads((output / filename).read_bytes())
    if (output / "private_receipt.json").exists():
        private = json.loads((output / "private_receipt.json").read_bytes())
        result["progress"] = {
            "status": private["status"],
            "committed_samples": len(private["shards"]),
        }
    return result

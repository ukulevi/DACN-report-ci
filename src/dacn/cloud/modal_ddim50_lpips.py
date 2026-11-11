"""Smoke -> validation calibration/freeze -> full test for single-phase DDIM50."""

import csv
import io
import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import modal

from dacn.cloud.modal_final_evaluation import (
    METADATA_SHA,
    MODEL,
    REMOTE,
    REVISION,
    _dataset,
    _runtime,
    _sha,
    data_volume,
    image,
    run_volume,
)

app = modal.App("dacn-ddim50-lpips", include_source=False)
BASE = Path("/mnt/runs/ddim50-lpips")
PARENT = Path("/mnt/runs/final-evaluation/final-fourarm-20261110-b")
SETTINGS = {
    "image": image,
    "volumes": {"/mnt/runs": run_volume, "/mnt/datasets": data_volume},
    "secrets": [modal.Secret.from_name("huggingface-secret-2")],
}


def root_for(protocol_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", protocol_id):
        raise ValueError("Invalid protocol ID")
    return BASE / protocol_id


def artifact_paths(root):
    from dacn.evaluation.ddim50_lpips import ARTIFACTS

    return {name: root / "artifacts" / name for name in ARTIFACTS}


@app.function(**SETTINGS, cpu=2, memory=8192, timeout=1800)
def prepare(protocol_id: str, code_revision: str):
    from dacn.evaluation.ddim50_lpips import CONFIG, capture_inputs
    from dacn.evaluation.dependency_snapshot import version_snapshot
    from dacn.evaluation.four_arm import atomic_json

    root = root_for(protocol_id)
    root.mkdir(parents=True, exist_ok=False)
    (root / "artifacts").mkdir()
    paths = artifact_paths(root)
    original = json.loads((REMOTE / "configs/final_evaluation_frozen_20261110.json").read_bytes())
    for key in (
        "checkpoint",
        "lpips_backbone",
        "lpips_calibration",
        "validation_index",
        "validation_partition",
    ):
        source = PARENT / "artifacts" / key
        if _sha(source) != original["artifacts"][key]:
            raise ValueError("Original artifact checksum differs: " + key)
        shutil.copyfile(source, paths[key])
    if _sha(PARENT / "dataset_metadata.json") != METADATA_SHA:
        raise ValueError("Dataset metadata differs")
    shutil.copyfile(PARENT / "dataset_metadata.json", paths["dataset_metadata"])
    shutil.copyfile(REMOTE / "configs/ddim50_lpips_t300.json", paths["config"])
    freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--exclude-editable"], text=True
    )
    paths["dependency_lock"].write_text(version_snapshot(freeze))
    with zipfile.ZipFile(paths["source_snapshot"], "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in ("src", "scripts"):
            for path in sorted((REMOTE / folder).rglob("*.py")):
                archive.writestr(
                    zipfile.ZipInfo(path.relative_to(REMOTE).as_posix(), (1980, 1, 1, 0, 0, 0)),
                    path.read_bytes(),
                )
    manifest = {
        "schema_version": 1,
        "status": "prepared",
        "method": "single_phase_DDIM50_LPIPS",
        "config": CONFIG,
        "dataset_revision": REVISION,
        "artifacts": {key: _sha(path) for key, path in paths.items()},
        "code_revision": code_revision,
        "working_tree_dirty": True,
        "scope": "User-requested exploratory extension after prior final test; no test tuning",
        "checkpoint_source": "Gaussian EMA30; same weights as DDIM20/20+30",
        "reference_artifact_revision": "fae9b50a4b78114fb51d87e803c77d019668f523",
        "checkpoint_path_in_repo": "runs/ddpm-partial-t4-b8-e30-cont-20261102T034644Z/ddpm_epoch_030.pt",
        "score": "raw spatial LPIPS0.1 squeeze; grayscale repeatedRGB [-1,1],normalizeFalse",
        "metric_region": "full240crop",
        "image_score": "mean top576pixels",
        "seed_policy": "base17+private UUID-derived sampling; batch1",
        "checkpoint_every_samples": 25,
        "resume_verification_log_interval": 100,
    }
    captured = capture_inputs(manifest, paths)
    _runtime(captured)
    atomic_json(root / "manifest.json", manifest)
    run_volume.commit()
    return {"status": "prepared", "protocol_id": protocol_id, "manifest": manifest}


@app.function(**SETTINGS, gpu="T4", cpu=4, memory=32768, timeout=8 * 3600)
def evaluate(protocol_id: str, stage: str):
    import os

    import numpy as np
    import torch
    from huggingface_hub import HfApi

    from dacn.evaluation.ddim50_lpips import (
        DDIM50Adapter,
        capture_inputs,
        check_stage,
        method_contract,
        run_samples,
        test_metrics,
        validation_metrics,
    )
    from dacn.evaluation.four_arm import atomic_json, digest, json_bytes
    from dacn.evaluation.registered_partition import registered_partition

    if stage not in {"smoke", "validation", "test"}:
        raise ValueError("Invalid stage")
    root = root_for(protocol_id)
    output = root / stage
    output.mkdir(exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_bytes())
    record = {
        "protocol_id": protocol_id,
        "stage": stage,
        "status": "started",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest,
        "held_out_test_used": False,
        "gpu": torch.cuda.get_device_name(0),
        "batch_size": 1,
        "cpu": 4,
        "timeout_seconds": 8 * 3600,
    }
    record_path = output / "run_manifest.json"
    if record_path.exists():
        old = json.loads(record_path.read_bytes())
        if (
            method_contract(old["manifest"]) != method_contract(manifest)
            or old["stage"] != stage
            or (stage == "test" and old["manifest"] != manifest)
        ):
            raise ValueError("Resume manifest differs")
        record["started_at"] = old["started_at"]
        record["held_out_test_used"] = old.get("held_out_test_used", False)
        record["resumes"] = old.get("resumes", []) + [
            {
                "resumed_at": datetime.now(timezone.utc).isoformat(),
                "previous_status": old["status"],
                "previous_exception": old.get("exception_type"),
            }
        ]
    atomic_json(record_path, record)
    run_volume.commit()
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    tick = time.perf_counter()
    try:
        paths = artifact_paths(root)
        captured = capture_inputs(manifest, paths)
        _runtime(captured)
        smoke = (
            json.loads((root / "smoke/aggregate.json").read_bytes())
            if stage == "validation"
            else None
        )
        acceptance = (root / "validation/aggregate.json").read_bytes() if stage == "test" else None
        check_stage(manifest, stage, acceptance, smoke)

        def samples():
            if stage == "test":
                record["held_out_test_used"] = True
                atomic_json(record_path, record)
                run_volume.commit()
            data = _dataset(PARENT, "test" if stage == "test" else "val")
            selected = (
                range(len(data))
                if stage != "smoke"
                else [int(np.flatnonzero(data.labels == target)[0]) for target in (0, 1)]
            )
            for i in selected:
                value, mask, _ = data[i]
                yield (
                    data.sample_ids[i],
                    {
                        "sample_id": data.sample_ids[i],
                        "image": value[0].numpy(),
                        "mask": mask.numpy(),
                    },
                )

        receipt = run_samples(
            manifest=manifest,
            captured=captured,
            stage=stage,
            samples=samples,
            infer=DDIM50Adapter("cuda"),
            output=output,
            commit=run_volume.commit,
            validation_receipt=acceptance,
            smoke_receipt=smoke,
        )
        report = {
            "status": "completed",
            "stage": stage,
            "sample_count": len(receipt["shards"]),
            "method_contract": method_contract(manifest),
            "config": manifest["config"],
            "dataset_revision": REVISION,
            "training_seeds": 1,
            "sampling_seeds": 1,
        }
        if stage == "validation":
            rows = list(csv.DictReader(io.StringIO(captured["validation_index"].decode())))
            groups = np.array([r["patient_group_id"] for r in rows])
            cal, hold = registered_partition(
                captured["validation_partition"], captured["validation_index"], groups
            )
            report.update(validation_metrics(output, cal, hold))
        elif stage == "test":
            report["metrics"] = test_metrics(output, manifest["thresholds"])
            report["thresholds"] = manifest["thresholds"]
        timing = [sample["adapter_seconds"] for sample in receipt["shards"]]
        count = report["sample_count"]
        report.update(
            unet_calls_completed_trajectories=count * 50,
            lpips_calls_completed_trajectories=count,
            adapter_seconds_sum=sum(timing),
            adapter_seconds_median=float(np.median(timing)),
            adapter_seconds_p95=float(np.quantile(timing, 0.95)),
            timing_scope="Adapter incl cold model load per continuation; excludes data/checksum/shard/global-metric I/O",
            cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            gpu=record["gpu"],
            training_performed=False,
            test_accessed=stage == "test",
        )
        atomic_json(output / "aggregate.json", report)
        if stage == "validation":
            manifest.update(
                status="frozen",
                thresholds=report["thresholds"],
                validation_acceptance_sha256=digest(json_bytes(report)),
                frozen_at=datetime.now(timezone.utc).isoformat(),
            )
            atomic_json(root / "manifest.json", manifest)
        receipt["status"] = "completed"
        atomic_json(output / "private_receipt.json", receipt)
        record.update(
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            total_seconds=time.perf_counter() - tick,
            aggregate_sha256=_sha(output / "aggregate.json"),
        )
        atomic_json(record_path, record)
        run_volume.commit()
        handoff = output / "handoff"
        handoff.mkdir(exist_ok=True)
        for filename in (
            "aggregate.json",
            "run_manifest.json",
            "frozen_thresholds.json",
            "curves.json",
        ):
            if (output / filename).exists():
                shutil.copyfile(output / filename, handoff / filename)
        shutil.copyfile(root / "manifest.json", handoff / "protocol_manifest.json")
        exported = handoff / "artifacts"
        exported.mkdir(exist_ok=True)
        for name in (
            "source_snapshot",
            "dependency_lock",
            "config",
            "lpips_backbone",
            "lpips_calibration",
        ):
            shutil.copyfile(paths[name], exported / name)
        api = HfApi(token=os.environ["HF_TOKEN"])
        if not api.repo_info(MODEL).private:
            raise ValueError("Private artifact repo required")
        commit = api.upload_folder(
            repo_id=MODEL,
            folder_path=str(handoff),
            path_in_repo=f"evaluations/{protocol_id}-{stage}",
        )
        record.update(artifact_sync_status="uploaded", model_revision=commit.oid)
        atomic_json(record_path, record)
        run_volume.commit()
        return {"manifest": manifest, "run_manifest": record, "aggregate": report}
    except BaseException as exc:
        record.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            exception_type=type(exc).__name__,
        )
        atomic_json(record_path, record)
        run_volume.commit()
        raise


@app.local_entrypoint()
def main(protocol_id: str, stage: str, output: str, code_revision: str = "", detach: bool = False):
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    call = (
        prepare.spawn(protocol_id, code_revision)
        if stage == "prepare"
        else evaluate.spawn(protocol_id, stage)
    )
    submission = {
        "protocol_id": protocol_id,
        "stage": stage,
        "call_id": call.object_id,
        "app_id": app.app_id,
    }
    (folder / "submission.json").write_text(json.dumps(submission, indent=2) + "\n")
    print(json.dumps(submission), flush=True)
    if detach:
        return
    result = call.get()
    (folder / "result.json").write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    for key, filename in (
        ("manifest", "protocol_manifest.json"),
        ("run_manifest", "run_manifest.json"),
        ("aggregate", "aggregate.json"),
    ):
        if key in result:
            (folder / filename).write_text(
                json.dumps(result[key], sort_keys=True, indent=2, allow_nan=False) + "\n"
            )
    print(
        json.dumps(
            {
                "stage": stage,
                "status": result.get("status", result.get("run_manifest", {}).get("status")),
            }
        ),
        flush=True,
    )

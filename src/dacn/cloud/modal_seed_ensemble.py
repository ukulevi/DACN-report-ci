"""Single frozen test evaluation of the seed ensemble (definition: seed_ensemble).

Stage ``maps`` (T4): the unchanged FrozenDDIMAdapter with sampling seed base 18 or 19
and that seed's validation mask threshold. ``smoke`` runs two VALIDATION images and
requires the Final LPIPS maps of the validation variants run (no test access).
``full`` writes resumable private test chunks on this Workspace's Volume.

Stage ``aggregate`` (CPU): loads seed-17 Final LPIPS, brightness and truth from the
test-CI chunks and the seed-18/19 chunks, all identity-checked. It requires the seed-17
maps to reproduce the published test metrics and the frozen single-seed hybrid result,
then applies the frozen ensemble thresholds and reports patient-group CIs. Only
aggregates are uploaded.

    DACN_HF_SECRET=huggingface modal run --detach -m dacn.cloud.modal_seed_ensemble \
        --stage maps --seed 18 --evaluation-id ens-s18-UNIQUE --code-revision SHA --mode full
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import modal

from dacn.cloud.modal_hybrid_test import PUBLISHED, REPLAY_ATOL, SOURCE_RUN, TEST_INDEX_SHA
from dacn.cloud.modal_mask_rule_ablation import CHECKPOINT, CHECKPOINT_SHA, LPIPS_STATE_SHA
from dacn.cloud.modal_mask_rule_ablation import image as gpu_image
from dacn.cloud.modal_sweep_bootstrap import (
    DATA_ROOT,
    DATASET,
    HF_SECRET,
    MODEL_REPO,
    MODEL_REVISION,
    REVISION,
    RUN_ROOT,
    data_volume,
    run_volume,
)
from dacn.cloud.modal_sweep_bootstrap import image as cpu_image

app = modal.App("dacn-seed-ensemble")
CHUNK = 250


def _start(evaluation_id, code_revision, working_tree_dirty, mode, method, resume, extra):
    from dacn.cloud.modal_train import _write_run_manifest, _write_source_snapshot

    if mode not in {"smoke", "full"} or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", evaluation_id
    ):
        raise ValueError("Invalid seed-ensemble request")
    output = RUN_ROOT / "evaluations" / evaluation_id
    output.mkdir(parents=True, exist_ok=resume)
    path = output / "run_manifest.json"
    if resume and not path.is_file():
        raise ValueError("Nothing to resume")
    manifest = json.loads(path.read_text()) if resume else {}
    if resume and (manifest.get("code_revision") != code_revision or manifest.get("mode") != mode):
        raise ValueError("Resume requires the same code revision and mode")
    manifest.update(
        evaluation_id=evaluation_id,
        method=method,
        mode=mode,
        status="running",
        dataset_repo_id=DATASET,
        dataset_revision=REVISION,
        code_revision=code_revision,
        working_tree_dirty=working_tree_dirty,
        threshold_fitting=False,
        selection="none; configuration frozen on validation before this run",
        **extra,
    )
    manifest.setdefault("started_at", datetime.now(timezone.utc).isoformat())
    manifest["source_snapshot_sha256"] = _write_source_snapshot(output)
    manifest.setdefault("attempts", []).append(datetime.now(timezone.utc).isoformat())

    def persist():
        _write_run_manifest(output, manifest)
        run_volume.commit()

    persist()
    return output, manifest, persist


def _finish(output, manifest, persist, report, api, evaluation_id, started):
    import time

    from dacn.cloud.modal_train import _file_checksums, _json_safe, _reject_patient_identifiers

    report = _json_safe(report)
    _reject_patient_identifiers(report)
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    manifest.update(
        status="completed",
        completed_at=datetime.now(timezone.utc).isoformat(),
        total_seconds_last_attempt=time.perf_counter() - started,
        artifacts={k: v for k, v in _file_checksums(output).items() if not k.startswith("chunks/")},
    )
    manifest.pop("progress", None)
    persist()
    try:
        commit = api.upload_folder(
            folder_path=str(output),
            path_in_repo=f"evaluations/{evaluation_id}",
            repo_id=MODEL_REPO,
            allow_patterns=["results.json", "run_manifest.json", "source_snapshot.zip"],
            commit_message=f"Store seed ensemble {evaluation_id}",
        )
        manifest.update(artifact_sync_status="uploaded", hub_commit=commit.oid)
    except Exception as exc:  # noqa: BLE001 - upload failure is recorded; results stay on the Volume
        manifest.update(artifact_sync_status="failed", artifact_sync_error=type(exc).__name__)
    persist()
    return {"manifest": manifest, "results": report}


def _test_index():
    import csv
    import hashlib
    import io

    import numpy as np

    from dacn.evaluation.frozen_test_ci import TEST_SAMPLES

    key = hashlib.sha256(f"{DATASET}@{REVISION}".encode()).hexdigest()[:20]
    test_dir = DATA_ROOT / key / "data" / "test"
    index_bytes = (test_dir / "index.csv").read_bytes()
    if hashlib.sha256(index_bytes).hexdigest() != TEST_INDEX_SHA:
        raise ValueError("Pinned test index differs from the test-CI source run")
    rows = list(csv.DictReader(io.StringIO(index_bytes.decode())))
    if len(rows) != TEST_SAMPLES:
        raise ValueError("Unexpected test sample count")
    return test_dir, [r["sample_id"] for r in rows], np.array([r["patient_group_id"] for r in rows])


def _identity(sample_ids, first, last):
    import hashlib

    return hashlib.sha256("\n".join(sample_ids[first:last]).encode()).hexdigest()


@app.function(
    image=gpu_image,
    gpu="T4",
    cpu=8,
    memory=32768,
    timeout=6 * 60 * 60,
    volumes={str(DATA_ROOT): data_volume, str(RUN_ROOT): run_volume},
    secrets=[modal.Secret.from_name(HF_SECRET)],
)
def run_seed_maps_remote(
    evaluation_id: str,
    code_revision: str,
    working_tree_dirty: bool,
    mode: str,
    seed: int,
    resume: bool,
):
    import hashlib
    import time

    import lpips
    import numpy as np
    import torch
    from huggingface_hub import HfApi, hf_hub_download

    from dacn.cloud.modal_sweep_bootstrap import _pinned_validation, _variant_maps
    from dacn.data.shards import ShardDataset
    from dacn.evaluation.autoddpm_lpips import perceptual_provenance
    from dacn.evaluation.ddim_ablation import resolve_ablation_config
    from dacn.evaluation.ddim_inpainting import ddim_indices
    from dacn.evaluation.frozen_ddim import FrozenDDIMAdapter
    from dacn.evaluation.frozen_test_ci import TEST_SAMPLES
    from dacn.evaluation.postprocess_hybrid import SEED_VARIANTS
    from dacn.evaluation.seed_ensemble import MASK_THRESHOLDS
    from dacn.training.ddpm_partial import build_diffusion, load_ema_checkpoint

    if seed not in (18, 19):
        raise ValueError("Only seed bases 18 and 19 need new test maps")
    output, manifest, persist = _start(
        evaluation_id,
        code_revision,
        working_tree_dirty,
        mode,
        "frozen_ddim20_30_final_lpips_test_maps_for_seed_ensemble",
        resume,
        {
            "seed_base": seed,
            "mask_threshold": MASK_THRESHOLDS[seed],
            "checkpoint_sha256": CHECKPOINT_SHA,
            "split": "validation-smoke" if mode == "smoke" else "test",
            "held_out_test_used": mode == "full",
            "gpu": torch.cuda.get_device_name(0),
        },
    )
    started = time.perf_counter()
    try:
        token = os.environ["HF_TOKEN"]
        api = HfApi(token=token)
        if not api.repo_info(MODEL_REPO).private:
            raise ValueError("Private artifact repository required")
        checkpoint_path = Path(
            hf_hub_download(
                MODEL_REPO, CHECKPOINT, revision=MODEL_REVISION, token=token, cache_dir="/tmp/hf"
            )
        )
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != CHECKPOINT_SHA:
            raise ValueError("Checkpoint checksum differs")
        device = torch.device("cuda")
        # SHA-verified team checkpoint; same loader contract as the frozen adapter.
        train_config = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["config"]
        adapter = FrozenDDIMAdapter(device)
        adapter.config = {**resolve_ablation_config({}), "seed": seed}
        adapter.mask_threshold = MASK_THRESHOLDS[seed]
        adapter.indices = [ddim_indices(300, 20), ddim_indices(50, 30)]
        adapter.model = load_ema_checkpoint(checkpoint_path, train_config, device)
        adapter.diffusion = build_diffusion(train_config)
        perceptual = lpips.LPIPS(net="squeeze", spatial=True, version="0.1").to(device)
        perceptual.eval().requires_grad_(False)
        if perceptual_provenance(perceptual)["state_sha256"] != LPIPS_STATE_SHA:
            raise ValueError("LPIPS weights differ from the accepted state")
        adapter.perceptual = perceptual
        adapter.ready = True

        def final_lpips(image_value, mask, sample_id):
            adapter({"image": image_value, "mask": mask, "sample_id": sample_id}, None)
            return np.asarray(adapter.last_score_maps["final_lpips"], np.float32)

        if mode == "smoke":
            _, _, dataset = _pinned_validation()
            reference = _variant_maps(token, *SEED_VARIANTS[seed], (83, 240, 240))
            labels = np.asarray(dataset["labels"])
            error = 0.0
            for target in (0, 1):
                index = int(np.flatnonzero(labels == target)[0])
                value = final_lpips(
                    dataset["images"][index],
                    dataset["masks"][index].astype(np.uint8),
                    dataset["sample_ids"][index],
                )
                error = max(
                    error, float(np.max(np.abs(value - reference["autoddpm_lpips"][index])))
                )
            if error > 1e-5:
                raise ValueError(f"Seed adapter does not replay validation maps: {error}")
            report = {
                "mode": "smoke",
                "test_accessed": False,
                "validation_replay_max_abs_error": error,
            }
        else:
            test_dir, sample_ids, _ = _test_index()
            data = ShardDataset(test_dir, "test", include_masks=True)
            if list(data.sample_ids) != sample_ids:
                raise ValueError("Pinned test split order differs")
            chunk_dir = output / "chunks"
            chunk_dir.mkdir(exist_ok=True)
            seconds = []
            for first in range(0, TEST_SAMPLES, CHUNK):
                last = min(first + CHUNK, TEST_SAMPLES)
                path = chunk_dir / f"private_chunk_{first:05d}.npz"
                identity = _identity(sample_ids, first, last)
                if path.is_file():
                    with np.load(path, allow_pickle=False) as existing:
                        if str(existing["identity"]) == identity:
                            continue
                    raise ValueError("Existing chunk identity differs")
                values = []
                for index in range(first, last):
                    image_value, mask, _ = data[index]
                    tick = time.perf_counter()
                    values.append(
                        final_lpips(
                            image_value[0].numpy(), mask.numpy().astype(np.uint8), sample_ids[index]
                        )
                    )
                    seconds.append(time.perf_counter() - tick)
                temporary = chunk_dir / f".chunk_{first:05d}.tmp.npz"
                np.savez_compressed(
                    temporary, identity=np.array(identity), final_lpips=np.stack(values)
                )
                temporary.replace(path)
                manifest["progress"] = {"completed_samples": last, "total": TEST_SAMPLES}
                persist()
                print(json.dumps(manifest["progress"]), flush=True)
            report = {
                "mode": "full",
                "test_accessed": True,
                "samples": TEST_SAMPLES,
                "seed_base": seed,
                "adapter_seconds_median_this_attempt": float(np.median(seconds))
                if seconds
                else None,
                "note": "Private Final LPIPS chunks stay on this Workspace's Volume.",
            }
        return _finish(output, manifest, persist, report, api, evaluation_id, started)
    except BaseException as exc:
        manifest.update(status="failed", error=type(exc).__name__)
        manifest["error_message"] = str(exc)[:300]
        persist()
        raise


@app.function(
    image=cpu_image,
    cpu=8,
    memory=65536,
    timeout=3 * 60 * 60,
    volumes={str(DATA_ROOT): data_volume, str(RUN_ROOT): run_volume},
    secrets=[modal.Secret.from_name(HF_SECRET)],
)
def run_aggregate_remote(
    evaluation_id: str, code_revision: str, working_tree_dirty: bool, seed_runs: str
):
    import time

    import numpy as np
    from huggingface_hub import HfApi

    from dacn.evaluation.frozen_test_ci import (
        FROZEN_THRESHOLDS,
        TEST_SAMPLES,
        image_group_bootstrap,
        image_scores,
    )
    from dacn.evaluation.group_bootstrap import paired_group_bootstrap
    from dacn.evaluation.postprocess_hybrid import FROZEN_HYBRID_THRESHOLDS, frozen_hybrid_map
    from dacn.evaluation.seed_ensemble import (
        ENSEMBLE_ARM,
        FROZEN_ENSEMBLE_THRESHOLDS,
        HYBRID_TEST_AP,
        ensemble_map,
    )
    from dacn.evaluation.validation import evaluate_maps

    runs = dict(zip((18, 19), seed_runs.split(","), strict=True))
    output, manifest, persist = _start(
        evaluation_id,
        code_revision,
        working_tree_dirty,
        "full",
        "frozen_seed_ensemble_final_lpips_x_brightness_single_test_evaluation",
        False,
        {
            "split": "test",
            "held_out_test_used": True,
            "test_access_count_after_freeze": 6,
            "source_map_runs": {17: SOURCE_RUN, **runs},
            "frozen_thresholds": FROZEN_ENSEMBLE_THRESHOLDS,
            "new_model_calls": 0,
            "gpu": None,
        },
    )
    started = time.perf_counter()
    try:
        api = HfApi(token=os.environ["HF_TOKEN"])
        if not api.repo_info(MODEL_REPO).private:
            raise ValueError("Private artifact repository required")
        _, sample_ids, groups = _test_index()
        for seed, run in runs.items():
            source = json.loads((RUN_ROOT / "evaluations" / run / "run_manifest.json").read_text())
            if (
                source.get("status") != "completed"
                or source.get("mode") != "full"
                or source.get("seed_base") != seed
            ):
                raise ValueError(f"Seed {seed} map run is not a completed full run")
        truth = np.empty((TEST_SAMPLES, 240, 240), dtype=np.uint8)
        finals = {s: np.empty((TEST_SAMPLES, 240, 240), dtype=np.float32) for s in (17, 18, 19)}
        bright = np.empty((TEST_SAMPLES, 240, 240), dtype=np.float32)
        for first in range(0, TEST_SAMPLES, CHUNK):
            last = min(first + CHUNK, TEST_SAMPLES)
            identity = _identity(sample_ids, first, last)
            name = f"private_chunk_{first:05d}.npz"
            with np.load(RUN_ROOT / "evaluations" / SOURCE_RUN / "chunks" / name) as archive:
                if str(archive["identity"]) != identity:
                    raise ValueError("Test-CI chunk identity differs")
                truth[first:last] = archive["truth"]
                finals[17][first:last] = archive["final_lpips"]
                bright[first:last] = archive["brightness"]
            for seed, run in runs.items():
                with np.load(RUN_ROOT / "evaluations" / run / "chunks" / name) as archive:
                    if str(archive["identity"]) != identity:
                        raise ValueError(f"Seed {seed} chunk identity differs")
                    finals[seed][first:last] = archive["final_lpips"]
        labels = truth.reshape(TEST_SAMPLES, -1).any(axis=1).astype(int)

        def metrics(report):
            return (
                report["pixel"]["auprc"],
                report["pixel"]["f1"],
                report["image"]["auroc"],
                report["image"]["auprc"],
            )

        replay = {}
        for arm, values in (("final_lpips", finals[17]), ("brightness", bright)):
            ours = metrics(
                evaluate_maps(
                    values, truth, labels, split="test", thresholds=FROZEN_THRESHOLDS[arm]
                )
            )
            replay[arm] = max(abs(a - b) for a, b in zip(ours, PUBLISHED[arm], strict=True))
        hybrid = frozen_hybrid_map(finals[17], bright)
        hybrid_point = evaluate_maps(
            hybrid, truth, labels, split="test", thresholds=FROZEN_HYBRID_THRESHOLDS
        )
        replay["frozen_hybrid_ap"] = abs(hybrid_point["pixel"]["auprc"] - HYBRID_TEST_AP)
        if max(replay.values()) > REPLAY_ATOL:
            raise ValueError(f"Source maps do not reproduce published test metrics: {replay}")
        ensemble = ensemble_map([finals[s] for s in (17, 18, 19)], bright)
        point = evaluate_maps(
            ensemble, truth, labels, split="test", thresholds=FROZEN_ENSEMBLE_THRESHOLDS
        )
        manifest["progress"] = {"stage": "bootstrap"}
        persist()
        maps = {
            ENSEMBLE_ARM: ensemble,
            "frozen_hybrid": hybrid,
            "final_lpips": finals[17].astype(np.float64),
            "brightness": bright.astype(np.float64),
        }
        del finals
        thresholds = {
            ENSEMBLE_ARM: FROZEN_ENSEMBLE_THRESHOLDS["pixel"],
            "frozen_hybrid": FROZEN_HYBRID_THRESHOLDS["pixel"],
            "final_lpips": FROZEN_THRESHOLDS["final_lpips"]["pixel"],
            "brightness": FROZEN_THRESHOLDS["brightness"]["pixel"],
        }
        scores = {name: image_scores(values) for name, values in maps.items()}
        report = {
            "mode": "full",
            "test_accessed": True,
            "samples": TEST_SAMPLES,
            "normal_slices": int((labels == 0).sum()),
            "abnormal_slices": int(labels.sum()),
            "source_replay_max_abs_error": replay,
            "frozen_ensemble_test": point,
            "frozen_hybrid_test": hybrid_point,
            "image_bootstrap": image_group_bootstrap(
                labels, scores, groups, reference=ENSEMBLE_ARM
            ),
            "pixel_bootstrap": paired_group_bootstrap(
                truth, maps, groups, thresholds, reference=ENSEMBLE_ARM, replicates=1000, seed=17
            ),
            "scope": (
                "Single test evaluation of a seed ensemble frozen on validation by a rule "
                "fixed beforehand; sixth test access; one trained model; three sampling "
                "seeds; intervals conditional on this model and cohort."
            ),
        }
        return _finish(output, manifest, persist, report, api, evaluation_id, started)
    except BaseException as exc:
        manifest.update(status="failed", error=type(exc).__name__)
        manifest["error_message"] = str(exc)[:300]
        persist()
        raise


@app.local_entrypoint()
def main(
    evaluation_id: str,
    code_revision: str,
    stage: str = "maps",
    mode: str = "smoke",
    seed: int = 18,
    seed_runs: str = "",
    working_tree_dirty: bool = True,
    resume: bool = False,
):
    if stage == "maps":
        result = run_seed_maps_remote.remote(
            evaluation_id, code_revision, working_tree_dirty, mode, seed, resume
        )
    elif stage == "aggregate":
        result = run_aggregate_remote.remote(
            evaluation_id, code_revision, working_tree_dirty, seed_runs
        )
    else:
        raise ValueError("stage must be maps or aggregate")
    destination = Path("runs") / "seed-ensemble" / evaluation_id
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in result.items():
        (destination / f"{name}.json").write_bytes(
            (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        )
    print(json.dumps(result["manifest"], indent=2))

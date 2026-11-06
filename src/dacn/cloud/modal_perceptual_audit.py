"""LPIPS-only ablation and cached pseudo-mask diagnostics; no diffusion calls."""

import csv
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

from dacn.cloud.modal_feature_condition import DATASET, MODEL_REPO, REVISION
from dacn.cloud.modal_train import (
    DATA_ROOT,
    RUN_ROOT,
    _file_checksums,
    _write_run_manifest,
    _write_source_snapshot,
    app,
    base_image,
    dataset_volume,
    run_volume,
)
from dacn.cloud.modal_uncertainty import BASELINE_SHA, INDEX_SHA

SOURCE_EVALUATION = "autoddpm-lpips-full83-t4-20261106T061833Z"
SOURCE_RESULTS_SHA = "6fb277491308c63b2cc52fa614fc912486266a9c3322f7f099aff850cb51a7e0"
SOURCE_MAPS_SHA = "f4b37638c5f6b2aca92ecaa072402847e4f18f371b4c4b178483d50ef1f4bce2"

audit_image = (
    base_image.pip_install("torchvision", "lpips==0.1.4", "matplotlib>=3.8,<4")
    .run_commands(
        "python -c \"import lpips; lpips.LPIPS(net='squeeze', spatial=True, version='0.1')\""
    )
    .add_local_python_source("dacn")
)


@app.function(
    image=audit_image,
    gpu="T4",
    timeout=30 * 60,
    volumes={DATA_ROOT: dataset_volume, RUN_ROOT: run_volume},
    secrets=[modal.Secret.from_name("huggingface-secret-2")],
)
def run_perceptual_audit_remote(
    evaluation_id: str, code_revision: str, working_tree_dirty: bool = True, mode: str = "smoke"
):
    import lpips
    import numpy as np
    import torch
    from huggingface_hub import HfApi

    from dacn.data.shards import ShardDataset
    from dacn.evaluation.ddpm_memory_fusion import verified_cache
    from dacn.evaluation.perceptual_audit import evaluate_perceptual_audit

    if mode not in {"smoke", "full"} or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", evaluation_id
    ):
        raise ValueError("Invalid perceptual audit request")
    output = Path(RUN_ROOT) / "evaluations" / evaluation_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "evaluation_id": evaluation_id,
        "method": "LPIPS_only_pseudo_mask_audit",
        "mode": mode,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset_repo_id": DATASET,
        "dataset_revision": REVISION,
        "source_evaluation": SOURCE_EVALUATION,
        "source_maps_sha256": SOURCE_MAPS_SHA,
        "source_results_sha256": SOURCE_RESULTS_SHA,
        "baseline_cache_sha256": BASELINE_SHA,
        "index_sha256": INDEX_SHA,
        "source_snapshot_sha256": _write_source_snapshot(output),
        "code_revision": code_revision,
        "working_tree_dirty": working_tree_dirty,
        "held_out_test_used": False,
        "diffusion_model_calls": 0,
        "mask_redesign": False,
        "gpu": torch.cuda.get_device_name(0),
    }

    def persist():
        _write_run_manifest(output, manifest)
        run_volume.commit()

    persist()
    started = time.perf_counter()
    try:
        api = HfApi(token=os.environ["HF_TOKEN"])
        if not api.repo_info(MODEL_REPO).private:
            raise ValueError("Private artifact repository required")
        key = hashlib.sha256(f"{DATASET}@{REVISION}".encode()).hexdigest()[:20]
        validation_dir = Path(DATA_ROOT) / key / "data" / "val"
        data = ShardDataset(validation_dir, "val", include_masks=True)
        baseline_path = Path(RUN_ROOT) / "evaluations" / "ddpm-val-t4b8-e30-t300-20261102T071516Z"
        baseline = verified_cache(
            baseline_path / "ddpm_epoch_030_t0300_ancestral_reconstructions.npz",
            BASELINE_SHA,
            data,
            INDEX_SHA,
        )
        source = Path(RUN_ROOT) / "evaluations" / SOURCE_EVALUATION
        parent = json.loads((source / "run_manifest.json").read_text())
        if parent["status"] != "completed" or parent["artifact_sync_status"] != "uploaded":
            raise ValueError("Original AutoDDPM evaluation not completed/uploaded")
        for path, checksum in (
            (source / "results.json", SOURCE_RESULTS_SHA),
            (source / "scores" / "private_maps.npz", SOURCE_MAPS_SHA),
        ):
            if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError("Registered original result/cache checksum differs")
        original = json.loads((source / "results.json").read_text())
        if original["smoke"] or original["samples"] != 83 or original["held_out_test_used"]:
            raise ValueError("Registered original full evaluation required")
        with np.load(source / "scores" / "private_maps.npz", allow_pickle=False) as archive:
            cached = {
                k: archive[k].copy()
                for k in (
                    "images",
                    "initials",
                    "masks",
                    "labels",
                    "predicted_masks",
                    "initial_heatmap",
                    "weighted_resampled_absolute",
                )
            }
        for target, expected in (
            ("images", "images"),
            ("initials", "reconstructions"),
            ("masks", "masks"),
            ("labels", "labels"),
        ):
            if not np.array_equal(cached[target], baseline[expected]):
                raise ValueError("Cached branch ordering/content differs")
        del baseline
        with (validation_dir / "index.csv").open(newline="") as stream:
            records = list(csv.DictReader(stream))
        if [r["sample_id"] for r in records] != data.sample_ids:
            raise ValueError("Group index ordering differs")
        perceptual = lpips.LPIPS(net="squeeze", spatial=True, version="0.1").to("cuda")
        perceptual.eval().requires_grad_(False)
        torch.cuda.reset_peak_memory_stats()

        def calibrated(parameters):
            manifest["stage"] = "diagnostic_and_score_parameters_frozen"
            manifest["parameters"] = parameters
            persist()

        result = evaluate_perceptual_audit(
            cached=cached,
            groups=np.asarray([r["patient_group_id"] for r in records]),
            perceptual_model=perceptual,
            original_results=original,
            output_dir=output / "scores",
            device=torch.device("cuda"),
            smoke=mode == "smoke",
            on_calibrated=calibrated,
        )
        result.update(
            total_seconds=time.perf_counter() - started,
            peak_vram_bytes=int(torch.cuda.max_memory_allocated()),
            source_maps_sha256=SOURCE_MAPS_SHA,
            source_results_sha256=SOURCE_RESULTS_SHA,
        )
        (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        manifest.update(
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            artifacts=_file_checksums(output),
        )
        persist()
        commit = api.upload_folder(
            repo_id=MODEL_REPO,
            folder_path=str(output),
            path_in_repo=f"evaluations/{evaluation_id}",
            commit_message=f"Store perceptual audit {evaluation_id}",
        )
        manifest.update(artifact_sync_status="uploaded", model_revision=commit.oid)
        persist()
        print(
            json.dumps(
                {
                    "evaluation_id": evaluation_id,
                    "mode": mode,
                    "revision": commit.oid,
                    "status": "completed",
                }
            ),
            flush=True,
        )
        return {"evaluation_id": evaluation_id, "revision": commit.oid}
    except Exception as exc:
        manifest.update(status="failed", failure={"exception_type": type(exc).__name__})
        persist()
        raise

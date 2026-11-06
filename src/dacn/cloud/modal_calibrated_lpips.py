"""Normal-calibrated LPIPS on immutable DDIM20+30 finals; no diffusion/model training."""

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
from dacn.cloud.modal_perceptual_audit import audit_image
from dacn.cloud.modal_train import (
    DATA_ROOT,
    RUN_ROOT,
    _file_checksums,
    _write_run_manifest,
    _write_source_snapshot,
    app,
    dataset_volume,
    run_volume,
)
from dacn.cloud.modal_uncertainty import INDEX_SHA

SOURCE_EVALUATION = "autoddpm-ddim2030-full83-t4-20261106T074200Z"
SOURCE_RESULTS_SHA = "a8bae24bd7270b078d49f923461e43fb51aea3163ede5a2fe71c90a70b127730"
SOURCE_MAPS_SHA = "22656e1f63a0047d3dbe32f7833eea8e303988414e5942dec1ba5a1d9ed1ef19"


@app.function(
    image=audit_image,
    gpu="T4",
    timeout=30 * 60,
    volumes={DATA_ROOT: dataset_volume, RUN_ROOT: run_volume},
    secrets=[modal.Secret.from_name("huggingface-secret-2")],
)
def run_calibrated_lpips_remote(
    evaluation_id: str, code_revision: str, working_tree_dirty: bool = True, mode: str = "smoke"
):
    import lpips
    import numpy as np
    import torch
    from huggingface_hub import HfApi

    from dacn.data.shards import ShardDataset
    from dacn.evaluation.calibrated_lpips import evaluate_calibrated_lpips

    if mode not in {"smoke", "full"} or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", evaluation_id
    ):
        raise ValueError("Invalid normal-layer LPIPS request")
    output = Path(RUN_ROOT) / "evaluations" / evaluation_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "evaluation_id": evaluation_id,
        "method": "normal_calibrated_seven_layer_LPIPS",
        "mode": mode,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset_repo_id": DATASET,
        "dataset_revision": REVISION,
        "code_revision": code_revision,
        "working_tree_dirty": working_tree_dirty,
        "source_snapshot_sha256": _write_source_snapshot(output),
        "source_evaluation": SOURCE_EVALUATION,
        "source_maps_sha256": SOURCE_MAPS_SHA,
        "source_results_sha256": SOURCE_RESULTS_SHA,
        "index_sha256": INDEX_SHA,
        "held_out_test_used": False,
        "diffusion_model_calls": 0,
        "encoder_changed": False,
        "mask_changed": False,
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
        source = Path(RUN_ROOT) / "evaluations" / SOURCE_EVALUATION
        parent = json.loads((source / "run_manifest.json").read_text())
        if parent["status"] != "completed" or parent["artifact_sync_status"] != "uploaded":
            raise ValueError("Source DDIM20+30 not completed/uploaded")
        for path, checksum in (
            (source / "results.json", SOURCE_RESULTS_SHA),
            (source / "comparison" / "private_ablation.npz", SOURCE_MAPS_SHA),
        ):
            if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError("Registered source artifact checksum differs")
        original = json.loads((source / "results.json").read_text())
        if (
            original["samples"] != 83
            or original["smoke"]
            or original["held_out_test_used"]
            or set(original["unet_calls_per_output_image"]) != {50}
        ):
            raise ValueError("Registered full DDIM20+30 source required")
        with np.load(source / "comparison" / "private_ablation.npz", allow_pickle=False) as archive:
            cached = {
                k: archive[k].copy()
                for k in (
                    "images",
                    "initials",
                    "finals",
                    "masks",
                    "labels",
                    "predicted_masks",
                    "autoddpm_lpips",
                )
            }
        key = hashlib.sha256(f"{DATASET}@{REVISION}".encode()).hexdigest()[:20]
        validation_dir = Path(DATA_ROOT) / key / "data" / "val"
        if hashlib.sha256((validation_dir / "index.csv").read_bytes()).hexdigest() != INDEX_SHA:
            raise ValueError("Pinned validation index differs")
        data = ShardDataset(validation_dir, "val", include_masks=True)
        if len(data) != 83:
            raise ValueError("Expected83validation samples")
        for index in range(83):
            image, mask, label = data[index]
            if (
                not np.array_equal(cached["images"][index], image[0].numpy())
                or not np.array_equal(cached["masks"][index], mask.numpy())
                or int(cached["labels"][index]) != int(label)
            ):
                raise ValueError("Source image/mask/label alignment differs")
        with (validation_dir / "index.csv").open(newline="") as stream:
            records = list(csv.DictReader(stream))
        if [r["sample_id"] for r in records] != data.sample_ids:
            raise ValueError("Group order differs")
        model = lpips.LPIPS(net="squeeze", spatial=True, version="0.1").to("cuda")
        model.eval().requires_grad_(False)
        torch.cuda.reset_peak_memory_stats()

        def progress(values):
            manifest["progress"] = values
            persist()
            print(json.dumps({k: v for k, v in values.items() if k != "parameters"}), flush=True)

        result = evaluate_calibrated_lpips(
            cached=cached,
            groups=np.asarray([r["patient_group_id"] for r in records]),
            perceptual_model=model,
            original_results=original,
            output_dir=output / "scores",
            device=torch.device("cuda"),
            smoke=mode == "smoke",
            on_progress=progress,
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
            commit_message=f"Store normal LPIPS calibration {evaluation_id}",
        )
        manifest.update(artifact_sync_status="uploaded", model_revision=commit.oid)
        persist()
        print(
            json.dumps(
                {
                    "evaluation_id": evaluation_id,
                    "mode": mode,
                    "status": "completed",
                    "revision": commit.oid,
                }
            ),
            flush=True,
        )
        return {"evaluation_id": evaluation_id, "revision": commit.oid}
    except Exception as exc:
        manifest.update(status="failed", failure={"exception_type": type(exc).__name__})
        persist()
        raise

"""AutoDDPM-style LPIPS pseudo-mask and resampling, without retraining."""

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

lpips_image = (
    base_image.pip_install("torchvision", "lpips==0.1.4")
    .run_commands(
        "python -c \"import lpips; lpips.LPIPS(net='squeeze', spatial=True, version='0.1')\""
    )
    .add_local_python_source("dacn")
)


@app.function(
    image=lpips_image,
    gpu="T4",
    timeout=60 * 60,
    volumes={DATA_ROOT: dataset_volume, RUN_ROOT: run_volume},
    secrets=[modal.Secret.from_name("huggingface-secret-2")],
)
def run_autoddpm_lpips_remote(
    evaluation_id: str, code_revision: str, working_tree_dirty: bool = True, mode: str = "smoke"
):
    import lpips
    import numpy as np
    import torch
    import torchvision
    from huggingface_hub import HfApi

    from dacn.data.shards import ShardDataset
    from dacn.evaluation.autoddpm_lpips import evaluate_lpips_resampling
    from dacn.evaluation.ddpm_memory_fusion import verified_cache
    from dacn.training.ddpm_partial import build_diffusion, load_ema_checkpoint
    from dacn.training.ddpm_unconditional_control import SOURCE_RUN, SOURCE_SHA

    if mode not in {"smoke", "full"} or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", evaluation_id
    ):
        raise ValueError("Invalid AutoDDPM request")
    output = Path(RUN_ROOT) / "evaluations" / evaluation_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "evaluation_id": evaluation_id,
        "method": "autoddpm_lpips_EMA30_t300_adaptation",
        "mode": mode,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset_repo_id": DATASET,
        "dataset_revision": REVISION,
        "code_revision": code_revision,
        "working_tree_dirty": working_tree_dirty,
        "source_snapshot_sha256": _write_source_snapshot(output),
        "source_run_id": SOURCE_RUN,
        "checkpoint_sha256": SOURCE_SHA,
        "baseline_cache_sha256": BASELINE_SHA,
        "index_sha256": INDEX_SHA,
        "initial_timestep_one_based": 300,
        "resampling_steps": 50,
        "repeats": 5,
        "seed": 17,
        "batch_size": 1,
        "held_out_test_used": False,
        "gpu": torch.cuda.get_device_name(0),
        "lpips_package_version": "0.1.4",
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }

    def persist():
        _write_run_manifest(output, manifest)
        run_volume.commit()

    persist()
    started = time.perf_counter()
    try:
        api = HfApi(token=os.environ["HF_TOKEN"])
        if not api.repo_info(MODEL_REPO).private:
            raise ValueError("Private model repository required")
        checkpoint = Path(RUN_ROOT) / SOURCE_RUN / "ddpm_epoch_030.pt"
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != SOURCE_SHA:
            raise ValueError("Original Gaussian EMA30 checksum differs")
        parent = json.loads((checkpoint.parent / "run_manifest.json").read_text())
        if parent["status"] != "completed" or parent["artifact_sync_status"] != "uploaded":
            raise ValueError("Original source not completed/uploaded")
        key = hashlib.sha256(f"{DATASET}@{REVISION}".encode()).hexdigest()[:20]
        validation_dir = Path(DATA_ROOT) / key / "data" / "val"
        data = ShardDataset(validation_dir, "val", include_masks=True)
        cached = verified_cache(
            Path(RUN_ROOT)
            / "evaluations"
            / "ddpm-val-t4b8-e30-t300-20261102T071516Z"
            / "ddpm_epoch_030_t0300_ancestral_reconstructions.npz",
            BASELINE_SHA,
            data,
            INDEX_SHA,
        )
        with (validation_dir / "index.csv").open(newline="") as stream:
            records = list(csv.DictReader(stream))
        if [r["sample_id"] for r in records] != data.sample_ids:
            raise ValueError("Sample ordering differs")
        config = torch.load(checkpoint, map_location="cpu", weights_only=False)["config"]
        model = load_ema_checkpoint(checkpoint, config, "cuda")
        diffusion = build_diffusion(config)
        perceptual = lpips.LPIPS(net="squeeze", spatial=True, version="0.1").to("cuda")
        perceptual.eval().requires_grad_(False)
        torch.cuda.reset_peak_memory_stats()

        def progress(value):
            manifest["progress"] = value
            persist()
            print(json.dumps({k: v for k, v in value.items() if k != "parameters"}), flush=True)

        result = evaluate_lpips_resampling(
            cached=cached,
            groups=np.asarray([r["patient_group_id"] for r in records]),
            sample_ids=data.sample_ids,
            model=model,
            diffusion=diffusion,
            perceptual_model=perceptual,
            output_dir=output / "scores",
            device=torch.device("cuda"),
            smoke=mode == "smoke",
            on_progress=progress,
        )
        result.update(
            total_seconds=time.perf_counter() - started,
            peak_vram_bytes=int(torch.cuda.max_memory_allocated()),
            checkpoint_sha256=SOURCE_SHA,
            baseline_cache_sha256=BASELINE_SHA,
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
            commit_message=f"Store AutoDDPM LPIPS {evaluation_id}",
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

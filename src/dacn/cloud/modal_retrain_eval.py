"""Validation maps of a retrained partial DDPM for the AF-11 variability study.

Loads the EMA epoch-30 checkpoint of one retraining run from this Workspace's Volume,
checked against that run's manifest SHA-256 and config. It then runs the frozen
producer-equivalent pipeline (``noise_extension.run_point``: t300 DDIM20+30, producer
pseudo-mask, p95/d3 rule refit on this model's normal calibration slices) for sampling
seeds 17/18/19 on all 83 validation slices. Saves score maps only. No test access.
``smoke`` runs two images with sampling seed 17 only.

    DACN_HF_SECRET=huggingface modal run -m dacn.cloud.modal_retrain_eval \
        --model a100_s18 --evaluation-id retrain-eval-s18-UNIQUE --code-revision SHA --mode full
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import modal

from dacn.cloud.modal_mask_rule_ablation import LPIPS_STATE_SHA, image
from dacn.cloud.modal_sweep_bootstrap import (
    DATA_ROOT,
    DATASET,
    HF_SECRET,
    INDEX_SHA,
    MODEL_REPO,
    PARTITION_SHA,
    REVISION,
    RUN_ROOT,
    data_volume,
    run_volume,
)

app = modal.App("dacn-retrain-eval")
REFERENCE_CONFIG = "configs/ddpm_partial_t4_batch8_epoch30.json"


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=16384,
    timeout=2 * 60 * 60,
    volumes={str(DATA_ROOT): data_volume, str(RUN_ROOT): run_volume},
    secrets=[modal.Secret.from_name(HF_SECRET)],
)
def run_retrain_eval_remote(
    evaluation_id: str,
    code_revision: str,
    working_tree_dirty: bool,
    mode: str,
    model: str,
    reference_config_json: str,
):
    import hashlib
    import time

    import lpips
    import numpy as np
    import torch
    from huggingface_hub import HfApi

    from dacn.cloud.modal_sweep_bootstrap import _pinned_validation
    from dacn.cloud.modal_train import (
        _file_checksums,
        _json_safe,
        _reject_patient_identifiers,
        _write_run_manifest,
        _write_source_snapshot,
    )
    from dacn.evaluation.autoddpm_lpips import perceptual_provenance
    from dacn.evaluation.noise_extension import run_point
    from dacn.evaluation.retrain_variability import (
        CHECKPOINT_NAME,
        SAMPLING_SEEDS,
        TRAINING_RUNS,
    )
    from dacn.training.ddpm_partial import build_diffusion, load_ema_checkpoint

    if (
        mode not in {"smoke", "full"}
        or model not in TRAINING_RUNS
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", evaluation_id)
    ):
        raise ValueError("Invalid retrain-eval request")
    training_run = TRAINING_RUNS[model]
    output = RUN_ROOT / "evaluations" / evaluation_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "evaluation_id": evaluation_id,
        "method": "retrained_ddpm_validation_maps_ddim20_30_t300_sampling_seeds",
        "mode": mode,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "split": "validation",
        "model": model,
        "training_run": training_run,
        "dataset_repo_id": DATASET,
        "dataset_revision": REVISION,
        "index_sha256": INDEX_SHA,
        "validation_partition_sha256": PARTITION_SHA,
        "code_revision": code_revision,
        "working_tree_dirty": working_tree_dirty,
        "source_snapshot_sha256": _write_source_snapshot(output),
        "held_out_test_used": False,
        "gpu": torch.cuda.get_device_name(0),
    }

    def persist():
        _write_run_manifest(output, manifest)
        run_volume.commit()

    persist()
    started = time.perf_counter()
    try:
        token = os.environ["HF_TOKEN"]
        api = HfApi(token=token)
        if not api.repo_info(MODEL_REPO).private:
            raise ValueError("Private artifact repository required")
        run_dir = RUN_ROOT / training_run
        source = json.loads((run_dir / "run_manifest.json").read_text())
        if source.get("status") != "completed":
            raise ValueError("Training run is not completed")
        checkpoint_path = run_dir / CHECKPOINT_NAME
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        checkpoint_sha = digest.hexdigest()
        if checkpoint_sha != source["artifacts"][CHECKPOINT_NAME]:
            raise ValueError("Checkpoint checksum differs from the training manifest")
        # SHA-verified checkpoint written by this team's trainer; same loader contract.
        config = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["config"]
        reference = json.loads(reference_config_json)
        # Compare in JSON form: the trainer normalises lists to tuples.
        saved = json.loads(json.dumps(config))
        differing = sorted(k for k, v in reference.items() if k != "seed" and saved.get(k) != v)
        if differing:
            raise ValueError(f"Retrained config differs from the reference: {differing}")
        manifest.update(
            checkpoint_sha256=checkpoint_sha,
            training_seed=config.get("seed"),
            training_model_revision=source.get("model_revision"),
            training_gpu=source.get("gpu"),
        )
        persist()
        _, _, dataset = _pinned_validation()
        labels = np.asarray(dataset["labels"])
        selected = (
            [int(np.flatnonzero(labels == t)[0]) for t in (0, 1)]
            if mode == "smoke"
            else list(range(len(labels)))
        )
        device = torch.device("cuda")
        net = load_ema_checkpoint(checkpoint_path, config, device)
        diffusion = build_diffusion(config)
        perceptual = lpips.LPIPS(net="squeeze", spatial=True, version="0.1").to(device)
        perceptual.eval().requires_grad_(False)
        if perceptual_provenance(perceptual)["state_sha256"] != LPIPS_STATE_SHA:
            raise ValueError("LPIPS weights differ from the accepted state")
        counter = {"unet": 0}
        hook = net.register_forward_hook(
            lambda *_: counter.__setitem__("unet", counter["unet"] + 1)
        )
        seeds = (17,) if mode == "smoke" else SAMPLING_SEEDS
        points = {}
        for seed in seeds:
            result = run_point(
                model=net,
                diffusion=diffusion,
                perceptual=perceptual,
                device=device,
                images=dataset["images"],
                labels=labels,
                sample_ids=dataset["sample_ids"],
                calibration=dataset["calibration"],
                selected=selected,
                point=(300, 20, 30),
                counter=counter,
                seed_base=seed,
            )
            name = f"private_{model}_s{seed}_maps.npz"
            np.savez_compressed(
                output / name,
                autoddpm_lpips=result["maps"]["autoddpm_lpips"],
                initial_lpips_only=result["maps"]["initial_lpips_only"],
                predicted_masks=result["predicted_masks"],
            )
            points[seed] = {
                "maps_file": name,
                "mask_threshold": result["mask_threshold"],
                "phase1_median_seconds": result["phase1_median_seconds"],
                "phase2_median_seconds": result["phase2_median_seconds"],
            }
            manifest["progress"] = {"completed_sampling_seeds": list(points)}
            persist()
        hook.remove()
        report = {
            "method": manifest["method"],
            "mode": mode,
            "model": model,
            "training_run": training_run,
            "checkpoint_sha256": checkpoint_sha,
            "training_seed": config.get("seed"),
            "samples": len(selected),
            "unet_calls_total": counter["unet"],
            "points": points,
            "selection": "none; maps only; analysis rules in dacn.evaluation.retrain_variability",
            "test_accessed": False,
        }
        report = _json_safe(report)
        _reject_patient_identifiers(report)
        (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        manifest.update(
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            total_seconds=time.perf_counter() - started,
            artifacts=_file_checksums(output),
        )
        manifest.pop("progress", None)
        persist()
        try:
            commit = api.upload_folder(
                folder_path=str(output),
                path_in_repo=f"evaluations/{evaluation_id}",
                repo_id=MODEL_REPO,
                commit_message=f"Store retrain eval {evaluation_id}",
            )
            manifest.update(artifact_sync_status="uploaded", hub_commit=commit.oid)
        except Exception as exc:  # noqa: BLE001 - upload failure is recorded; results stay on the Volume
            manifest.update(artifact_sync_status="failed", artifact_sync_error=type(exc).__name__)
        persist()
        return {"manifest": manifest, "results": report}
    except BaseException as exc:
        manifest.update(status="failed", error=type(exc).__name__)
        manifest["error_message"] = str(exc)[:300]
        persist()
        raise


@app.local_entrypoint()
def main(
    evaluation_id: str,
    code_revision: str,
    model: str,
    mode: str = "smoke",
    working_tree_dirty: bool = True,
):
    reference = Path(REFERENCE_CONFIG).read_text(encoding="utf-8")
    result = run_retrain_eval_remote.remote(
        evaluation_id, code_revision, working_tree_dirty, mode, model, reference
    )
    destination = Path("runs") / "retrain-eval" / evaluation_id
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in result.items():
        (destination / f"{name}.json").write_bytes(
            (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        )
    print(json.dumps(result["manifest"], indent=2))

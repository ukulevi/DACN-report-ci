"""T4 validation-only extension of the DDIM20+30 noise sweep to t50 and t400 (AF-05).

The producer-equivalent t300 DDIM20+30 control must replay the cached reference
arrays before new points run. Full mode reports producer-style metrics for all
eight score maps, mask diagnostics, reconstruction-quality aggregates and a
paired patient-group bootstrap against the t300 control. No test access.

    DACN_HF_SECRET=huggingface modal run -m dacn.cloud.modal_noise_extension \
        --evaluation-id noise-ext-smoke-UNIQUE --code-revision GIT_SHA --mode smoke \
        --no-working-tree-dirty
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import modal

from dacn.cloud.modal_mask_rule_ablation import (
    CHECKPOINT,
    CHECKPOINT_SHA,
    LPIPS_STATE_SHA,
    image,
)
from dacn.cloud.modal_sweep_bootstrap import (
    DATA_ROOT,
    DATASET,
    HF_SECRET,
    INDEX_SHA,
    MODEL_REPO,
    MODEL_REVISION,
    PARTITION_SHA,
    REVISION,
    RUN_ROOT,
    data_volume,
    run_volume,
)

app = modal.App("dacn-noise-extension")


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=16384,
    timeout=2 * 60 * 60,
    volumes={str(DATA_ROOT): data_volume, str(RUN_ROOT): run_volume},
    secrets=[modal.Secret.from_name(HF_SECRET)],
)
def run_noise_extension_remote(
    evaluation_id: str, code_revision: str, working_tree_dirty: bool, mode: str
):
    import hashlib
    import time

    import lpips
    import numpy as np
    import torch
    from huggingface_hub import HfApi, hf_hub_download

    from dacn.cloud.modal_sweep_bootstrap import _iter_verified_runs, _pinned_validation
    from dacn.cloud.modal_train import (
        _file_checksums,
        _json_safe,
        _reject_patient_identifiers,
        _write_run_manifest,
        _write_source_snapshot,
    )
    from dacn.evaluation.autoddpm_lpips import perceptual_provenance
    from dacn.evaluation.group_bootstrap import paired_group_bootstrap
    from dacn.evaluation.mask_rule_ablation import mask_statistics
    from dacn.evaluation.noise_extension import (
        CONTROL_POINT,
        NEW_POINTS,
        SCORE_NAMES,
        SMOKE_POINTS,
        check_point_replay,
        point_name,
        run_point,
    )
    from dacn.evaluation.noise_quality import reconstruction_quality
    from dacn.evaluation.sweep_bootstrap import REFERENCE_RUN
    from dacn.evaluation.validation import evaluate_maps
    from dacn.training.ddpm_partial import build_diffusion, load_ema_checkpoint

    if mode not in {"smoke", "full"} or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", evaluation_id
    ):
        raise ValueError("Invalid noise-extension request")
    points = SMOKE_POINTS if mode == "smoke" else (CONTROL_POINT, *NEW_POINTS)
    output = RUN_ROOT / "evaluations" / evaluation_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "evaluation_id": evaluation_id,
        "method": "ddim20_30_noise_sweep_extension_t50_t400",
        "mode": mode,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "split": "validation",
        "dataset_repo_id": DATASET,
        "dataset_revision": REVISION,
        "model_repo_revision": MODEL_REVISION,
        "reference_run": REFERENCE_RUN,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "index_sha256": INDEX_SHA,
        "validation_partition_sha256": PARTITION_SHA,
        "points": [point_name(*p) for p in points],
        "code_revision": code_revision,
        "working_tree_dirty": working_tree_dirty,
        "source_snapshot_sha256": _write_source_snapshot(output),
        "held_out_test_used": False,
        "training_performed": False,
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
        groups, holdout, dataset = _pinned_validation()
        calibration, sample_ids = dataset["calibration"], dataset["sample_ids"]
        [(_, _, cached, results, verified)] = list(
            _iter_verified_runs(token, dataset, run_ids={REFERENCE_RUN}, keep_reconstructions=True)
        )
        cached["mask_threshold"] = float(results["parameters"]["mask_threshold"])
        images, labels, masks = cached["images"], cached["labels"], cached["masks"]
        selected = (
            [int(np.flatnonzero(labels == target)[0]) for target in (0, 1)]
            if mode == "smoke"
            else list(range(len(labels)))
        )
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
        # SHA-verified team checkpoint; same loader contract as the producer runs.
        config = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["config"]
        device = torch.device("cuda")
        model = load_ema_checkpoint(checkpoint_path, config, device)
        diffusion = build_diffusion(config)
        perceptual = lpips.LPIPS(net="squeeze", spatial=True, version="0.1").to(device)
        perceptual.eval().requires_grad_(False)
        if perceptual_provenance(perceptual)["state_sha256"] != LPIPS_STATE_SHA:
            raise ValueError("LPIPS weights differ from the accepted state")
        counter = {"unet": 0}
        hook = model.register_forward_hook(
            lambda *_: counter.__setitem__("unet", counter["unet"] + 1)
        )
        torch.cuda.reset_peak_memory_stats()
        outputs, replay = {}, None
        for point in points:
            name = point_name(*point)
            result = run_point(
                model=model,
                diffusion=diffusion,
                perceptual=perceptual,
                device=device,
                images=images,
                labels=labels,
                sample_ids=sample_ids,
                calibration=calibration,
                selected=selected,
                point=point,
                counter=counter,
            )
            if point == CONTROL_POINT:
                replay = check_point_replay(result, cached, selected)
            outputs[name] = result
            manifest["progress"] = {"completed_points": list(outputs), "control_replay": replay}
            persist()
        hook.remove()
        report = {
            "method": manifest["method"],
            "mode": mode,
            "question": (
                "Advisor AF-05: add the suggested t50 and t400 phase-1 noise levels with all "
                "other components fixed (DDIM 50-call budget, phase 2 from t50, p95/d3 mask)."
            ),
            "control": point_name(*CONTROL_POINT),
            "control_replay": replay,
            "producer_artifacts": {REFERENCE_RUN: verified},
            "samples": len(selected),
            "unet_calls_total": counter["unet"],
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "selection": "none; all points reported; no test access",
            "points": {},
        }
        if mode == "full":
            truth = masks[:, 8:-8, 8:-8]
            boot_maps, boot_thresholds = {}, {}
            everything = np.arange(len(labels))
            for name, result in outputs.items():
                metrics, thresholds = {}, {}
                for score in SCORE_NAMES:
                    values = result["maps"][score]
                    fitted = evaluate_maps(
                        values[calibration],
                        truth[calibration],
                        labels[calibration],
                        split="validation",
                    )
                    thresholds[score] = {lv: fitted[lv]["threshold"] for lv in ("pixel", "image")}
                    metrics[score] = {
                        "calibration": fitted,
                        "group_holdout": evaluate_maps(
                            values[holdout],
                            truth[holdout],
                            labels[holdout],
                            split="validation",
                            thresholds=thresholds[score],
                        ),
                    }
                quality_arrays = {
                    "images": images,
                    "initials": result["initials"],
                    "finals": result["finals"],
                    "masks": masks,
                    "labels": labels,
                    "predicted_masks": result["predicted_masks"],
                    "initial_lpips_only": result["maps"]["initial_lpips_only"],
                    "autoddpm_lpips": result["maps"]["autoddpm_lpips"],
                }
                report["points"][name] = {
                    "mask_threshold": result["mask_threshold"],
                    "phase1_indices_zero_based": result["phase1_indices"],
                    "phase2_indices_zero_based": result["phase2_indices"],
                    "phase1_median_seconds": result["phase1_median_seconds"],
                    "phase2_median_seconds": result["phase2_median_seconds"],
                    "score_thresholds": thresholds,
                    "metrics": metrics,
                    "mask_all83": mask_statistics(
                        result["predicted_masks"], images, masks, labels, everything
                    ),
                    "quality_all83": reconstruction_quality(quality_arrays, everything),
                    "quality_holdout43": reconstruction_quality(quality_arrays, holdout),
                }
                for score, label in (
                    ("autoddpm_lpips", "final"),
                    ("initial_lpips_only", "initial"),
                ):
                    key = f"{label}_lpips_{name}"
                    boot_maps[key] = result["maps"][score][holdout]
                    boot_thresholds[key] = thresholds[score]["pixel"]
                np.savez_compressed(
                    output / f"private_{name}.npz",
                    images=images,
                    initials=result["initials"],
                    finals=result["finals"],
                    masks=masks,
                    labels=labels,
                    predicted_masks=result["predicted_masks"],
                    **result["maps"],
                )
            report["paired_bootstrap_vs_control"] = paired_group_bootstrap(
                truth[holdout],
                boot_maps,
                groups[holdout],
                boot_thresholds,
                reference=f"final_lpips_{point_name(*CONTROL_POINT)}",
                replicates=1000,
                seed=17,
            )
        report = _json_safe(report)
        _reject_patient_identifiers(report)
        (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        manifest.update(
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            total_seconds=time.perf_counter() - started,
            artifacts=_file_checksums(output),
        )
        persist()
        try:
            commit = api.upload_folder(
                folder_path=str(output),
                path_in_repo=f"evaluations/{evaluation_id}",
                repo_id=MODEL_REPO,
                commit_message=f"Store noise-sweep extension {evaluation_id}",
            )
            manifest.update(artifact_sync_status="uploaded", hub_commit=commit.oid)
        except Exception as exc:  # Results stay on the Volume and are returned.
            manifest.update(artifact_sync_status="failed", artifact_sync_error=type(exc).__name__)
        persist()
        return {"manifest": manifest, "results": report}
    except BaseException as exc:
        manifest.update(
            status="failed", error=type(exc).__name__, total_seconds=time.perf_counter() - started
        )
        manifest["error_message"] = str(exc)[:300]
        persist()
        raise


@app.local_entrypoint()
def main(
    evaluation_id: str, code_revision: str, mode: str = "smoke", working_tree_dirty: bool = True
):
    result = run_noise_extension_remote.remote(
        evaluation_id, code_revision, working_tree_dirty, mode
    )
    destination = Path("runs") / "noise-extension" / evaluation_id
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in result.items():
        (destination / f"{name}.json").write_bytes(
            (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        )
    print(json.dumps(result["manifest"], indent=2))

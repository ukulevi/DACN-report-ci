"""T4 K4 uncertainty scoring on fixed Gaussian EMA30; no training."""

import csv
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

from dacn.cloud.modal_feature_condition import (
    DATA_ROOT,
    DATASET,
    MODEL_REPO,
    REVISION,
    RUN_ROOT,
    app,
    dataset_volume,
    feature_image,
    run_volume,
)
from dacn.cloud.modal_train import _file_checksums, _write_run_manifest, _write_source_snapshot

INDEX_SHA = "57ad2e31df3ed4e625da00efaecd6fddc1e2a132329dfba6e60867caf1802db3"
BASELINE_SHA = "8278ad18543dd87f698c36b510564b9fcd4445d4a1b0ae5d7a859093e2a4e959"


@app.function(
    image=feature_image,
    gpu="T4",
    timeout=2 * 60 * 60,
    volumes={DATA_ROOT: dataset_volume, RUN_ROOT: run_volume},
    secrets=[modal.Secret.from_name("huggingface-secret-2")],
)
def run_uncertainty_remote(
    evaluation_id: str, code_revision: str, working_tree_dirty: bool = True, mode: str = "full"
):
    import numpy as np
    import torch
    from huggingface_hub import HfApi

    from dacn.data.shards import ShardDataset
    from dacn.evaluation.patch_validation import gaussian_ancestral_with_noisy
    from dacn.evaluation.uncertainty_scores import (
        ensemble_statistics,
        fit_normal_calibration,
        mahalanobis_score,
        normalized_score,
    )
    from dacn.evaluation.uncertainty_validation import (
        BASE_SEEDS,
        evaluate_ensemble,
        load_draw_prefix,
        save_draw_prefix,
    )
    from dacn.training.ddpm_partial import (
        build_diffusion,
        derive_sampling_seed,
        load_ema_checkpoint,
    )
    from dacn.training.ddpm_unconditional_control import SOURCE_RUN, SOURCE_SHA

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", evaluation_id) or mode not in {
        "smoke",
        "full",
    }:
        raise ValueError("Invalid uncertainty evaluation request")
    output = Path(RUN_ROOT) / "evaluations" / evaluation_id
    output.mkdir(parents=True, exist_ok=True)
    path = output / "run_manifest.json"
    previous = json.loads(path.read_text()) if path.exists() else None
    snapshot = _write_source_snapshot(output, persist=previous is None)
    if previous and (
        previous["source_snapshot_sha256"] != snapshot
        or previous["mode"] != mode
        or previous["code_revision"] != code_revision
        or previous["working_tree_dirty"] != working_tree_dirty
    ):
        raise ValueError("Uncertainty resume provenance differs")
    manifest = previous or {
        "schema_version": 1,
        "evaluation_id": evaluation_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "method": "Gaussian_EMA30_K4_uncertainty_scoring",
        "mode": mode,
        "ensemble_size": 4,
        "draw_base_seeds": list(BASE_SEEDS),
        "timesteps_one_based": [300],
        "split": "validation",
        "held_out_test_used": False,
        "source_run_id": SOURCE_RUN,
        "checkpoint": "ddpm_epoch_030.pt",
        "checkpoint_sha256": SOURCE_SHA,
        "dataset_repo_id": DATASET,
        "dataset_revision": REVISION,
        "index_sha256": INDEX_SHA,
        "baseline_cache_sha256": BASELINE_SHA,
        "code_revision": code_revision,
        "working_tree_dirty": working_tree_dirty,
        "source_snapshot_sha256": snapshot,
        "gpu": torch.cuda.get_device_name(0),
        "progress": {},
    }
    if manifest.get("status") == "completed" and manifest.get("artifact_sync_status") == "uploaded":
        return {"evaluation_id": evaluation_id, "revision": manifest["model_revision"]}

    def persist():
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_run_manifest(output, manifest)
        run_volume.commit()

    manifest.update(status="running", stage="verify_source_and_cached_draw")
    persist()
    try:
        started = time.perf_counter()
        api = HfApi(token=os.environ["HF_TOKEN"])
        if not api.repo_info(MODEL_REPO).private:
            raise ValueError("Private model repository required")
        checkpoint = Path(RUN_ROOT) / SOURCE_RUN / "ddpm_epoch_030.pt"
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != SOURCE_SHA:
            raise ValueError("Original Gaussian EMA30 checkpoint differs")
        parent = json.loads((checkpoint.parent / "run_manifest.json").read_text())
        if (
            parent.get("status") != "completed"
            or parent.get("artifact_sync_status") != "uploaded"
            or parent["artifacts"]["ddpm_epoch_030.pt"] != SOURCE_SHA
        ):
            raise ValueError("Source training not completed/uploaded")
        cached_path = (
            Path(RUN_ROOT)
            / "evaluations"
            / "ddpm-val-t4b8-e30-t300-20261102T071516Z"
            / "ddpm_epoch_030_t0300_ancestral_reconstructions.npz"
        )
        if hashlib.sha256(cached_path.read_bytes()).hexdigest() != BASELINE_SHA:
            raise ValueError("Registered first-draw cache differs")
        with np.load(cached_path, allow_pickle=False) as cache:
            cached = {k: cache[k].copy() for k in ("images", "reconstructions", "masks", "labels")}
        key = hashlib.sha256(f"{DATASET}@{REVISION}".encode()).hexdigest()[:20]
        validation_dir = Path(DATA_ROOT) / key / "data" / "val"
        if hashlib.sha256((validation_dir / "index.csv").read_bytes()).hexdigest() != INDEX_SHA:
            raise ValueError("Pinned validation index differs")
        with (validation_dir / "index.csv").open(newline="") as stream:
            records = list(csv.DictReader(stream))
        data = ShardDataset(validation_dir, "val", include_masks=True)
        if len(data) != 83 or [r["sample_id"] for r in records] != data.sample_ids:
            raise ValueError("Expected ordered83 validation images")
        for index in range(len(data)):
            image, mask, label = data[index]
            if (
                not np.array_equal(image[0].numpy(), cached["images"][index])
                or not np.array_equal(mask.numpy(), cached["masks"][index])
                or int(label) != int(cached["labels"][index])
            ):
                raise ValueError("Cached first draw/data alignment differs")
        config = torch.load(checkpoint, map_location="cpu", weights_only=False)["config"]
        model = load_ema_checkpoint(checkpoint, config, "cuda")
        diffusion = build_diffusion(config)
        if diffusion.noise_type != "gaussian":
            raise ValueError("Gaussian baseline required")
        selected = (
            [int(np.flatnonzero(data.labels == label)[0]) for label in (0, 1)]
            if mode == "smoke"
            else list(range(83))
        )
        manifest["stage"] = "bounded_cached_draw_replay"
        persist()
        errors = []
        for label in (0, 1):
            index = int(np.flatnonzero(data.labels == label)[0])
            value, _ = gaussian_ancestral_with_noisy(
                model,
                diffusion,
                data[index][0][None].to("cuda"),
                steps=300,
                seed=derive_sampling_seed(17, data.sample_ids[index]),
            )
            error = float(
                np.abs(value[0, 0].cpu().numpy() - cached["reconstructions"][index]).max()
            )
            if error > 1e-6:
                raise ValueError("Cached original reconstruction replay differs")
            errors.append(error)
        manifest["preflight"] = {"status": "passed", "original_draw_max_errors": errors}
        persist()
        torch.cuda.reset_peak_memory_stats()
        draws = [cached["reconstructions"][selected]]
        for draw_index, base_seed in enumerate(BASE_SEEDS[1:], 1):
            manifest["stage"] = f"reconstruction_draw_{draw_index + 1}_of4"
            persist()
            prefix = output / f"private_draw_{draw_index:02d}.npz"
            provenance = {
                "base_seed": base_seed,
                "checkpoint_sha": SOURCE_SHA,
                "index_sha": INDEX_SHA,
            }
            previous_values = load_draw_prefix(prefix, **provenance, max_samples=len(selected))
            values = list(previous_values)
            for position in range(len(values), len(selected)):
                index = selected[position]
                seeds = [derive_sampling_seed(s, data.sample_ids[index]) for s in BASE_SEEDS]
                if len(set(seeds)) != 4:
                    raise ValueError("Ensemble sampling seed collision")
                value, _ = gaussian_ancestral_with_noisy(
                    model,
                    diffusion,
                    data[index][0][None].to("cuda"),
                    steps=300,
                    seed=seeds[draw_index],
                )
                prediction = value[0, 0].cpu().numpy().copy()
                if not np.isfinite(prediction).all():
                    raise ValueError("Nonfinite reconstruction")
                values.append(prediction)
                completed = position + 1
                if completed == 1 or completed % 5 == 0 or completed == len(selected):
                    save_draw_prefix(prefix, values, **provenance)
                    manifest["progress"][f"draw_{draw_index + 1}"] = {
                        "completed_samples": completed,
                        "total_samples": len(selected),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                    persist()
                    print(
                        json.dumps(
                            {
                                "draw": draw_index + 1,
                                "completed_samples": completed,
                                "total_samples": len(selected),
                            }
                        ),
                        flush=True,
                    )
            draws.append(np.asarray(values))
        inference_seconds = time.perf_counter() - started
        peak = int(torch.cuda.max_memory_allocated())
        del model, diffusion, value
        torch.cuda.empty_cache()
        draws = np.stack(draws)
        if mode == "smoke":
            mean, variance = ensemble_statistics(draws)
            residual = (cached["images"][selected] - mean)[:, 8:-8, 8:-8]
            fitted = fit_normal_calibration(
                residual, variance[:, 8:-8, 8:-8], cached["labels"][selected], [0]
            )
            z = normalized_score(residual, variance[:, 8:-8, 8:-8], fitted)
            mahal = np.stack([mahalanobis_score(r, fitted) for r in residual])
            if not np.isfinite(z).all() or not np.isfinite(mahal).all():
                raise ValueError("Uncertainty scoring smoke failed")
            result = {
                "smoke": "passed",
                "samples": 2,
                "ensemble_size": 4,
                "preflight": manifest["preflight"],
                "variance_mean": float(variance.mean()),
                "normalized_and_mahalanobis_finite": True,
            }
            result_path = output / "smoke_result.json"
        else:
            manifest["stage"] = "normal_calibration_and_scoring"
            persist()

            def calibrated(values):
                manifest["calibration"] = values
                manifest["stage"] = "fixed_calibration_group_evaluation"
                persist()

            result = evaluate_ensemble(
                cached["images"],
                draws,
                cached["masks"],
                cached["labels"],
                np.asarray([r["patient_group_id"] for r in records]),
                output / "scores",
                on_calibrated=calibrated,
            )
            reference_path = (
                Path(RUN_ROOT)
                / "evaluations"
                / "ddpm-uncond-val-best36-t300-20261105T094934Z"
                / "results"
                / "results.json"
            )
            if (
                hashlib.sha256(reference_path.read_bytes()).hexdigest()
                != "babbd335d6a9c40e64d23b2e7dded4dd8f195e3522669586713488ced3f62c7e"
            ):
                raise ValueError("Registered grouped baseline reference differs")
            reference = json.loads(reference_path.read_text())["baseline_metrics"]
            for target, source in (
                ("single_absolute", "pixel_difference"),
                ("single_ssim", "local_ssim"),
            ):
                if not np.isclose(
                    result["metrics"][target]["group_holdout"]["pixel"]["average_precision"],
                    reference[source]["group_holdout"]["pixel"]["average_precision"],
                    rtol=0,
                    atol=1e-10,
                ):
                    raise ValueError("Single-draw baseline scoring contract differs")
            result_path = output / "scores" / "results.json"
        result.update(
            checkpoint_sha256=SOURCE_SHA,
            checkpoint_epoch=30,
            ema=True,
            held_out_test_used=False,
            baseline_cache_sha256=BASELINE_SHA,
            index_sha256=INDEX_SHA,
            gpu_inference_phase_seconds=inference_seconds,
            peak_vram_bytes=peak,
            total_seconds=time.perf_counter() - started,
        )
        result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        manifest.update(
            status="completed",
            stage="upload_artifacts",
            completed_at=datetime.now(timezone.utc).isoformat(),
            artifacts=_file_checksums(output),
        )
        persist()
        commit = api.upload_folder(
            repo_id=MODEL_REPO,
            folder_path=str(output),
            path_in_repo=f"evaluations/{evaluation_id}",
            commit_message=f"Store K4 uncertainty evaluation {evaluation_id}",
        )
        manifest.update(artifact_sync_status="uploaded", model_revision=commit.oid)
        persist()
        print(
            json.dumps(
                {
                    "evaluation_id": evaluation_id,
                    "mode": mode,
                    "status": "passed",
                    "revision": commit.oid,
                }
            ),
            flush=True,
        )
        return {"evaluation_id": evaluation_id, "mode": mode, "revision": commit.oid}
    except Exception as exc:
        manifest.update(status="failed", failure={"exception_type": type(exc).__name__})
        persist()
        raise

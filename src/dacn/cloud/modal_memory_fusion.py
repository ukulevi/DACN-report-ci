"""Frozen normal-memory construction and cached DDPM fusion on T4."""

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
from dacn.cloud.modal_spatial_condition import spatial_image
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
from dacn.cloud.modal_uncertainty import BASELINE_SHA, INDEX_SHA


@app.function(
    image=spatial_image,
    gpu="T4",
    timeout=60 * 60,
    volumes={DATA_ROOT: dataset_volume, RUN_ROOT: run_volume},
    secrets=[modal.Secret.from_name("huggingface-secret-2")],
)
def run_memory_fusion_remote(
    evaluation_id: str, code_revision: str, working_tree_dirty: bool = True, mode: str = "smoke"
):
    import numpy as np
    import torch
    from huggingface_hub import HfApi

    from dacn.data.shards import ShardDataset
    from dacn.evaluation.ddpm_memory_fusion import evaluate_fusion, verified_cache
    from dacn.models.memory_bank import MemoryFeatureEncoder, load_bank, memory_maps
    from dacn.training.build_memory_bank import build_memory_bank
    from dacn.training.ddpm_unconditional_control import SOURCE_SHA

    if mode not in {"smoke", "full"} or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", evaluation_id
    ):
        raise ValueError("Invalid memory fusion request")
    output = Path(RUN_ROOT) / "evaluations" / evaluation_id
    # Never overwrite a prior bank or evaluation, including failed attempts.
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "evaluation_id": evaluation_id,
        "method": "DDPM_normal_memory_fusion_v1",
        "mode": mode,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset_repo_id": DATASET,
        "dataset_revision": REVISION,
        "code_revision": code_revision,
        "working_tree_dirty": working_tree_dirty,
        "source_snapshot_sha256": _write_source_snapshot(output),
        "baseline_cache_sha256": BASELINE_SHA,
        "index_sha256": INDEX_SHA,
        "ddpm_checkpoint_sha256": SOURCE_SHA,
        "ddpm_ema_epoch": 30,
        "timestep_one_based": 300,
        "sampler": "ancestral_cached_base17",
        "held_out_test_used": False,
        "alpha": 0.5,
        "seed": 17,
        "bank_budget": 128 if mode == "smoke" else 10000,
        "max_training_samples": 16 if mode == "smoke" else None,
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
            raise ValueError("Private model repository required")
        key = hashlib.sha256(f"{DATASET}@{REVISION}".encode()).hexdigest()[:20]
        data_root = Path(DATA_ROOT) / key / "data"
        validation = ShardDataset(data_root / "val", "val", include_masks=True)
        baseline = Path(RUN_ROOT) / "evaluations" / "ddpm-val-t4b8-e30-t300-20261102T071516Z"
        cached = verified_cache(
            baseline / "ddpm_epoch_030_t0300_ancestral_reconstructions.npz",
            BASELINE_SHA,
            validation,
            INDEX_SHA,
        )
        with (data_root / "val" / "index.csv").open(newline="") as stream:
            records = list(csv.DictReader(stream))
        if [r["sample_id"] for r in records] != validation.sample_ids:
            raise ValueError("Validation sample ordering differs")
        training = ShardDataset(data_root / "train", "train", include_masks=False)
        encoder = MemoryFeatureEncoder().to("cuda").eval()
        torch.cuda.reset_peak_memory_stats()
        manifest["stage"] = "build_normal_bank"
        persist()

        def progress(count):
            manifest["training_samples_processed"] = count
            persist()
            print(json.dumps({"bank_training_samples": count}), flush=True)

        bank_path = output / "private_normal_bank.npz"
        _, metadata, checksum = build_memory_bank(
            training,
            encoder,
            bank_path,
            revision=REVISION,
            budget=manifest["bank_budget"],
            seed=17,
            max_samples=manifest["max_training_samples"],
            on_progress=progress,
        )
        values = load_bank(bank_path, checksum=checksum, expected_metadata=metadata)
        bank = torch.from_numpy(values).to("cuda")
        manifest.update(
            bank_sha256=checksum,
            bank_metadata=metadata,
            stage="memory_inference",
            bank_build_seconds=time.perf_counter() - started,
        )
        persist()
        selected = (
            [int(np.flatnonzero(validation.labels == label)[0]) for label in (0, 1)]
            if mode == "smoke"
            else list(range(83))
        )
        scores = []
        for position, index in enumerate(selected):
            image = torch.from_numpy(cached["images"][index])[None, None].to("cuda")
            score = memory_maps(encoder, image, bank)[0].cpu().numpy()
            if not np.isfinite(score).all() or score.shape != (240, 240):
                raise ValueError("Invalid aligned memory score")
            scores.append(score)
            if (position + 1) % 10 == 0:
                manifest["validation_samples_processed"] = position + 1
                persist()
        scores = np.stack(scores)
        manifest["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
        np.savez_compressed(output / "private_memory_maps.npz", scores=scores)
        if mode == "smoke":
            # Finite pipeline and bank persistence check only, no AP or threshold selection.
            from dacn.evaluation.ddpm_memory_fusion import fit_robust_normalization, fuse_maps
            from dacn.evaluation.scores import paired_score_maps

            paired = [
                paired_score_maps(cached["images"][i], cached["reconstructions"][i])
                for i in selected
            ]
            maps = {
                "ddpm_absolute": np.stack([p["pixel_difference"] for p in paired]),
                "ddpm_ssim": np.stack([p["local_ssim"] for p in paired]),
                "memory_bank": scores,
            }
            fitted = fit_robust_normalization(maps, np.array([0, 1]), np.array([0]))
            fused = fuse_maps(maps, fitted)
            if not all(np.isfinite(v).all() for v in fused.values()):
                raise ValueError("Nonfinite fusion smoke")
            result = {
                "smoke": "passed",
                "validation_samples": 2,
                "normal_training_samples": metadata["training_samples"],
                "bank_vectors": len(bank),
                "descriptor_dimensions": 384,
                "finite_aligned_scores": True,
                "AP_measured": False,
            }
        else:

            def calibrated(parameters):
                manifest["stage"] = "calibration_frozen"
                manifest["calibration"] = parameters
                persist()

            result = evaluate_fusion(
                cached,
                scores,
                np.asarray([r["patient_group_id"] for r in records]),
                output / "scores",
                provenance={**manifest},
                on_calibrated=calibrated,
            )
            # Registered controls must reproduce before publishing new fusion metrics.
            for name, expected in (
                ("ddpm_absolute", 0.101065919667),
                ("ddpm_ssim", 0.102402212970),
                ("brightness", 0.389505655716),
            ):
                measured = result["metrics"][name]["group_holdout"]["pixel"]["average_precision"]
                if not np.isclose(measured, expected, rtol=0, atol=1e-10):
                    raise ValueError("Registered grouped control metric differs")
        result.update(
            bank_sha256=checksum,
            baseline_cache_sha256=BASELINE_SHA,
            total_seconds=time.perf_counter() - started,
            held_out_test_used=False,
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
            commit_message=f"Store memory fusion {evaluation_id}",
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

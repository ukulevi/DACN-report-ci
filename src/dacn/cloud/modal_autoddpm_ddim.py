"""T4 two-phase DDIM acceleration for the existing AutoDDPM adaptation."""

import csv
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

from dacn.cloud.modal_autoddpm_lpips import lpips_image
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
from dacn.cloud.modal_uncertainty import BASELINE_SHA, INDEX_SHA


@app.function(
    image=lpips_image,
    gpu="T4",
    timeout=60 * 60,
    volumes={DATA_ROOT: dataset_volume, RUN_ROOT: run_volume},
    secrets=[modal.Secret.from_name("huggingface-secret-2")],
)
def run_autoddpm_ddim_remote(
    evaluation_id: str,
    code_revision: str,
    working_tree_dirty: bool = True,
    mode: str = "smoke",
    experiment: str = "legacy10_10x5",
    config_json: str = "{}",
):
    import lpips
    import numpy as np
    import torch
    from huggingface_hub import HfApi

    from dacn.data.shards import ShardDataset
    from dacn.evaluation.autoddpm_ddim import evaluate_ddim_autoddpm
    from dacn.evaluation.autoddpm_lpips import perceptual_provenance
    from dacn.evaluation.ddim_ablation import evaluate_ddim20_30, resolve_ablation_config
    from dacn.evaluation.ddim_inpainting import INPAINT_INDICES, ddim_indices
    from dacn.evaluation.ddpm_memory_fusion import verified_cache
    from dacn.evaluation.sampler_ablation import DDIM_INDICES
    from dacn.training.ddpm_partial import build_diffusion, load_ema_checkpoint
    from dacn.training.ddpm_unconditional_control import SOURCE_RUN, SOURCE_SHA

    if mode not in {"smoke", "full"} or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", evaluation_id
    ):
        raise ValueError("Invalid two-phase DDIM request")
    if experiment not in {"legacy10_10x5", "ddim20_30"}:
        raise ValueError("Unsupported registered DDIM experiment")
    ablation_config = (
        resolve_ablation_config(json.loads(config_json)) if experiment == "ddim20_30" else None
    )
    output = Path(RUN_ROOT) / "evaluations" / evaluation_id
    output.mkdir(parents=True, exist_ok=False)
    source_id = "autoddpm-lpips-full83-t4-20261106T061833Z"
    source_result_sha = "6fb277491308c63b2cc52fa614fc912486266a9c3322f7f099aff850cb51a7e0"
    source_map_sha = "f4b37638c5f6b2aca92ecaa072402847e4f18f371b4c4b178483d50ef1f4bce2"
    manifest = {
        "evaluation_id": evaluation_id,
        "method": "AutoDDPM_two_phase_DDIM10_eta0",
        "mode": mode,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset_repo_id": DATASET,
        "dataset_revision": REVISION,
        "code_revision": code_revision,
        "working_tree_dirty": working_tree_dirty,
        "source_snapshot_sha256": _write_source_snapshot(output),
        "checkpoint_sha256": SOURCE_SHA,
        "baseline_cache_sha256": BASELINE_SHA,
        "index_sha256": INDEX_SHA,
        "source_evaluation": source_id,
        "source_result_sha256": source_result_sha,
        "source_maps_sha256": source_map_sha,
        "phase1_indices_zero_based": list(DDIM_INDICES),
        "phase2_indices_zero_based": list(INPAINT_INDICES),
        "eta": 0,
        "repeats": 5,
        "held_out_test_used": False,
        "gpu": torch.cuda.get_device_name(0),
    }
    if ablation_config is not None:
        manifest.update(
            method="AutoDDPM_DDIM20_DDIM30_single_pass_four_score_ablation",
            config=ablation_config,
            repeats=1,
            phase1_indices_zero_based=list(ddim_indices(300, 20)),
            phase2_indices_zero_based=list(ddim_indices(50, 30)),
        )

    def persist():
        _write_run_manifest(output, manifest)
        run_volume.commit()

    persist()
    started = time.perf_counter()
    try:
        api = HfApi(token=os.environ["HF_TOKEN"])
        if not api.repo_info(MODEL_REPO).private:
            raise ValueError("Private artifact repository required")
        checkpoint = Path(RUN_ROOT) / SOURCE_RUN / "ddpm_epoch_030.pt"
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != SOURCE_SHA:
            raise ValueError("Original EMA30 checkpoint differs")
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
        source = Path(RUN_ROOT) / "evaluations" / source_id
        parent = json.loads((source / "run_manifest.json").read_text())
        if parent["status"] != "completed" or parent["artifact_sync_status"] != "uploaded":
            raise ValueError("Original AutoDDPM evaluation not completed/uploaded")
        for path, checksum in (
            (source / "results.json", source_result_sha),
            (source / "scores" / "private_maps.npz", source_map_sha),
        ):
            if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError("Original AutoDDPM artifacts differ")
        original = json.loads((source / "results.json").read_text())
        with np.load(source / "scores" / "private_maps.npz", allow_pickle=False) as archive:
            for old, current in (
                ("images", "images"),
                ("initials", "reconstructions"),
                ("masks", "masks"),
                ("labels", "labels"),
            ):
                if not np.array_equal(archive[old], cached[current]):
                    raise ValueError("Original AutoDDPM sample alignment differs")
            original_private = {
                k: archive[k].copy() for k in ("predicted_masks", "reconstructions")
            }
        with (validation_dir / "index.csv").open(newline="") as stream:
            records = list(csv.DictReader(stream))
        if [r["sample_id"] for r in records] != data.sample_ids:
            raise ValueError("Group record ordering differs")
        config = torch.load(checkpoint, map_location="cpu", weights_only=False)["config"]
        model = load_ema_checkpoint(checkpoint, config, "cuda")
        diffusion = build_diffusion(config)
        perceptual = lpips.LPIPS(net="squeeze", spatial=True, version="0.1").to("cuda")
        perceptual.eval().requires_grad_(False)
        if perceptual_provenance(perceptual) != original["parameters"]["perceptual"]:
            raise ValueError("LPIPS weights/preprocessing differ")
        torch.cuda.reset_peak_memory_stats()

        def progress(value):
            manifest["progress"] = value
            persist()
            print(json.dumps({k: v for k, v in value.items() if k != "parameters"}), flush=True)

        arguments = {
            "cached": cached,
            "original_private": original_private,
            "original_results": original,
            "groups": np.asarray([r["patient_group_id"] for r in records]),
            "sample_ids": data.sample_ids,
            "model": model,
            "diffusion": diffusion,
            "perceptual_model": perceptual,
            "output_dir": output / "comparison",
            "device": torch.device("cuda"),
            "smoke": mode == "smoke",
            "on_progress": progress,
        }
        if ablation_config is None:
            result = evaluate_ddim_autoddpm(**arguments)
        else:
            arguments.pop("original_private")
            arguments.pop("original_results")
            timing_path = (
                Path(RUN_ROOT)
                / "evaluations"
                / "autoddpm-ddim-smoke-t4-20261106T072722Z"
                / "results.json"
            )
            if (
                hashlib.sha256(timing_path.read_bytes()).hexdigest()
                != "40ce933e3c848f9da8066f2235d5ebe72a7046a96c65e2cb4492b8314340a396"
            ):
                raise ValueError("Recorded ancestral timing reference differs")
            timing_reference = json.loads(timing_path.read_text())[
                "ancestral_two_image_timing_probe"
            ]
            historical_path = (
                Path(RUN_ROOT)
                / "evaluations"
                / "perceptual-audit-full83-t4-20261106T065328Z"
                / "results.json"
            )
            if (
                hashlib.sha256(historical_path.read_bytes()).hexdigest()
                != "224e17448d671bf96de619efe4196a1d0a2992bc07e29915a7c0499df63ba0fa"
            ):
                raise ValueError("Historical LPIPS-only reference differs")
            historical = json.loads(historical_path.read_text())
            result = evaluate_ddim20_30(
                **arguments,
                config=ablation_config,
                timing_reference=timing_reference,
                historical_metrics={
                    "ancestral300_lpips_only": historical["new_metrics"]["lpips_only"],
                    "ancestral_AutoDDPM": original["metrics"],
                },
            )
        result.update(
            total_seconds=time.perf_counter() - started,
            peak_vram_bytes=int(torch.cuda.max_memory_allocated()),
            checkpoint_sha256=SOURCE_SHA,
            source_maps_sha256=source_map_sha,
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
            commit_message=f"Store AutoDDPM DDIM comparison {evaluation_id}",
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


@app.function(
    image=audit_image,
    gpu="T4",
    timeout=60 * 60,
    volumes={DATA_ROOT: dataset_volume, RUN_ROOT: run_volume},
    secrets=[modal.Secret.from_name("huggingface-secret-2")],
)
def run_ddim_ablation_remote(
    evaluation_id: str,
    code_revision: str,
    working_tree_dirty: bool = True,
    mode: str = "smoke",
    config_json: str = "{}",
):
    return run_autoddpm_ddim_remote.local(
        evaluation_id=evaluation_id,
        code_revision=code_revision,
        working_tree_dirty=working_tree_dirty,
        mode=mode,
        experiment="ddim20_30",
        config_json=config_json,
    )

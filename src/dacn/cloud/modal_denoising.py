"""Fixed epoch30 Gaussian denoising validation experiment on Modal T4."""

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import modal

from dacn.cloud.modal_train import (
    DATA_ROOT,
    RUN_ROOT,
    _file_checksums,
    _write_run_manifest,
    _write_source_snapshot,
    app,
    dataset_volume,
    image,
    run_volume,
)


@app.function(
    image=image,
    gpu="T4",
    timeout=3 * 60 * 60,
    volumes={DATA_ROOT: dataset_volume, RUN_ROOT: run_volume},
    secrets=[modal.Secret.from_name("huggingface-secret-2")],
)
def evaluate_remote(
    evaluation_id: str,
    code_revision: str,
    working_tree_dirty: bool = True,
    method: str = "denoising",
):
    """Evaluate fixed model/protocol; publish results only to the private Hub."""
    import torch
    from huggingface_hub import HfApi

    from dacn.data.hub import validate_hf_export
    from dacn.evaluation.contract_audit import evaluate_contract_audit
    from dacn.evaluation.denoising import evaluate_denoising
    from dacn.evaluation.masked_diffusion import evaluate_masked
    from dacn.evaluation.patch_preview import evaluate_patch_preview
    from dacn.evaluation.patch_validation import evaluate_patch_validation
    from dacn.evaluation.sampler_ablation import evaluate_sampler_ablation

    if method not in {
        "denoising",
        "masked_ssim_proxy",
        "contract_audit",
        "patch_preview",
        "patch_validation",
        "patch_one_step",
        "gaussian_one_step",
        "sampler_t300",
    }:
        raise ValueError("Unsupported experiment method")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", evaluation_id):
        raise ValueError("Invalid evaluation_id")
    repo = "dacn-bmad-team/dacn-runs"
    dataset_repo = "dacn-bmad-team/bmad-flair-patient-disjoint-v1"
    revision = "350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7"
    source_run = "ddpm-partial-t4-b8-e30-cont-20261102T034644Z"
    checkpoint = Path(RUN_ROOT) / source_run / "ddpm_epoch_030.pt"
    expected_sha = "d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946"
    if method in {"patch_preview", "patch_validation", "patch_one_step"}:
        source_run = "patch-context-ft5-t4-b8-20261102T155628Z"
        checkpoint = Path(RUN_ROOT) / source_run / "patch_context_best.pt"
        expected_sha = "5d3dbaaf69e80502eab692b545917caf0fcbd609c5864c636a3d1c84098082b3"
    output = Path(RUN_ROOT) / "evaluations" / evaluation_id
    if output.exists():
        raise FileExistsError("Use a fresh evaluation_id")
    output.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "evaluation_id": evaluation_id,
        "status": "started",
        "stage": "validate_source",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "split": "validation",
        "held_out_test_used": False,
        "source_run_id": source_run,
        "checkpoint": checkpoint.name,
        "dataset_repo_id": dataset_repo,
        "dataset_revision": revision,
        "model_repo_id": repo,
        "code_revision": code_revision,
        "working_tree_dirty": working_tree_dirty,
        "timesteps_one_based": [100, 200, 300],
        "noise_repeats": 4,
        "sampling_base_seed": 17,
        "partition_seed": 17,
        "batch_size": 1,
        "method": method,
        "artifacts": {},
    }
    if method == "masked_ssim_proxy":
        manifest.pop("timesteps_one_based")
        manifest.pop("noise_repeats")
        manifest.update(
            {
                "initial_timestep": 200,
                "resampling_timestep": 50,
                "resamples_per_step": 5,
                "mask_proxy": "norm95 absolute difference times clipped local SSIM; threshold0.5; dilation3",
            }
        )
    if method == "contract_audit":
        manifest.pop("noise_repeats")
        manifest["timesteps_one_based"] = [100, 300]
        manifest["normal_samples"] = 4
        manifest["controls"] = ["online_vs_ema_single_step", "large_small_mean_chains_common_noise"]
    if method == "gaussian_one_step":
        manifest.pop("noise_repeats")
        manifest.update(
            {
                "timesteps_one_based": [100],
                "normal_samples": 39,
                "abnormal_samples": 44,
                "reconstruction_mode": "full_image_one_step_x0",
                "model_calls_per_image": 1,
            }
        )
    if method == "sampler_t300":
        manifest.pop("noise_repeats")
        manifest.update(
            {
                "timesteps_one_based": [300],
                "normal_samples": 39,
                "abnormal_samples": 44,
                "samplers": ["one_step_x0", "ddim10_eta0", "cached_ancestral300"],
                "shared_initial_noise": True,
                "ddim_eta": 0,
            }
        )
    if method in {"patch_preview", "patch_validation", "patch_one_step"}:
        manifest.pop("noise_repeats")
        manifest.update(
            {
                "timesteps_one_based": [100],
                "normal_samples": 2 if method == "patch_preview" else 39,
                "abnormal_samples": 2 if method == "patch_preview" else 44,
                "patch_size": 64,
                "stride": 48,
                "patches_per_image": 25,
            }
        )
        manifest["reconstruction_mode"] = (
            "one_step_x0" if method == "patch_one_step" else "ancestral_reverse_chain"
        )

    def persist():
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_run_manifest(output, manifest)
        run_volume.commit()

    persist()
    try:
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != expected_sha:
            raise ValueError("Source checkpoint checksum mismatch")
        source_manifest = json.loads((checkpoint.parent / "run_manifest.json").read_text())
        if source_manifest["dataset_revision"] != revision:
            raise ValueError("Source dataset mismatch")
        manifest["checkpoint_sha256"] = digest
        manifest["source_snapshot_sha256"] = _write_source_snapshot(output)
        manifest["stage"] = "validate_dataset"
        persist()
        cache_key = hashlib.sha256(f"{dataset_repo}@{revision}".encode()).hexdigest()[:20]
        dataset_dir = Path(DATA_ROOT) / cache_key
        validate_hf_export(dataset_dir, verify_checksums=True, allow_hf_cache=True)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this T4 experiment")
        api = HfApi(token=os.environ["HF_TOKEN"])
        if not api.repo_info(repo).private:
            raise ValueError("Refusing public artifact upload")
        manifest["gpu"] = torch.cuda.get_device_name(0)
        manifest["status"] = "running"
        manifest["stage"] = "validation_evaluation"
        persist()
        evaluator = {
            "denoising": evaluate_denoising,
            "masked_ssim_proxy": evaluate_masked,
            "contract_audit": evaluate_contract_audit,
            "patch_preview": evaluate_patch_preview,
            "patch_validation": evaluate_patch_validation,
            "patch_one_step": evaluate_patch_validation,
            "gaussian_one_step": evaluate_patch_validation,
            "sampler_t300": evaluate_sampler_ablation,
        }[method]
        evaluator_kwargs = {}
        if method in {"patch_validation", "patch_one_step", "gaussian_one_step", "sampler_t300"}:

            def progress_callback(progress):
                manifest["progress"] = progress
                persist()

            evaluator_kwargs["on_progress"] = progress_callback
            if method in {"patch_one_step", "gaussian_one_step"}:
                evaluator_kwargs["one_step"] = True
            if method == "gaussian_one_step":
                evaluator_kwargs["gaussian_baseline"] = True
        result = evaluator(
            validation_dir=dataset_dir / "data" / "val",
            checkpoint_path=checkpoint,
            output_dir=output / "results",
            device=torch.device("cuda"),
            **evaluator_kwargs,
        )
        manifest["status"] = "completed"
        manifest["stage"] = "upload_artifacts"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["partition"] = result["partition"]
        manifest["inference_seconds"] = result["inference_seconds"]
        manifest["peak_vram_bytes"] = result["peak_vram_bytes"]
        manifest["artifacts"] = _file_checksums(output)
        persist()
        commit = api.upload_folder(
            repo_id=repo,
            folder_path=str(output),
            path_in_repo=f"evaluations/{evaluation_id}",
            commit_message=f"Store denoising validation {evaluation_id}",
        )
        manifest["artifact_sync_status"] = "uploaded"
        manifest["model_revision"] = commit.oid
        persist()
        print(json.dumps({"metrics": result["metrics"], "model_revision": commit.oid}))
        return {"evaluation_id": evaluation_id, "model_revision": commit.oid}
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = {"exception_type": type(exc).__name__}
        manifest["artifacts"] = _file_checksums(output)
        persist()
        raise

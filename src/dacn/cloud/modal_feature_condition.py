"""Resumable experiment A on T4: bounded smoke or joint training then evaluation."""

import hashlib
import json
import os
import re
from pathlib import Path

import modal

from dacn.cloud.modal_train import (
    DATA_ROOT,
    RUN_ROOT,
    _file_checksums,
    _train_remote_impl,
    _write_run_manifest,
    _write_source_snapshot,
    app,
    base_image,
    dataset_volume,
    run_volume,
)

DATASET = "dacn-bmad-team/bmad-flair-patient-disjoint-v1"
REVISION = "350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7"
MODEL_REPO = "dacn-bmad-team/dacn-runs"
feature_image = base_image.pip_install("matplotlib>=3.8,<4").add_local_python_source("dacn")


@app.function(
    image=feature_image,
    gpu="T4",
    timeout=4 * 60 * 60,
    volumes={DATA_ROOT: dataset_volume, RUN_ROOT: run_volume},
    secrets=[modal.Secret.from_name("huggingface-secret-2")],
)
def run_remote(
    run_id: str,
    config_json: str,
    code_revision: str,
    working_tree_dirty: bool = True,
    mode: str = "train_eval",
):
    import time
    from datetime import datetime, timezone

    import torch
    from huggingface_hub import HfApi

    from dacn.data.shards import ShardDataset
    from dacn.evaluation.feature_condition import evaluate_feature
    from dacn.training.feature_condition import (
        feature_reconstruct,
        load_feature_checkpoint,
        resolved_config,
    )

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_id):
        raise ValueError("Invalid run ID")
    config = json.loads(config_json)
    effective = resolved_config(config)
    spatial = effective["condition_architecture"] == "spatial_imagenet"
    experiment = (
        "spatial_imagenet"
        if spatial
        else "B_pretrained"
        if effective["encoder_pretrain_run_id"]
        else "A_joint"
    )
    if mode == "smoke":
        if (
            effective["epochs"] != 1
            or effective["batch_size"] > 2
            or not effective["max_train_samples"]
            or effective["max_train_samples"] > 32
            or not effective["max_normal_validation_samples"]
            or effective["max_normal_validation_samples"] > 4
        ):
            raise ValueError("Smoke must be bounded to1 epoch,32train,4normal-val,batch2")
    elif (
        mode != "train_eval"
        or effective["epochs"] != 10
        or effective["max_train_samples"] is not None
        or effective["max_normal_validation_samples"] is not None
    ):
        raise ValueError("Feature full run requires10 epochs/full normal data")
    run_dir = Path(RUN_ROOT) / run_id
    previous = (
        json.loads((run_dir / "run_manifest.json").read_text())
        if (run_dir / "run_manifest.json").is_file()
        else None
    )
    if previous:
        if (
            previous["config"] != config
            or previous["code_revision"] != code_revision
            or previous["working_tree_dirty"] != working_tree_dirty
        ):
            raise ValueError("Repeated request differs from saved configuration")
        if _write_source_snapshot(run_dir, persist=False) != previous["source_snapshot_sha256"]:
            raise ValueError("Repeated request source snapshot differs")
    if (
        previous is None
        or previous.get("status") != "completed"
        or previous.get("artifact_sync_status") != "uploaded"
    ):
        result = _train_remote_impl(
            DATASET,
            REVISION,
            MODEL_REPO,
            run_id,
            "dacn.training.feature_condition:train",
            config_json,
            code_revision,
            working_tree_dirty,
            resume=bool(previous),
            continue_from_run_id="",
        )
    else:
        result = {"run_id": run_id, "model_revision": previous["model_revision"]}
    key = hashlib.sha256(f"{DATASET}@{REVISION}".encode()).hexdigest()[:20]
    validation_dir = Path(DATA_ROOT) / key / "data" / "val"
    if mode == "smoke":
        model, diffusion = load_feature_checkpoint(run_dir / "feature_condition_best.pt", "cuda")
        data = ShardDataset(validation_dir, "val", include_masks=False)
        index = int(next(i for i, label in enumerate(data.labels) if label == 0))
        image_tensor = data[index][0][None].to("cuda")
        reconstructed = feature_reconstruct(model, diffusion, image_tensor, seed=17, steps=3)
        if reconstructed.shape != image_tensor.shape or not torch.isfinite(reconstructed).all():
            raise ValueError("Smoke conditional reconstruction failed")
        print(
            json.dumps(
                {"smoke": "passed", "training": result, "finite_conditional_reverse_steps": 3}
            ),
            flush=True,
        )
        return {**result, "smoke": "passed"}
    evaluation_id = f"{run_id}-val"
    output = Path(RUN_ROOT) / "evaluations" / evaluation_id
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.is_file()
        else {
            "schema_version": 1,
            "evaluation_id": evaluation_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "method": f"resnet18_condition_experiment_{experiment}",
            "source_run_id": run_id,
            "checkpoint": "feature_condition_best.pt",
            "dataset_repo_id": DATASET,
            "dataset_revision": REVISION,
            "model_repo_id": MODEL_REPO,
            "code_revision": code_revision,
            "working_tree_dirty": working_tree_dirty,
            "held_out_test_used": False,
            "split": "validation",
            "timesteps_one_based": [300],
            "sampling_base_seed": 17,
            "gpu": torch.cuda.get_device_name(0),
            "progress": {},
        }
    )
    if manifest.get("status") == "completed" and manifest.get("artifact_sync_status") == "uploaded":
        return {**result, "evaluation": manifest["model_revision"]}
    checkpoint = run_dir / "feature_condition_best.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if "checkpoint_sha256" in manifest and manifest["checkpoint_sha256"] != digest:
        raise ValueError("Evaluation checkpoint changed")
    manifest["checkpoint_sha256"] = digest
    manifest["source_snapshot_sha256"] = _write_source_snapshot(output)

    def persist():
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_run_manifest(output, manifest)
        run_volume.commit()

    def progress(values):
        manifest["progress"] = values
        persist()

    manifest["status"], manifest["stage"] = "running", "validation_evaluation"
    persist()
    try:
        api = HfApi(token=os.environ["HF_TOKEN"])
        if not api.repo_info(MODEL_REPO).private:
            raise ValueError("Private model repository required")
        baseline = (
            Path(RUN_ROOT)
            / "evaluations"
            / "ddpm-val-t4b8-e30-t300-20261102T071516Z"
            / "ddpm_epoch_030_t0300_ancestral_reconstructions.npz"
        )
        started = time.perf_counter()
        metrics = evaluate_feature(
            validation_dir=validation_dir,
            checkpoint_path=checkpoint,
            output_dir=output / "results",
            device=torch.device("cuda"),
            baseline_cache=baseline,
            on_progress=progress,
            resume_partial=(output / "results").exists(),
            experiment_A_results=(
                Path(RUN_ROOT)
                / "evaluations"
                / "feature-A-joint-t4-b8-e10-20261104T174552Z-val"
                / "results"
                / "results.json"
            )
            if experiment == "B_pretrained" or spatial
            else None,
            experiment_B_results=(
                Path(RUN_ROOT)
                / "evaluations"
                / "feature-B-pretrain5-joint10-t4-20261105T053132Z-val"
                / "results"
                / "results.json"
            )
            if spatial
            else None,
            experiment_control_results=(
                Path(RUN_ROOT)
                / "evaluations"
                / "ddpm-uncond-val-best36-t300-20261105T094934Z"
                / "results"
                / "results.json"
            )
            if spatial
            else None,
        )
        manifest.update(
            {
                "status": "completed",
                "stage": "upload_artifacts",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "evaluation_seconds": time.perf_counter() - started,
                "partition": metrics["partition"],
                "artifacts": _file_checksums(output),
            }
        )
        persist()
        commit = api.upload_folder(
            repo_id=MODEL_REPO,
            folder_path=str(output),
            path_in_repo=f"evaluations/{evaluation_id}",
            commit_message=f"Store feature-condition experiment {experiment} {run_id}",
        )
        manifest.update({"artifact_sync_status": "uploaded", "model_revision": commit.oid})
        persist()
        print(
            json.dumps(
                {
                    "evaluation_id": evaluation_id,
                    "model_revision": commit.oid,
                    "metrics": metrics["metrics"],
                    "baseline": metrics["baseline_metrics"],
                }
            ),
            flush=True,
        )
        return {**result, "evaluation_id": evaluation_id, "evaluation_revision": commit.oid}
    except Exception as exc:
        manifest.update({"status": "failed", "failure": {"exception_type": type(exc).__name__}})
        persist()
        raise

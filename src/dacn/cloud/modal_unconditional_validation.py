"""Fixed t300 validation of the matched unconditional EMA30 fine-tune control."""

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

TRAIN_RUN = "ddpm-uncond-control-t4-b8-e40-20261105T082502Z"
BEST_SHA = "350791c48fd99babc38c329af5644a4c53a70f6628b6d751e85e3a6cab8cf0f7"


@app.function(
    image=feature_image,
    gpu="T4",
    timeout=2 * 60 * 60,
    volumes={DATA_ROOT: dataset_volume, RUN_ROOT: run_volume},
    secrets=[modal.Secret.from_name("huggingface-secret-2")],
)
def evaluate_remote(
    evaluation_id: str,
    code_revision: str,
    working_tree_dirty: bool = True,
    mode: str = "full",
):
    import numpy as np
    import torch
    from huggingface_hub import HfApi

    from dacn.data.shards import ShardDataset
    from dacn.evaluation.condition_ablation import B_RUN, verified_reference
    from dacn.evaluation.feature_condition import evaluate_feature
    from dacn.evaluation.patch_validation import gaussian_ancestral_with_noisy
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
        raise ValueError("Invalid unconditional validation request")
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
        raise ValueError("Validation resume provenance differs")
    manifest = previous or {
        "schema_version": 1,
        "evaluation_id": evaluation_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "method": "matched_unconditional_finetune_validation",
        "mode": mode,
        "source_run_id": TRAIN_RUN,
        "checkpoint": "ddpm_best.pt",
        "checkpoint_sha256": BEST_SHA,
        "checkpoint_epoch": 36,
        "fine_tune_epoch": 6,
        "dataset_repo_id": DATASET,
        "dataset_revision": REVISION,
        "model_repo_id": MODEL_REPO,
        "source_snapshot_sha256": snapshot,
        "code_revision": code_revision,
        "working_tree_dirty": working_tree_dirty,
        "split": "validation",
        "held_out_test_used": False,
        "timesteps_one_based": [300],
        "sampling_base_seed": 17,
        "gpu": torch.cuda.get_device_name(0),
        "progress": {},
    }
    if manifest.get("status") == "completed" and manifest.get("artifact_sync_status") == "uploaded":
        return {"evaluation_id": evaluation_id, "revision": manifest["model_revision"]}

    def persist():
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_run_manifest(output, manifest)
        run_volume.commit()

    def progress(values):
        manifest["progress"] = values
        persist()

    manifest.update(status="running", stage="verify_training_and_references")
    persist()
    try:
        started = time.perf_counter()
        api = HfApi(token=os.environ["HF_TOKEN"])
        if not api.repo_info(MODEL_REPO).private:
            raise ValueError("Private model repo required")
        training = Path(RUN_ROOT) / TRAIN_RUN
        parent = json.loads((training / "run_manifest.json").read_text())
        if (
            parent.get("status") != "completed"
            or parent.get("artifact_sync_status") != "uploaded"
            or parent["trainer_result"]["best_epoch"] != 36
            or parent["trainer_result"]["optimizer_steps"] != 9380
            or parent["dataset_revision"] != REVISION
            or parent["artifacts"]["ddpm_best.pt"] != BEST_SHA
        ):
            raise ValueError("Unconditional training provenance differs")
        checkpoint = training / "ddpm_best.pt"
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != BEST_SHA:
            raise ValueError("Unconditional best checkpoint differs")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            state["epoch"] != 36
            or state["ema_initialization"]["source_sha256"] != SOURCE_SHA
            or state["ema_initialization"]["encoder_added"] is not False
        ):
            raise ValueError("EMA initialization/checkpoint selection differs")
        target_config = state["config"]
        del state
        key = hashlib.sha256(f"{DATASET}@{REVISION}".encode()).hexdigest()[:20]
        validation_dir = Path(DATA_ROOT) / key / "data" / "val"
        b_directory = Path(RUN_ROOT) / "evaluations" / f"{B_RUN}-val"
        _, paired = verified_reference(
            b_directory, Path(RUN_ROOT) / B_RUN / "feature_condition_best.pt", validation_dir
        )
        a_results = (
            Path(RUN_ROOT)
            / "evaluations"
            / "feature-A-joint-t4-b8-e10-20261104T174552Z-val"
            / "results"
            / "results.json"
        )
        if (
            hashlib.sha256(a_results.read_bytes()).hexdigest()
            != "a8e57ed824c9e6a00404d82296f15fb208eabb9d8d911c06fcd983dfb8b5a962"
        ):
            raise ValueError("Registered A reference differs")
        baseline = (
            Path(RUN_ROOT)
            / "evaluations"
            / "ddpm-val-t4b8-e30-t300-20261102T071516Z"
            / "ddpm_epoch_030_t0300_ancestral_reconstructions.npz"
        )
        if (
            hashlib.sha256(baseline.read_bytes()).hexdigest()
            != "8278ad18543dd87f698c36b510564b9fcd4445d4a1b0ae5d7a859093e2a4e959"
        ):
            raise ValueError("Registered original baseline differs")
        with np.load(baseline, allow_pickle=False) as cache:
            for field in ("images", "masks", "labels"):
                if not np.array_equal(cache[field], paired[field]):
                    raise ValueError("Baseline/B input alignment differs")
            original_predictions = cache["reconstructions"].copy()
        data = ShardDataset(validation_dir, "val", include_masks=True)
        if len(data) != 83:
            raise ValueError("Expected83 validation slices")
        for index in range(len(data)):
            image, mask, label = data[index]
            if (
                not np.array_equal(image[0].numpy(), paired["images"][index])
                or not np.array_equal(mask.numpy(), paired["masks"][index])
                or int(label) != int(paired["labels"][index])
            ):
                raise ValueError("Validation and cached reference differ")
        manifest["stage"] = "bounded_preflight"
        persist()
        original_path = Path(RUN_ROOT) / SOURCE_RUN / "ddpm_epoch_030.pt"
        if hashlib.sha256(original_path.read_bytes()).hexdigest() != SOURCE_SHA:
            raise ValueError("Original EMA30 source differs")
        original_config = torch.load(original_path, map_location="cpu", weights_only=False)[
            "config"
        ]
        model = load_ema_checkpoint(original_path, original_config, "cuda")
        diffusion = build_diffusion(original_config)
        errors = []
        for label in (0, 1):
            index = int(np.flatnonzero(data.labels == label)[0])
            image = data[index][0][None].to("cuda")
            value, noisy = gaussian_ancestral_with_noisy(
                model,
                diffusion,
                image,
                steps=300,
                seed=derive_sampling_seed(17, data.sample_ids[index]),
            )
            error = float(np.abs(value[0, 0].cpu().numpy() - original_predictions[index]).max())
            if error > 1e-6 or not np.array_equal(
                noisy[0, 0].cpu().numpy(), paired["noisy_images"][index]
            ):
                raise ValueError("Original baseline replay/forward noise differs")
            errors.append(error)
        del model
        model = load_ema_checkpoint(checkpoint, target_config, "cuda")
        diffusion = build_diffusion(target_config)
        for label in (0, 1):
            index = int(np.flatnonzero(data.labels == label)[0])
            image = data[index][0][None].to("cuda")
            value, noisy = gaussian_ancestral_with_noisy(
                model,
                diffusion,
                image,
                steps=3,
                seed=derive_sampling_seed(17, data.sample_ids[index]),
            )
            if value.shape != image.shape or not torch.isfinite(value).all():
                raise ValueError("Target36 EMA reconstruction smoke failed")
        del model, diffusion, image, value, noisy
        torch.cuda.empty_cache()
        manifest["preflight"] = {
            "status": "passed",
            "baseline_replay_max_errors": errors,
            "target_reverse_steps": 3,
            "target_samples": 2,
        }
        persist()
        if mode == "full":
            manifest["stage"] = "validation_evaluation"
            persist()
            result = evaluate_feature(
                validation_dir=validation_dir,
                checkpoint_path=checkpoint,
                output_dir=output / "results",
                device=torch.device("cuda"),
                baseline_cache=baseline,
                on_progress=progress,
                resume_partial=(output / "results").exists(),
                unconditional_finetune=True,
                experiment_A_results=a_results,
                experiment_B_results=b_directory / "results" / "results.json",
            )
            with np.load(
                output / "results" / "private_reconstructions.npz", allow_pickle=False
            ) as cache:
                for field in ("images", "masks", "labels", "noisy_images"):
                    if not np.array_equal(cache[field], paired[field]):
                        raise ValueError("Control/A/B paired inputs or noise differ")
            result.update(
                fine_tune_epoch=6,
                source_epoch=30,
                preflight=manifest["preflight"],
                selection_criterion="normal_validation_epsilon_mse",
                fine_tune_optimizer_steps=9380,
            )
            (output / "results" / "results.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n"
            )
        else:
            (output / "smoke_result.json").write_text(
                json.dumps(manifest["preflight"], indent=2) + "\n"
            )
        manifest.update(
            status="completed",
            stage="upload_artifacts",
            completed_at=datetime.now(timezone.utc).isoformat(),
            evaluation_seconds=time.perf_counter() - started,
            artifacts=_file_checksums(output),
        )
        persist()
        commit = api.upload_folder(
            repo_id=MODEL_REPO,
            folder_path=str(output),
            path_in_repo=f"evaluations/{evaluation_id}",
            commit_message=f"Store unconditional fine-tune validation {evaluation_id}",
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

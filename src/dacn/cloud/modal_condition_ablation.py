"""Fixed B EMA condition intervention on T4; no training or timestep search."""

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
    from dacn.evaluation.condition_ablation import (
        B_RUN,
        B_SHA,
        ConditionIntervention,
        compare_reconstructions,
        donor_index,
        verified_reference,
    )
    from dacn.evaluation.feature_condition import make_figure
    from dacn.evaluation.patch_validation import evaluate_patch_validation
    from dacn.training.ddpm_partial import derive_sampling_seed
    from dacn.training.feature_condition import feature_reconstruct, load_feature_checkpoint

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", evaluation_id) or mode not in {
        "smoke",
        "full",
    }:
        raise ValueError("Invalid condition-ablation request")
    output = Path(RUN_ROOT) / "evaluations" / evaluation_id
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    snapshot = _write_source_snapshot(output, persist=previous is None)
    if previous and (
        previous["source_snapshot_sha256"] != snapshot
        or previous["code_revision"] != code_revision
        or previous["working_tree_dirty"] != working_tree_dirty
        or previous["mode"] != mode
    ):
        raise ValueError("Ablation resume provenance differs")
    manifest = previous or {
        "schema_version": 1,
        "evaluation_id": evaluation_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "method": "B_EMA_condition_intervention",
        "mode": mode,
        "dataset_repo_id": DATASET,
        "dataset_revision": REVISION,
        "checkpoint_sha256": B_SHA,
        "source_run_id": B_RUN,
        "source_snapshot_sha256": snapshot,
        "code_revision": code_revision,
        "working_tree_dirty": working_tree_dirty,
        "held_out_test_used": False,
        "split": "validation",
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

    manifest.update(status="running", stage="verify_reference")
    persist()
    try:
        started = time.perf_counter()
        api = HfApi(token=os.environ["HF_TOKEN"])
        if not api.repo_info(MODEL_REPO).private:
            raise ValueError("Private model repository required")
        key = hashlib.sha256(f"{DATASET}@{REVISION}".encode()).hexdigest()[:20]
        data_root = Path(DATA_ROOT) / key / "data"
        validation_dir, train_dir = data_root / "val", data_root / "train"
        checkpoint = Path(RUN_ROOT) / B_RUN / "feature_condition_best.pt"
        reference_dir = Path(RUN_ROOT) / "evaluations" / f"{B_RUN}-val"
        reference, cached = verified_reference(reference_dir, checkpoint, validation_dir)
        data = ShardDataset(validation_dir, "val", include_masks=True)
        donors = ShardDataset(train_dir, "train", include_masks=False)
        if len(data) != 83 or len(donors) != 7500 or np.any(donors.labels != 0):
            raise ValueError("Unexpected validation/normal donor set")
        for index in range(len(data)):
            image, mask, label = data[index]
            if (
                not np.array_equal(image[0].numpy(), cached["images"][index])
                or not np.array_equal(mask.numpy(), cached["masks"][index])
                or int(label) != int(cached["labels"][index])
            ):
                raise ValueError("Reference arrays and validation differ")
        # Bounded preflight: two cached300-step replays (one per class), then
        # finite three-step zero/donor controls. Abort before full job on mismatch.
        manifest["stage"] = "bounded_preflight"
        persist()
        model, diffusion = load_feature_checkpoint(checkpoint, "cuda")
        replay_errors = []
        for label in (0, 1):
            index = int(np.flatnonzero(data.labels == label)[0])
            image = data[index][0][None].to("cuda")
            seed = derive_sampling_seed(17, data.sample_ids[index])
            value, noisy = feature_reconstruct(
                model, diffusion, image, seed=seed, steps=300, return_noisy=True
            )
            error = float(
                np.abs(value[0, 0].cpu().numpy() - cached["reconstructions"][index]).max()
            )
            if error > 1e-6 or not np.array_equal(
                noisy[0, 0].cpu().numpy(), cached["noisy_images"][index]
            ):
                raise ValueError("Reference replay/noise mismatch")
            replay_errors.append(error)
            donor = donors[donor_index(data.sample_ids[index], len(donors))][0][None].to("cuda")
            for condition_mode in ("zero", "normal_donor"):
                modified = ConditionIntervention(model, condition_mode, donor)
                value = feature_reconstruct(modified, diffusion, image, seed=seed, steps=3)
                if value.shape != image.shape or not torch.isfinite(value).all():
                    raise ValueError("Condition intervention smoke failed")
        del model, diffusion, modified, value, noisy, image, donor
        torch.cuda.empty_cache()
        manifest["preflight"] = {
            "status": "passed",
            "reference_max_absolute_errors": replay_errors,
            "reverse_steps_per_control": 3,
        }
        persist()
        result = {
            "schema_version": 1,
            "checkpoint_sha256": B_SHA,
            "checkpoint_epoch": 6,
            "ema": True,
            "mode": mode,
            "split": "validation",
            "held_out_test_used": False,
            "sampling_base_seed": 17,
            "timesteps_one_based": [300],
            "preflight": manifest["preflight"],
            "metrics": {"self": reference["metrics"]} if mode == "full" else {},
            "partition": reference["partition"],
            "reconstruction_change": {},
            "protocol": "Zero bypasses all condition injection, including bias. Normal donor from train, deterministic UUID seed77 mapping. Same evaluated image/noise/reverse RNG; no mask as model input. Thresholds independently fitted on40calibration, frozen on43group holdout; all83 descriptive only. One exploratory seed, no training/test access.",
        }
        if mode == "full":
            for condition_mode in ("zero", "normal_donor"):
                manifest["stage"] = f"evaluate_{condition_mode}"
                persist()

                def progress(values, current=condition_mode):
                    manifest["progress"][current] = values
                    persist()

                directory = output / condition_mode
                measured = evaluate_patch_validation(
                    validation_dir=validation_dir,
                    checkpoint_path=checkpoint,
                    output_dir=directory,
                    device=torch.device("cuda"),
                    feature_condition=True,
                    starting_timestep=300,
                    condition_mode=condition_mode,
                    donor_train_dir=train_dir,
                    resume_partial=directory.exists(),
                    on_progress=progress,
                )
                result["metrics"][condition_mode] = measured["metrics"]
                result.setdefault("runtime", {})[condition_mode] = {
                    k: measured[k] for k in ("inference_seconds", "peak_vram_bytes")
                }
                with np.load(
                    directory / "private_reconstructions.npz", allow_pickle=False
                ) as cache:
                    for field in ("images", "masks", "labels", "noisy_images"):
                        if not np.array_equal(cache[field], cached[field]):
                            raise ValueError("Paired condition input/noise differs")
                    predictions = cache["reconstructions"].copy()
                result["reconstruction_change"][condition_mode] = compare_reconstructions(
                    cached["reconstructions"], predictions, data.labels
                )
                make_figure(
                    cached["images"],
                    cached["noisy_images"],
                    predictions,
                    cached["masks"],
                    cached["labels"],
                    directory / "private_examples.png",
                )
        result["samples"] = 83 if mode == "full" else 2
        result["total_seconds"] = time.perf_counter() - started
        (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
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
            commit_message=f"Store B condition intervention {evaluation_id}",
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

"""Pretrain B's encoder, independently fine-tune from Gaussian EMA30, evaluate."""

import hashlib
import json
import re
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
    run_remote,
    run_volume,
)
from dacn.cloud.modal_train import _train_remote_impl, _write_source_snapshot


@app.function(
    image=feature_image,
    gpu="T4",
    timeout=4 * 60 * 60,
    volumes={DATA_ROOT: dataset_volume, RUN_ROOT: run_volume},
    secrets=[modal.Secret.from_name("huggingface-secret-2")],
)
def run_b_remote(
    run_id: str,
    pretrain_config_json: str,
    finetune_config_json: str,
    code_revision: str,
    working_tree_dirty: bool = True,
    mode: str = "train_eval",
):
    from dacn.training.encoder_pretrain import resolved_config
    from dacn.training.feature_condition import resolved_config as resolve_finetune

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,55}", run_id):
        raise ValueError("Invalid B run ID")
    pretrain, fine = json.loads(pretrain_config_json), json.loads(finetune_config_json)
    effective, fine_effective = resolved_config(pretrain), resolve_finetune(fine)
    if fine_effective["encoder_pretrain_run_id"]:
        raise ValueError("B wrapper binds the newly produced encoder itself")
    for key in ("source_run_id", "source_sha256", "seed", "batch_size"):
        if effective[key] != fine_effective[key]:
            raise ValueError(f"Pretrain/fine-tune {key} differs")
    if mode == "smoke":
        for config in (effective, fine_effective):
            if (
                config["epochs"] != 1
                or config["batch_size"] > 2
                or not config["max_train_samples"]
                or config["max_train_samples"] > 32
                or not config["max_normal_validation_samples"]
                or config["max_normal_validation_samples"] > 4
            ):
                raise ValueError("B smoke requires1epoch/32train/4normal-val/batch2 maximum")
    elif (
        mode != "train_eval"
        or effective["epochs"] != 5
        or fine_effective["epochs"] != 10
        or any(
            config[key] is not None
            for config in (effective, fine_effective)
            for key in ("max_train_samples", "max_normal_validation_samples")
        )
    ):
        raise ValueError("Full B requires5pretrain+10joint epochs on full normal data")
    encoder_run_id = f"{run_id}-enc"
    encoder_dir = Path(RUN_ROOT) / encoder_run_id
    manifest_path = encoder_dir / "run_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and (
        previous["config"] != pretrain
        or previous["code_revision"] != code_revision
        or previous["working_tree_dirty"] != working_tree_dirty
        or _write_source_snapshot(encoder_dir, persist=False) != previous["source_snapshot_sha256"]
    ):
        raise ValueError("B pretraining repeat differs from saved provenance")
    if (
        previous is None
        or previous.get("status") != "completed"
        or previous.get("artifact_sync_status") != "uploaded"
    ):
        pretrain_result = _train_remote_impl(
            DATASET,
            REVISION,
            MODEL_REPO,
            encoder_run_id,
            "dacn.training.encoder_pretrain:train",
            pretrain_config_json,
            code_revision,
            working_tree_dirty,
            resume=bool(previous),
            continue_from_run_id="",
        )
    else:
        pretrain_result = {"run_id": encoder_run_id, "model_revision": previous["model_revision"]}
    fine.update(
        encoder_pretrain_run_id=encoder_run_id,
        encoder_pretrain_sha256=hashlib.sha256(
            (encoder_dir / "encoder_pretrain_best.pt").read_bytes()
        ).hexdigest(),
    )
    print(
        json.dumps({"encoder_ready": pretrain_result, "next": "fresh_EMA30_joint_finetune"}),
        flush=True,
    )
    # Reuse this T4 container; .local does not reserve another GPU.
    result = run_remote.local(
        run_id=run_id,
        config_json=json.dumps(fine),
        code_revision=code_revision,
        working_tree_dirty=working_tree_dirty,
        mode=mode,
    )
    return {"encoder_pretraining": pretrain_result, "joint_and_validation": result}

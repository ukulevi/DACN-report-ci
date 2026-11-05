"""Original U-Net EMA30 fine-tuning control for A/B; no encoder is instantiated."""

from pathlib import Path

from dacn.training import ddpm_partial

SOURCE_RUN = "ddpm-partial-t4-b8-e30-cont-20261102T034644Z"
SOURCE_SHA = "d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946"


def train(
    *,
    train_dir,
    validation_dir,
    output_dir,
    config,
    device,
    on_checkpoint=None,
    resume_from_checkpoint=None,
):
    resolved = ddpm_partial._resolved_config(config)
    if (
        resolved["epochs"] not in {31, 40}
        or resolved["learning_rate"] != 1e-5
        or resolved["ema_decay"] != 0.9992
        or resolved["seed"] != 17
        or resolved["weight_decay"] != 0
        or resolved["gradient_clip_norm"] != 1
        or resolved["noise_type"] != "gaussian"
    ):
        raise ValueError("Unconditional control must match A/B fine-tuning settings")
    if resolved["epochs"] == 31:
        if (
            resolved["batch_size"] > 2
            or not resolved["max_train_samples"]
            or resolved["max_train_samples"] > 32
            or not resolved["max_validation_samples"]
            or resolved["max_validation_samples"] > 8
        ):
            raise ValueError("Control smoke must be bounded")
    elif (
        resolved["batch_size"] != 8
        or resolved["validation_batch_size"] != 8
        or resolved["max_train_samples"] is not None
        or resolved["max_validation_samples"] is not None
    ):
        raise ValueError("Full control uses all7500normal train/39normal-val and batch8")
    result = ddpm_partial.train(
        train_dir=train_dir,
        validation_dir=validation_dir,
        output_dir=output_dir,
        config=config,
        device=device,
        on_checkpoint=on_checkpoint,
        resume_from_checkpoint=resume_from_checkpoint,
        initial_checkpoint=Path(output_dir).parent / SOURCE_RUN / "ddpm_epoch_030.pt",
        initial_checkpoint_sha256=SOURCE_SHA,
    )
    return {
        **result,
        "experiment": "unconditional_EMA30_finetune_control",
        "source_run_id": SOURCE_RUN,
        "fine_tune_epochs_completed": result["epochs_completed"] - 30,
    }

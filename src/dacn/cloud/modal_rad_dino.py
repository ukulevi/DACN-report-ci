"""T4 frozen RAD-DINO scoring; immutable final DDIM20+30 cache, no diffusion."""

import csv
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

from dacn.cloud.modal_calibrated_lpips import SOURCE_EVALUATION, SOURCE_MAPS_SHA, SOURCE_RESULTS_SHA
from dacn.cloud.modal_feature_condition import DATASET, MODEL_REPO, REVISION
from dacn.cloud.modal_train import (
    DATA_ROOT,
    RUN_ROOT,
    _file_checksums,
    _write_run_manifest,
    _write_source_snapshot,
    app,
    base_image,
    dataset_volume,
    run_volume,
)
from dacn.cloud.modal_uncertainty import INDEX_SHA
from dacn.evaluation.rad_dino import ENCODER_REPO, ENCODER_REVISION

# Pin the tested transformers API; official Dinov2Model, no remote Python execution.
rad_image = (
    base_image.pip_install(
        "transformers==4.46.3", "torchvision", "pillow>=10", "safetensors>=0.4,<1"
    )
    .run_commands(
        'python -c "from huggingface_hub import snapshot_download; '
        f"snapshot_download('{ENCODER_REPO}', revision='{ENCODER_REVISION}', "
        "local_dir='/opt/rad-dino', allow_patterns=['model.safetensors', 'config.json', "
        "'preprocessor_config.json', 'LICENSE', 'README.md'])\""
    )
    .add_local_python_source("dacn")
)


@app.function(
    image=rad_image,
    gpu="T4",
    timeout=30 * 60,
    volumes={DATA_ROOT: dataset_volume, RUN_ROOT: run_volume},
    secrets=[modal.Secret.from_name("huggingface-secret-2")],
)
def run_rad_dino_remote(
    evaluation_id: str, code_revision: str, working_tree_dirty: bool = True, mode: str = "smoke"
):
    import numpy as np
    import torch
    import transformers
    from huggingface_hub import HfApi
    from transformers import AutoImageProcessor, AutoModel

    from dacn.data.shards import ShardDataset
    from dacn.evaluation.rad_dino import evaluate_rad_dino

    if mode not in {"smoke", "full"} or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", evaluation_id
    ):
        raise ValueError("Invalid RAD-DINO request")
    output = Path(RUN_ROOT) / "evaluations" / evaluation_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "evaluation_id": evaluation_id,
        "status": "running",
        "mode": mode,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset_repo_id": DATASET,
        "dataset_revision": REVISION,
        "code_revision": code_revision,
        "working_tree_dirty": working_tree_dirty,
        "source_snapshot_sha256": _write_source_snapshot(output),
        "source_evaluation": SOURCE_EVALUATION,
        "source_maps_sha256": SOURCE_MAPS_SHA,
        "source_results_sha256": SOURCE_RESULTS_SHA,
        "index_sha256": INDEX_SHA,
        "encoder_repo": ENCODER_REPO,
        "encoder_revision": ENCODER_REVISION,
        "encoder_artifacts": _file_checksums(Path("/opt/rad-dino")),
        "held_out_test_used": False,
        "diffusion_model_calls": 0,
        "gpu": torch.cuda.get_device_name(0),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        },
    }

    def persist():
        _write_run_manifest(output, manifest)
        run_volume.commit()

    persist()
    started = time.perf_counter()
    try:
        api = HfApi(token=os.environ["HF_TOKEN"])
        if not api.repo_info(MODEL_REPO).private:
            raise ValueError("Private output required")
        source = Path(RUN_ROOT) / "evaluations" / SOURCE_EVALUATION
        parent = json.loads((source / "run_manifest.json").read_text())
        if parent["status"] != "completed" or parent["artifact_sync_status"] != "uploaded":
            raise ValueError("Completed source required")
        for path, sha in (
            (source / "results.json", SOURCE_RESULTS_SHA),
            (source / "comparison" / "private_ablation.npz", SOURCE_MAPS_SHA),
        ):
            if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise ValueError("Source checksum differs")
        with np.load(source / "comparison" / "private_ablation.npz", allow_pickle=False) as archive:
            cached = {
                k: archive[k].copy()
                for k in ("images", "finals", "masks", "labels", "autoddpm_lpips")
            }
        key = hashlib.sha256(f"{DATASET}@{REVISION}".encode()).hexdigest()[:20]
        validation_dir = Path(DATA_ROOT) / key / "data" / "val"
        if hashlib.sha256((validation_dir / "index.csv").read_bytes()).hexdigest() != INDEX_SHA:
            raise ValueError("Validation index differs")
        data = ShardDataset(validation_dir, "val", include_masks=True)
        if len(data) != 83:
            raise ValueError("Expected83validation samples")
        for index in range(83):
            image, mask, label = data[index]
            if (
                not np.array_equal(cached["images"][index], image[0].numpy())
                or not np.array_equal(cached["masks"][index], mask.numpy())
                or int(cached["labels"][index]) != int(label)
            ):
                raise ValueError("Cache ordering differs")
        with (validation_dir / "index.csv").open(newline="") as stream:
            records = list(csv.DictReader(stream))
        if [r["sample_id"] for r in records] != data.sample_ids:
            raise ValueError("Group order differs")
        torch.manual_seed(17)
        torch.backends.cuda.matmul.allow_tf32 = False
        model, loading = AutoModel.from_pretrained(
            "/opt/rad-dino",
            local_files_only=True,
            output_loading_info=True,
            attn_implementation="eager",
        )
        if any(
            loading[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        ):
            raise ValueError("Encoder weights not strictly matched")
        if (model.config.hidden_size, model.config.num_hidden_layers, model.config.patch_size) != (
            768,
            12,
            14,
        ):
            raise ValueError("Unexpected encoder architecture")
        model = model.eval().requires_grad_(False).to("cuda")
        processor = AutoImageProcessor.from_pretrained("/opt/rad-dino", local_files_only=True)
        manifest["processor"] = processor.to_dict()
        manifest["encoder_loading"] = loading
        torch.cuda.reset_peak_memory_stats()

        def progress(stage):
            manifest["stage"] = stage
            persist()
            print(json.dumps({"stage": stage}), flush=True)

        result = evaluate_rad_dino(
            cached=cached,
            groups=np.asarray([r["patient_group_id"] for r in records]),
            model=model,
            processor=processor,
            output_dir=output / "scores",
            smoke=mode == "smoke",
            progress=progress,
        )
        result.update(
            encoder_repo=ENCODER_REPO,
            encoder_revision=ENCODER_REVISION,
            total_seconds=time.perf_counter() - started,
            peak_vram_bytes=int(torch.cuda.max_memory_allocated()),
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
            commit_message=f"Store frozen RAD-DINO {evaluation_id}",
        )
        manifest.update(artifact_sync_status="uploaded", model_revision=commit.oid)
        persist()
        print(
            json.dumps(
                {"status": "completed", "evaluation_id": evaluation_id, "revision": commit.oid}
            ),
            flush=True,
        )
        return {"evaluation_id": evaluation_id, "revision": commit.oid}
    except Exception as exc:
        manifest.update(status="failed", failure={"exception_type": type(exc).__name__})
        persist()
        raise

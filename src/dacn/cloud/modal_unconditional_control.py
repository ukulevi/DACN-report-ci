"""Resumable T4 unconditional EMA30 fine-tuning control; training only."""

import json
import re
from pathlib import Path

import modal

from dacn.cloud.modal_train import (
    DATA_ROOT,
    RUN_ROOT,
    _train_remote_impl,
    _write_source_snapshot,
    app,
    dataset_volume,
    image,
    run_volume,
)


@app.function(
    image=image,
    gpu="T4",
    timeout=4 * 60 * 60,
    volumes={DATA_ROOT: dataset_volume, RUN_ROOT: run_volume},
    secrets=[modal.Secret.from_name("huggingface-secret-2")],
)
def run_remote(run_id: str, config_json: str, code_revision: str, working_tree_dirty: bool = True):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_id):
        raise ValueError("Invalid unconditional-control run ID")
    output = Path(RUN_ROOT) / run_id
    manifest_path = output / "run_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and (
        previous["config"] != json.loads(config_json)
        or previous["code_revision"] != code_revision
        or previous["working_tree_dirty"] != working_tree_dirty
        or _write_source_snapshot(output, persist=False) != previous["source_snapshot_sha256"]
    ):
        raise ValueError("Unconditional control repeat differs from saved provenance")
    if (
        previous
        and previous.get("status") == "completed"
        and previous.get("artifact_sync_status") == "uploaded"
    ):
        return {"run_id": run_id, "model_revision": previous["model_revision"]}
    return _train_remote_impl(
        "dacn-bmad-team/bmad-flair-patient-disjoint-v1",
        "350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7",
        "dacn-bmad-team/dacn-runs",
        run_id,
        "dacn.training.ddpm_unconditional_control:train",
        config_json,
        code_revision,
        working_tree_dirty,
        resume=bool(previous),
        continue_from_run_id="",
    )

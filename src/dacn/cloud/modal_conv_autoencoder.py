"""Shared lifecycle for matched normal-only ConvAE/ConvVAE baselines on T4."""

import json
import subprocess
from pathlib import Path

import modal

from dacn.cloud.modal_train import (
    DATA_ROOT,
    RUN_ROOT,
    _train_remote_impl,
    base_image,
    dataset_volume,
    run_volume,
)

app = modal.App("dacn-conv-ae-vae")
image = base_image.add_local_python_source("dacn")


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=8192,
    timeout=8 * 3600,
    volumes={DATA_ROOT: dataset_volume, RUN_ROOT: run_volume},
    secrets=[modal.Secret.from_name("huggingface-secret-2")],
)
def run_remote(
    run_id: str,
    config_json: str,
    code_revision: str,
    working_tree_dirty: bool = True,
    resume: bool = False,
):
    from dacn.training.conv_autoencoder import resolve_config

    resolved = resolve_config(json.loads(config_json))
    return _train_remote_impl(
        "dacn-bmad-team/bmad-flair-patient-disjoint-v1",
        "350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7",
        "dacn-bmad-team/dacn-runs",
        run_id,
        "dacn.training.conv_autoencoder:train",
        json.dumps(resolved),
        code_revision,
        working_tree_dirty,
        resume=resume,
    )


@app.local_entrypoint()
def main(
    run_id: str, config_path: str, output: str, submit_only: bool = False, resume: bool = False
):
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    config = Path(config_path).read_text()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    call = run_remote.spawn(run_id, config, revision, dirty, resume)
    submission = {
        "run_id": run_id,
        "config": json.loads(config),
        "app_id": app.app_id,
        "call_id": call.object_id,
    }
    (folder / "submission.json").write_text(json.dumps(submission, indent=2) + "\n")
    print(json.dumps(submission), flush=True)
    if submit_only:
        return
    result = call.get()
    (folder / "result.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result), flush=True)

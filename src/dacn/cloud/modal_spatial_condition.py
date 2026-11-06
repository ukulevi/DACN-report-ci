"""Spatial ImageNet conditioning pilot with cached official weights and T4 lifecycle."""

import shlex

import modal

from dacn.cloud.modal_feature_condition import run_remote as feature_remote
from dacn.cloud.modal_train import DATA_ROOT, RUN_ROOT, app, base_image, dataset_volume, run_volume

DOWNLOAD_SCRIPT = (
    "import hashlib, urllib.request; from pathlib import Path; "
    "p=Path('/opt/dacn/pretrained/resnet18-f37072fd.pth'); "
    "p.parent.mkdir(parents=True, exist_ok=True); "
    "urllib.request.urlretrieve('https://download.pytorch.org/models/resnet18-f37072fd.pth',p); "
    "assert hashlib.sha256(p.read_bytes()).hexdigest() == "
    "'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'"
)


spatial_image = (
    base_image.pip_install("matplotlib>=3.8,<4")
    .run_commands("python -c " + shlex.quote(DOWNLOAD_SCRIPT))
    .add_local_python_source("dacn")
)


@app.function(
    image=spatial_image,
    gpu="T4",
    timeout=4 * 60 * 60,
    volumes={DATA_ROOT: dataset_volume, RUN_ROOT: run_volume},
    secrets=[modal.Secret.from_name("huggingface-secret-2")],
)
def run_spatial_remote(
    run_id: str,
    config_json: str,
    code_revision: str,
    working_tree_dirty: bool = True,
    mode: str = "train_eval",
):
    import json

    if json.loads(config_json).get("condition_architecture") != "spatial_imagenet":
        raise ValueError("Spatial entrypoint requires spatial ImageNet config")
    return feature_remote.local(
        run_id=run_id,
        config_json=config_json,
        code_revision=code_revision,
        working_tree_dirty=working_tree_dirty,
        mode=mode,
    )

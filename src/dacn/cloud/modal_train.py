"""Run a DACN training callable on Modal and publish its artifacts to the Hub.

Training implementations provide ``train(train_dir, validation_dir, output_dir,
config, device)`` under ``dacn.training.*``. This module supplies the storage and
execution boundary and withholds the held-out test path from the trainer.
"""

import hashlib
import importlib
import json
import os
import platform
import re
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import modal

app = modal.App("dacn-bmad-flair")

DATA_ROOT = "/mnt/dacn/datasets"
RUN_ROOT = "/mnt/dacn/runs"
dataset_volume = modal.Volume.from_name("dacn-bmad-datasets", create_if_missing=True)
run_volume = modal.Volume.from_name("dacn-bmad-runs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub>=0.28,<3",
        "numpy>=1.24,<3",
        "scikit-learn>=1.3,<2",
        "PyYAML>=6,<7",
        "torch>=2.2",
        "tqdm>=4.66",
    )
    .add_local_python_source("dacn")
)


def _json_safe(value):
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if re.search(r"token|secret|password|credential", str(key), re.IGNORECASE)
                else _json_safe(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Training result is not JSON serializable: {type(value).__name__}")


def _reject_patient_identifiers(value) -> None:
    forbidden = {
        "patient_id",
        "patient_ids",
        "patient_identifier",
        "patient_identifiers",
        "sample_id",
        "source_path",
        "image_path",
        "mask_path",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in forbidden:
                raise ValueError("Run metadata must not contain patient IDs or source-level paths")
            _reject_patient_identifiers(item)
    elif isinstance(value, list):
        for item in value:
            _reject_patient_identifiers(item)


def _file_checksums(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Refusing to publish symlinked run output: {path.name}")
        if path.is_file():
            relative = path.relative_to(root)
            if any("patient" in part.lower() for part in relative.parts):
                raise ValueError(f"Refusing to publish patient-level run output: {relative}")
            checksum = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(block)
            digest = checksum.hexdigest()
            result[path.relative_to(root).as_posix()] = digest
    return result


def _write_source_snapshot(output_dir: Path) -> str:
    """Archive the exact DACN Python sources loaded by this Modal run."""
    package_file = importlib.import_module("dacn").__file__
    if package_file is None:
        raise RuntimeError("Cannot locate DACN package sources for the run snapshot")
    package_root = Path(package_file).resolve().parent
    source_files = sorted(package_root.rglob("*.py"))
    if not source_files:
        raise RuntimeError("DACN package contains no Python source files to snapshot")

    snapshot_path = output_dir / "source_snapshot.zip"
    with zipfile.ZipFile(snapshot_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source_file in source_files:
            if source_file.is_symlink():
                raise ValueError(f"Refusing to snapshot symlinked source: {source_file.name}")
            relative = source_file.relative_to(package_root)
            entry = zipfile.ZipInfo(f"dacn/{relative.as_posix()}", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, source_file.read_bytes())
    return hashlib.sha256(snapshot_path.read_bytes()).hexdigest()


@app.function(
    image=image,
    gpu="T4",
    timeout=6 * 60 * 60,
    volumes={DATA_ROOT: dataset_volume, RUN_ROOT: run_volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def train_remote(
    dataset_repo_id: str,
    dataset_revision: str,
    model_repo_id: str,
    run_id: str,
    trainer: str,
    config_json: str,
    code_revision: str,
    working_tree_dirty: bool,
) -> dict:
    """Fetch a pinned dataset, invoke one DACN trainer, and persist its outputs."""
    from importlib.metadata import version as package_version

    import torch
    from huggingface_hub import HfApi, snapshot_download

    from dacn.cloud.config import parse_config_json
    from dacn.data.hub import validate_hf_export

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_id):
        raise ValueError("run_id must be 1-64 path-safe characters")
    if not trainer.startswith("dacn.training.") or ":" not in trainer:
        raise ValueError("trainer must be a callable in dacn.training, written as module:function")
    if not torch.cuda.is_available():
        raise RuntimeError("Modal function started without a CUDA GPU")
    config = parse_config_json(config_json)
    module_name, function_name = trainer.split(":", maxsplit=1)
    training_function = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(training_function):
        raise TypeError(f"Trainer callable does not exist: {trainer}")

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Modal Secret 'huggingface' must contain a write-capable HF_TOKEN")
    api = HfApi(token=token)

    # Create the destination before spending GPU time; never silently publish to a public repo.
    api.create_repo(model_repo_id, repo_type="model", private=True, exist_ok=True)
    model_info = api.repo_info(model_repo_id, repo_type="model")
    if not model_info.private:
        raise ValueError("Refusing to upload run artifacts to a public model repository")

    source_info = api.repo_info(
        dataset_repo_id,
        repo_type="dataset",
        revision=dataset_revision,
    )
    pinned_revision = source_info.sha
    cache_key = hashlib.sha256(f"{dataset_repo_id}@{pinned_revision}".encode()).hexdigest()[:20]
    dataset_dir = Path(DATA_ROOT) / cache_key
    dataset_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=dataset_repo_id,
        repo_type="dataset",
        revision=pinned_revision,
        token=token,
        local_dir=str(dataset_dir),
    )
    dataset_metadata = validate_hf_export(
        dataset_dir,
        verify_checksums=True,
        allow_hf_cache=True,
    )
    dataset_volume.commit()

    run_dir = Path(RUN_ROOT) / run_id
    if run_dir.exists():
        raise ValueError(f"Run output already exists: {run_id}; choose a new run_id")
    run_dir.mkdir(parents=True)
    started_at = datetime.now(timezone.utc).isoformat()
    train_result = training_function(
        train_dir=dataset_dir / "data" / "train",
        validation_dir=dataset_dir / "data" / "val",
        output_dir=run_dir,
        config=config,
        device=torch.device("cuda"),
    )
    if not any(path.is_file() for path in run_dir.rglob("*")):
        raise RuntimeError("Trainer returned without writing a checkpoint or result artifact")
    source_snapshot_sha256 = _write_source_snapshot(run_dir)
    safe_result = _json_safe(train_result)
    _reject_patient_identifiers(safe_result)
    safe_config = _json_safe(config)
    _reject_patient_identifiers(safe_config)

    run_metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "dataset_repo_id": dataset_repo_id,
        "dataset_revision": pinned_revision,
        "dataset_version": dataset_metadata["dataset_version"],
        "source_archive_sha256": dataset_metadata.get("source_archive_sha256"),
        "source_manifest_sha256": dataset_metadata["source_manifest_sha256"],
        "trainer": trainer,
        "config": safe_config,
        "trainer_result": safe_result,
        "gpu": torch.cuda.get_device_name(0),
        "environment": {
            "python": platform.python_version(),
            # Modal injects its runtime package into containers without always
            # installing distribution metadata, so importlib.metadata.version
            # can raise PackageNotFoundError for ``modal`` even though the app
            # is running normally.
            "packages": {
                "modal": getattr(modal, "__version__", "runtime-managed"),
                **{
                    name: package_version(name)
                    for name in ("huggingface-hub", "numpy", "scikit-learn", "torch")
                },
            },
        },
        "code_revision": code_revision,
        "working_tree_dirty": working_tree_dirty,
        "source_snapshot": "source_snapshot.zip",
        "source_snapshot_sha256": source_snapshot_sha256,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": _file_checksums(run_dir),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run_volume.commit()

    commit = api.upload_folder(
        repo_id=model_repo_id,
        repo_type="model",
        folder_path=str(run_dir),
        path_in_repo=f"runs/{run_id}",
        commit_message=f"Store DACN training run {run_id}",
        token=token,
    )
    model_revision = getattr(commit, "oid", None)
    return {
        "run_id": run_id,
        "dataset_repo_id": dataset_repo_id,
        "dataset_revision": pinned_revision,
        "model_repo_id": model_repo_id,
        "model_revision": model_revision,
        "run_url": (
            f"https://huggingface.co/{model_repo_id}/tree/{model_revision or 'main'}/runs/{run_id}"
        ),
    }


@app.local_entrypoint()
def main(
    dataset_repo_id: str,
    dataset_revision: str,
    model_repo_id: str,
    run_id: str,
    trainer: str,
    config_json: str = "{}",
):
    """Launch one pinned training run; the Modal result is also synced to the Hub."""
    sha_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    status_result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    code_revision = sha_result.stdout.strip() if sha_result.returncode == 0 else "unknown"
    working_tree_dirty = (
        bool(status_result.stdout.strip()) if status_result.returncode == 0 else True
    )
    result = train_remote.remote(
        dataset_repo_id=dataset_repo_id,
        dataset_revision=dataset_revision,
        model_repo_id=model_repo_id,
        run_id=run_id,
        trainer=trainer,
        config_json=config_json,
        code_revision=code_revision,
        working_tree_dirty=working_tree_dirty,
    )
    print(json.dumps(result, indent=2))

"""Run a DACN training callable on Modal and publish its artifacts to the Hub.

Training implementations provide ``train(train_dir, validation_dir, output_dir,
config, device)`` under ``dacn.training.*``. This module supplies the storage and
execution boundary and withholds the held-out test path from the trainer.
"""

import hashlib
import importlib
import inspect
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
        "numba>=0.60,<1",
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
            if relative.as_posix() == "run_manifest.json" or path.name.endswith(".tmp"):
                continue
            if any("patient" in part.lower() for part in relative.parts):
                raise ValueError(f"Refusing to publish patient-level run output: {relative}")
            checksum = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(block)
            digest = checksum.hexdigest()
            result[path.relative_to(root).as_posix()] = digest
    return result


def _write_source_snapshot(output_dir: Path, *, persist: bool = True) -> str:
    """Hash or persist the exact DACN Python sources loaded by this Modal run."""
    package_file = importlib.import_module("dacn").__file__
    if package_file is None:
        raise RuntimeError("Cannot locate DACN package sources for the run snapshot")
    package_root = Path(package_file).resolve().parent
    source_files = sorted(package_root.rglob("*.py"))
    if not source_files:
        raise RuntimeError("DACN package contains no Python source files to snapshot")

    snapshot_path = output_dir / "source_snapshot.zip"
    temporary_path = output_dir / ".source_snapshot.zip.tmp"
    try:
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for source_file in source_files:
                if source_file.is_symlink():
                    raise ValueError(f"Refusing to snapshot symlinked source: {source_file.name}")
                relative = source_file.relative_to(package_root)
                entry = zipfile.ZipInfo(
                    f"dacn/{relative.as_posix()}", date_time=(1980, 1, 1, 0, 0, 0)
                )
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.external_attr = 0o100644 << 16
                archive.writestr(entry, source_file.read_bytes())
        digest = hashlib.sha256(temporary_path.read_bytes()).hexdigest()
        if persist:
            os.replace(temporary_path, snapshot_path)
        return digest
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_run_manifest(run_dir: Path, manifest: dict) -> None:
    temporary = run_dir / "run_manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, run_dir / "run_manifest.json")


def _read_run_manifest(run_dir: Path) -> dict:
    path = run_dir / "run_manifest.json"
    if not path.is_file():
        raise ValueError("Cannot resume: run manifest is missing from the Modal Volume")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 2:
        raise ValueError("Cannot resume: run manifest schema is unsupported")
    return value


def _remove_temporary_run_files(run_dir: Path) -> None:
    for path in run_dir.rglob("*.tmp"):
        if path.is_file() and not path.is_symlink():
            path.unlink()


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
    resume: bool = False,
) -> dict:
    """Fetch a pinned dataset, invoke one trainer, and durably record its lifecycle."""
    from importlib.metadata import version as package_version

    import torch
    from huggingface_hub import HfApi, snapshot_download

    from dacn.cloud.config import parse_config_json
    from dacn.data.hub import validate_hf_export

    run_dir = None
    manifest = None
    api = None
    token = None
    private_model_repo_verified = False
    stage = "validate_request"
    started_at = datetime.now(timezone.utc).isoformat()
    pinned_revision = None
    try:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_id):
            raise ValueError("run_id must be 1-64 path-safe characters")
        if not trainer.startswith("dacn.training.") or ":" not in trainer:
            raise ValueError(
                "trainer must be a callable in dacn.training, written as module:function"
            )
        if not isinstance(resume, bool):
            raise TypeError("resume must be a boolean")
        if resume and trainer != "dacn.training.ddpm_partial:train":
            raise ValueError("Resume is currently supported only by the partial DDPM trainer")

        config = parse_config_json(config_json)
        safe_config = _json_safe(config)
        _reject_patient_identifiers(safe_config)
        module_name, function_name = trainer.split(":", maxsplit=1)

        run_dir = Path(RUN_ROOT) / run_id
        previous_manifest = None
        if resume:
            if not run_dir.is_dir():
                raise ValueError(f"Cannot resume missing run: {run_id}")
            previous_manifest = _read_run_manifest(run_dir)
            if (
                previous_manifest.get("status") == "completed"
                and previous_manifest.get("artifact_sync_status") == "uploaded"
            ):
                raise ValueError("Cannot resume a completed and uploaded run; choose a new run_id")
            if not (run_dir / "ddpm_last.pt").is_file():
                raise ValueError("Cannot resume: no completed-epoch DDPM checkpoint is available")
        elif run_dir.exists():
            raise ValueError(f"Run output already exists: {run_id}; choose a new run_id")
        else:
            run_dir.mkdir(parents=True)
            manifest = {
                "schema_version": 2,
                "run_id": run_id,
                "status": "started",
                "stage": stage,
                "started_at": started_at,
                "updated_at": started_at,
                "requested_dataset_revision": dataset_revision,
                "dataset_repo_id": dataset_repo_id,
                "model_repo_id": model_repo_id,
                "trainer": trainer,
                "config": safe_config,
                "requested_device": "cuda",
                "code_revision": code_revision,
                "working_tree_dirty": working_tree_dirty,
                "attempt": 1,
                "attempt_history": [],
                "progress": {},
                "artifacts": {},
            }
            _write_run_manifest(run_dir, manifest)
            run_volume.commit()

        if not torch.cuda.is_available():
            raise RuntimeError("Modal function started without a CUDA GPU")
        if previous_manifest is not None:
            expected = {
                "dataset_repo_id": dataset_repo_id,
                "requested_dataset_revision": dataset_revision,
                "model_repo_id": model_repo_id,
                "trainer": trainer,
                "config": safe_config,
                "code_revision": code_revision,
                "working_tree_dirty": working_tree_dirty,
            }
            mismatches = [
                key for key, value in expected.items() if previous_manifest.get(key) != value
            ]
            if mismatches:
                raise ValueError(
                    "Cannot resume: request differs from the saved run metadata ("
                    + ", ".join(mismatches)
                    + ")"
                )
            current_source_sha = _write_source_snapshot(run_dir, persist=False)
            if current_source_sha != previous_manifest.get("source_snapshot_sha256"):
                raise ValueError("Cannot resume: loaded DACN source snapshot differs")
            manifest = previous_manifest
            previous_status = manifest.get("status")
            if previous_status in {"started", "running"} or (
                previous_status == "completed"
                and manifest.get("artifact_sync_status") != "uploaded"
            ):
                manifest.setdefault("attempt_history", []).append(
                    {
                        "attempt": manifest.get("attempt", 1),
                        "status": "interrupted",
                        "last_stage": manifest.get("stage"),
                        "recorded_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
            manifest["attempt"] = int(manifest.get("attempt", 1)) + 1
            manifest["resumed_at"] = datetime.now(timezone.utc).isoformat()
            manifest["status"] = "running"
            manifest["stage"] = stage
            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            manifest.pop("failure", None)
            _write_run_manifest(run_dir, manifest)
            run_volume.commit()

        def persist_stage(stage_name: str) -> None:
            nonlocal stage
            stage = stage_name
            manifest["stage"] = stage_name
            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_run_manifest(run_dir, manifest)
            run_volume.commit()

        persist_stage("load_trainer")
        training_function = getattr(importlib.import_module(module_name), function_name, None)
        if not callable(training_function):
            raise TypeError(f"Trainer callable does not exist: {trainer}")
        resolve_config = getattr(importlib.import_module(module_name), "_resolved_config", None)
        if callable(resolve_config):
            manifest["effective_config"] = _json_safe(resolve_config(config))
            _reject_patient_identifiers(manifest["effective_config"])
            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_run_manifest(run_dir, manifest)
            run_volume.commit()

        persist_stage("source_snapshot")
        if previous_manifest is None:
            source_snapshot_sha256 = _write_source_snapshot(run_dir)
            manifest["source_snapshot"] = "source_snapshot.zip"
            manifest["source_snapshot_sha256"] = source_snapshot_sha256
            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_run_manifest(run_dir, manifest)
            run_volume.commit()
        else:
            source_snapshot_sha256 = previous_manifest["source_snapshot_sha256"]

        persist_stage("hub_authentication")
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("Modal Secret 'huggingface' must contain a write-capable HF_TOKEN")
        api = HfApi(token=token)

        # Verify private destination access before any GPU training is invoked.
        persist_stage("model_repo_access")
        api.create_repo(model_repo_id, repo_type="model", private=True, exist_ok=True)
        model_info = api.repo_info(model_repo_id, repo_type="model")
        if not model_info.private:
            raise ValueError("Refusing to upload run artifacts to a public model repository")
        private_model_repo_verified = True

        persist_stage("resolve_dataset_revision")
        source_info = api.repo_info(
            dataset_repo_id,
            repo_type="dataset",
            revision=dataset_revision,
        )
        pinned_revision = source_info.sha
        if (
            previous_manifest is not None
            and previous_manifest.get("dataset_revision") != pinned_revision
        ):
            raise ValueError(
                "Cannot resume: dataset reference now resolves to a different revision"
            )
        manifest["dataset_revision"] = pinned_revision
        manifest["source_snapshot_sha256"] = source_snapshot_sha256
        manifest["gpu"] = torch.cuda.get_device_name(0)
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_run_manifest(run_dir, manifest)
        run_volume.commit()

        persist_stage("download_dataset")
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
        persist_stage("validate_dataset")
        dataset_metadata = validate_hf_export(
            dataset_dir,
            verify_checksums=True,
            allow_hf_cache=True,
        )
        dataset_volume.commit()
        manifest["dataset_version"] = dataset_metadata["dataset_version"]
        manifest["source_archive_sha256"] = dataset_metadata.get("source_archive_sha256")
        manifest["source_manifest_sha256"] = dataset_metadata["source_manifest_sha256"]
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_run_manifest(run_dir, manifest)
        run_volume.commit()

        def checkpoint_callback(progress: dict) -> None:
            allowed = ("epoch", "optimizer_steps", "checkpoint", "best_epoch")
            manifest["progress"] = {key: progress[key] for key in allowed if key in progress}
            manifest["status"] = "running"
            manifest["stage"] = "training"
            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            manifest["artifacts"] = _file_checksums(run_dir)
            _write_run_manifest(run_dir, manifest)
            run_volume.commit()

        stage = "training"
        manifest["status"] = "running"
        manifest["stage"] = stage
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_run_manifest(run_dir, manifest)
        run_volume.commit()
        train_kwargs = {
            "train_dir": dataset_dir / "data" / "train",
            "validation_dir": dataset_dir / "data" / "val",
            "output_dir": run_dir,
            "config": config,
            "device": torch.device("cuda"),
        }
        parameters = inspect.signature(training_function).parameters
        if "on_checkpoint" in parameters:
            train_kwargs["on_checkpoint"] = checkpoint_callback
        if resume:
            if "resume_from_checkpoint" not in parameters:
                raise TypeError("DDPM trainer does not expose checkpoint resume")
            train_kwargs["resume_from_checkpoint"] = run_dir / "ddpm_last.pt"
        train_result = training_function(**train_kwargs)
        if trainer == "dacn.training.ddpm_partial:train" and not all(
            (run_dir / filename).is_file() for filename in ("ddpm_best.pt", "ddpm_last.pt")
        ):
            raise RuntimeError("Partial DDPM trainer returned without both checkpoints")
        if not any(
            path.is_file() and path.name != "run_manifest.json" for path in run_dir.rglob("*")
        ):
            raise RuntimeError("Trainer returned without writing a checkpoint or result artifact")
        safe_result = _json_safe(train_result)
        _reject_patient_identifiers(safe_result)
        manifest["trainer_result"] = safe_result
        manifest["environment"] = {
            "python": platform.python_version(),
            # Modal injects its runtime package without always installing its distribution metadata.
            "packages": {
                "modal": getattr(modal, "__version__", "runtime-managed"),
                **{
                    name: package_version(name)
                    for name in ("huggingface-hub", "numpy", "scikit-learn", "torch")
                },
            },
        }
        manifest["status"] = "completed"
        manifest["stage"] = "upload_artifacts"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["updated_at"] = manifest["completed_at"]
        manifest["artifacts"] = _file_checksums(run_dir)
        manifest.setdefault("attempt_history", []).append(
            {
                "attempt": manifest.get("attempt", 1),
                "status": "training_completed",
                "stage": "training",
                "recorded_at": manifest["completed_at"],
            }
        )
        _write_run_manifest(run_dir, manifest)
        run_volume.commit()

        persist_stage("upload_artifacts")
        commit = api.upload_folder(
            repo_id=model_repo_id,
            repo_type="model",
            folder_path=str(run_dir),
            path_in_repo=f"runs/{run_id}",
            commit_message=f"Store DACN training run {run_id}",
            token=token,
        )
        model_revision = getattr(commit, "oid", None)
        manifest["artifact_sync_status"] = "uploaded"
        manifest["model_revision"] = model_revision
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_run_manifest(run_dir, manifest)
        run_volume.commit()
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
    except Exception as exc:
        if run_dir is not None and run_dir.is_dir() and manifest is not None:
            try:
                _remove_temporary_run_files(run_dir)
                manifest["status"] = "failed"
                manifest["stage"] = stage
                manifest["failure"] = {"exception_type": type(exc).__name__}
                manifest["failed_at"] = datetime.now(timezone.utc).isoformat()
                manifest["updated_at"] = manifest["failed_at"]
                manifest["artifacts"] = _file_checksums(run_dir)
                manifest.setdefault("attempt_history", []).append(
                    {
                        "attempt": manifest.get("attempt", 1),
                        "status": "failed",
                        "stage": stage,
                        "exception_type": type(exc).__name__,
                        "recorded_at": manifest["failed_at"],
                    }
                )
                _write_run_manifest(run_dir, manifest)
                run_volume.commit()
                if api is not None and private_model_repo_verified:
                    try:
                        commit = api.upload_folder(
                            repo_id=model_repo_id,
                            repo_type="model",
                            folder_path=str(run_dir),
                            path_in_repo=f"runs/{run_id}",
                            commit_message=f"Store failed DACN run {run_id}",
                            token=token,
                        )
                        manifest["artifact_sync_status"] = "uploaded_partial"
                        manifest["model_revision"] = getattr(commit, "oid", None)
                    except Exception as upload_exc:
                        manifest["artifact_sync_status"] = "modal_volume_only"
                        manifest["artifact_sync_error_type"] = type(upload_exc).__name__
                    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
                    _write_run_manifest(run_dir, manifest)
                    run_volume.commit()
            except Exception:
                # Preserve the primary error; Modal logs remain the diagnostic source.
                pass
        raise


@app.local_entrypoint()
def main(
    dataset_repo_id: str,
    dataset_revision: str,
    model_repo_id: str,
    run_id: str,
    trainer: str,
    config_json: str = "{}",
    resume: bool = False,
) -> str:
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
        resume=resume,
    )
    result_json = json.dumps(result, indent=2)
    print(result_json)
    return result_json

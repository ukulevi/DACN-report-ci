"""Run aggregate validation-only DDPM evaluation on Modal."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path

import modal

from dacn.cloud.modal_train import (
    DATA_ROOT,
    RUN_ROOT,
    _file_checksums,
    _json_safe,
    _reject_patient_identifiers,
    _write_run_manifest,
    _write_source_snapshot,
    app,
    dataset_volume,
    image,
    run_volume,
)


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_string_list(value: str, *, field: str) -> list[str]:
    parsed = json.loads(value)
    if (
        not isinstance(parsed, list)
        or not parsed
        or not all(isinstance(item, str) for item in parsed)
    ):
        raise ValueError(f"{field} must be a nonempty JSON string list")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{field} must not contain duplicates")
    return parsed


def _parse_timestep_list(value: str) -> list[int]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("starting_timesteps_json must be a nonempty JSON list")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or not 1 <= item <= 1000
        for item in parsed
    ):
        raise ValueError("starting timesteps must be integers in [1, 1000]")
    if len(set(parsed)) != len(parsed):
        raise ValueError("starting timesteps must not contain duplicates")
    return parsed


@app.function(
    image=image,
    gpu="T4",
    timeout=12 * 60 * 60,
    volumes={DATA_ROOT: dataset_volume, RUN_ROOT: run_volume},
    secrets=[modal.Secret.from_name("huggingface-secret-2")],
)
def evaluate_remote(
    dataset_repo_id: str,
    dataset_revision: str,
    model_repo_id: str,
    source_run_id: str,
    evaluation_id: str,
    checkpoint_names_json: str,
    starting_timesteps_json: str,
    code_revision: str,
    working_tree_dirty: bool,
    max_validation_samples: int = 0,
    sampling_base_seed: int = 17,
    reverse_sampler: str = "ancestral",
    batch_size: int = 1,
    save_reconstruction_cache: bool = False,
) -> dict:
    """Evaluate stored checkpoints on validation; the held-out test path is unavailable."""
    import torch
    from huggingface_hub import HfApi, snapshot_download

    from dacn.data.hub import validate_hf_export
    from dacn.evaluation.ddpm_validation import evaluate_checkpoint
    from dacn.evaluation.handoff import validate_dataset_binding

    evaluation_dir = None
    manifest = None
    api = None
    stage = "validate_request"
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        path_safe = r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}"
        if not re.fullmatch(path_safe, source_run_id):
            raise ValueError("source_run_id must be 1-96 path-safe characters")
        if not re.fullmatch(path_safe, evaluation_id):
            raise ValueError("evaluation_id must be 1-96 path-safe characters")
        checkpoint_names = _parse_string_list(
            checkpoint_names_json,
            field="checkpoint_names_json",
        )
        if any(not re.fullmatch(r"ddpm_epoch_[0-9]{3}\.pt", name) for name in checkpoint_names):
            raise ValueError("only explicit ddpm_epoch_NNN.pt milestones may be evaluated")
        timesteps = _parse_timestep_list(starting_timesteps_json)
        if isinstance(max_validation_samples, bool) or not isinstance(max_validation_samples, int):
            raise TypeError("max_validation_samples must be an integer")
        if max_validation_samples < 0 or max_validation_samples == 1:
            raise ValueError("max_validation_samples must be 0 or at least 2")
        sample_limit = max_validation_samples or None
        if reverse_sampler not in {"ancestral", "posterior_mean"}:
            raise ValueError("reverse_sampler must be ancestral or posterior_mean")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if reverse_sampler == "ancestral" and batch_size != 1:
            raise ValueError("ancestral evaluation requires batch_size=1")
        if not isinstance(save_reconstruction_cache, bool):
            raise TypeError("save_reconstruction_cache must be a boolean")

        source_dir = Path(RUN_ROOT) / source_run_id
        if not source_dir.is_dir():
            raise ValueError("source training run is missing from the Modal Volume")
        source_manifest_path = source_dir / "run_manifest.json"
        if not source_manifest_path.is_file():
            raise ValueError("source training manifest is missing")
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        source_artifacts = source_manifest.get("artifacts", {})
        checkpoint_records = []
        for name in checkpoint_names:
            path = source_dir / name
            if not path.is_file():
                raise ValueError(f"source checkpoint does not exist: {name}")
            checksum = _sha256(path)
            if source_artifacts.get(name) != checksum:
                raise ValueError(f"source checkpoint checksum disagrees with manifest: {name}")
            checkpoint_records.append({"name": name, "sha256": checksum})

        evaluation_dir = Path(RUN_ROOT) / "evaluations" / evaluation_id
        if evaluation_dir.exists():
            raise ValueError("evaluation output already exists; choose a new evaluation_id")
        evaluation_dir.mkdir(parents=True)
        manifest = {
            "schema_version": 1,
            "evaluation_id": evaluation_id,
            "status": "started",
            "stage": stage,
            "started_at": started_at,
            "updated_at": started_at,
            "split": "validation",
            "held_out_test_used": False,
            "dataset_repo_id": dataset_repo_id,
            "requested_dataset_revision": dataset_revision,
            "model_repo_id": model_repo_id,
            "source_run_id": source_run_id,
            "source_checkpoints": checkpoint_records,
            "starting_timesteps": timesteps,
            "max_validation_samples": sample_limit,
            "sampling_base_seed": sampling_base_seed,
            "reverse_sampler": reverse_sampler,
            "batch_size": batch_size,
            "save_reconstruction_cache": save_reconstruction_cache,
            "code_revision": code_revision,
            "working_tree_dirty": working_tree_dirty,
            "progress": {"completed_combinations": 0},
            "artifacts": {},
        }
        _write_run_manifest(evaluation_dir, manifest)
        run_volume.commit()

        def persist_stage(name: str) -> None:
            nonlocal stage
            stage = name
            manifest["stage"] = name
            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_run_manifest(evaluation_dir, manifest)
            run_volume.commit()

        if not torch.cuda.is_available():
            raise RuntimeError("Modal evaluator started without a CUDA GPU")
        manifest["gpu"] = torch.cuda.get_device_name(0)
        persist_stage("source_snapshot")
        manifest["source_snapshot"] = "source_snapshot.zip"
        manifest["source_snapshot_sha256"] = _write_source_snapshot(evaluation_dir)
        _write_run_manifest(evaluation_dir, manifest)
        run_volume.commit()

        persist_stage("hub_authentication")
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("Modal Secret must contain HF_TOKEN")
        api = HfApi(token=token)
        api.create_repo(model_repo_id, repo_type="model", private=True, exist_ok=True)
        if not api.repo_info(model_repo_id, repo_type="model").private:
            raise ValueError("refusing to upload evaluation artifacts to a public repository")

        persist_stage("resolve_dataset_revision")
        pinned_revision = api.repo_info(
            dataset_repo_id,
            repo_type="dataset",
            revision=dataset_revision,
        ).sha
        validate_dataset_binding(source_manifest, dataset_repo_id, pinned_revision)
        manifest["dataset_revision"] = pinned_revision
        _write_run_manifest(evaluation_dir, manifest)
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
        validate_hf_export(dataset_dir, verify_checksums=True, allow_hf_cache=True)
        manifest["dataset_metadata_sha256"] = _sha256(dataset_dir / "dataset_metadata.json")
        manifest["train_seed"] = source_manifest.get(
            "effective_config", source_manifest.get("config", {})
        ).get("seed")
        manifest["source_training_snapshot_sha256"] = source_manifest.get("source_snapshot_sha256")
        dataset_volume.commit()

        results = {
            "schema_version": 1,
            "evaluation_id": evaluation_id,
            "split": "validation",
            "held_out_test_used": False,
            "source_run_id": source_run_id,
            "dataset_revision": pinned_revision,
            "checkpoints": {},
        }
        completed = 0
        persist_stage("validation_evaluation")
        for checkpoint_record in checkpoint_records:
            checkpoint_name = checkpoint_record["name"]

            def save_timestep(
                result: dict,
                checkpoint_name: str = checkpoint_name,
                checkpoint_sha256: str = checkpoint_record["sha256"],
            ) -> None:
                nonlocal completed
                checkpoint_result = results["checkpoints"].setdefault(
                    checkpoint_name,
                    {"checkpoint_sha256": checkpoint_sha256, "timesteps": {}},
                )
                checkpoint_result["timesteps"][str(result["starting_timestep"])] = result
                completed += 1
                manifest["progress"] = {
                    "checkpoint": checkpoint_name,
                    "starting_timestep": result["starting_timestep"],
                    "completed_combinations": completed,
                    "total_combinations": len(checkpoint_records) * len(timesteps),
                }
                manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
                _write_json(evaluation_dir / "validation_results.json", _json_safe(results))
                manifest["artifacts"] = _file_checksums(evaluation_dir)
                _write_run_manifest(evaluation_dir, manifest)
                run_volume.commit()

            checkpoint_result = evaluate_checkpoint(
                validation_dir=dataset_dir / "data" / "val",
                checkpoint_path=source_dir / checkpoint_name,
                starting_timesteps=timesteps,
                device=torch.device("cuda"),
                max_validation_samples=sample_limit,
                sampling_base_seed=sampling_base_seed,
                reverse_sampler=reverse_sampler,
                batch_size=batch_size,
                reconstruction_cache_dir=evaluation_dir if save_reconstruction_cache else None,
                artifact_dir=evaluation_dir,
                on_timestep_complete=save_timestep,
            )
            stored = results["checkpoints"][checkpoint_name]
            stored["checkpoint_epoch"] = checkpoint_result["checkpoint_epoch"]
            stored["validation_samples"] = checkpoint_result["validation_samples"]
            _write_json(evaluation_dir / "validation_results.json", _json_safe(results))
            run_volume.commit()

        safe_results = _json_safe(results)
        _reject_patient_identifiers(safe_results)
        manifest["status"] = "completed"
        manifest["stage"] = "upload_artifacts"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["updated_at"] = manifest["completed_at"]
        manifest["environment"] = {
            "python": platform.python_version(),
            "packages": {
                name: package_version(name)
                for name in ("huggingface-hub", "numpy", "scikit-learn", "torch")
            },
        }
        manifest["artifacts"] = _file_checksums(evaluation_dir)
        _write_run_manifest(evaluation_dir, manifest)
        run_volume.commit()

        commit = api.upload_folder(
            repo_id=model_repo_id,
            repo_type="model",
            folder_path=str(evaluation_dir),
            path_in_repo=f"evaluations/{evaluation_id}",
            commit_message=f"Store DACN validation evaluation {evaluation_id}",
            token=token,
        )
        manifest["artifact_sync_status"] = "uploaded"
        manifest["model_revision"] = getattr(commit, "oid", None)
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_run_manifest(evaluation_dir, manifest)
        run_volume.commit()
        return {
            "evaluation_id": evaluation_id,
            "status": "completed",
            "split": "validation",
            "held_out_test_used": False,
            "model_revision": manifest["model_revision"],
        }
    except Exception as exc:
        if evaluation_dir is not None and evaluation_dir.is_dir() and manifest is not None:
            manifest["status"] = "failed"
            manifest["stage"] = stage
            manifest["failure"] = {"exception_type": type(exc).__name__}
            manifest["failed_at"] = datetime.now(timezone.utc).isoformat()
            manifest["updated_at"] = manifest["failed_at"]
            manifest["artifacts"] = _file_checksums(evaluation_dir)
            _write_run_manifest(evaluation_dir, manifest)
            run_volume.commit()
        raise

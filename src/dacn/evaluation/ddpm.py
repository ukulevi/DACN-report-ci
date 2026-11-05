"""Explicit terminal bridge from a local DDPM checkpoint to validation metrics.

Running this command performs inference on every validation image. It never
trains, selects a timestep, loads test data or freezes thresholds. Install the
train extra before using the real backend. GPU execution is an explicit action.
"""

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dacn.evaluation.artifacts import (
    _provenance,
    _validation_data,
    evaluate_validation_artifact,
)


def produce_validation_artifact(dataset_dir, output, provenance, reconstruct):
    """Call a model adapter once per validation UUID, without passing labels/masks.

    Adapter: reconstruct(image, *, sample_id, starting_timestep,
    sampling_base_seed) -> float32 array (1,1,256,256). It must implement the
    per-UUID seed policy. The real adapter below delegates that policy to C.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError("reconstruction artifact already exists")
    provenance = _provenance(provenance)
    by_id, images, _, _ = _validation_data(dataset_dir, provenance["dataset_metadata_sha256"])
    ids, pairs = [], []
    for identifier, position in by_id.items():
        prediction = np.asarray(
            reconstruct(
                images[position][None, None].copy(),
                sample_id=identifier,
                starting_timestep=provenance["starting_timestep"],
                sampling_base_seed=provenance["sampling_seed"],
            )
        )
        if (
            prediction.shape != (1, 1, 256, 256)
            or prediction.dtype != np.float32
            or not np.isfinite(prediction).all()
            or np.any(np.abs(prediction) > 1)
        ):
            raise ValueError("reconstruction must be float32 (1,1,256,256), finite in [-1,1]")
        ids.append(identifier)
        pairs.append(prediction[0].copy())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(
            stream, sample_ids=np.asarray(ids, dtype="U36"), reconstructions=np.stack(pairs)
        )
    return {
        "samples": len(ids),
        "reconstruction_artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def _ddpm_backend(checkpoint_bytes, config, device):
    # Lazy imports keep the package and data/metric checks usable without Torch.
    import torch

    from dacn.training.ddpm_partial import (
        build_diffusion,
        derive_sampling_seed,
        load_ema_checkpoint,
        partial_reconstruct,
    )

    # C's loader accepts a path. Snapshot already-hashed bytes privately so it
    # cannot reopen a producer-controlled path that changed after checksum QC.
    with tempfile.TemporaryDirectory(prefix="dacn-checkpoint-") as directory:
        snapshot = Path(directory) / "verified.pt"
        snapshot.write_bytes(checkpoint_bytes)
        model = load_ema_checkpoint(snapshot, config, device)
    diffusion = build_diffusion(config)

    def reconstruct(image, *, sample_id, starting_timestep, sampling_base_seed):
        tensor = torch.from_numpy(image).to(device)
        prediction = partial_reconstruct(
            model,
            tensor,
            starting_timestep=starting_timestep,
            seed=derive_sampling_seed(sampling_base_seed, sample_id),
            diffusion=diffusion,
        )
        return prediction.detach().cpu().numpy().astype(np.float32)

    return reconstruct


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "checkpoint", "config", "provenance", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    # Reject existing run directories, even empty ones; never overwrite evidence.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output_dir / "evaluation_manifest.json"
    manifest = {
        "schema_version": 1,
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "integration validation; not a sweep or final research result",
        "device_requested": args.device,
        "split": "validation",
        "test_evaluated": False,
        "thresholds_frozen": False,
        "sampling_policy": "C derive_sampling_seed UUID-v1, batch1, one pair for both scores",
        "evaluation_source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("ddpm.py", "artifacts.py", "validation.py", "scores.py")
        },
    }

    def save():
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        temporary.replace(manifest_path)

    save()
    try:
        provenance = _provenance(json.loads(args.provenance.read_text(encoding="utf-8")))
        config_bytes = args.config.read_bytes()
        config = json.loads(config_bytes)
        if not isinstance(config, dict) or config.get("seed", 17) != provenance["train_seed"]:
            raise ValueError("config train seed does not match provenance")
        checkpoint_bytes = args.checkpoint.read_bytes()
        if hashlib.sha256(checkpoint_bytes).hexdigest() != provenance["checkpoint_sha256"]:
            raise ValueError("checkpoint checksum mismatch")
        manifest["provenance_declared_by_producer"] = provenance
        manifest["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
        manifest["stage"] = "load_checkpoint"
        save()
        reconstruct = _ddpm_backend(checkpoint_bytes, config, args.device)
        del checkpoint_bytes
        manifest["stage"] = "reconstruct_validation"
        save()
        artifact = args.output_dir / "validation_pairs.npz"
        receipt = produce_validation_artifact(args.dataset, artifact, provenance, reconstruct)
        manifest.update(receipt)
        manifest["stage"] = "evaluate_paired_scores"
        save()
        report = evaluate_validation_artifact(artifact, args.dataset, provenance)
        if report["reconstruction_artifact_sha256"] != receipt["reconstruction_artifact_sha256"]:
            raise ValueError("reconstruction artifact changed between production and scoring")
        with (args.output_dir / "validation_metrics.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        manifest.update(status="completed", stage="completed")
    except BaseException as exc:
        # Record class only: exception messages may contain paths/private UUIDs.
        manifest.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error_class=type(exc).__name__,
        )
        raise
    finally:
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    print(json.dumps({"status": manifest["status"], "samples": manifest["samples"]}))


if __name__ == "__main__":
    main()

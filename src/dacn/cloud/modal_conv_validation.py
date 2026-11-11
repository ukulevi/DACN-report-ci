"""Validation-only AE/VAE evaluation with registered calibration membership."""

import csv
import io
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

from dacn.cloud.modal_train import RUN_ROOT, _write_source_snapshot, base_image, run_volume

app = modal.App("dacn-conv-validation")
image = base_image.add_local_python_source("dacn")
PINS = {
    "conv_ae": (
        "conv-ae-z256-e30-20261111T141800Z",
        "fc34f511e4910c688d9e7c5f4f075957f291cb801ced3f1d77850294f271bfbb",
        13,
    ),
    "conv_vae": (
        "conv-vae-z256-e30-20261111T141800Z",
        "b85619af1f53f74f09136c26c92e5ac960f9ebdea7d84329cc582c1d56f0ab8e",
        15,
    ),
}


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=8192,
    timeout=3600,
    volumes={RUN_ROOT: run_volume},
    secrets=[modal.Secret.from_name("huggingface-secret-2")],
)
def evaluate(evaluation_id: str, code_revision: str, smoke: bool = False):
    import re

    import numpy as np
    import torch
    from huggingface_hub import HfApi

    from dacn.evaluation.ddim20_local_ssim import (
        CACHE_SHA256,
        DATASET_REVISION,
        evaluate_partitions,
        sha256_file,
        write_json,
    )
    from dacn.evaluation.registered_partition import registered_partition
    from dacn.evaluation.scores import paired_score_maps
    from dacn.models.conv_autoencoder import ConvAutoencoder

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,90}", evaluation_id):
        raise ValueError("Invalid evaluation ID")
    output = Path(RUN_ROOT) / "evaluations" / evaluation_id
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "evaluation_id": evaluation_id,
        "status": "started",
        "split": "validation",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": code_revision,
        "working_tree_dirty": True,
        "dataset_revision": DATASET_REVISION,
        "smoke": smoke,
        "test_accessed": False,
        "training_seeds": 1,
        "inference": "deterministic mean latent",
        "scores": ["pixel_difference", "local_ssim"],
        "gpu": "T4",
        "source_snapshot_sha256": _write_source_snapshot(output),
    }

    def persist():
        write_json(output / "run_manifest.json", record)
        run_volume.commit()

    persist()
    started = time.perf_counter()
    try:
        protocol = Path(RUN_ROOT) / "final-evaluation/final-fourarm-20261110-b"
        reference = protocol / "private_reference.npz"
        original = json.loads((protocol / "manifest.json").read_text())
        index = protocol / "artifacts/validation_index"
        partition = protocol / "artifacts/validation_partition"
        if (
            original["dataset_revision"] != DATASET_REVISION
            or sha256_file(reference) != CACHE_SHA256
        ):
            raise ValueError("Validation cache provenance differs")
        for key, path in [("validation_index", index), ("validation_partition", partition)]:
            if sha256_file(path) != original["artifacts"][key]:
                raise ValueError("Validation membership checksum differs")
        rows = list(csv.DictReader(io.StringIO(index.read_text())))
        if len(rows) != 83 or any(r["split"] != "val" for r in rows):
            raise ValueError("Validation only")
        cal, hold = registered_partition(
            partition.read_bytes(),
            index.read_bytes(),
            np.array([r["patient_group_id"] for r in rows]),
        )
        with np.load(reference, allow_pickle=False) as a:
            images, masks, labels = a["images"].copy(), a["masks"].copy(), a["labels"].copy()
        if (
            images.shape != (83, 256, 256)
            or masks.shape != images.shape
            or not np.array_equal(labels, [int(r["label"]) for r in rows])
        ):
            raise ValueError("Validation shape/alignment differs")
        results = {}
        for kind, (run_id, checkpoint_sha, epoch) in PINS.items():
            training = Path(RUN_ROOT) / run_id
            checkpoint = training / "autoencoder_best.pt"
            manifest = json.loads((training / "run_manifest.json").read_text())
            if (
                manifest["status"] != "completed"
                or manifest["dataset_revision"] != DATASET_REVISION
                or manifest["artifacts"]["autoencoder_best.pt"] != checkpoint_sha
                or sha256_file(checkpoint) != checkpoint_sha
            ):
                raise ValueError("Checkpoint provenance differs")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved["epoch"] != epoch or saved["config"]["model_type"] != kind:
                raise ValueError("Checkpoint identity differs")
            model = ConvAutoencoder(kind, 256, 32).cuda().eval()
            model.load_state_dict(saved["model"], strict=True)
            scores = {k: [] for k in record["scores"]}
            reconstructions, seconds = [], []
            torch.cuda.reset_peak_memory_stats()
            count = 2 if smoke else 83
            for position in range(count):
                x = torch.from_numpy(images[position].copy())[None, None].cuda()
                torch.cuda.synchronize()
                tick = time.perf_counter()
                with torch.inference_mode():
                    reconstructed = model(x, sample=False)[0][0, 0].cpu().numpy()
                seconds.append(time.perf_counter() - tick)
                reconstructions.append(reconstructed)
                paired = paired_score_maps(images[position], reconstructed)
                for key, values in scores.items():
                    values.append(paired[key])
                if (position + 1) % 10 == 0 or position + 1 == count:
                    print(
                        json.dumps(
                            {
                                "model": kind,
                                "completed_samples": position + 1,
                                "expected_samples": count,
                            }
                        ),
                        flush=True,
                    )
            model_output = output / kind
            model_output.mkdir()
            measured = {}
            for key, maps in scores.items():
                maps = np.stack(maps)
                score_output = model_output / key
                score_output.mkdir()
                if not smoke:
                    measured[key] = evaluate_partitions(
                        maps, masks[:, 8:-8, 8:-8], labels, cal, hold, score_output
                    )
                np.savez_compressed(
                    score_output / "private_maps.npz",
                    scores=maps,
                    masks=masks[:count, 8:-8, 8:-8],
                    labels=labels[:count],
                )
            np.savez_compressed(
                model_output / "private_reconstructions.npz",
                reconstructions=np.stack(reconstructions),
            )
            results[kind] = {
                "samples_scored": count,
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_epoch": epoch,
                "training_run_id": run_id,
                "scores": measured,
                "reconstruction_seconds_sum": sum(seconds),
                "reconstruction_seconds_median": float(np.median(seconds)),
                "timing_scope": "one image transfer/inference/output copy; excludes scoring, metric calculation and I/O; no warmup",
                "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            }
            del model
        result = {
            "models": results,
            "split": "validation",
            "calibration_samples": 40,
            "group_holdout_samples": 43,
            "test_accessed": False,
            "smoke": smoke,
            "metric_region": "full240crop; image top576mean",
            "threshold_rule": "max F1 on40calibration; largest threshold on exact tie; score>=threshold",
            "local_ssim": "11x11Gaussian sigma1.5, data_range2 on signed[-1,1], no clipping or foreground masking",
            "limitations": "Exploratory validation; normal-validation checkpoint selection overlaps both partitions. Internal holdout is NOT independent model evaluation. One training seed; no test access.",
        }
        write_json(output / "results.json", result)
        record.update(
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            total_seconds=time.perf_counter() - started,
            artifacts={
                p.relative_to(output).as_posix(): sha256_file(p)
                for p in output.rglob("*")
                if p.is_file() and p.name != "run_manifest.json"
            },
        )
        persist()
        api = HfApi(token=os.environ["HF_TOKEN"])
        repo = "dacn-bmad-team/dacn-runs"
        if not api.repo_info(repo).private:
            raise ValueError("Private artifact repository required")
        commit = api.upload_folder(
            repo_id=repo, folder_path=str(output), path_in_repo=f"evaluations/{evaluation_id}"
        )
        record.update(artifact_sync_status="uploaded", model_revision=commit.oid)
        persist()
        return {"manifest": record, "results": result}
    except BaseException as exc:
        record.update(status="failed", exception_type=type(exc).__name__)
        persist()
        raise


@app.local_entrypoint()
def main(evaluation_id: str, output: str, smoke: bool = False):
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    call = evaluate.spawn(evaluation_id, revision, smoke)
    submission = {"evaluation_id": evaluation_id, "app_id": app.app_id, "call_id": call.object_id}
    (folder / "submission.json").write_text(json.dumps(submission, indent=2) + "\n")
    print(json.dumps(submission), flush=True)
    result = call.get()
    for key in ("manifest", "results"):
        (folder / (key + ".json")).write_text(
            json.dumps(result[key], indent=2, allow_nan=False) + "\n"
        )
    print(
        json.dumps(
            {
                "status": result["manifest"]["status"],
                "revision": result["manifest"]["model_revision"],
            }
        ),
        flush=True,
    )

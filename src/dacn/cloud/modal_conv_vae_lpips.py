"""Raw LPIPS on immutable cached ConvVAE validation reconstructions only."""

import csv
import io
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import modal

from dacn.cloud.modal_final_evaluation import image
from dacn.cloud.modal_train import _write_source_snapshot

app = modal.App("dacn-conv-vae-lpips", include_source=False)
volume = modal.Volume.from_name("dacn-bmad-runs")
ROOT = Path("/mnt/runs")
RECONSTRUCTION_SHA = "fa3afc69d70a56bec23791c22c7f63ae31c3143c7707722c56bdd54252991df6"
CHECKPOINT_SHA = "b85619af1f53f74f09136c26c92e5ac960f9ebdea7d84329cc582c1d56f0ab8e"


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=8192,
    timeout=3600,
    volumes={str(ROOT): volume},
    secrets=[modal.Secret.from_name("huggingface-secret-2")],
)
def evaluate(evaluation_id: str, code_revision: str, smoke: bool = False):
    import lpips
    import numpy as np
    import torch
    from huggingface_hub import HfApi

    from dacn.evaluation.autoddpm_lpips import perceptual_provenance
    from dacn.evaluation.calibrated_lpips import LPIPS_STATE_SHA, layer_contributions
    from dacn.evaluation.ddim20_local_ssim import (
        CACHE_SHA256,
        DATASET_REVISION,
        evaluate_partitions,
        sha256_file,
        write_json,
    )
    from dacn.evaluation.registered_partition import registered_partition

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,90}", evaluation_id):
        raise ValueError("Invalid evaluation ID")
    output = ROOT / "evaluations" / evaluation_id
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "evaluation_id": evaluation_id,
        "status": "started",
        "split": "validation",
        "smoke": smoke,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": code_revision,
        "working_tree_dirty": True,
        "source_snapshot_sha256": _write_source_snapshot(output),
        "dataset_revision": DATASET_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "checkpoint_epoch": 15,
        "reconstruction_sha256": RECONSTRUCTION_SHA,
        "model": "ConvVAE z256",
        "training_seeds": 1,
        "training_seed": 17,
        "new_vae_calls": 0,
        "training_performed": False,
        "test_accessed": False,
        "gpu": "T4",
        "cpu": 4,
        "memory_mib": 8192,
        "versions": {
            k: version(k)
            for k in ["torch", "torchvision", "lpips", "numpy", "scipy", "scikit-learn"]
        },
    }

    def persist():
        write_json(output / "run_manifest.json", record)
        volume.commit()

    persist()
    tick = time.perf_counter()
    try:
        protocol = ROOT / "final-evaluation/final-fourarm-20261110-b"
        original = json.loads((protocol / "manifest.json").read_text())
        index, partition = (
            protocol / "artifacts/validation_index",
            protocol / "artifacts/validation_partition",
        )
        reference = protocol / "private_reference.npz"
        previous = ROOT / "evaluations/conv-val-full83-20261111-a"
        reconstruction = previous / "conv_vae/private_reconstructions.npz"
        parent = json.loads((previous / "run_manifest.json").read_text())
        if (
            original["dataset_revision"] != DATASET_REVISION
            or parent["dataset_revision"] != DATASET_REVISION
            or parent["status"] != "completed"
            or parent["artifacts"]["conv_vae/private_reconstructions.npz"] != RECONSTRUCTION_SHA
            or sha256_file(reconstruction) != RECONSTRUCTION_SHA
            or sha256_file(reference) != CACHE_SHA256
        ):
            raise ValueError("Cached validation provenance differs")
        prior_results = previous / "results.json"
        if sha256_file(prior_results) != parent["artifacts"]["results.json"]:
            raise ValueError("Reconstruction provenance differs")
        prior = json.loads(prior_results.read_text())["models"]["conv_vae"]
        if (
            prior["checkpoint_sha256"] != CHECKPOINT_SHA
            or prior["checkpoint_epoch"] != 15
            or prior["samples_scored"] != 83
        ):
            raise ValueError("Expected fixed best ConvVAE checkpoint")
        for key, path in [("validation_index", index), ("validation_partition", partition)]:
            if sha256_file(path) != original["artifacts"][key]:
                raise ValueError("Membership checksum differs")
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
        with np.load(reconstruction, allow_pickle=False) as a:
            reconstructed = a["reconstructions"].copy()
        if (
            images.shape != (83, 256, 256)
            or reconstructed.shape != images.shape
            or masks.shape != images.shape
            or not np.array_equal(labels, [int(r["label"]) for r in rows])
        ):
            raise ValueError("Cache alignment differs")
        model = lpips.LPIPS(net="squeeze", spatial=True, version="0.1").eval().cuda()
        provenance = perceptual_provenance(model)
        if provenance["state_sha256"] != LPIPS_STATE_SHA:
            raise ValueError("Pretrained LPIPS state differs")
        record["lpips"] = provenance
        persist()
        torch.cuda.reset_peak_memory_stats()
        scores, seconds = [], []
        count = 2 if smoke else 83
        for i in range(count):
            torch.cuda.synchronize()
            start = time.perf_counter()
            # Raw LPIPS only; no residual product or calibrated layer fusion.
            score, _, _ = layer_contributions(model, images[i], reconstructed[i], "cuda")
            seconds.append(time.perf_counter() - start)
            scores.append(score)
            if (i + 1) % 10 == 0 or i + 1 == count:
                print(
                    json.dumps({"completed_samples": i + 1, "expected_samples": count}), flush=True
                )
        scores = np.stack(scores)
        result = {
            "samples_scored": count,
            "score": "raw spatial LPIPS0.1 squeeze; clamp minimum0",
            "lpips": provenance,
            "checkpoint_sha256": CHECKPOINT_SHA,
            "checkpoint_epoch": 15,
            "split": "validation",
            "smoke": smoke,
            "test_accessed": False,
            "new_vae_calls": 0,
            "calibration_samples": 40,
            "holdout_samples": 43,
            "training_seeds": 1,
            "metric_region": "full240crop; grayscale repeatRGB [-1,1]; normalize=False; no foreground or smoothing; image top576mean",
            "scoring_seconds_sum": sum(seconds),
            "scoring_seconds_median": float(np.median(seconds)),
            "timing_scope": "cached image-pair transfer + LPIPS layer contributions + map copy; no warmup; excludes input loading, metrics and upload",
            "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "limitations": "Post-inspection exploratory validation; normal-validation checkpoint selection overlaps both partitions. Not independent model holdout; one training seed; no test access.",
        }
        if not smoke:
            result.update(
                evaluate_partitions(scores, masks[:, 8:-8, 8:-8], labels, cal, hold, output)
            )
        np.savez_compressed(
            output / "private_lpips_maps.npz",
            scores=scores,
            masks=masks[:count, 8:-8, 8:-8],
            labels=labels[:count],
        )
        write_json(output / "results.json", result)
        record.update(
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            total_seconds=time.perf_counter() - tick,
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
            raise ValueError("Private artifacts required")
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
    for key in ["manifest", "results"]:
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

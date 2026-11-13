"""CPU-only analyses of the 06–09 Nov DDIM noise/step sweep producer artifacts.

Validation only. Reuses producer arrays, score maps and calibration thresholds; no
model calls, threshold fitting, selection or test access. Volumes are looked up
without creation and must already hold the pinned dataset cache. The registered
40/43 partition is rebuilt from the pinned index and must reproduce its recorded
SHA-256. The Hub Secret name defaults to the documented name; override it with
DACN_HF_SECRET (each member uses their own Modal Workspace).

Analyses:
  bootstrap  paired patient-group bootstrap of pixel AP/Dice (smoke: 20 replicates)
  quality    reconstruction-quality aggregates per noise level (smoke: reference run)

    DACN_HF_SECRET=huggingface modal run -m dacn.cloud.modal_sweep_bootstrap \
        --analysis quality --evaluation-id sweep-quality-smoke-UNIQUE \
        --code-revision GIT_SHA --mode smoke --no-working-tree-dirty
"""

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import modal

app = modal.App("dacn-sweep-bootstrap")
HF_SECRET = os.environ.get("DACN_HF_SECRET", "huggingface-secret-2")
data_volume = modal.Volume.from_name("dacn-bmad-datasets")
run_volume = modal.Volume.from_name("dacn-bmad-runs")
DATA_ROOT = Path("/mnt/datasets")
RUN_ROOT = Path("/mnt/runs")
DATASET = "dacn-bmad-team/bmad-flair-patient-disjoint-v1"
REVISION = "350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7"
MODEL_REPO = "dacn-bmad-team/dacn-runs"
# Immutable Hub revision containing all twelve producer sweep artifacts.
MODEL_REVISION = "de97d764ef360742cb181346afbb9f0da59bc218"
INDEX_SHA = "57ad2e31df3ed4e625da00efaecd6fddc1e2a132329dfba6e60867caf1802db3"
PARTITION_SHA = "5e8a8ce1ad3a952293a6798ae72ef342aa23676e5559fd468acb76422a27d184"
REPLICATES = {"smoke": 20, "full": 1000}
METHODS = {
    "bootstrap": "paired_patient_group_bootstrap_ddim_noise_step_sweep",
    "quality": "reconstruction_quality_ddim_noise_step_sweep",
}
SCOPE = (
    "Exploratory validation; producer arrays, maps and thresholds fixed; one training "
    "and sampling seed; test unused."
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "numpy>=1.24,<3", "numba>=0.60,<1", "scikit-learn>=1.3,<2", "huggingface_hub>=0.28,<3"
    )
    .add_local_python_source("dacn")
)


def _sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _pinned_validation():
    """Return pinned index groups, labels/arrays and the verified registered partition."""
    import csv
    import io

    import numpy as np

    from dacn.data.shards import ShardDataset
    from dacn.evaluation.registered_partition import registered_partition
    from dacn.evaluation.sweep_bootstrap import rebuild_registered_partition

    key = hashlib.sha256(f"{DATASET}@{REVISION}".encode()).hexdigest()[:20]
    validation_dir = DATA_ROOT / key / "data" / "val"
    index_bytes = (validation_dir / "index.csv").read_bytes()
    if _sha_bytes(index_bytes) != INDEX_SHA:
        raise ValueError("Pinned validation index differs")
    rows = list(csv.DictReader(io.StringIO(index_bytes.decode())))
    groups = np.array([row["patient_group_id"] for row in rows])
    data = ShardDataset(validation_dir, "val", include_masks=True)
    if len(data) != 83:
        raise ValueError("Expected 83 validation samples")
    partition_bytes = rebuild_registered_partition(index_bytes, data.labels, groups)
    if _sha_bytes(partition_bytes) != PARTITION_SHA:
        raise ValueError("Rebuilt partition differs from the registered 40/43 membership")
    _, holdout = registered_partition(partition_bytes, index_bytes, groups)
    images, masks = [], []
    for i in range(len(data)):
        image_value, mask, _ = data[i]
        images.append(image_value[0].numpy())
        masks.append(mask.numpy())
    dataset = {"images": np.stack(images), "masks": np.stack(masks), "labels": data.labels}
    return groups, holdout, dataset


def _iter_verified_runs(token, dataset, run_ids=None, keep_reconstructions=False):
    """Yield verified producer runs; arrays must equal the pinned validation shards."""
    import io

    import numpy as np
    from huggingface_hub import hf_hub_download

    from dacn.evaluation.sweep_bootstrap import SWEEP_RUNS

    for start, steps1, steps2, run_id, npz_sha, results_sha in SWEEP_RUNS:
        if run_ids is not None and run_id not in run_ids:
            continue
        payloads = {}
        for name, digest in (
            ("results.json", results_sha),
            ("comparison/private_ablation.npz", npz_sha),
        ):
            path = hf_hub_download(
                MODEL_REPO,
                f"evaluations/{run_id}/{name}",
                revision=MODEL_REVISION,
                token=token,
                cache_dir="/tmp/hf-cache",
            )
            payloads[name] = Path(path).read_bytes()
            if _sha_bytes(payloads[name]) != digest:
                raise ValueError("Producer artifact checksum differs: " + run_id)
        skip = () if keep_reconstructions else ("initials", "finals")
        with np.load(
            io.BytesIO(payloads["comparison/private_ablation.npz"]), allow_pickle=False
        ) as archive:
            arrays = {k: archive[k].copy() for k in archive.files if k not in skip}
        if not (
            np.array_equal(arrays["images"], dataset["images"])
            and np.array_equal(arrays["masks"], dataset["masks"])
            and np.array_equal(arrays["labels"].astype(int), np.asarray(dataset["labels"], int))
        ):
            raise ValueError("Cached arrays differ from pinned dataset order/content: " + run_id)
        verified = {"results_sha256": results_sha, "private_ablation_sha256": npz_sha}
        yield (
            (start, steps1, steps2),
            run_id,
            arrays,
            json.loads(payloads["results.json"]),
            verified,
        )


def _bootstrap(token, mode, groups, holdout, dataset):
    from dacn.evaluation.sweep_bootstrap import REFERENCE_RUN, bootstrap_sweep, collect_candidates

    runs, verified = {}, {}
    for spec, run_id, arrays, results, checks in _iter_verified_runs(token, dataset):
        runs[run_id] = (spec, arrays, results)
        verified[run_id] = checks
    truth, maps, thresholds, expected = collect_candidates(runs, holdout)
    del runs
    report = bootstrap_sweep(
        truth, maps, groups[holdout], thresholds, expected, replicates=REPLICATES[mode]
    )
    report.update(
        scope=SCOPE + " Intervals are conditional on this cohort and protocol.",
        holdout_samples=int(len(holdout)),
        producer_artifacts=verified,
        reference_run=REFERENCE_RUN,
    )
    return report


def _quality(token, mode, holdout, dataset):
    import numpy as np

    from dacn.evaluation.noise_quality import producer_detection, reconstruction_quality
    from dacn.evaluation.sweep_bootstrap import REFERENCE_RUN

    run_ids = {REFERENCE_RUN} if mode == "smoke" else None
    everything = np.arange(len(dataset["labels"]))
    rows = []
    for spec, run_id, arrays, results, checks in _iter_verified_runs(
        token, dataset, run_ids=run_ids, keep_reconstructions=True
    ):
        rows.append(
            {
                "evaluation_id": run_id,
                "phase1_starting_timestep": spec[0],
                "phase1_inference_steps": spec[1],
                "phase2_inference_steps": spec[2],
                "phase2_starting_timestep": results["config"]["phase2_starting_timestep"],
                "mask_threshold": results["parameters"]["mask_threshold"],
                "all83": reconstruction_quality(arrays, everything),
                "holdout43": reconstruction_quality(arrays, holdout),
                "producer_holdout_detection": producer_detection(results),
                **checks,
            }
        )
        del arrays
    return {
        "method": METHODS["quality"],
        "scope": SCOPE + " Ground-truth masks used only for post-hoc aggregates.",
        "foreground_rule": "input > -0.95 inside the original 240x240 crop",
        "aggregation": "pixel-micro means over the stated tissue class",
        "runs": rows,
    }


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=60 * 60,
    volumes={str(DATA_ROOT): data_volume, str(RUN_ROOT): run_volume},
    secrets=[modal.Secret.from_name(HF_SECRET)],
)
def run_sweep_analysis_remote(
    analysis: str, evaluation_id: str, code_revision: str, working_tree_dirty: bool, mode: str
):
    import time

    from huggingface_hub import HfApi

    from dacn.cloud.modal_train import (
        _file_checksums,
        _json_safe,
        _reject_patient_identifiers,
        _write_run_manifest,
        _write_source_snapshot,
    )

    if (
        analysis not in METHODS
        or mode not in REPLICATES
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", evaluation_id)
    ):
        raise ValueError("Invalid sweep analysis request")
    output = RUN_ROOT / "evaluations" / evaluation_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "evaluation_id": evaluation_id,
        "method": METHODS[analysis],
        "mode": mode,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "split": "validation",
        "dataset_repo_id": DATASET,
        "dataset_revision": REVISION,
        "model_repo_revision": MODEL_REVISION,
        "index_sha256": INDEX_SHA,
        "validation_partition_sha256": PARTITION_SHA,
        "code_revision": code_revision,
        "working_tree_dirty": working_tree_dirty,
        "source_snapshot_sha256": _write_source_snapshot(output),
        "held_out_test_used": False,
        "new_model_calls": 0,
        "threshold_fitting": False,
        "gpu": None,
    }

    def persist():
        _write_run_manifest(output, manifest)
        run_volume.commit()

    persist()
    started = time.perf_counter()
    try:
        token = os.environ["HF_TOKEN"]
        if not HfApi(token=token).repo_info(MODEL_REPO).private:
            raise ValueError("Private artifact repository required")
        groups, holdout, dataset = _pinned_validation()
        if analysis == "bootstrap":
            report = _bootstrap(token, mode, groups, holdout, dataset)
        else:
            report = _quality(token, mode, holdout, dataset)
        report = _json_safe(report)
        _reject_patient_identifiers(report)
        (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        manifest.update(
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            total_seconds=time.perf_counter() - started,
            artifacts=_file_checksums(output),
        )
        persist()
        try:
            commit = HfApi(token=token).upload_folder(
                folder_path=str(output),
                path_in_repo=f"evaluations/{evaluation_id}",
                repo_id=MODEL_REPO,
                commit_message=f"Store sweep {analysis} {evaluation_id}",
            )
            manifest.update(artifact_sync_status="uploaded", hub_commit=commit.oid)
        except Exception as exc:  # Results stay on the Volume and are returned.
            manifest.update(artifact_sync_status="failed", artifact_sync_error=type(exc).__name__)
        persist()
        return {"manifest": manifest, "results": report}
    except BaseException as exc:
        manifest.update(
            status="failed", error=type(exc).__name__, total_seconds=time.perf_counter() - started
        )
        persist()
        raise


@app.local_entrypoint()
def main(
    evaluation_id: str,
    code_revision: str,
    analysis: str = "bootstrap",
    mode: str = "smoke",
    working_tree_dirty: bool = True,
):
    result = run_sweep_analysis_remote.remote(
        analysis, evaluation_id, code_revision, working_tree_dirty, mode
    )
    destination = Path("runs") / "bootstrap" / evaluation_id
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in result.items():
        (destination / f"{name}.json").write_bytes(
            (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        )
    print(json.dumps(result["manifest"], indent=2))

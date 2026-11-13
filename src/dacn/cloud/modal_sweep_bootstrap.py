"""CPU-only paired patient-group bootstrap for the 09 Nov noise/DDIM sweep.

Validation only. Reuses producer score maps and calibration thresholds; no model
calls, threshold fitting, selection or test access. Requires the shared team
Workspace: Volumes are looked up without creation and the Hub Secret must exist.

    modal run --detach -m dacn.cloud.modal_sweep_bootstrap \
        --evaluation-id sweep-bootstrap-smoke-UNIQUE --code-revision GIT_SHA --mode smoke
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import modal

app = modal.App("dacn-sweep-bootstrap")
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
PARTITION_PATH = "final-evaluation/final-fourarm-20261110-b/artifacts/validation_partition"
PARTITION_SHA = "5e8a8ce1ad3a952293a6798ae72ef342aa23676e5559fd468acb76422a27d184"
REPLICATES = {"smoke": 20, "full": 1000}

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


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=60 * 60,
    volumes={str(DATA_ROOT): data_volume, str(RUN_ROOT): run_volume},
    secrets=[modal.Secret.from_name("huggingface-secret-2")],
)
def run_sweep_bootstrap_remote(
    evaluation_id: str, code_revision: str, working_tree_dirty: bool, mode: str
):
    import csv
    import io
    import os
    import time

    import numpy as np
    from huggingface_hub import HfApi, hf_hub_download

    from dacn.cloud.modal_train import (
        _file_checksums,
        _json_safe,
        _reject_patient_identifiers,
        _write_run_manifest,
        _write_source_snapshot,
    )
    from dacn.data.shards import ShardDataset
    from dacn.evaluation.registered_partition import registered_partition
    from dacn.evaluation.sweep_bootstrap import SWEEP_RUNS, bootstrap_sweep, collect_candidates

    if mode not in REPLICATES or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", evaluation_id
    ):
        raise ValueError("Invalid sweep bootstrap request")
    output = RUN_ROOT / "evaluations" / evaluation_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "evaluation_id": evaluation_id,
        "method": "paired_patient_group_bootstrap_ddim_noise_step_sweep",
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
        key = hashlib.sha256(f"{DATASET}@{REVISION}".encode()).hexdigest()[:20]
        validation_dir = DATA_ROOT / key / "data" / "val"
        index_bytes = (validation_dir / "index.csv").read_bytes()
        partition_bytes = (RUN_ROOT / PARTITION_PATH).read_bytes()
        if _sha_bytes(index_bytes) != INDEX_SHA or _sha_bytes(partition_bytes) != PARTITION_SHA:
            raise ValueError("Pinned validation index or registered partition differs")
        rows = list(csv.DictReader(io.StringIO(index_bytes.decode())))
        groups = np.array([row["patient_group_id"] for row in rows])
        _, holdout = registered_partition(partition_bytes, index_bytes, groups)
        data = ShardDataset(validation_dir, "val", include_masks=True)

        runs, verified = {}, {}
        for start, steps1, steps2, run_id, npz_sha, results_sha in SWEEP_RUNS:
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
            with np.load(
                io.BytesIO(payloads["comparison/private_ablation.npz"]), allow_pickle=False
            ) as archive:
                arrays = {
                    k: archive[k].copy() for k in archive.files if k not in ("initials", "finals")
                }
            runs[run_id] = ((start, steps1, steps2), arrays, json.loads(payloads["results.json"]))
            verified[run_id] = {"results_sha256": results_sha, "private_ablation_sha256": npz_sha}
            del payloads
        reference = runs["autoddpm-ddim2030-full83-t4-20261106T074200Z"][1]
        if len(data) != 83:
            raise ValueError("Expected 83 validation samples")
        for i in range(len(data)):
            image_value, mask, label = data[i]
            if not (
                np.array_equal(reference["images"][i], image_value[0].numpy())
                and np.array_equal(reference["masks"][i], mask.numpy())
                and int(reference["labels"][i]) == int(label)
            ):
                raise ValueError("Cached arrays differ from pinned dataset order/content")
        truth, maps, thresholds, expected = collect_candidates(runs, holdout)
        del runs
        report = bootstrap_sweep(
            truth, maps, groups[holdout], thresholds, expected, replicates=REPLICATES[mode]
        )
        report.update(
            scope=(
                "Exploratory validation on the repeatedly inspected 43-image group holdout; "
                "producer maps and calibration thresholds fixed; one training and sampling seed; "
                "test unused. Intervals are conditional on this cohort and protocol."
            ),
            holdout_samples=int(len(holdout)),
            producer_artifacts=verified,
            reference_run="autoddpm-ddim2030-full83-t4-20261106T074200Z",
        )
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
        HfApi(token=token).upload_folder(
            folder_path=str(output),
            path_in_repo=f"evaluations/{evaluation_id}",
            repo_id=MODEL_REPO,
            commit_message=f"Store sweep bootstrap {evaluation_id}",
        )
        manifest["artifact_sync_status"] = "uploaded"
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
    evaluation_id: str, code_revision: str, mode: str = "smoke", working_tree_dirty: bool = True
):
    result = run_sweep_bootstrap_remote.remote(
        evaluation_id, code_revision, working_tree_dirty, mode
    )
    destination = Path("runs") / "bootstrap" / evaluation_id
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in result.items():
        (destination / f"{name}.json").write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n"
        )
    print(json.dumps(result["manifest"], indent=2))

"""Single frozen test evaluation of Final LPIPS x brightness (CPU, no model calls).

Configuration and thresholds were frozen on validation before this run
(``FROZEN_HYBRID_*`` in ``dacn.evaluation.postprocess_hybrid``). The Final LPIPS
and brightness test maps are the private chunks of ``test-ci-full-20261114T042202Z``
on this Workspace's Volume. Each chunk's sample identity is checked. Gates before
any hybrid metric is reported:
  1. the unprocessed Final LPIPS and brightness maps reproduce the published
     frozen four-arm test metrics with their frozen thresholds;
  2. ``smoke`` (validation only, no test access) reproduces the hybrid holdout
     metrics from cached validation maps with the frozen thresholds.
Nothing is fitted or selected on test. Only aggregates are uploaded.

    DACN_HF_SECRET=huggingface modal run --detach -m dacn.cloud.modal_hybrid_test \
        --evaluation-id hybrid-test-UNIQUE --code-revision GIT_SHA --mode full \
        --no-working-tree-dirty
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import modal

from dacn.cloud.modal_sweep_bootstrap import (
    DATA_ROOT,
    DATASET,
    HF_SECRET,
    MODEL_REPO,
    REVISION,
    RUN_ROOT,
    data_volume,
    image,
    run_volume,
)

app = modal.App("dacn-hybrid-test")
SOURCE_RUN = "test-ci-full-20261114T042202Z"
TEST_INDEX_SHA = "eb6b79086e16f2c5affa73495d2501f52267cb70f9bbb6ef4f6e49018d8cd613"
CHUNK = 250
# docs/evidence/2026-11-14-test-ci-results.json (pixel AP, Dice, image AUROC, image AP);
# those values reproduced the published 10 Nov aggregate within 5.2e-7.
PUBLISHED = {
    "final_lpips": (
        0.31822613370648617,
        0.36429250238689437,
        0.805989152892562,
        0.9458630561844523,
    ),
    "brightness": (0.5114340237017682, 0.416976905015636, 0.6436497933884298, 0.9013407249766718),
}
REPLAY_ATOL = 1e-6


def _metrics(report):
    return (
        report["pixel"]["auprc"],
        report["pixel"]["f1"],
        report["image"]["auroc"],
        report["image"]["auprc"],
    )


@app.function(
    image=image,
    cpu=8,
    memory=65536,
    timeout=3 * 60 * 60,
    volumes={str(DATA_ROOT): data_volume, str(RUN_ROOT): run_volume},
    secrets=[modal.Secret.from_name(HF_SECRET)],
)
def run_hybrid_test_remote(
    evaluation_id: str, code_revision: str, working_tree_dirty: bool, mode: str
):
    import csv
    import hashlib
    import io
    import time

    import numpy as np
    from huggingface_hub import HfApi

    from dacn.cloud.modal_sweep_bootstrap import _iter_verified_runs, _pinned_validation
    from dacn.cloud.modal_train import (
        _file_checksums,
        _json_safe,
        _reject_patient_identifiers,
        _write_run_manifest,
        _write_source_snapshot,
    )
    from dacn.evaluation.frozen_test_ci import (
        FROZEN_THRESHOLDS,
        TEST_SAMPLES,
        image_group_bootstrap,
        image_scores,
    )
    from dacn.evaluation.group_bootstrap import paired_group_bootstrap
    from dacn.evaluation.postprocess_hybrid import (
        FROZEN_HYBRID_ARM,
        FROZEN_HYBRID_THRESHOLDS,
        HYBRID_REPLAY,
        frozen_hybrid_map,
    )
    from dacn.evaluation.sweep_bootstrap import REFERENCE_RUN
    from dacn.evaluation.validation import evaluate_maps

    if mode not in {"smoke", "full"} or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", evaluation_id
    ):
        raise ValueError("Invalid hybrid test request")
    output = RUN_ROOT / "evaluations" / evaluation_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "evaluation_id": evaluation_id,
        "method": "frozen_final_lpips_x_brightness_single_test_evaluation",
        "mode": mode,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "split": "validation-smoke" if mode == "smoke" else "test",
        "dataset_repo_id": DATASET,
        "dataset_revision": REVISION,
        "source_map_run": SOURCE_RUN,
        "frozen_arm": FROZEN_HYBRID_ARM,
        "frozen_thresholds": FROZEN_HYBRID_THRESHOLDS,
        "code_revision": code_revision,
        "working_tree_dirty": working_tree_dirty,
        "source_snapshot_sha256": _write_source_snapshot(output),
        "held_out_test_used": mode == "full",
        "test_access_count_after_freeze": 5 if mode == "full" else None,
        "new_model_calls": 0,
        "threshold_fitting": False,
        "selection": "none; configuration frozen on validation before this run",
        "gpu": None,
    }

    def persist():
        _write_run_manifest(output, manifest)
        run_volume.commit()

    persist()
    started = time.perf_counter()
    try:
        token = os.environ["HF_TOKEN"]
        api = HfApi(token=token)
        if not api.repo_info(MODEL_REPO).private:
            raise ValueError("Private artifact repository required")
        if mode == "smoke":
            groups, holdout, dataset = _pinned_validation()
            [(_, _, cached, _, verified)] = list(
                _iter_verified_runs(token, dataset, run_ids={REFERENCE_RUN})
            )
            hybrid = frozen_hybrid_map(cached["autoddpm_lpips"], cached["brightness"])[holdout]
            truth = cached["masks"][holdout, 8:-8, 8:-8]
            held = evaluate_maps(
                hybrid,
                truth,
                cached["labels"][holdout],
                split="validation",
                thresholds=FROZEN_HYBRID_THRESHOLDS,
            )
            ours = (held["pixel"]["auprc"], held["pixel"]["f1"], held["image"]["auroc"])
            error = max(abs(a - b) for a, b in zip(ours, HYBRID_REPLAY, strict=True))
            if error > 1e-9:
                raise ValueError("Frozen hybrid does not replay validation holdout metrics")
            report = {
                "mode": "smoke",
                "test_accessed": False,
                "validation_holdout_replay_abs_error": error,
                "producer_artifacts": {REFERENCE_RUN: verified},
            }
        else:
            key = hashlib.sha256(f"{DATASET}@{REVISION}".encode()).hexdigest()[:20]
            index_bytes = (DATA_ROOT / key / "data" / "test" / "index.csv").read_bytes()
            if hashlib.sha256(index_bytes).hexdigest() != TEST_INDEX_SHA:
                raise ValueError("Pinned test index differs from the source map run")
            rows = list(csv.DictReader(io.StringIO(index_bytes.decode())))
            if len(rows) != TEST_SAMPLES:
                raise ValueError("Unexpected test sample count")
            sample_ids = [row["sample_id"] for row in rows]
            groups = np.array([row["patient_group_id"] for row in rows])
            chunk_dir = RUN_ROOT / "evaluations" / SOURCE_RUN / "chunks"
            truth = np.empty((TEST_SAMPLES, 240, 240), dtype=np.uint8)
            final = np.empty((TEST_SAMPLES, 240, 240), dtype=np.float32)
            bright = np.empty((TEST_SAMPLES, 240, 240), dtype=np.float32)
            for first in range(0, TEST_SAMPLES, CHUNK):
                last = min(first + CHUNK, TEST_SAMPLES)
                identity = hashlib.sha256("\n".join(sample_ids[first:last]).encode()).hexdigest()
                path = chunk_dir / f"private_chunk_{first:05d}.npz"
                with np.load(path, allow_pickle=False) as archive:
                    if str(archive["identity"]) != identity:
                        raise ValueError("Source chunk identity differs")
                    truth[first:last] = archive["truth"]
                    final[first:last] = archive["final_lpips"]
                    bright[first:last] = archive["brightness"]
            labels = truth.reshape(TEST_SAMPLES, -1).any(axis=1).astype(int)
            replay = {}
            for arm, values in (("final_lpips", final), ("brightness", bright)):
                ours = _metrics(
                    evaluate_maps(
                        values, truth, labels, split="test", thresholds=FROZEN_THRESHOLDS[arm]
                    )
                )
                replay[arm] = max(abs(a - b) for a, b in zip(ours, PUBLISHED[arm], strict=True))
            if max(replay.values()) > REPLAY_ATOL:
                raise ValueError(f"Source maps do not reproduce published test metrics: {replay}")
            hybrid = frozen_hybrid_map(final, bright)
            point = evaluate_maps(
                hybrid, truth, labels, split="test", thresholds=FROZEN_HYBRID_THRESHOLDS
            )
            manifest["progress"] = {"stage": "bootstrap"}
            persist()
            maps = {
                FROZEN_HYBRID_ARM: hybrid,
                "final_lpips": final.astype(np.float64),
                "brightness": bright.astype(np.float64),
            }
            del final, bright
            thresholds = {
                FROZEN_HYBRID_ARM: FROZEN_HYBRID_THRESHOLDS["pixel"],
                "final_lpips": FROZEN_THRESHOLDS["final_lpips"]["pixel"],
                "brightness": FROZEN_THRESHOLDS["brightness"]["pixel"],
            }
            image_values = {name: image_scores(values) for name, values in maps.items()}
            report = {
                "mode": "full",
                "test_accessed": True,
                "samples": TEST_SAMPLES,
                "normal_slices": int((labels == 0).sum()),
                "abnormal_slices": int(labels.sum()),
                "source_replay_max_abs_error": replay,
                "frozen_hybrid_test": point,
                "image_bootstrap": image_group_bootstrap(
                    labels, image_values, groups, reference=FROZEN_HYBRID_ARM
                ),
                "pixel_bootstrap": paired_group_bootstrap(
                    truth,
                    maps,
                    groups,
                    thresholds,
                    reference=FROZEN_HYBRID_ARM,
                    replicates=1000,
                    seed=17,
                ),
                "scope": (
                    "Single test evaluation of a configuration frozen on validation after "
                    "several development rounds; fifth test access; one training and "
                    "sampling seed; intervals conditional on this model and cohort."
                ),
            }
        report = _json_safe(report)
        _reject_patient_identifiers(report)
        (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        manifest.update(
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            total_seconds=time.perf_counter() - started,
            artifacts=_file_checksums(output),
        )
        manifest.pop("progress", None)
        persist()
        try:
            commit = api.upload_folder(
                folder_path=str(output),
                path_in_repo=f"evaluations/{evaluation_id}",
                repo_id=MODEL_REPO,
                allow_patterns=["results.json", "run_manifest.json", "source_snapshot.zip"],
                commit_message=f"Store hybrid test {evaluation_id}",
            )
            manifest.update(artifact_sync_status="uploaded", hub_commit=commit.oid)
        except Exception as exc:  # noqa: BLE001 - upload failure is recorded; results stay on the Volume
            manifest.update(artifact_sync_status="failed", artifact_sync_error=type(exc).__name__)
        persist()
        return {"manifest": manifest, "results": report}
    except BaseException as exc:
        manifest.update(status="failed", error=type(exc).__name__)
        manifest["error_message"] = str(exc)[:300]
        persist()
        raise


@app.local_entrypoint()
def main(
    evaluation_id: str, code_revision: str, mode: str = "smoke", working_tree_dirty: bool = True
):
    result = run_hybrid_test_remote.remote(evaluation_id, code_revision, working_tree_dirty, mode)
    destination = Path("runs") / "hybrid-test" / evaluation_id
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in result.items():
        (destination / f"{name}.json").write_bytes(
            (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        )
    print(json.dumps(result["manifest"], indent=2))

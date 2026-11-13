import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from dacn.evaluation import sweep_bootstrap as sb

REF = sb.REFERENCE_RUN
OTHER = "autoddpm-ddim2030-t500-full-20261109T095805Z"


def _arrays(rng, n=6):
    masks = np.zeros((n, 256, 256), dtype=np.uint8)
    masks[n // 2 :, 100:120, 100:120] = 1
    arrays = {
        "images": rng.standard_normal((n, 256, 256)).astype(np.float32),
        "masks": masks,
        "labels": (np.arange(n) >= n // 2).astype(np.uint8),
    }
    truth = masks[:, 8:-8, 8:-8].astype(float)
    for arm in [*sb.SWEEP_ARMS, *sb.REFERENCE_CONTROLS]:
        arrays[arm] = (truth + rng.uniform(0, 1.5, truth.shape)).astype(np.float32)
    return arrays


def _results(arrays, spec, holdout, threshold=1.0):
    start, steps1, steps2 = spec
    truth = arrays["masks"][holdout, 8:-8, 8:-8].ravel()
    metrics, thresholds = {}, {}
    for arm in [*sb.SWEEP_ARMS, *sb.REFERENCE_CONTROLS]:
        scores = arrays[arm][holdout].ravel()
        pred = scores >= threshold
        tp = (pred & (truth == 1)).sum()
        metrics[arm] = {
            "group_holdout": {
                "pixel": {
                    "auprc": average_precision_score(truth, scores),
                    "f1": 2 * tp / (pred.sum() + truth.sum()),
                }
            }
        }
        thresholds[arm] = {"pixel": threshold, "image": threshold}
    return {
        "held_out_test_used": False,
        "smoke": False,
        "samples": 83,
        "config": {
            "phase1_starting_timestep": start,
            "phase1_inference_steps": steps1,
            "phase2_inference_steps": steps2,
        },
        "parameters": {"score_thresholds": thresholds},
        "metrics": metrics,
    }


def _runs(rng, holdout):
    shared = _arrays(rng)
    other = {k: (v.copy() if k in sb.SHARED_ARRAYS else v) for k, v in _arrays(rng).items()}
    other.update({k: shared[k] for k in sb.SHARED_ARRAYS})
    return {
        REF: ((300, 20, 30), shared, _results(shared, (300, 20, 30), holdout)),
        OTHER: ((500, 20, 30), other, _results(other, (500, 20, 30), holdout)),
    }


def test_collect_candidates_names_thresholds_and_controls():
    holdout = np.array([0, 2, 3, 5])
    truth, maps, thresholds, expected = sb.collect_candidates(
        _runs(np.random.default_rng(0), holdout), holdout
    )
    assert truth.shape == (4, 240, 240)
    assert set(maps) == {
        "final_lpips_t300_20_30",
        "initial_lpips_t300_20_30",
        "final_lpips_t500_20_30",
        "initial_lpips_t500_20_30",
        *sb.REFERENCE_CONTROLS.values(),
    }
    assert set(thresholds) == set(maps) == set(expected)
    assert all(values.shape == truth.shape for values in maps.values())


def test_bootstrap_replays_producer_metrics_and_pairs_reference():
    holdout = np.array([0, 2, 3, 5])
    groups = np.array(["a", "b", "c", "d"])
    truth, maps, thresholds, expected = sb.collect_candidates(
        _runs(np.random.default_rng(1), holdout), holdout
    )
    report = sb.bootstrap_sweep(truth, maps, groups, thresholds, expected, replicates=10)
    assert report["reference"] == sb.REFERENCE
    assert report["producer_replay_max_abs_error"] < 1e-9
    assert report["candidates"][sb.REFERENCE]["delta_ap_ci95"] == [0, 0]


def test_bootstrap_rejects_mismatched_producer_metrics():
    holdout = np.array([0, 2, 3, 5])
    truth, maps, thresholds, expected = sb.collect_candidates(
        _runs(np.random.default_rng(2), holdout), holdout
    )
    expected["brightness"] = (expected["brightness"][0] + 1e-3, expected["brightness"][1])
    with pytest.raises(ValueError, match="replay failed: brightness"):
        sb.bootstrap_sweep(truth, maps, np.array(list("abcd")), thresholds, expected, replicates=5)


def test_collect_candidates_rejects_unpaired_samples_and_test_use():
    holdout = np.array([0, 2, 3, 5])
    runs = _runs(np.random.default_rng(3), holdout)
    runs[OTHER][1]["masks"] = runs[OTHER][1]["masks"].copy()
    runs[OTHER][1]["masks"][0, 0, 0] = 1
    with pytest.raises(ValueError, match="Sample order/content differs"):
        sb.collect_candidates(runs, holdout)
    runs = _runs(np.random.default_rng(3), holdout)
    runs[OTHER][2]["held_out_test_used"] = True
    with pytest.raises(ValueError, match="Unexpected producer configuration"):
        sb.collect_candidates(runs, holdout)


def test_sweep_registry_matches_reference_and_has_unique_runs():
    ids = [run[3] for run in sb.SWEEP_RUNS]
    assert len(ids) == len(set(ids)) == 12 and sb.REFERENCE_RUN in ids
    specs = [(run[0], run[1], run[2]) for run in sb.SWEEP_RUNS]
    assert len(specs) == len(set(specs))
    assert all(len(run[4]) == len(run[5]) == 64 for run in sb.SWEEP_RUNS)


def test_rebuilt_partition_is_deterministic_group_disjoint_and_canonical():
    import hashlib
    import json

    rng = np.random.default_rng(5)
    groups = np.repeat([f"g{i}" for i in range(20)], 4)
    labels = rng.integers(0, 2, size=len(groups))
    index = b"sample_id,patient_group_id\n"
    payload = sb.rebuild_registered_partition(index, labels, groups)
    assert payload == sb.rebuild_registered_partition(index, labels, groups)
    record = json.loads(payload)
    assert list(record) == sorted(record) and payload.endswith(b"}\n")
    assert record["index_sha256"] == hashlib.sha256(index).hexdigest()
    cal, hold = record["calibration_positions"], record["holdout_positions"]
    assert sorted(cal + hold) == list(range(len(groups)))
    assert set(groups[cal]).isdisjoint(groups[hold])

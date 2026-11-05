import argparse
import csv
import hashlib
import io
import json
import time
import uuid
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from dacn.evaluation.group_bootstrap import paired_group_bootstrap
from dacn.evaluation.noise_fusion import GAUSSIAN_SHA, INDEX_SHA, SIMPLEX20_SHA, normal_cdf
from dacn.evaluation.scores import paired_score_maps

parser = argparse.ArgumentParser(
    description="Bootstrap registered Gaussian30/Simplex20 validation caches; private inputs, aggregate output"
)
for name in [
    "cache-root",
    "pairs-root",
    "index",
    "frozen-selection",
    "reference-results",
    "output",
]:
    parser.add_argument("--" + name, type=Path, required=True)
args = parser.parse_args()
root = args.output
root.mkdir(parents=True, exist_ok=False)
manifest = {
    "status": "started",
    "split": "exploratory_validation_group_holdout",
    "test_used": False,
    "seed": 17,
    "replicates": 1000,
}
(root / "manifest.json").write_text(json.dumps(manifest))
started = time.perf_counter()
try:
    base = args.cache_root
    paths = [
        base
        / "ddpm-val-t4b8-e30-t300-20261102T071516Z/ddpm_epoch_030_t0300_ancestral_reconstructions.npz",
        base
        / "ddpm-val-simplex-e20-t300-20261104T105703Z/ddpm_epoch_020_t0300_ancestral_reconstructions.npz",
    ]
    payloads = [p.read_bytes() for p in paths]
    assert [hashlib.sha256(p).hexdigest() for p in payloads] == [GAUSSIAN_SHA, SIMPLEX20_SHA]
    caches = []
    for b in payloads:
        with np.load(io.BytesIO(b), allow_pickle=False) as z:
            caches.append({k: z[k] for k in z.files})
    x, y, labels = [caches[0][k] for k in ["images", "masks", "labels"]]
    for k in ["images", "masks", "labels"]:
        assert np.array_equal(caches[0][k], caches[1][k])
    index = args.index.read_bytes()
    assert hashlib.sha256(index).hexdigest() == INDEX_SHA
    rows = list(csv.DictReader(io.StringIO(index.decode())))
    assert all(r["split"] == "val" for r in rows) and len(rows) == len(x) == 83
    assert np.array_equal(labels, [int(r["label"]) for r in rows])
    # Verify the UUID-bound private conversions retain exactly this index order.
    for name, cache in zip(["gaussian30", "simplex20"], caches):
        evidence = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / f"docs/evidence/2026-11-05-b-{name}-spatial.json"
            ).read_text()
        )
        pair = (args.pairs_root / f"{name}-pairs.npz").read_bytes()
        assert hashlib.sha256(pair).hexdigest() == evidence["reconstruction_artifact_sha256"]
        with np.load(io.BytesIO(pair), allow_pickle=False) as z:
            assert z["sample_ids"].tolist() == [str(uuid.UUID(r["sample_id"])) for r in rows]
            assert np.array_equal(z["reconstructions"][:, 0], cache["reconstructions"])
    groups = np.array([r["patient_group_id"] for r in rows])
    cal, hold = next(
        StratifiedGroupKFold(2, shuffle=True, random_state=17).split(np.zeros(83), labels, groups)
    )
    assert set(groups[cal]).isdisjoint(groups[hold])
    frozen_path = args.frozen_selection
    frozen_bytes = frozen_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    assert frozen["hashes"] == {
        "gaussian": GAUSSIAN_SHA,
        "simplex": SIMPLEX20_SHA,
        "index": INDEX_SHA,
    }
    maps = []
    for cache in caches:
        maps.append(
            np.stack(
                [paired_score_maps(a, b)["local_ssim"] for a, b in zip(x, cache["reconstructions"])]
            )
        )
    gc = normal_cdf(maps[0], cal[labels[cal] == 0])
    sc = normal_cdf(maps[1], cal[labels[cal] == 0])
    candidates = {
        "gaussian_cdf": gc[hold],
        "simplex_raw": maps[1][hold],
        "fusion_gaussian_0.75": (0.75 * gc + 0.25 * sc)[hold],
        "brightness_control": x[hold, 8:-8, 8:-8].astype(np.float64),
    }
    thresholds = {n: frozen["selection"][n]["thresholds"]["pixel"] for n in candidates}
    report = paired_group_bootstrap(
        y[hold, 8:-8, 8:-8],
        candidates,
        groups[hold],
        thresholds,
        reference=frozen["selected_by_calibration_ap"],
    )
    previous = json.loads(args.reference_results.read_text())
    for n, m in report["candidates"].items():
        assert np.isclose(
            m["pixel_ap"],
            previous["holdout_results"][n]["pixel"]["average_precision"],
            atol=1e-12,
            rtol=0,
        )
        assert np.isclose(
            m["pixel_dice_micro"],
            previous["holdout_results"][n]["pixel"]["dice_micro"],
            atol=1e-12,
            rtol=0,
        )
    report.update(
        source_cache_hashes=[GAUSSIAN_SHA, SIMPLEX20_SHA],
        index_sha256=INDEX_SHA,
        frozen_selection_sha256=hashlib.sha256(frozen_bytes).hexdigest(),
        cpu_seconds=time.perf_counter() - started,
        scope="Exploratory previously inspected validation; differing training budgets; one trained seed per model. No candidate reselection. Conditional intervals do not cover training variability.",
        test_used=False,
    )
    (root / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    manifest.update(status="completed", cpu_seconds=report["cpu_seconds"])
    print(json.dumps(report, indent=2))
except BaseException as e:
    manifest.update(
        status="interrupted" if isinstance(e, KeyboardInterrupt) else "failed",
        error_class=type(e).__name__,
    )
    raise
finally:
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

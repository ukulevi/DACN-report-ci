import argparse
import csv
import hashlib
import io
import json
import time
import uuid
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from dacn.evaluation.artifacts import load_validation_pairs
from dacn.evaluation.group_bootstrap import paired_group_bootstrap
from dacn.evaluation.scores import paired_score_maps
from dacn.evaluation.validation import evaluate_maps, select_validation_threshold

p = argparse.ArgumentParser()
for field in [
    "gaussian",
    "gaussian-provenance",
    "dataset",
    "simplex",
    "simplex-provenance",
    "output",
]:
    p.add_argument("--" + field, type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
manifest = {"status": "started", "split": "exploratory_validation", "test_used": False}
(a.output / "manifest.json").write_text(json.dumps(manifest))
started = time.perf_counter()
try:
    ps = [json.loads(path.read_text()) for path in [a.gaussian_provenance, a.simplex_provenance]]
    expected = [
        "b4fc177db493fce2143a6412161068c7af73a25e8ae7ef9d88c117863c7e37f1",
        "1b9f50988f8a57634ddc9b3b9de1db93c4e324c5de7d42132db50525e4a36d07",
    ]
    for provenance, h in zip(ps, expected):
        assert (
            provenance["checkpoint_sha256"] == h
            and provenance["sampling_seed"] == 17
            and provenance["train_seed"] == 17
            and provenance["starting_timestep"] == 300
        )
    data = [
        load_validation_pairs(path, a.dataset, provenance)
        for path, provenance in zip([a.gaussian, a.simplex], ps)
    ]
    ids = []
    for path in [a.gaussian, a.simplex]:
        with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as z:
            ids.append([str(uuid.UUID(s)) for s in z["sample_ids"]])
    assert set(ids[0]) == set(ids[1])
    order = [ids[1].index(s) for s in ids[0]]
    x, g, y, labels, _, gh = data[0]
    sx, s, sy, sl, _, sh = data[1]
    s = s[order]
    assert (
        np.array_equal(x, sx[order])
        and np.array_equal(y, sy[order])
        and np.array_equal(labels, sl[order])
    )
    index = (a.dataset / "data/val/index.csv").read_bytes()
    assert (
        hashlib.sha256(index).hexdigest()
        == "57ad2e31df3ed4e625da00efaecd6fddc1e2a132329dfba6e60867caf1802db3"
    )
    rows = list(csv.DictReader(io.StringIO(index.decode())))
    index_labels = np.array([int(r["label"]) for r in rows])
    index_groups = np.array([r["patient_group_id"] for r in rows])
    ci, hi = next(
        StratifiedGroupKFold(2, shuffle=True, random_state=17).split(
            np.zeros(83), index_labels, index_groups
        )
    )
    positions = {v: i for i, v in enumerate(ids[0])}
    cal = np.array([positions[str(uuid.UUID(rows[i]["sample_id"]))] for i in ci])
    hold = np.array([positions[str(uuid.UUID(rows[i]["sample_id"]))] for i in hi])
    group_by_id = {str(uuid.UUID(r["sample_id"])): r["patient_group_id"] for r in rows}
    groups = np.array([group_by_id[sid] for sid in ids[0]])
    assert set(groups[cal]).isdisjoint(groups[hold])
    maps = {}
    for model, recon in [("gaussian20", g), ("simplex20", s)]:
        paired = [paired_score_maps(i, j) for i, j in zip(x, recon)]
        for method in ["pixel_difference", "local_ssim"]:
            maps[model + "_" + method] = np.stack([z[method] for z in paired])
    maps["brightness_control"] = x[:, 8:-8, 8:-8].astype(np.float64)
    truth = y[:, 8:-8, 8:-8]
    selection = {}
    for name, m in maps.items():
        cm = m[cal]
        flat = cm.reshape(len(cal), -1)
        im = np.partition(flat, -576, axis=1)[:, -576:].mean(axis=1)
        selection[name] = {
            "calibration_ap": float(average_precision_score(truth[cal].ravel(), cm.ravel())),
            "thresholds": {
                "pixel": select_validation_threshold(truth[cal].ravel(), cm.ravel()),
                "image": select_validation_threshold(labels[cal], im),
            },
        }
    winner = max(
        [n for n in maps if n != "brightness_control"], key=lambda n: selection[n]["calibration_ap"]
    )
    frozen = {
        "selection": selection,
        "selected_by_calibration_ap": winner,
        "candidate_order": list(maps),
        "ties": "first in candidate_order",
        "brightness_excluded": True,
        "gaussian_artifact_sha256": gh,
        "simplex_artifact_sha256": sh,
        "calibration_samples": len(cal),
        "holdout_samples": len(hold),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (a.output / "frozen_selection.json").write_text(json.dumps(frozen, indent=2))
    full = {n: evaluate_maps(m, truth, labels, split="validation") for n, m in maps.items()}
    # Check Simplex full-cohort outputs against independently reproduced saved metrics.
    expected_simplex = json.loads(
        Path("docs/evidence/2026-11-05-b-simplex20-reproduction.json").read_text()
    )["metrics"]
    for method in ["pixel_difference", "local_ssim"]:
        for key in ["average_precision", "auroc", "precision", "recall", "f1"]:
            assert np.isclose(
                full["simplex20_" + method]["pixel"][key],
                expected_simplex[method]["pixel"][key],
                atol=1e-12,
                rtol=0,
            )
    held = {
        n: evaluate_maps(
            m[hold],
            truth[hold],
            labels[hold],
            split="validation",
            thresholds=selection[n]["thresholds"],
        )
        for n, m in maps.items()
    }
    selected_maps = {
        n: maps[n][hold]
        for n in ["gaussian20_local_ssim", "simplex20_local_ssim", "brightness_control"]
    }
    boot = paired_group_bootstrap(
        truth[hold],
        selected_maps,
        groups[hold],
        {n: selection[n]["thresholds"]["pixel"] for n in selected_maps},
        reference="gaussian20_local_ssim",
    )
    report = {
        "frozen_selection": frozen,
        "descriptive_all83": full,
        "grouped_holdout": held,
        "bootstrap": boot,
        "provenance": [ps[0], ps[1]],
        "matched_optimizer_steps": 18760,
        "epochs_each": 20,
        "cpu_seconds": time.perf_counter() - started,
        "test_used": False,
        "caveats": "Previously examined validation, one train/sampling seed. Same optimizer budget does not isolate every training/source difference.",
    }
    (a.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    manifest.update(status="completed", cpu_seconds=report["cpu_seconds"])
    print(
        json.dumps(
            {
                "selected": winner,
                "holdout_pixel_metrics": {n: m["pixel"] for n, m in held.items()},
                "bootstrap": boot,
            },
            indent=2,
        )
    )
except BaseException as e:
    manifest.update(
        status="interrupted" if isinstance(e, KeyboardInterrupt) else "failed",
        error_class=type(e).__name__,
    )
    raise
finally:
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2))

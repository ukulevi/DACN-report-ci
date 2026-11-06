import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from huggingface_hub import try_to_load_from_cache
from sklearn.model_selection import StratifiedGroupKFold

from dacn.data.shards import ShardDataset
from dacn.evaluation.group_bootstrap import paired_group_bootstrap

root = Path("runs/audit-2026-11-06")
out = root / "b-ddim-bootstrap-results.json"
rev = "fae9b50a4b78114fb51d87e803c77d019668f523"
base = "evaluations/autoddpm-ddim2030-full83-t4-20261106T074200Z/"
other = "evaluations/lpips-normalcal-full83-t4-20261106T083649Z/"


def manifest(p):
    return json.loads((root / "hf-latest" / p / "run_manifest.json").read_text())


def cache(prefix, name):
    p = Path(try_to_load_from_cache("dacn-bmad-team/dacn-runs", prefix + name, revision=rev))
    assert (
        hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
        == manifest(prefix)["artifacts"][name]
    )
    return np.load(p, allow_pickle=False)


a = cache(base, "comparison/private_ablation.npz")
b = cache(other, "scores/private_layer_maps.npz")
assert (
    manifest(other)["source_maps_sha256"]
    == manifest(base)["artifacts"]["comparison/private_ablation.npz"]
)
split = Path("runs/b-validation/pinned-export/data/val")
assert (
    hashlib.file_digest((split / "index.csv").open("rb"), "sha256").hexdigest()
    == manifest(base)["index_sha256"]
)
rows = list(csv.DictReader((split / "index.csv").open(newline="")))
data = ShardDataset(split, "val", include_masks=True)
assert len(data) == 83 and data.sample_ids == [r["sample_id"] for r in rows]
for i in range(83):
    image, mask, label = data[i]
    assert np.array_equal(a["images"][i], image[0].numpy())
    assert np.array_equal(a["masks"][i], mask.numpy()) and int(a["labels"][i]) == int(label)
groups = np.array([r["patient_group_id"] for r in rows])
labels = a["labels"]
cal, hold = next(
    StratifiedGroupKFold(2, shuffle=True, random_state=17).split(np.zeros(83), labels, groups)
)
assert (
    len(cal) == 40
    and len(hold) == 43
    and len(set(groups[hold])) == 25
    and set(groups[cal]).isdisjoint(groups[hold])
)
assert np.array_equal(b["converted_positions"], np.arange(83))
assert np.allclose(b["original_lpips"], a["autoddpm_lpips"], atol=1e-6, rtol=0)
results = json.loads((root / "hf-latest" / base / "results.json").read_text())["metrics"]
nr = json.loads((root / "hf-latest" / other / "results.json").read_text())["metrics"][
    "naive_stitch_lpips"
]
maps = {k: a[k][hold] for k in ["autoddpm_lpips", "initial_lpips_only", "brightness"]}
maps["naive_stitch_lpips"] = b["naive_stitch_lpips"][hold]
metrics = {k: results[k]["group_holdout"]["pixel"] for k in maps if k != "naive_stitch_lpips"}
metrics["naive_stitch_lpips"] = nr["group_holdout"]["pixel"]
thresholds = {k: v["threshold"] for k, v in metrics.items()}
report = paired_group_bootstrap(
    a["masks"][hold, 8:-8, 8:-8],
    maps,
    groups[hold],
    thresholds,
    reference="autoddpm_lpips",
    replicates=1000,
    seed=17,
)
for k, v in report["candidates"].items():
    assert np.isclose(v["pixel_ap"], metrics[k]["average_precision"], atol=1e-12, rtol=0)
    assert np.isclose(v["pixel_dice_micro"], metrics[k]["dice_micro"], atol=1e-12, rtol=0)
report["scope"] = (
    "Exploratory validation, thresholds fixed; conditional on one model and selected protocol; test unused."
)
out.write_text(json.dumps(report, indent=2, allow_nan=False))
print(json.dumps(report, indent=2))

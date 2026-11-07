"""Bounded two-sample CPU controls acceptance; no fitting or test access."""

import argparse
import ctypes
import hashlib
import io
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import try_to_load_from_cache

from dacn.data.shards import ShardDataset
from dacn.evaluation.dependency_snapshot import version_snapshot
from dacn.evaluation.frozen_ddim import FrozenDDIMAdapter
from dacn.evaluation.frozen_inputs import capture_frozen_inputs
from dacn.evaluation.validation import evaluate_maps


def windows_peak_working_set():
    """Windows process lifetime peak resident bytes, including startup/data/model."""
    if sys.platform != "win32":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (key, ctypes.c_size_t)
            for key in (
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.PeakWorkingSetSize)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true", help="All83 validation; default bounded2")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    manifest = json.loads(Path("configs/final_evaluation_candidate.json").read_text())
    contracts_path = Path("configs/final_control_candidates.json")
    contracts = json.loads(contracts_path.read_bytes())
    if (
        contracts["dataset_revision"] != manifest["dataset_revision"]
        or contracts["reference_artifact_revision"] != manifest["reference_artifact_revision"]
        or contracts["arms"]["final_lpips"] != manifest["thresholds"]
    ):
        raise ValueError("Control and primary contracts differ")
    paths = {
        k: args.environment / v
        for k, v in {
            "checkpoint": "checkpoint.pt",
            "lpips_backbone": "lpips_backbone.pth",
            "lpips_calibration": "lpips_calibration.pth",
            "dependency_lock": "requirements.freeze.txt",
        }.items()
    }
    prefix = "evaluations/autoddpm-ddim2030-full83-t4-20261106T074200Z/"
    paths.update(
        config=Path("configs/autoddpm_ddim20_30.json"),
        validation_index=args.validation / "index.csv",
        mask_parameters=Path("runs/audit-2026-11-06/hf-latest")
        / prefix
        / "comparison/mask_parameters.json",
        source_snapshot=args.output / "source.zip",
    )
    with zipfile.ZipFile(paths["source_snapshot"], "w") as archive:
        for path in sorted(Path("src").rglob("*.py")):
            archive.writestr(path.as_posix(), path.read_bytes())
        for path in (Path(__file__), contracts_path):
            archive.writestr(path.name, path.read_bytes())
    manifest["artifacts"]["source_snapshot"] = hashlib.sha256(
        paths["source_snapshot"].read_bytes()
    ).hexdigest()
    manifest["controls_sha256"] = hashlib.sha256(contracts_path.read_bytes()).hexdigest()
    captured = capture_frozen_inputs(manifest, paths, validation_only=True)
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--exclude-editable"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if version_snapshot(freeze) != version_snapshot(captured["dependency_lock"].decode()):
        raise ValueError("Dependency versions differ")
    if args.validation.name != "val":
        raise ValueError("Only validation allowed")
    data = ShardDataset(args.validation, "val", include_masks=True)
    p = try_to_load_from_cache(
        "dacn-bmad-team/dacn-runs",
        prefix + "comparison/private_ablation.npz",
        revision=manifest["reference_artifact_revision"],
    )
    payload = Path(p).read_bytes()
    if (
        hashlib.sha256(payload).hexdigest()
        != "22656e1f63a0047d3dbe32f7833eea8e303988414e5942dec1ba5a1d9ed1ef19"
    ):
        raise ValueError("Cache checksum differs")
    with np.load(io.BytesIO(payload), allow_pickle=False) as reference:
        if len(data) != 83:
            raise ValueError("Validation83 required")
        for i in range(83):
            image, mask, label = data[i]
            if not (
                np.array_equal(image[0].numpy(), reference["images"][i])
                and np.array_equal(mask.numpy(), reference["masks"][i])
                and int(label) == int(reference["labels"][i])
            ):
                raise ValueError("Validation content differs")
    adapter = FrozenDDIMAdapter("cpu", include_controls=True)
    maps = {key: [] for key in contracts["arms"]}
    masks, labels, timings = [], [], []
    selected = (
        list(range(83))
        if args.full
        else [int(np.flatnonzero(data.labels == target)[0]) for target in (0, 1)]
    )
    for position, i in enumerate(selected):
        image, mask, label = data[i]
        started = time.perf_counter()
        adapter(
            {"image": image[0].numpy(), "mask": mask.numpy(), "sample_id": data.sample_ids[i]},
            captured,
        )
        timings.append(time.perf_counter() - started)
        for key, values in maps.items():
            values.append(adapter.last_score_maps[key])
        masks.append(mask.numpy()[8:-8, 8:-8])
        labels.append(int(label))
        print("Completed control sample", position + 1, flush=True)
    report = {
        "status": "completed",
        "samples": len(selected),
        "unet_calls": 50 * len(selected),
        "lpips_calls": 3 * len(selected),
        "device": "cpu",
        "threads": 2,
        "batch": 1,
        "cold_sample_seconds_including_model_load": timings[0],
        "warm_sample_seconds": timings[1],
        "peak_process_working_set_bytes": windows_peak_working_set(),
        "memory_scope": "Process lifetime resident peak incl startup/data/capture/model; no VRAM",
        "timing_scope": "Adapter call incl transfers/scores, first includes load; excludes dataset/checksum/global metrics",
        "metrics": {
            key: evaluate_maps(
                np.stack(values),
                np.stack(masks),
                np.array(labels),
                split="validation",
                thresholds=contracts["arms"][key],
            )
            for key, values in maps.items()
        },
        "candidate_manifest": manifest,
        "scope": "CPU validation; selected budget; no threshold fit or held-out test",
    }
    np.savez_compressed(
        args.output / "private_control_maps.npz", **{k: np.stack(v) for k, v in maps.items()}
    )
    (args.output / "aggregate_receipt.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"metrics", "candidate_manifest"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

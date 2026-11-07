"""Replay LPIPS on a checksum-pinned private 83-slice validation cache."""

import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

from dacn.evaluation.ddim_ablation import ablation_pair_scores
from dacn.evaluation.frozen_ddim import load_perceptual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a fresh receipt path")
    payload = args.cache.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != args.expected_sha256:
        raise ValueError("Cache checksum differs")
    torch.set_num_threads(2)
    model = load_perceptual(
        {
            key: (args.environment / (key + ".pth")).read_bytes()
            for key in ("lpips_backbone", "lpips_calibration")
        },
        "cpu",
    )
    errors = []
    started = time.perf_counter()
    with np.load(io.BytesIO(payload), allow_pickle=False) as cache:
        if cache["images"].shape != (83, 256, 256):
            raise ValueError("Registered validation83 cache required")
        chosen = (
            list(range(83))
            if args.full
            else [int(np.flatnonzero(cache["labels"] == label)[0]) for label in (0, 1)]
        )
        for index in chosen:
            for pair, key in (("initials", "initial_lpips_only"), ("finals", "autoddpm_lpips")):
                scores, _ = ablation_pair_scores(
                    model, cache["images"][index], cache[pair][index], torch.device("cpu")
                )
                errors.append(float(np.max(np.abs(scores["lpips"] - cache[key][index]))))
    report = {
        "status": "completed",
        "cache_sha256": digest,
        "samples": len(chosen),
        "lpips_calls": len(errors),
        "max_abs_error": max(errors),
        "all_maps_match_atol_1e-6": max(errors) <= 1e-6,
        "seconds": time.perf_counter() - started,
        "scope": "CPU cached validation pair replay; no U-Net or test",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

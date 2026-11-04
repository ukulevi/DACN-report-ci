"""Aggregate validation-only reconstruction diagnostics; no GPU inference."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from dacn.evaluation.scores import paired_score_maps
from dacn.evaluation.validation import evaluate_maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new diagnostic output path")
    digest = hashlib.sha256(args.cache.read_bytes()).hexdigest()
    if digest != args.expected_sha256:
        raise ValueError("Cache checksum mismatch")
    with np.load(args.cache, allow_pickle=False) as data:
        images, recon, masks, labels = [
            data[key] for key in ("images", "reconstructions", "masks", "labels")
        ]
    if images.shape != recon.shape or masks.shape != images.shape:
        raise ValueError("Cache arrays disagree")
    x, y = images[:, 8:-8, 8:-8], recon[:, 8:-8, 8:-8]
    truth = masks[:, 8:-8, 8:-8].astype(bool)
    # Intensity heuristic for diagnosis, not an anatomical brain mask.
    foreground = x > -0.95
    interior = np.stack([distance_transform_edt(fg) > 5 for fg in foreground])
    regions = {
        "lesion": truth,
        "nonlesion_foreground_interior": interior & ~truth,
        "nonlesion_foreground_boundary_5px": foreground & ~interior & ~truth,
        "nonlesion_background": ~foreground & ~truth,
    }
    paired = [paired_score_maps(a, b) for a, b in zip(images, recon, strict=True)]
    output = {
        "split": "validation",
        "held_out_test_used": False,
        "cache_sha256": digest,
        "samples": len(x),
        "lesion_pixel_prevalence": float(truth.mean()),
        "foreground_heuristic": "input > -0.95; interior distance > 5 pixels",
        "lesion_fraction_outside_foreground": float((truth & ~foreground).sum() / truth.sum()),
        "methods": {},
    }
    for method in ("pixel_difference", "local_ssim", "input_intensity_control"):
        scores = (
            x.astype(np.float64)
            if method == "input_intensity_control"
            else np.stack([item[method] for item in paired])
        )
        metrics = evaluate_maps(scores, truth, labels, split="validation")
        predicted = scores >= metrics["pixel"]["threshold"]
        fps = predicted & ~truth
        regional = {}
        for name, region in regions.items():
            values = scores[region]
            regional[name] = {
                "pixel_count": int(region.sum()),
                "mean_score": float(values.mean()) if len(values) else None,
                "score_quantiles_10_50_90": np.quantile(values, [0.1, 0.5, 0.9]).tolist()
                if len(values)
                else None,
                "fraction_predicted_positive": float(predicted[region].mean())
                if len(values)
                else None,
                "share_of_all_false_positive_pixels": float(
                    (fps & region).sum() / max(fps.sum(), 1)
                ),
                "reconstruction_mae": float(np.abs(x - y)[region].mean()) if len(values) else None,
            }
        # Zero scores outside heuristic foreground: same full-image metric ROI.
        gated = np.where(foreground, scores, scores.min() - 1)
        gated_metrics = evaluate_maps(gated, truth, labels, split="validation")
        output["methods"][method] = {
            "baseline": metrics,
            "regions": regional,
            "foreground_gating_diagnostic": gated_metrics,
        }
        print(
            json.dumps(
                {
                    "method": method,
                    "pixel": metrics["pixel"],
                    "foreground_gated_pixel": gated_metrics["pixel"],
                }
            ),
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

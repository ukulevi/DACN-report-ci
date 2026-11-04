"""Measure normal-slice reconstruction quality in input-derived foreground ROIs."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

from dacn.evaluation.scores import paired_score_maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Choose a fresh output directory")
    started = time.perf_counter()
    sources = {
        100: (
            "reconstructions_t100.npz",
            "c1aa679db880a13a860109dee028db80d979eca684fad161c4726d5c4000db6d",
        ),
        200: (
            "reconstructions_t200.npz",
            "cffefdc5d2cf599fc3903eb8a8fa9a192dddd2dabe4cffeb04ee28cfa60bc06a",
        ),
        300: (
            "reconstructions.npz",
            "8278ad18543dd87f698c36b510564b9fcd4445d4a1b0ae5d7a859093e2a4e959",
        ),
    }
    reconstructions = {}
    reference = None
    for timestep, (name, digest) in sources.items():
        path = args.cache_dir / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        with np.load(path, allow_pickle=False) as archive:
            images, masks, labels = [archive[k] for k in ("images", "masks", "labels")]
            reconstructed = archive["reconstructions"].copy()
        if reference is None:
            reference = (images, masks, labels)
        else:
            assert all(
                np.array_equal(a, b)
                for a, b in zip(reference, (images, masks, labels), strict=True)
            )
        reconstructions[timestep] = reconstructed[labels == 0]
    images, masks, labels = reference
    assert int((labels == 0).sum()) == 39 and not masks[labels == 0].any()
    original = images[labels == 0]
    x = original[:, 8:-8, 8:-8].astype(np.float64)
    foreground = x > -0.95
    interior = np.stack([distance_transform_edt(fg) > 5 for fg in foreground])
    regions = {
        "foreground": foreground,
        "interior": interior,
        "boundary_5px": foreground & ~interior,
        "background": ~foreground,
        "full_crop": np.ones_like(foreground, dtype=bool),
    }
    result = {
        "split": "validation",
        "normal_slices": 39,
        "held_out_test_used": False,
        "model": "Gaussian EMA epoch30",
        "sampling_seed": 17,
        "roi": "input > -0.95, interior distance >5px; heuristic, not anatomical segmentation",
        "intensity_scale": "MAE/RMSE/PSNR on [0,1]; SSIM range=1 equivalent to original [-1,1] range2",
        "source_cache_hashes": {str(t): digest for t, (_, digest) in sources.items()},
        "regions": {name: int(roi.sum()) for name, roi in regions.items()},
        "metrics": {},
    }
    for t, recon in reconstructions.items():
        y = recon[:, 8:-8, 8:-8].astype(np.float64)
        absolute = np.abs(x - y) / 2
        squared = ((x - y) / 2) ** 2
        ssim = np.stack(
            [
                1 - paired_score_maps(a, b)["local_ssim"]
                for a, b in zip(original, recon, strict=True)
            ]
        )
        result["metrics"][str(t)] = {}
        for name, roi in regions.items():
            per_image_mae = np.asarray([absolute[i][roi[i]].mean() for i in range(39)])
            mse = float(squared[roi].mean())
            result["metrics"][str(t)][name] = {
                "mae": float(absolute[roi].mean()),
                "rmse": float(np.sqrt(mse)),
                "psnr_db": float(-10 * np.log10(max(mse, 1e-15))),
                "mean_local_ssim": float(ssim[roi].mean()),
                "signed_recon_minus_input": float(((y - x) / 2)[roi].mean()),
                "per_image_mae_median": float(np.median(per_image_mae)),
                "per_image_mae_q25_q75": np.quantile(per_image_mae, [0.25, 0.75]).tolist(),
            }
        print(json.dumps({"timestep": t, "metrics": result["metrics"][str(t)]}), flush=True)
    # Paired examples selected by t300 foreground MAE, not cherry-picked visually.
    errors = np.abs(x - reconstructions[300][:, 8:-8, 8:-8]) / 2
    order = np.argsort([errors[i][foreground[i]].mean() for i in range(39)])
    selected = [order[0], order[len(order) // 2], order[-1]]
    fig, axes = plt.subplots(3, 5, figsize=(13, 8), layout="constrained")
    for row, (idx, caption) in enumerate(
        zip(selected, ("Lowest", "Median", "Highest"), strict=True)
    ):
        values = [
            x[idx],
            foreground[idx],
            *[reconstructions[t][idx, 8:-8, 8:-8] for t in (100, 200, 300)],
        ]
        for col, (value, title) in enumerate(
            zip(
                values,
                ("Input", "Foreground ROI", "Recon t*=100", "Recon t*=200", "Recon t*=300"),
                strict=True,
            )
        ):
            axes[row, col].imshow(value, cmap="gray", vmin=0 if col == 1 else -1, vmax=1)
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(f"{caption} t300 error", fontsize=10)
    fig.suptitle("Normal validation slices | Gaussian EMA epoch30 | input-derived foreground ROI")
    args.output_dir.mkdir(parents=True)
    fig.savefig(args.output_dir / "normal_examples.png", dpi=160)
    plt.close(fig)
    result["cpu_seconds"] = time.perf_counter() - started
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()

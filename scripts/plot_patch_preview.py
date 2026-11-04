"""Render private, de-identified diagnostic arrays into a local comparison PNG."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with np.load(args.archive, allow_pickle=False) as data:
        images = data["images"][:, 8:-8, 8:-8]
        reconstructed = data["reconstructions"][:, 8:-8, 8:-8]
        masks = data["masks"][:, 8:-8, 8:-8]
        differences = data["difference"]
        labels = data["labels"]
    vmax = max(float(np.quantile(differences, 0.99)), 1e-6)
    fig, axes = plt.subplots(len(images), 4, figsize=(12, 3 * len(images)), squeeze=False)
    for row in range(len(images)):
        for col, value in enumerate(
            (images[row], reconstructed[row], differences[row], masks[row])
        ):
            ax = axes[row, col]
            if col < 2:
                ax.imshow(value, cmap="gray", vmin=-1, vmax=1)
            elif col == 2:
                artist = ax.imshow(value, cmap="inferno", vmin=0, vmax=vmax)
                fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.imshow(value, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(
                    (
                        "Input",
                        "Patch/context reconstruction",
                        "Absolute difference [0,1]",
                        "Tumor annotation",
                    )[col]
                )
        axes[row, 0].set_ylabel(f"Example {row + 1}: {'tumor' if labels[row] else 'normal'}")
    fig.suptitle("EMA fine-tune epoch 5 | t*=100 | 64px patches, stride48 | validation diagnostic")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Plot recorded DDPM losses without smoothing or additional inference."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Choose a new output directory")
    raw = args.history.read_bytes()
    if (
        hashlib.sha256(raw).hexdigest()
        != "30908ee939a37470ba47549f9d92a607ad1bfeb21600842e40ec12f325888003"
    ):
        raise ValueError("Unexpected epoch30 training-history checksum")
    records = json.loads(raw)["epochs"]
    epochs = np.asarray([row["epoch"] for row in records])
    assert epochs.tolist() == list(range(1, 31))
    assert all(row["optimizer_steps"] == row["epoch"] * 938 for row in records)
    train = np.asarray([row["train_epsilon_mse"] for row in records])
    validation = np.asarray([row["normal_validation_epsilon_mse"] for row in records])
    assert np.isfinite(train).all() and np.isfinite(validation).all()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    for ax in axes:
        ax.plot(epochs, train, "o-", markersize=3, label="Train: online model, random t/noise")
        ax.plot(epochs, validation, "o-", markersize=3, label="Normal val: EMA, fixed t/noise")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Epsilon-prediction MSE")
        ax.grid(alpha=0.25)
    axes[0].set_yscale("log")
    axes[0].set_title("All epochs (log scale)")
    axes[0].legend(fontsize=8)
    axes[1].set_xlim(5, 30)
    axes[1].set_ylim(0.0071, 0.011)
    axes[1].set_title("Epochs 5-30 (linear zoom)")
    fig.suptitle("Gaussian DDPM | T4 batch 8 | seed 17 | 7,500 train / 39 normal val slices")
    args.output_dir.mkdir(parents=True)
    fig.savefig(args.output_dir / "loss_curves.png", dpi=170)
    fig.savefig(args.output_dir / "loss_curves.pdf")
    plt.close(fig)
    with (args.output_dir / "loss_history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Best normal-val epoch: {int(epochs[validation.argmin()])}")
    print(f"Final normal-val MSE: {validation[-1]:.10f}")


if __name__ == "__main__":
    main()

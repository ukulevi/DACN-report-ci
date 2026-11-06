"""Full feature-conditioned validation, baseline comparison and private figures."""

import csv
import hashlib
import io
import json

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from dacn.evaluation.patch_validation import evaluate_patch_validation
from dacn.evaluation.scores import paired_score_maps
from dacn.evaluation.validation import evaluate_maps


def evaluate_feature(
    *,
    validation_dir,
    checkpoint_path,
    output_dir,
    device,
    baseline_cache,
    on_progress=None,
    resume_partial=False,
    experiment_A_results=None,
    unconditional_finetune=False,
    experiment_B_results=None,
    experiment_control_results=None,
):
    baseline_sha = "8278ad18543dd87f698c36b510564b9fcd4445d4a1b0ae5d7a859093e2a4e959"
    baseline_bytes = baseline_cache.read_bytes()
    if hashlib.sha256(baseline_bytes).hexdigest() != baseline_sha:
        raise ValueError("Registered Gaussian baseline cache differs")
    result = evaluate_patch_validation(
        validation_dir=validation_dir,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        device=device,
        feature_condition=not unconditional_finetune,
        unconditional_finetune=unconditional_finetune,
        starting_timestep=300,
        resume_partial=resume_partial,
        on_progress=on_progress,
    )
    with np.load(output_dir / "private_reconstructions.npz", allow_pickle=False) as cache:
        images, reconstructed, masks, labels, noisy = [
            cache[key] for key in ("images", "reconstructions", "masks", "labels", "noisy_images")
        ]
    with np.load(io.BytesIO(baseline_bytes), allow_pickle=False) as cache:
        for key, expected in (("images", images), ("masks", masks), ("labels", labels)):
            if not np.array_equal(cache[key], expected):
                raise ValueError("Baseline/conditional evaluation samples differ")
        baseline = cache["reconstructions"].copy()
    with (validation_dir / "index.csv").open(newline="") as stream:
        groups = np.asarray([row["patient_group_id"] for row in csv.DictReader(stream)])
    calibration, holdout = next(
        StratifiedGroupKFold(2, shuffle=True, random_state=17).split(
            np.zeros(len(labels)), labels, groups
        )
    )
    result["baseline_metrics"], result["reconstruction_quality"] = {}, {}
    target_name = "unconditional_finetune" if unconditional_finetune else "feature_condition"
    for name, predictions in (("baseline", baseline), (target_name, reconstructed)):
        paired = [paired_score_maps(x, y) for x, y in zip(images, predictions, strict=True)]
        quality = []
        for i, pair in enumerate(paired):
            foreground = images[i, 8:-8, 8:-8] > -0.95
            lesion = masks[i, 8:-8, 8:-8].astype(bool)
            quality.append(
                {
                    "normal": not bool(labels[i]),
                    "foreground_mae_01": float((pair["pixel_difference"][foreground] / 2).mean()),
                    "foreground_ssim": float((1 - pair["local_ssim"][foreground]).mean()),
                    "lesion_ssim_error": float(pair["local_ssim"][lesion].mean())
                    if lesion.any()
                    else None,
                    "nonlesion_ssim_error": float(pair["local_ssim"][foreground & ~lesion].mean()),
                    "lesion_absolute_error": float(pair["pixel_difference"][lesion].mean())
                    if lesion.any()
                    else None,
                    "nonlesion_absolute_error": float(
                        pair["pixel_difference"][foreground & ~lesion].mean()
                    ),
                }
            )
        result["reconstruction_quality"][name] = {
            "normal_foreground_mae_01": float(
                np.mean([q["foreground_mae_01"] for q in quality if q["normal"]])
            ),
            "normal_foreground_ssim": float(
                np.mean([q["foreground_ssim"] for q in quality if q["normal"]])
            ),
            "abnormal_slice_macro": {
                key: float(np.mean([q[key] for q in quality if not q["normal"]]))
                for key in (
                    "lesion_ssim_error",
                    "nonlesion_ssim_error",
                    "lesion_absolute_error",
                    "nonlesion_absolute_error",
                )
            },
            "foreground_definition": "input>-0.95 in240crop, heuristic not anatomical segmentation",
        }
        if name == "baseline":
            for score in ("pixel_difference", "local_ssim"):
                values = np.stack([pair[score] for pair in paired])
                cropped = masks[:, 8:-8, 8:-8]
                fitted = evaluate_maps(
                    values[calibration],
                    cropped[calibration],
                    labels[calibration],
                    split="validation",
                )
                thresholds = {level: fitted[level]["threshold"] for level in ("pixel", "image")}
                result["baseline_metrics"][score] = {
                    "calibration": fitted,
                    "group_holdout": evaluate_maps(
                        values[holdout],
                        cropped[holdout],
                        labels[holdout],
                        split="validation",
                        thresholds=thresholds,
                    ),
                    "all_validation_descriptive": evaluate_maps(
                        values, cropped, labels, split="validation"
                    ),
                }
    result["baseline_cache_sha256"] = baseline_sha
    result["initialization_and_budget_caveat"] = (
        "Unconditional control and A/B each start from original EMA30 with fresh AdamW, U-Net LR1e-5,9380 updates and normal-validation selection. No encoder in control. One seed on previously inspected validation; no significance claim. B encoder pretraining compute separate."
        if unconditional_finetune
        else "Original baseline epoch30 vs +10epochs joint fine-tuning; improvement, if any, cannot isolate conditioning from extra training without a matched unconditional fine-tune control. B additionally uses normal-only encoder pretraining, whose compute must be reported separately. Encoder sees original image; tumor leakage remains possible."
    )
    if experiment_A_results is not None:
        result["experiment_A_comparison"] = compare_experiment_A(result, experiment_A_results)
    if experiment_B_results is not None:
        payload = experiment_B_results.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != "17cea7ce87afc15332835ffdafa514b2758883d92510339f86bb45b521ac247d":
            raise ValueError("Registered experiment B results differ")
        reference = json.loads(payload)
        for key in (
            "index_sha256",
            "partition",
            "samples",
            "ema",
            "reconstruction_mode",
            "timesteps_one_based",
            "sampling_base_seed",
            "partition_seed",
            "baseline_cache_sha256",
        ):
            if result[key] != reference[key]:
                raise ValueError(f"B/control evaluation protocol differs: {key}")
        result["experiment_B_comparison"] = {
            "B_results_sha256": digest,
            "B_checkpoint_sha256": reference["checkpoint_sha256"],
            "B_metrics": reference["metrics"],
            "B_reconstruction_quality": reference["reconstruction_quality"]["feature_condition"],
        }
    if experiment_control_results is not None:
        payload = experiment_control_results.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != "babbd335d6a9c40e64d23b2e7dded4dd8f195e3522669586713488ced3f62c7e":
            raise ValueError("Registered matched control results differ")
        reference = json.loads(payload)
        for key in (
            "index_sha256",
            "partition",
            "samples",
            "timesteps_one_based",
            "sampling_base_seed",
            "partition_seed",
            "ema",
            "reconstruction_mode",
            "baseline_cache_sha256",
        ):
            if result[key] != reference[key]:
                raise ValueError(f"Spatial/control evaluation protocol differs: {key}")
        result["unconditional_control_comparison"] = {
            "results_sha256": digest,
            "metrics": reference["metrics"],
            "reconstruction_quality": reference["reconstruction_quality"]["unconditional_finetune"],
        }
        result["initialization_and_budget_caveat"] = (
            "Spatial ImageNet model vs matched unconditional/A/B: same EMA30 start and9380fine-tune updates. This changes both spatial fusion and encoder pretraining; encoder LR1e-5 vs1e-4 in A/B. Not an isolated spatial-only ablation. One exploratory seed; encoder sees original pathology."
        )
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    make_figure(images, noisy, reconstructed, masks, labels, output_dir / "private_examples.png")
    return result


def compare_experiment_A(result, path):
    """Compare registered A aggregates without selecting B from anomaly metrics."""
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != "a8e57ed824c9e6a00404d82296f15fb208eabb9d8d911c06fcd983dfb8b5a962":
        raise ValueError("Registered experiment A results differ")
    reference = json.loads(payload)
    for key in (
        "index_sha256",
        "partition",
        "timesteps_one_based",
        "sampling_base_seed",
        "partition_seed",
        "samples",
        "ema",
        "reconstruction_mode",
        "baseline_cache_sha256",
    ):
        if result[key] != reference[key]:
            raise ValueError(f"A/B evaluation protocol differs: {key}")
    return {
        "A_results_sha256": digest,
        "A_checkpoint_sha256": reference["checkpoint_sha256"],
        "A_selected_finetune_epoch": reference["checkpoint_epoch"],
        "A_metrics": reference["metrics"],
        "A_reconstruction_quality": reference["reconstruction_quality"]["feature_condition"],
        "budget": "A and B each9380 joint updates; B adds5 encoder-pretraining epochs. Both best checkpoints selected by normal-validation noise MSE, not Pixel AP.",
    }


def make_figure(images, noisy, reconstructed, masks, labels, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(17)
    selected = [
        int(i)
        for label in (0, 1)
        for i in rng.choice(np.flatnonzero(labels == label), 2, replace=False)
    ]
    fig, axes = plt.subplots(4, 6, figsize=(16, 11))
    for row, i in enumerate(selected):
        scores = paired_score_maps(images[i], reconstructed[i])
        values = (
            images[i, 8:-8, 8:-8],
            noisy[i, 8:-8, 8:-8],
            reconstructed[i, 8:-8, 8:-8],
            scores["pixel_difference"],
            scores["local_ssim"],
            masks[i, 8:-8, 8:-8],
        )
        for col, value in enumerate(values):
            axes[row, col].imshow(
                value,
                cmap="gray" if col in (0, 1, 2, 5) else "inferno",
                **(
                    {"vmin": -1, "vmax": 1}
                    if col < 3
                    else {"vmin": 0, "vmax": 1}
                    if col == 5
                    else {}
                ),
            )
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(
                    (
                        "Input",
                        "Noisy t300",
                        "Reconstruction",
                        "Absolute difference",
                        "1-SSIM",
                        "Annotation",
                    )[col]
                )
        axes[row, 0].set_ylabel("Normal" if labels[i] == 0 else "Tumor")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

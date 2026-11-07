"""CPU orchestration boundary for frozen inference backends.

The backend receives captured artifacts and frozen thresholds. Dataset access
is deferred until after verification. Private shards must stay private.
"""

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from dacn.evaluation.frozen_inputs import capture_frozen_inputs
from dacn.evaluation.metrics import compute_binary_metrics


def run_frozen_evaluation(
    *, manifest, artifact_paths, samples, infer, output_dir, validation_only=False
):
    """Run/resume ordered samples with one private NPZ shard per sample.

    samples is a zero-argument factory yielding (stable_id, sample); infer returns four
    arrays: pixel_labels, pixel_scores, image_labels, image_scores. The backend
    owns preprocessing and model loading; this function does not claim that an
    arbitrary backend implements the registered scientific method.
    """
    manifest = json.loads(json.dumps(manifest, allow_nan=False))
    manifest["execution_scope"] = "validation_acceptance" if validation_only else "final"
    captured = capture_frozen_inputs(manifest, artifact_paths, validation_only=validation_only)
    contract = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "receipt.json"
    receipt = {"contract": contract, "status": "running", "shards": []}
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["contract"] != contract:
            raise ValueError("Resume contract differs")

    def save():
        temporary = output / "receipt.partial"
        temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        temporary.replace(receipt_path)

    keys = ("pixel_labels", "pixel_scores", "image_labels", "image_scores")
    arrays = {key: [] for key in keys}
    save()
    try:
        count = 0
        for count, (stable_id, sample) in enumerate(samples(), start=1):
            index = count - 1
            identity = hashlib.sha256(str(stable_id).encode()).hexdigest()
            path = output / f"shard-{index:06d}.npz"
            if index < len(receipt["shards"]):
                payload = path.read_bytes()
                expected = receipt["shards"][index]
                if (
                    hashlib.sha256(payload).hexdigest() != expected["sha256"]
                    or identity != expected["identity"]
                ):
                    raise ValueError("Resume shard checksum differs")
            else:
                values = infer(sample, captured)
                if set(values) != set(keys):
                    raise ValueError("Inference must return the fixed score/label schema")
                values = {key: np.asarray(values[key]).reshape(-1) for key in keys}
                for level in ("pixel", "image"):
                    labels, scores = values[f"{level}_labels"], values[f"{level}_scores"]
                    if (
                        len(labels) == 0
                        or labels.shape != scores.shape
                        or not np.isin(labels, [0, 1]).all()
                        or not np.isfinite(scores).all()
                    ):
                        raise ValueError("Invalid inference labels/scores")
                buffer = io.BytesIO()
                np.savez_compressed(buffer, **values)
                payload = buffer.getvalue()
                temporary = path.with_suffix(".partial")
                temporary.write_bytes(payload)
                temporary.replace(path)
                receipt["shards"].append(
                    {"sha256": hashlib.sha256(payload).hexdigest(), "identity": identity}
                )
                save()
            with np.load(io.BytesIO(payload), allow_pickle=False) as shard:
                for key in keys:
                    arrays[key].append(shard[key])
        if count != len(receipt["shards"]) or not count:
            raise ValueError("Empty or shortened resume sample sequence")
        metrics = {
            level: compute_binary_metrics(
                np.concatenate(arrays[f"{level}_labels"]),
                np.concatenate(arrays[f"{level}_scores"]),
                manifest["thresholds"][level],
            )
            for level in ("pixel", "image")
        }
        metrics["pixel"]["dice_micro"] = metrics["pixel"]["f1"]
        lesion_dice, normal_fp, normal_pixels = [], 0, 0
        for labels, scores in zip(arrays["pixel_labels"], arrays["pixel_scores"], strict=True):
            predicted = scores >= manifest["thresholds"]["pixel"]
            if labels.any():
                lesion_dice.append(
                    float(
                        2
                        * np.count_nonzero(predicted & (labels > 0))
                        / (np.count_nonzero(predicted) + np.count_nonzero(labels))
                    )
                )
            else:
                normal_fp += int(np.count_nonzero(predicted))
                normal_pixels += len(labels)
        metrics["pixel"]["macro_lesion_dice"] = float(np.mean(lesion_dice)) if lesion_dice else None
        metrics["pixel"]["normal_pixel_fpr"] = normal_fp / normal_pixels if normal_pixels else None
        receipt.update(status="completed", metrics=metrics, sample_count=count)
        save()
        return receipt
    except BaseException:
        receipt["status"] = "interrupted"
        save()
        raise

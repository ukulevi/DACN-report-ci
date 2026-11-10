"""Durable four-arm inference with one shared trajectory and fixed thresholds.

All per-sample files are private. Only aggregate.json and curves.json are safe
to copy into project records. No threshold fitting occurs in this module.
"""

import gc
import hashlib
import io
import json
import math
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from dacn.evaluation.frozen_inputs import capture_frozen_inputs
from dacn.evaluation.metrics import compute_binary_metrics

ARMS = ("final_lpips", "initial_lpips", "naive_stitch_lpips", "brightness")


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def restore_pinned_text(payload, expected_sha256):
    """Recover a known LF/CRLF artifact, accepting only its exact historical hash."""
    normalized = payload.replace(b"\r\n", b"\n")
    for value in (payload, normalized, normalized.replace(b"\n", b"\r\n")):
        if digest(value) == expected_sha256:
            return value
    raise ValueError("Text artifact does not match historical checksum")


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".partial")
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def verify_controls(manifest, controls):
    if manifest.get("controls_sha256") != digest(json_bytes(controls)):
        raise ValueError("Control contract checksum differs")
    if controls.get("dataset_revision") != manifest["dataset_revision"]:
        raise ValueError("Control dataset revision differs")
    if set(controls.get("arms", {})) != set(ARMS):
        raise ValueError("Require exactly the four registered arms")
    if controls["arms"]["final_lpips"] != manifest["thresholds"]:
        raise ValueError("Primary thresholds differ")
    for thresholds in controls["arms"].values():
        if set(thresholds) != {"pixel", "image"} or any(
            type(x) not in (int, float) or not math.isfinite(x) for x in thresholds.values()
        ):
            raise ValueError("Require finite pixel/image control thresholds")


def method_contract(manifest):
    """Acceptance binds actual runtime/source/data/controls; status may then freeze."""
    return digest(
        json_bytes(
            {
                k: manifest[k]
                for k in (
                    "artifacts",
                    "dataset_revision",
                    "thresholds",
                    "controls_sha256",
                    "dataset_metadata_sha256",
                )
            }
        )
    )


def verify_acceptance(manifest, receipt_bytes):
    receipt = json.loads(receipt_bytes)
    if (
        digest(receipt_bytes) != manifest.get("validation_acceptance_sha256")
        or receipt.get("status") != "completed"
        or receipt.get("split") != "validation"
        or receipt.get("sample_count") != 43
        or receipt.get("reference_maps_verified") is not True
        or receipt.get("method_contract") != method_contract(manifest)
    ):
        raise ValueError("Final requires matching real-validation acceptance")


def run_four_arm(
    *,
    manifest,
    controls,
    artifact_paths,
    samples,
    infer,
    output_dir,
    expected_samples,
    split,
    runtime_check,
    validation_receipt=None,
    on_commit=None,
):
    """Gate before dataset access; commit all four maps together for each sample.

    runtime_check validates the captured source/dependency bytes. samples is a
    deferred revision/content-verified factory. infer is a shared-trajectory
    adapter exposing last_score_maps. Resume checks input identity AND content.
    """
    if split not in {"validation", "test"} or type(expected_samples) is not int:
        raise ValueError("Invalid evaluation scope")
    if expected_samples < 2 or (split == "test" and expected_samples != 3665):
        raise ValueError("Require registered full test or bounded validation")
    manifest = json.loads(json_bytes(manifest))
    controls = json.loads(json_bytes(controls))
    verify_controls(manifest, controls)
    captured = capture_frozen_inputs(
        manifest, artifact_paths, validation_only=split == "validation"
    )
    if split == "test":
        if controls.get("status") != "frozen" or validation_receipt is None:
            raise ValueError("Final requires frozen controls and validation acceptance")
        verify_acceptance(manifest, Path(validation_receipt).read_bytes())
    runtime_check(captured)
    contract = digest(
        json_bytes(
            {
                "manifest": manifest,
                "split": split,
                "expected_samples": expected_samples,
            }
        )
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "private_receipt.json"
    receipt = {"contract": contract, "status": "running", "shards": []}
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_bytes())
        if receipt["contract"] != contract:
            raise ValueError("Resume contract differs")

    def persist():
        atomic_json(receipt_path, receipt)
        if on_commit:
            on_commit()

    persist()
    try:
        seen, count = set(), 0
        for count, (stable_id, sample) in enumerate(samples(), 1):
            if count > expected_samples or stable_id in seen:
                raise ValueError("Unexpected sample count or duplicate identity")
            seen.add(stable_id)
            image, mask = np.asarray(sample["image"]), np.asarray(sample["mask"])
            if image.shape != (256, 256) or mask.shape != image.shape:
                raise ValueError("Expected padded image/mask")
            if not np.isfinite(image).all() or (np.abs(image) > 1).any():
                raise ValueError("Expected finite normalized image")
            if not np.isin(mask, [0, 1]).all():
                raise ValueError("Expected binary mask")
            identity = digest(str(stable_id).encode())
            content = digest(image.tobytes() + mask.tobytes())
            path = output / f"shard-{count - 1:06d}.npz"
            if count <= len(receipt["shards"]):
                old = receipt["shards"][count - 1]
                if (
                    identity != old["identity"]
                    or content != old["content"]
                    or digest(path.read_bytes()) != old["sha256"]
                ):
                    raise ValueError("Resume input/shard checksum differs")
                continue
            started = time.perf_counter()
            infer(sample, captured)
            elapsed = time.perf_counter() - started
            maps = infer.last_score_maps
            if set(maps) != set(ARMS) or any(
                np.asarray(v).shape != (240, 240) or not np.isfinite(v).all() for v in maps.values()
            ):
                raise ValueError("Invalid four-arm score maps")
            buffer = io.BytesIO()
            np.savez_compressed(
                buffer,
                mask=mask[8:-8, 8:-8].astype(np.uint8),
                **{k: np.asarray(v, dtype=np.float32) for k, v in maps.items()},
            )
            payload = buffer.getvalue()
            temporary = path.with_suffix(".partial")
            temporary.write_bytes(payload)
            temporary.replace(path)
            receipt["shards"].append(
                {
                    "identity": identity,
                    "content": content,
                    "sha256": digest(payload),
                    "adapter_seconds": elapsed,
                }
            )
            persist()
            if count % 25 == 0 or count == expected_samples:
                print(
                    json.dumps({"completed_samples": count, "expected_samples": expected_samples}),
                    flush=True,
                )
        if count != expected_samples or len(receipt["shards"]) != count:
            raise ValueError("Empty, shortened or incomplete sample sequence")
        receipt["status"] = "inference_completed"
        persist()
        report, curves = aggregate_shards(output, controls, count)
        timings = [s["adapter_seconds"] for s in receipt["shards"]]
        report.update(
            status="completed",
            split=split,
            sample_count=count,
            method_contract=method_contract(manifest),
            unet_calls=50 * count,
            lpips_calls=3 * count,
            adapter_seconds_sum=sum(timings),
            adapter_seconds_median=float(np.median(timings)),
            adapter_seconds_p95=float(np.quantile(timings, 0.95)),
            timing_scope="Adapter incl model load on first call, scores/transfers; excludes I/O/global metrics",
            training_seeds=1,
            sampling_seeds=1,
            pixel_aggregation="global micro over 240x240 crop; Dice=F1",
            image_aggregation="mean of top576 pixels; score>=fixed threshold",
            threshold_source="original40 validation calibration; no refitting",
        )
        atomic_json(output / "curves.json", curves)
        atomic_json(output / "aggregate.json", report)
        receipt["status"] = "completed"
        persist()
        return report
    except BaseException as exc:
        receipt.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            exception_type=type(exc).__name__,
        )
        persist()
        raise


def aggregate_shards(output, controls, count):
    """Exact global ranking per arm, never average AUROC/AP across shards.

    One arm at a time bounds memory; float32 disk arrays avoid keeping four
    full test maps in memory simultaneously. Curves are display-downsampled;
    scalar metrics always use all pixels with exact score ties.
    """
    output = Path(output)
    labels = np.lib.format.open_memmap(
        output / "private_labels.npy", mode="w+", dtype=np.uint8, shape=(count, 57600)
    )
    for i in range(count):
        with np.load(output / f"shard-{i:06d}.npz", allow_pickle=False) as shard:
            labels[i] = shard["mask"].reshape(-1)
    image_labels = labels.any(axis=1).astype(np.uint8)
    report, curves = {"arms": {}}, {}
    for arm in ARMS:
        scores = np.lib.format.open_memmap(
            output / "private_scores.npy", mode="w+", dtype=np.float32, shape=labels.shape
        )
        image_scores, dice, fp, normals = [], [], 0, 0
        for i in range(count):
            with np.load(output / f"shard-{i:06d}.npz", allow_pickle=False) as shard:
                scores[i] = shard[arm].reshape(-1)
            image_scores.append(
                float(np.partition(scores[i].astype(np.float64), -576)[-576:].mean())
            )
            pred = scores[i] >= controls["arms"][arm]["pixel"]
            if image_labels[i]:
                dice.append(
                    float(
                        2
                        * np.count_nonzero(pred & (labels[i] > 0))
                        / (np.count_nonzero(pred) + np.count_nonzero(labels[i]))
                    )
                )
            else:
                fp += int(pred.sum())
                normals += 57600
        flat_labels, flat_scores = labels.reshape(-1), scores.reshape(-1)
        metrics = {
            "pixel": compute_binary_metrics(
                flat_labels, flat_scores, controls["arms"][arm]["pixel"]
            ),
            "image": compute_binary_metrics(
                image_labels, image_scores, controls["arms"][arm]["image"]
            ),
        }
        metrics["pixel"].update(
            dice_micro=metrics["pixel"]["f1"],
            macro_lesion_dice=float(np.mean(dice)) if dice else None,
            normal_pixel_fpr=fp / normals if normals else None,
        )
        report["arms"][arm] = metrics
        curves[arm] = {}
        for level, y, s in (
            ("pixel", flat_labels, flat_scores),
            ("image", image_labels, np.asarray(image_scores)),
        ):
            fpr, tpr, _ = roc_curve(y, s)
            ix = np.unique(np.linspace(0, len(fpr) - 1, min(2000, len(fpr))).astype(int))
            curve = {"fpr": fpr[ix].tolist(), "tpr": tpr[ix].tolist()}
            del fpr, tpr
            precision, recall, _ = precision_recall_curve(y, s)
            ix = np.unique(np.linspace(0, len(recall) - 1, min(2000, len(recall))).astype(int))
            curve.update(precision=precision[ix].tolist(), recall=recall[ix].tolist())
            curves[arm][level] = curve
            del precision, recall
        del scores, flat_scores
        gc.collect()
    del labels
    return report, curves

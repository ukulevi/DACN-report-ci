"""One-phase DDIM50+LPIPS, with calibration-before-test and durable resume."""

import io
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_curve

from dacn.evaluation.ddim_ablation import ablation_pair_scores
from dacn.evaluation.ddim_inpainting import ddim_indices, ddim_initial_reconstruction
from dacn.evaluation.four_arm import atomic_json, digest, json_bytes
from dacn.evaluation.frozen_ddim import load_perceptual
from dacn.evaluation.metrics import compute_binary_metrics
from dacn.evaluation.validation import evaluate_maps
from dacn.training.ddpm_partial import build_diffusion, derive_sampling_seed, load_ema_checkpoint

CONFIG = {
    "seed": 17,
    "starting_timestep": 300,
    "inference_steps": 50,
    "eta": 0,
    "lpips_net": "squeeze",
    "lpips_version": "0.1",
    "crop": 240,
    "image_top_pixels": 576,
}
ARTIFACTS = {
    "checkpoint",
    "lpips_backbone",
    "lpips_calibration",
    "source_snapshot",
    "dependency_lock",
    "config",
    "validation_index",
    "validation_partition",
    "dataset_metadata",
}


def capture_inputs(manifest, paths):
    if (
        manifest.get("schema_version") != 1
        or manifest.get("config") != CONFIG
        or manifest.get("dataset_revision") != "350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7"
    ):
        raise ValueError("Require the fixed one-phase DDIM50 configuration")
    if any(type(manifest["config"][k]) is not type(v) for k, v in CONFIG.items()):
        raise ValueError("Config field types differ")
    if set(manifest.get("artifacts", {})) != ARTIFACTS or set(paths) != ARTIFACTS:
        raise ValueError("Require all DDIM50 artifact pins")
    captured = {}
    for key, path in paths.items():
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing or symlinked artifact: " + key)
        payload = path.read_bytes()
        if digest(payload) != manifest["artifacts"][key]:
            raise ValueError("Artifact checksum differs: " + key)
        captured[key] = payload
    if json.loads(captured["config"]) != CONFIG:
        raise ValueError("Captured sampler config differs")
    return captured


def method_contract(manifest):
    return digest(
        json_bytes({key: manifest[key] for key in ("config", "dataset_revision", "artifacts")})
    )


def check_stage(manifest, stage, validation_receipt=None, smoke_receipt=None):
    if stage not in {"smoke", "validation", "test"}:
        raise ValueError("Invalid DDIM50 stage")
    if stage == "validation" and (
        not smoke_receipt
        or smoke_receipt.get("status") != "completed"
        or smoke_receipt.get("sample_count") != 2
        or smoke_receipt.get("method_contract") != method_contract(manifest)
    ):
        raise ValueError("Matching bounded smoke required before validation")
    if stage == "test":
        thresholds = manifest.get("thresholds", {})
        if (
            manifest.get("status") != "frozen"
            or set(thresholds) != {"pixel", "image"}
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in thresholds.values())
            or not validation_receipt
        ):
            raise ValueError("Test requires frozen finite thresholds and validation receipt")
        receipt = json.loads(validation_receipt)
        if (
            digest(validation_receipt) != manifest.get("validation_acceptance_sha256")
            or receipt.get("status") != "completed"
            or receipt.get("stage") != "validation"
            or receipt.get("sample_count") != 83
            or receipt.get("holdout_samples") != 43
            or receipt.get("method_contract") != method_contract(manifest)
            or receipt.get("thresholds") != thresholds
        ):
            raise ValueError("Validation acceptance differs from final inputs")


class DDIM50Adapter:
    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.ready = False

    def initialize(self, captured):
        train_config = torch.load(
            io.BytesIO(captured["checkpoint"]), map_location="cpu", weights_only=False
        )["config"]
        self.model = load_ema_checkpoint(
            io.BytesIO(captured["checkpoint"]), train_config, self.device
        )
        self.diffusion = build_diffusion(train_config)
        self.perceptual = load_perceptual(captured, self.device)
        self.ready = True

    @torch.inference_mode()
    def __call__(self, sample, captured):
        if not self.ready:
            self.initialize(captured)
        image = np.asarray(sample["image"])
        if image.shape != (256, 256) or not np.isfinite(image).all() or (np.abs(image) > 1).any():
            raise ValueError("Expected one normalized padded image")
        seed = derive_sampling_seed(17, sample["sample_id"])
        tensor = torch.from_numpy(image.copy()).float()[None, None].to(self.device)
        counts = [0, 0]

        def count_model(*_):
            counts[0] += 1

        def count_lpips(*_):
            counts[1] += 1

        hooks = [
            self.model.register_forward_hook(count_model),
            self.perceptual.register_forward_hook(count_lpips),
        ]
        try:
            reconstruction = ddim_initial_reconstruction(
                self.model, self.diffusion, tensor, seed=seed, indices=ddim_indices(300, 50)
            )
            value = reconstruction[0, 0].cpu().numpy()
            maps, _ = ablation_pair_scores(self.perceptual, image, value, self.device)
            if counts != [50, 1]:
                raise ValueError("Expected exactly50 U-Net and1 LPIPS calls")
            return {"scores": maps["lpips"], "reconstruction": value}
        finally:
            for hook in hooks:
                hook.remove()


def run_samples(
    *,
    manifest,
    captured,
    stage,
    samples,
    infer,
    output,
    commit,
    validation_receipt=None,
    smoke_receipt=None,
):
    """Gate before dataset factory; checkpoint every25outputs, log resume verification."""
    check_stage(manifest, stage, validation_receipt, smoke_receipt)
    expected = {"smoke": 2, "validation": 83, "test": 3665}[stage]
    contract = digest(
        json_bytes(
            {
                "method": method_contract(manifest),
                "stage": stage,
                "thresholds": manifest.get("thresholds") if stage == "test" else None,
            }
        )
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "private_receipt.json"
    receipt = {"contract": contract, "status": "running", "shards": []}
    if path.exists():
        receipt = json.loads(path.read_bytes())
        if receipt["contract"] != contract:
            raise ValueError("Resume contract differs")
    if len(receipt["shards"]) > expected:
        raise ValueError("Unexpected stored sample count")

    def persist(durable=False):
        atomic_json(path, receipt)
        if durable:
            commit()

    persist(True)
    seen, count = set(), 0
    try:
        for count, (identity, sample) in enumerate(samples(), 1):
            if count > expected or identity in seen:
                raise ValueError("Unexpected sample count or duplicate identity")
            if sample.get("sample_id") != identity:
                raise ValueError("Sampling identity differs from ordered dataset identity")
            seen.add(identity)
            image, mask = np.asarray(sample["image"]), np.asarray(sample["mask"])
            if (
                image.shape != (256, 256)
                or mask.shape != image.shape
                or not np.isfinite(image).all()
                or (np.abs(image) > 1).any()
                or not np.isin(mask, [0, 1]).all()
            ):
                raise ValueError("Invalid image or binary mask")
            key, content = digest(str(identity).encode()), digest(image.tobytes() + mask.tobytes())
            shard = output / f"shard-{count - 1:06d}.npz"
            if count <= len(receipt["shards"]):
                prior = receipt["shards"][count - 1]
                if (
                    prior["identity"] != key
                    or prior["content"] != content
                    or digest(shard.read_bytes()) != prior["sha256"]
                ):
                    raise ValueError("Resume input/shard checksum differs")
                if count == 1 or count % 100 == 0 or count == len(receipt["shards"]):
                    print(
                        json.dumps(
                            {
                                "verified_cached_samples": count,
                                "cached_samples": len(receipt["shards"]),
                            }
                        ),
                        flush=True,
                    )
                continue
            tick = time.perf_counter()
            values = infer(sample, captured)
            elapsed = time.perf_counter() - tick
            if (
                set(values) != {"scores", "reconstruction"}
                or np.asarray(values["scores"]).shape != (240, 240)
                or np.asarray(values["reconstruction"]).shape != (256, 256)
                or any(not np.isfinite(v).all() for v in values.values())
                or (np.abs(values["reconstruction"]) > 1).any()
            ):
                raise ValueError("Invalid LPIPS map/reconstruction")
            buffer = io.BytesIO()
            np.savez_compressed(
                buffer,
                scores=np.asarray(values["scores"], dtype=np.float32),
                reconstruction=np.asarray(values["reconstruction"], dtype=np.float32),
                mask=mask[8:-8, 8:-8].astype(np.uint8),
            )
            payload = buffer.getvalue()
            temporary = shard.with_suffix(".partial")
            temporary.write_bytes(payload)
            temporary.replace(shard)
            receipt["shards"].append(
                {
                    "identity": key,
                    "content": content,
                    "sha256": digest(payload),
                    "adapter_seconds": elapsed,
                }
            )
            checkpoint = count % 25 == 0 or count == expected
            persist(checkpoint)
            if count == 1 or checkpoint:
                print(
                    json.dumps(
                        {
                            "completed_samples": count,
                            "expected_samples": expected,
                            "durable_checkpoint": checkpoint,
                        }
                    ),
                    flush=True,
                )
        if count != expected or count != len(receipt["shards"]):
            raise ValueError("Empty or shortened sample sequence")
        receipt["status"] = "inference_completed"
        persist(True)
        return receipt
    except BaseException as exc:
        receipt.update(status="interrupted", exception_type=type(exc).__name__)
        persist(True)
        raise


def validation_metrics(output, calibration, holdout):
    output = Path(output)
    maps, masks = [], []
    for i in range(83):
        with np.load(output / f"shard-{i:06d}.npz", allow_pickle=False) as sample:
            maps.append(sample["scores"])
            masks.append(sample["mask"])
    maps, masks = np.stack(maps), np.stack(masks)
    labels = masks.any(axis=(1, 2)).astype(np.uint8)
    fit = evaluate_maps(
        maps[calibration], masks[calibration], labels[calibration], split="validation"
    )
    thresholds = {level: fit[level]["threshold"] for level in ("pixel", "image")}
    atomic_json(
        output / "frozen_thresholds.json",
        {
            "thresholds": thresholds,
            "calibration_samples": 40,
            "rule": "calibration max F1, largest threshold on tie; score>=threshold",
        },
    )
    measured = evaluate_maps(
        maps[holdout], masks[holdout], labels[holdout], split="validation", thresholds=thresholds
    )
    return {
        "thresholds": thresholds,
        "calibration": fit,
        "group_holdout": measured,
        "holdout_samples": 43,
        "calibration_samples": 40,
    }


def test_metrics(output, thresholds):
    """One score at a time; exact global pixel metrics, no per-shard AUC averaging."""
    output = Path(output)
    shape = (3665, 57600)
    labels = np.lib.format.open_memmap(
        output / "private_labels.npy", mode="w+", dtype=np.uint8, shape=shape
    )
    scores = np.lib.format.open_memmap(
        output / "private_scores.npy", mode="w+", dtype=np.float32, shape=shape
    )
    image_scores, image_labels, dice, fp, normals = [], [], [], 0, 0
    for i in range(shape[0]):
        with np.load(output / f"shard-{i:06d}.npz", allow_pickle=False) as sample:
            labels[i], scores[i] = sample["mask"].ravel(), sample["scores"].ravel()
        positive = bool(labels[i].any())
        image_labels.append(int(positive))
        image_scores.append(float(np.partition(scores[i].astype(np.float64), -576)[-576:].mean()))
        predicted = scores[i] >= thresholds["pixel"]
        if positive:
            dice.append(
                float(
                    2
                    * np.count_nonzero(predicted & (labels[i] > 0))
                    / (np.count_nonzero(predicted) + np.count_nonzero(labels[i]))
                )
            )
        else:
            fp += int(predicted.sum())
            normals += 57600
    report = {
        "pixel": compute_binary_metrics(labels.ravel(), scores.ravel(), thresholds["pixel"]),
        "image": compute_binary_metrics(image_labels, image_scores, thresholds["image"]),
    }
    report.update(
        pixel_dice_macro_lesion_slices=float(np.mean(dice)),
        empty_mask_pixel_false_positive_rate=fp / normals,
        pixel_aggregation="global micro240crop",
        image_aggregation="mean top576pixels",
    )
    report["pixel"]["dice_micro"] = report["pixel"]["f1"]
    curves = {}
    for level, y, s in (
        ("pixel", labels.ravel(), scores.ravel()),
        ("image", image_labels, image_scores),
    ):
        fpr, tpr, _ = roc_curve(y, s)
        ix = np.unique(np.linspace(0, len(fpr) - 1, min(2000, len(fpr))).astype(int))
        curves[level] = {"fpr": fpr[ix].tolist(), "tpr": tpr[ix].tolist()}
        del fpr, tpr
        precision, recall, _ = precision_recall_curve(y, s)
        ix = np.unique(np.linspace(0, len(recall) - 1, min(2000, len(recall))).astype(int))
        curves[level].update(precision=precision[ix].tolist(), recall=recall[ix].tolist())
        del precision, recall
    atomic_json(output / "curves.json", curves)
    del labels, scores
    return report

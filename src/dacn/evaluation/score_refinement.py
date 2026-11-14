"""Two LPIPS-centred refinements of Final LPIPS x brightness (validation only).

Fixed before running (14 Nov), after the frozen hybrid's test evaluation. Uses the
cached t300 DDIM20+30 producer inputs and Final reconstructions; no DDPM calls and
no test access.

Hypotheses
  H1 (LPIPS backbone): SqueezeNet is the smallest LPIPS trunk. AlexNet or VGG16
     LPIPS (same lpips 0.1 linear heads, same input contract) may localise
     differences more sharply and raise pixel AP, both raw and gated by brightness.
  H2 (normalised gate): raw brightness varies between slices and patients. A gate
     expressing brightness relative to normal brain tissue should suppress
     uniformly bright normal slices. Two fixed forms:
       normal_cdf  midrank CDF of brightness fitted on foreground pixels
                   (input > -0.95) of the normal calibration slices;
       slice_rank  midrank of each foreground pixel within its own slice.
     Background pixels get gate 0 in both.

Arms: backbone (squeeze, alex, vgg) x gate (none, brightness, normal_cdf,
slice_rank) = 12, plus the reference. The reference is the frozen hybrid:
producer SqueezeNet LPIPS x brightness. Thresholds are max-F1 on the 40-image
calibration half; paired patient-group bootstrap on the 43-image holdout.

Rule: an arm improves on the frozen hybrid only if the holdout pixel-AP delta CI
(arm - reference) is above 0, calibration AP is higher, and image AUROC is not lower
beyond uncertainty (delta CI upper bound >= 0). Otherwise the frozen hybrid stands.
With 12 arms, a pass is a candidate for a later decision, not a confirmed result.
Replay gate: CPU SqueezeNet LPIPS must match the producer maps within 1e-4.
"""

import numpy as np

from dacn.evaluation.final_fusion import apply_midrank_cdf

FOREGROUND_LEVEL = -0.95
BACKBONES = ("squeeze", "alex", "vgg")
GATES = ("none", "brightness", "normal_cdf", "slice_rank")
REFERENCE = "frozen_hybrid"
MAP_REPLAY_ATOL = 1e-4


def foreground(images):
    """Foreground of the 240 crop from (N,256,256) inputs in [-1, 1]."""
    images = np.asarray(images)
    if images.ndim != 3 or images.shape[1:] != (256, 256):
        raise ValueError("images must have shape (N, 256, 256)")
    return images[:, 8:-8, 8:-8] > FOREGROUND_LEVEL


def normal_cdf_gate(brightness, images, labels, calibration):
    brightness = np.asarray(brightness, dtype=np.float64)
    fg = foreground(images)
    calibration = np.asarray(calibration)
    normal = calibration[np.asarray(labels)[calibration] == 0]
    if not len(normal):
        raise ValueError("Normal calibration slices are required")
    reference = np.sort(brightness[normal][fg[normal]])
    return np.where(fg, apply_midrank_cdf(reference, brightness), 0.0)


def slice_rank_gate(brightness, images):
    from scipy.stats import rankdata

    brightness = np.asarray(brightness, dtype=np.float64)
    fg = foreground(images)
    out = np.zeros_like(brightness)
    for i in range(len(brightness)):
        values = brightness[i][fg[i]]
        if len(values):
            out[i][fg[i]] = (rankdata(values, method="average") - 0.5) / len(values)
    return out


def gates(brightness, images, labels, calibration):
    brightness = np.asarray(brightness, dtype=np.float64)
    return {
        "none": np.ones_like(brightness),
        "brightness": brightness,
        "normal_cdf": normal_cdf_gate(brightness, images, labels, calibration),
        "slice_rank": slice_rank_gate(brightness, images),
    }


def lpips_maps(model, images, reconstructions, device="cpu"):
    """Producer contract: 240 crop, grayscale repeated RGB, normalize=False, floor 0."""
    import torch

    out = np.empty((len(images), 240, 240), dtype=np.float32)
    with torch.inference_mode():
        for i, (image, recon) in enumerate(zip(images, reconstructions, strict=True)):
            tensors = [
                torch.from_numpy(np.asarray(v)[8:-8, 8:-8].copy())
                .float()[None, None]
                .repeat(1, 3, 1, 1)
                .to(device)
                for v in (image, recon)
            ]
            value = model(*tensors, normalize=False)[0, 0].cpu().numpy()
            if value.shape != (240, 240) or not np.isfinite(value).all():
                raise ValueError("Invalid spatial LPIPS map")
            out[i] = np.maximum(value, 0)
    return out


def refinement_arms(lpips_by_backbone, gate_maps, producer_lpips, brightness):
    """Frozen-hybrid reference plus every backbone x gate product."""
    arms = {REFERENCE: np.asarray(producer_lpips, np.float64) * np.asarray(brightness, np.float64)}
    for backbone in BACKBONES:
        lpips = np.asarray(lpips_by_backbone[backbone], dtype=np.float64)
        for gate in GATES:
            arms[f"{backbone}__{gate}"] = lpips * gate_maps[gate]
    return arms


def improves(name, summary, pixel_vs_reference, image_vs_reference):
    if name == REFERENCE:
        return False
    return bool(
        pixel_vs_reference["candidates"][name]["delta_ap_ci95"][0] > 0
        and summary[name]["calibration_ap"] > summary[REFERENCE]["calibration_ap"]
        and image_vs_reference["candidates"][name]["delta_auroc_ci95"][1] >= 0
    )

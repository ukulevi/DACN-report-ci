"""Paired anomaly maps for the BMAD 256-pixel input contract."""

import numpy as np
from scipy.signal import convolve2d


def paired_score_maps(image, reconstruction):
    """Return absolute and local SSIM differences for one cropped image pair.

    Inputs are 256x256 images in [-1, 1]. The result contains 240x240 maps;
    larger values mean more anomalous. No postprocessing or clipping is applied.
    """
    image = np.asarray(image, dtype=np.float64)
    reconstruction = np.asarray(reconstruction, dtype=np.float64)
    if image.shape != (256, 256) or reconstruction.shape != (256, 256):
        raise ValueError("image and reconstruction must both have shape (256, 256)")
    if not np.isfinite(image).all() or not np.isfinite(reconstruction).all():
        raise ValueError("image and reconstruction must be finite")
    if np.any(np.abs(image) > 1) or np.any(np.abs(reconstruction) > 1):
        raise ValueError("image and reconstruction must be in [-1, 1]")
    x, y = image[8:-8, 8:-8], reconstruction[8:-8, 8:-8]
    axis = np.arange(-5, 6, dtype=np.float64)
    gaussian = np.exp(-(axis**2) / (2 * 1.5**2))
    kernel = np.outer(gaussian, gaussian)
    kernel /= kernel.sum()

    def local_mean(a):
        return convolve2d(np.pad(a, 5, mode="reflect"), kernel, mode="valid")

    mx, my = local_mean(x), local_mean(y)
    vx = local_mean(x * x) - mx * mx
    vy = local_mean(y * y) - my * my
    covariance = local_mean(x * y) - mx * my
    c1, c2 = (0.01 * 2) ** 2, (0.03 * 2) ** 2
    ssim = ((2 * mx * my + c1) * (2 * covariance + c2)) / (
        (mx * mx + my * my + c1) * (vx + vy + c2)
    )
    return {"pixel_difference": np.abs(x - y), "local_ssim": 1 - ssim}

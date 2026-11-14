"""Frozen seed ensemble of Final LPIPS x brightness for one test evaluation.

Chosen by rule R3 (``dacn.evaluation.mask_gate``) before the GPU mask-gate run and
frozen here before any seed-18/19 test map exists. Definition: the frozen
DDIM20+30 pipeline (producer pseudo-mask: initial LPIPS x normalised residual, p95,
dilation 3) runs once per sampling seed base 17, 18 and 19. Each seed uses its own
pseudo-mask threshold, fitted on the validation normal-calibration slices in the
validation runs. The score is mean(Final LPIPS over the three seeds) x brightness,
with pixel/image thresholds fitted on the 40-image validation calibration half
(``maskgate-cpu-full-20261114T135821Z``, arm product_ensemble). The test is never
used for fitting or selection. This is the sixth test access.
"""

import numpy as np

SEEDS = (17, 18, 19)
# Validation pseudo-mask thresholds per seed: 17 from the producer, 18/19 from
# variants-full-20261114T041258Z (seed18/seed19_t300_20_30).
MASK_THRESHOLDS = {17: 0.13344676792621613, 18: 0.1258777529001236, 19: 0.1294720470905304}
FROZEN_ENSEMBLE_THRESHOLDS = {"pixel": 0.13542788565883218, "image": 0.15379337911124097}
ENSEMBLE_ARM = "seed_ensemble"
VALIDATION_HOLDOUT_AP = 0.7107883108575528
# hybrid-test-full-20261114T105717Z: frozen single-seed hybrid on test (pixel AP).
HYBRID_TEST_AP = 0.6505047984912514


def ensemble_map(finals, brightness):
    """Mean of per-seed Final LPIPS maps (float64) times brightness."""
    finals = [np.asarray(f, dtype=np.float64) for f in finals]
    if len(finals) != len(SEEDS) or len({f.shape for f in finals}) != 1:
        raise ValueError("Need one Final LPIPS map per seed with a shared shape")
    return np.mean(finals, axis=0) * np.asarray(brightness, dtype=np.float64)

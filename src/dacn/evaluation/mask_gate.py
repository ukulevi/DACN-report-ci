"""Brightness-gated pseudo-mask: decision rules for the CPU analysis (validation).

Fixed before the GPU ``maskgate`` run (14 Nov). The GPU run regenerates Final LPIPS
with the phase-1 pseudo-mask heatmap changed from initial LPIPS x normalised
residual (``product``, the producer rule) to initial LPIPS x brightness. Everything
else is unchanged: p95 threshold on normal calibration, dilation 3, masked DDIM30
from t50. Seeds 17/18/19. Scores are always gated by brightness afterwards
(Final LPIPS x brightness).

Arms
  product_s{17,18,19}    frozen-hybrid pipeline; s17 is the producer, s18/s19 come
                         from the variants run
  bmask_s{17,18,19}      brightness-gated mask
  product_ensemble, bmask_ensemble   mean of the three seeds' Final LPIPS, x brightness

Rules
  R1 (single seed): bmask_s17 improves on product_s17 if the pooled-83 AP delta CI is
     above 0, the holdout and calibration AP points are both higher, the pooled image
     AUROC is not lower beyond uncertainty, and bmask beats product on holdout AP for
     at least 2 of 3 seeds.
  R2 (secondary): holdout pixel recall of lesions >= 1000 px rises under bmask_s17
     (the failure the hypothesis targets). Reported; not a gate.
  R3 (test candidate, decided before seeing the results): if bmask_ensemble passes R1's
     first three conditions against product_ensemble, the single test candidate is
     bmask_ensemble; otherwise it is product_ensemble. Running the test needs a separate
     decision because it would be the sixth test access.
"""

SEEDS = (17, 18, 19)
# GPU source run; filled in after it completes. The CPU analysis checks its maps
# against that run's own manifest checksums.
MASKGATE_SOURCE_RUN = "maskgate-full-20261114T134252Z"
MASKGATE_SOURCE_REVISION = "d8441e8ba602ec068b3f24566af91b79ce5bd193"
LARGE_LESION_BIN = ">=1000"


def passes_core(name, reference, summary, pooled_pixel, pooled_image):
    """First three R1 conditions; ``pooled_*`` bootstraps use ``reference``."""
    return bool(
        pooled_pixel["candidates"][name]["delta_ap_ci95"][0] > 0
        and summary[name]["holdout_ap"] > summary[reference]["holdout_ap"]
        and summary[name]["calibration_ap"] > summary[reference]["calibration_ap"]
        and pooled_image["candidates"][name]["delta_auroc_ci95"][1] >= 0
    )


def seed_wins(summary, seeds=SEEDS):
    return sum(
        summary[f"bmask_s{s}"]["holdout_ap"] > summary[f"product_s{s}"]["holdout_ap"] for s in seeds
    )


def r1(summary, pooled_pixel_s17, pooled_image_s17):
    return bool(
        passes_core("bmask_s17", "product_s17", summary, pooled_pixel_s17, pooled_image_s17)
        and seed_wins(summary) >= 2
    )


def choose_test_candidate(summary, pooled_pixel_ens, pooled_image_ens):
    ok = passes_core(
        "bmask_ensemble", "product_ensemble", summary, pooled_pixel_ens, pooled_image_ens
    )
    return "bmask_ensemble" if ok else "product_ensemble"

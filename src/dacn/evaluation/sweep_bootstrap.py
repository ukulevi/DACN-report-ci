"""Paired patient-group bootstrap over the fixed 09 Nov DDIM noise/step sweep.

Scores, thresholds and partition are reused unchanged from producer artifacts;
this module selects nothing and never touches the held-out test split.
"""

import numpy as np

from dacn.evaluation.group_bootstrap import paired_group_bootstrap

# (phase-1 start t, phase-1 DDIM steps, phase-2 DDIM steps, evaluation ID,
#  private_ablation.npz SHA-256, results.json SHA-256) from producer manifests.
SWEEP_RUNS = (
    (
        300,
        10,
        40,
        "autoddpm-ddim1040-full83-20261109-a",
        "eb0078f73e132219f4be73b7bd7bdf472a0452e8b2d1dcb884269e9db4a4a02d",
        "77250cce8bfa3cfbdae0ec557a95335393cbf057e46deb553b7ce16cd3bed680",
    ),
    (
        300,
        20,
        30,
        "autoddpm-ddim2030-full83-t4-20261106T074200Z",
        "22656e1f63a0047d3dbe32f7833eea8e303988414e5942dec1ba5a1d9ed1ef19",
        "a8bae24bd7270b078d49f923461e43fb51aea3163ede5a2fe71c90a70b127730",
    ),
    (
        100,
        20,
        30,
        "autoddpm-ddim2030-t100-full-20261109T092149Z",
        "962ee089cd0fc9f5c26f46f9dea57b13dbe95dac0c1f339a1b5a01ec62722f75",
        "c24b0178d216daeebaf59769269326845838899659e207eb8149981aed9e4548",
    ),
    (
        200,
        20,
        30,
        "autoddpm-ddim2030-t200-full-20261109T092149Z",
        "58cc10fdf664145dba5356fbe0fe2f55491757e9e6d6a0832c81d7faf0667ff2",
        "30a16590c86dc1876d970de2265c9ff79d065658d6e1c19af6254833553d78eb",
    ),
    (
        500,
        20,
        30,
        "autoddpm-ddim2030-t500-full-20261109T095805Z",
        "8b4c2f460264cbcec13aba1f063ada18916d3bd1b57861c3e51110694512d6e9",
        "9302ebdd87ca7c50b06dc30acbff318f1a63542c72a07fa3a76f0bbf0b959217",
    ),
    (
        800,
        20,
        30,
        "autoddpm-ddim2030-t800-full-20261109T100840Z",
        "61fe7c298b364e9c30ce2f91c6de9784509b6e6efe25b5204744bda06e6622f7",
        "72c9f1e0fff3bd387961162e9250ee2f79e72f3778bf469a9ab316d05e2d97f9",
    ),
    (
        300,
        25,
        25,
        "autoddpm-ddim2525-full83-20261109-a",
        "2594fc01243430ebbce89cdd1800c1be19b183b78128cb04d4bb38e498bc0ca5",
        "47e1dd7554a3205cc8109a91b325d9f0d8ef591894b61b05053ddb289c43818b",
    ),
    (
        300,
        30,
        20,
        "autoddpm-ddim3020-full83-20261109-a",
        "9b931083a79183812f47a1190c581b80f1222592f3508e2bdf55b09606a7022a",
        "4dd22769529ef34a848dac9202bd049846d86a44881408c3bd0e80ae4e35b6aa",
    ),
    (
        100,
        30,
        20,
        "autoddpm-ddim3020-t100-full-20261109T092149Z",
        "7e6c253e87139cefb12619f464d362656ca9b6895b303242deb0fcf8f5a2acdf",
        "1de2a4ee7ade2cea7020b00205ccc886752d898d3a598429bdb6a3573bccf075",
    ),
    (
        200,
        30,
        20,
        "autoddpm-ddim3020-t200-full-20261109T092149Z",
        "ccb72b0305675b081613a61f37ed8ab2eedb2c189faefa1eca862eda99fdfbd1",
        "e6a32da1a89044e230c5e53c495e159caad4ce126328ce9bc8a26f7d1e5d2f21",
    ),
    (
        500,
        30,
        20,
        "autoddpm-ddim3020-t500-full-20261109T095805Z",
        "18e215b190519d1914677f92c0d946b19abfab56e5cd856d5fd0c95919736fe5",
        "5d82d882868fd304f2e47dda925d14f835472b1a623c3bf398bf7810ff3acbed",
    ),
    (
        800,
        30,
        20,
        "autoddpm-ddim3020-t800-full-20261109T100840Z",
        "85eb4289c7c8ffb0028296cd51bfaf22479c00191508ab98ab4d3fcc2abd8e34",
        "f118d2f66fbf23e4bc085b2d0c3471a95b7cfeb8afabe29845c8647f8a27342f",
    ),
)
REFERENCE_RUN = "autoddpm-ddim2030-full83-t4-20261106T074200Z"
REFERENCE = "final_lpips_t300_20_30"
# Arms bootstrapped from every run; extra controls only from the reference run.
SWEEP_ARMS = {"autoddpm_lpips": "final_lpips", "initial_lpips_only": "initial_lpips"}
REFERENCE_CONTROLS = {
    "autoddpm_lpips_residual": "final_lpips_x_residual_t300_20_30",
    "naive_stitch_l1": "naive_stitch_l1_t300_20_30",
    "ddpm_l1": "initial_l1_t300_20_30",
    "brightness": "brightness",
}
SHARED_ARRAYS = ("images", "masks", "labels")


def candidate_name(arm_label, start, steps1, steps2):
    return f"{arm_label}_t{start}_{steps1}_{steps2}"


def collect_candidates(runs, holdout):
    """Build fixed holdout maps, thresholds and producer point metrics.

    ``runs`` maps evaluation ID to ``(spec, arrays, results)`` where ``arrays`` is
    a mapping of the producer NPZ members and ``results`` the parsed results.json.
    Requires identical images/masks/labels across runs so draws stay paired.
    """
    if REFERENCE_RUN not in runs:
        raise ValueError("Reference run is required")
    reference_arrays = runs[REFERENCE_RUN][1]
    maps, thresholds, expected = {}, {}, {}
    for evaluation_id, (spec, arrays, results) in runs.items():
        start, steps1, steps2 = spec
        config = results["config"]
        if (
            results.get("held_out_test_used") is not False
            or results.get("smoke") is not False
            or results.get("samples") != 83
            or (
                config["phase1_starting_timestep"],
                config["phase1_inference_steps"],
                config["phase2_inference_steps"],
            )
            != (start, steps1, steps2)
        ):
            raise ValueError("Unexpected producer configuration: " + evaluation_id)
        for key in SHARED_ARRAYS:
            if not np.array_equal(arrays[key], reference_arrays[key]):
                raise ValueError("Sample order/content differs across runs: " + evaluation_id)
        arms = dict.fromkeys(SWEEP_ARMS)
        if evaluation_id == REFERENCE_RUN:
            arms.update(dict.fromkeys(REFERENCE_CONTROLS))
        for arm in arms:
            label = SWEEP_ARMS.get(arm)
            name = (
                candidate_name(label, start, steps1, steps2) if label else REFERENCE_CONTROLS[arm]
            )
            maps[name] = np.asarray(arrays[arm])[holdout]
            thresholds[name] = float(results["parameters"]["score_thresholds"][arm]["pixel"])
            pixel = results["metrics"][arm]["group_holdout"]["pixel"]
            expected[name] = (float(pixel["auprc"]), float(pixel["f1"]))
    truth = np.asarray(reference_arrays["masks"])[holdout, 8:-8, 8:-8]
    if any(values.shape != truth.shape for values in maps.values()):
        raise ValueError("Score maps must match the cropped 240x240 masks")
    return truth, maps, thresholds, expected


def bootstrap_sweep(truth, maps, groups, thresholds, expected, *, replicates, seed=17, atol=1e-9):
    """Run the paired bootstrap and require producer AP/Dice replay before reporting."""
    report = paired_group_bootstrap(
        truth, maps, groups, thresholds, reference=REFERENCE, replicates=replicates, seed=seed
    )
    for name, (ap, dice) in expected.items():
        measured = report["candidates"][name]
        if abs(measured["pixel_ap"] - ap) > atol or abs(measured["pixel_dice_micro"] - dice) > atol:
            raise ValueError("Producer point-metric replay failed: " + name)
    report["producer_replay_max_abs_error"] = max(
        max(
            abs(report["candidates"][n]["pixel_ap"] - ap),
            abs(report["candidates"][n]["pixel_dice_micro"] - dice),
        )
        for n, (ap, dice) in expected.items()
    )
    return report

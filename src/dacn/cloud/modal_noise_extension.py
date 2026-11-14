"""T4 validation-only extension of the DDIM20+30 noise sweep to t50 and t400 (AF-05).

The producer-equivalent t300 DDIM20+30 control must replay the cached reference
arrays before new points run. Full mode reports producer-style metrics for all
eight score maps, mask diagnostics, reconstruction-quality aggregates and a
paired patient-group bootstrap against the t300 control. No test access.

    DACN_HF_SECRET=huggingface modal run -m dacn.cloud.modal_noise_extension \
        --evaluation-id noise-ext-smoke-UNIQUE --code-revision GIT_SHA --mode smoke \
        --no-working-tree-dirty
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import modal

from dacn.cloud.modal_mask_rule_ablation import (
    CHECKPOINT,
    CHECKPOINT_SHA,
    LPIPS_STATE_SHA,
    image,
)
from dacn.cloud.modal_sweep_bootstrap import (
    DATA_ROOT,
    DATASET,
    HF_SECRET,
    INDEX_SHA,
    MODEL_REPO,
    MODEL_REVISION,
    PARTITION_SHA,
    REVISION,
    RUN_ROOT,
    data_volume,
    run_volume,
)

app = modal.App("dacn-noise-extension")


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=16384,
    timeout=2 * 60 * 60,
    volumes={str(DATA_ROOT): data_volume, str(RUN_ROOT): run_volume},
    secrets=[modal.Secret.from_name(HF_SECRET)],
)
def run_noise_extension_remote(
    evaluation_id: str, code_revision: str, working_tree_dirty: bool, mode: str, suite: str
):
    import hashlib
    import time

    import lpips
    import numpy as np
    import torch
    from huggingface_hub import HfApi, hf_hub_download

    from dacn.cloud.modal_sweep_bootstrap import _iter_verified_runs, _pinned_validation
    from dacn.cloud.modal_train import (
        _file_checksums,
        _json_safe,
        _reject_patient_identifiers,
        _write_run_manifest,
        _write_source_snapshot,
    )
    from dacn.evaluation.autoddpm_lpips import perceptual_provenance
    from dacn.evaluation.group_bootstrap import paired_group_bootstrap
    from dacn.evaluation.mask_rule_ablation import mask_statistics
    from dacn.evaluation.noise_extension import (
        CONTROL_POINT,
        MASKGATE_VARIANTS,
        NEW_POINTS,
        PUBLISHED_HOLDOUT_AP,
        PUBLISHED_LPIPS_MASK_THRESHOLD,
        SCORE_NAMES,
        SMOKE_MASKGATE,
        SMOKE_POINTS,
        SMOKE_VARIANTS,
        VARIANTS,
        check_point_replay,
        point_name,
        run_point,
    )
    from dacn.evaluation.noise_quality import reconstruction_quality
    from dacn.evaluation.sweep_bootstrap import REFERENCE_RUN
    from dacn.evaluation.validation import evaluate_maps
    from dacn.training.ddpm_partial import build_diffusion, load_ema_checkpoint

    if (
        mode not in {"smoke", "full"}
        or suite not in {"noise", "variants", "maskgate"}
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", evaluation_id)
    ):
        raise ValueError("Invalid noise-extension request")
    if suite == "noise":
        points = SMOKE_POINTS if mode == "smoke" else (CONTROL_POINT, *NEW_POINTS)
        specs = [(point_name(*p), p, {}, p == CONTROL_POINT) for p in points]
    else:
        table = VARIANTS if suite == "variants" else MASKGATE_VARIANTS
        smoke = SMOKE_VARIANTS if suite == "variants" else SMOKE_MASKGATE
        names = smoke if mode == "smoke" else tuple(table)
        specs = []
        for name in names:
            point, phase2_start, seed_base, mask_score = table[name]
            kwargs = {
                "phase2_start": phase2_start,
                "seed_base": seed_base,
                "mask_score": mask_score,
            }
            specs.append((name, point, kwargs, name == "control_t300_20_30"))
    if not specs[0][3]:
        raise ValueError("The t300 control must run first")
    control_name = specs[0][0]
    output = RUN_ROOT / "evaluations" / evaluation_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "evaluation_id": evaluation_id,
        "method": {
            "noise": "ddim20_30_noise_sweep_extension_t50_t400",
            "variants": "ddim_validation_variants_seeds_budget_phase2_lpips_mask",
            "maskgate": "ddim20_30_brightness_gated_pseudo_mask_seeds",
        }[suite],
        "suite": suite,
        "mode": mode,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "split": "validation",
        "dataset_repo_id": DATASET,
        "dataset_revision": REVISION,
        "model_repo_revision": MODEL_REVISION,
        "reference_run": REFERENCE_RUN,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "index_sha256": INDEX_SHA,
        "validation_partition_sha256": PARTITION_SHA,
        "points": [spec[0] for spec in specs],
        "code_revision": code_revision,
        "working_tree_dirty": working_tree_dirty,
        "source_snapshot_sha256": _write_source_snapshot(output),
        "held_out_test_used": False,
        "training_performed": False,
        "gpu": torch.cuda.get_device_name(0),
    }

    def persist():
        _write_run_manifest(output, manifest)
        run_volume.commit()

    persist()
    started = time.perf_counter()
    try:
        token = os.environ["HF_TOKEN"]
        api = HfApi(token=token)
        if not api.repo_info(MODEL_REPO).private:
            raise ValueError("Private artifact repository required")
        groups, holdout, dataset = _pinned_validation()
        calibration, sample_ids = dataset["calibration"], dataset["sample_ids"]
        [(_, _, cached, results, verified)] = list(
            _iter_verified_runs(token, dataset, run_ids={REFERENCE_RUN}, keep_reconstructions=True)
        )
        cached["mask_threshold"] = float(results["parameters"]["mask_threshold"])
        images, labels, masks = cached["images"], cached["labels"], cached["masks"]
        selected = (
            [int(np.flatnonzero(labels == target)[0]) for target in (0, 1)]
            if mode == "smoke"
            else list(range(len(labels)))
        )
        checkpoint_path = Path(
            hf_hub_download(
                MODEL_REPO, CHECKPOINT, revision=MODEL_REVISION, token=token, cache_dir="/tmp/hf"
            )
        )
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != CHECKPOINT_SHA:
            raise ValueError("Checkpoint checksum differs")
        # SHA-verified team checkpoint; same loader contract as the producer runs.
        config = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["config"]
        device = torch.device("cuda")
        model = load_ema_checkpoint(checkpoint_path, config, device)
        diffusion = build_diffusion(config)
        perceptual = lpips.LPIPS(net="squeeze", spatial=True, version="0.1").to(device)
        perceptual.eval().requires_grad_(False)
        if perceptual_provenance(perceptual)["state_sha256"] != LPIPS_STATE_SHA:
            raise ValueError("LPIPS weights differ from the accepted state")
        counter = {"unet": 0}
        hook = model.register_forward_hook(
            lambda *_: counter.__setitem__("unet", counter["unet"] + 1)
        )
        torch.cuda.reset_peak_memory_stats()
        outputs, replay = {}, None
        for name, point, kwargs, is_control in specs:
            result = run_point(
                model=model,
                diffusion=diffusion,
                perceptual=perceptual,
                device=device,
                images=images,
                labels=labels,
                sample_ids=sample_ids,
                calibration=calibration,
                selected=selected,
                point=point,
                counter=counter,
                **kwargs,
            )
            if is_control:
                replay = check_point_replay(result, cached, selected)
            outputs[name] = result
            manifest["progress"] = {"completed_points": list(outputs), "control_replay": replay}
            persist()
        hook.remove()
        report = {
            "method": manifest["method"],
            "mode": mode,
            "question": {
                "noise": "Advisor AF-05: add the suggested t50 and t400 phase-1 noise levels "
                "with all other components fixed (DDIM 50-call budget, phase 2 from t50, "
                "p95/d3 mask).",
                "variants": "Advisor AF-11/AF-07/AF-02: sampling-seed stability (base 18/19 "
                "at t300 and t400), total DDIM budget (10+15, 40+60), phase-2 start (t30, "
                "t100) and an LPIPS-only pseudo-mask, each changing one component of the "
                "t300 control.",
                "maskgate": "AF-07 follow-up: pseudo-mask from initial LPIPS x brightness "
                "instead of LPIPS x residual (same p95/d3 rule), sampling seeds 17/18/19.",
            }[suite],
            "variants": {
                spec[0]: {"point": spec[1], **spec[2]} for spec in specs if suite != "noise"
            },
            "control": control_name,
            "control_replay": replay,
            "producer_artifacts": {REFERENCE_RUN: verified},
            "samples": len(selected),
            "unet_calls_total": counter["unet"],
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "selection": "none; all points reported; no test access",
            "points": {},
        }
        if mode == "full":
            truth = masks[:, 8:-8, 8:-8]
            boot_maps, boot_thresholds = {}, {}
            everything = np.arange(len(labels))
            for name, result in outputs.items():
                metrics, thresholds = {}, {}
                for score in SCORE_NAMES:
                    values = result["maps"][score]
                    fitted = evaluate_maps(
                        values[calibration],
                        truth[calibration],
                        labels[calibration],
                        split="validation",
                    )
                    thresholds[score] = {lv: fitted[lv]["threshold"] for lv in ("pixel", "image")}
                    metrics[score] = {
                        "calibration": fitted,
                        "group_holdout": evaluate_maps(
                            values[holdout],
                            truth[holdout],
                            labels[holdout],
                            split="validation",
                            thresholds=thresholds[score],
                        ),
                    }
                quality_arrays = {
                    "images": images,
                    "initials": result["initials"],
                    "finals": result["finals"],
                    "masks": masks,
                    "labels": labels,
                    "predicted_masks": result["predicted_masks"],
                    "initial_lpips_only": result["maps"]["initial_lpips_only"],
                    "autoddpm_lpips": result["maps"]["autoddpm_lpips"],
                }
                report["points"][name] = {
                    "mask_threshold": result["mask_threshold"],
                    "phase1_indices_zero_based": result["phase1_indices"],
                    "phase2_indices_zero_based": result["phase2_indices"],
                    "phase1_median_seconds": result["phase1_median_seconds"],
                    "phase2_median_seconds": result["phase2_median_seconds"],
                    "score_thresholds": thresholds,
                    "metrics": metrics,
                    "mask_all83": mask_statistics(
                        result["predicted_masks"], images, masks, labels, everything
                    ),
                    "quality_all83": reconstruction_quality(quality_arrays, everything),
                    "quality_holdout43": reconstruction_quality(quality_arrays, holdout),
                }
                for score, label in (
                    ("autoddpm_lpips", "final"),
                    ("initial_lpips_only", "initial"),
                ):
                    key = f"{label}_lpips_{name}"
                    boot_maps[key] = result["maps"][score][holdout]
                    boot_thresholds[key] = thresholds[score]["pixel"]
                if suite == "noise":
                    np.savez_compressed(
                        output / f"private_{name}.npz",
                        images=images,
                        initials=result["initials"],
                        finals=result["finals"],
                        masks=masks,
                        labels=labels,
                        predicted_masks=result["predicted_masks"],
                        **result["maps"],
                    )
                else:  # Score maps and masks only; no images or reconstructions.
                    np.savez_compressed(
                        output / f"private_{name}_maps.npz",
                        autoddpm_lpips=result["maps"]["autoddpm_lpips"],
                        initial_lpips_only=result["maps"]["initial_lpips_only"],
                        predicted_masks=result["predicted_masks"],
                    )
            if suite == "variants":
                published = {}
                for name, (source, value) in PUBLISHED_HOLDOUT_AP.items():
                    measured = report["points"][name]["metrics"]["autoddpm_lpips"]
                    measured = measured["group_holdout"]["pixel"]["auprc"]
                    published[name] = {
                        "source": source,
                        "published_holdout_ap": value,
                        "measured_holdout_ap": measured,
                        "abs_error": abs(measured - value),
                    }
                measured = report["points"]["lpips_mask_t300_20_30"]["mask_threshold"]
                published["lpips_mask_threshold"] = {
                    "published": PUBLISHED_LPIPS_MASK_THRESHOLD,
                    "measured": measured,
                    "abs_error": abs(measured - PUBLISHED_LPIPS_MASK_THRESHOLD),
                }
                report["published_comparisons"] = published
            report["paired_bootstrap_vs_control"] = paired_group_bootstrap(
                truth[holdout],
                boot_maps,
                groups[holdout],
                boot_thresholds,
                reference=f"final_lpips_{control_name}",
                replicates=1000,
                seed=17,
            )
        report = _json_safe(report)
        _reject_patient_identifiers(report)
        (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        manifest.update(
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            total_seconds=time.perf_counter() - started,
            artifacts=_file_checksums(output),
        )
        persist()
        try:
            commit = api.upload_folder(
                folder_path=str(output),
                path_in_repo=f"evaluations/{evaluation_id}",
                repo_id=MODEL_REPO,
                commit_message=f"Store noise-sweep extension {evaluation_id}",
            )
            manifest.update(artifact_sync_status="uploaded", hub_commit=commit.oid)
        except Exception as exc:  # noqa: BLE001 - upload failure is recorded; results stay on the Volume
            manifest.update(artifact_sync_status="failed", artifact_sync_error=type(exc).__name__)
        persist()
        return {"manifest": manifest, "results": report}
    except BaseException as exc:
        manifest.update(
            status="failed", error=type(exc).__name__, total_seconds=time.perf_counter() - started
        )
        manifest["error_message"] = str(exc)[:300]
        persist()
        raise


@app.local_entrypoint()
def main(
    evaluation_id: str,
    code_revision: str,
    mode: str = "smoke",
    suite: str = "noise",
    working_tree_dirty: bool = True,
):
    result = run_noise_extension_remote.remote(
        evaluation_id, code_revision, working_tree_dirty, mode, suite
    )
    destination = Path("runs") / "noise-extension" / evaluation_id
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in result.items():
        (destination / f"{name}.json").write_bytes(
            (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        )
    print(json.dumps(result["manifest"], indent=2))

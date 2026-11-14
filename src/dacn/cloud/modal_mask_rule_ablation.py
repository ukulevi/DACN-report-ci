"""T4 validation-only pseudo-mask rule ablation for Final LPIPS (advisor AF-07).

Reuses the verified reference sweep run (DDIM20 t300 initials and heatmaps); per
mask rule it fits the pooled normal-calibration percentile threshold, re-runs the
masked DDIM30(t50) phase 2 and scores raw spatial LPIPS. The p95/dilation3
control must reproduce the producer threshold, masks, finals and LPIPS maps
before any other arm runs. No test access, model selection or retraining.

    DACN_HF_SECRET=huggingface modal run -m dacn.cloud.modal_mask_rule_ablation \
        --evaluation-id mask-rule-smoke-UNIQUE --code-revision GIT_SHA --mode smoke \
        --no-working-tree-dirty
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import modal

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

app = modal.App("dacn-mask-rule-ablation")
CHECKPOINT = "runs/ddpm-partial-t4-b8-e30-cont-20261102T034644Z/ddpm_epoch_030.pt"
CHECKPOINT_SHA = "d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946"
LPIPS_STATE_SHA = "18fed4d1fc5c04f2918f64df3a2463344108576d1888f0d75538ca6d5c4dffe6"
BOOTSTRAP_REPLICATES = 1000

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2",
        "torchvision",
        "lpips==0.1.4",
        "numpy>=1.24,<3",
        "scipy",
        "scikit-learn>=1.3,<2",
        "numba>=0.60,<1",
        "huggingface_hub>=0.28,<3",
        "PyYAML>=6,<7",
        "tqdm>=4.66",
    )
    .run_commands(
        "python -c \"import lpips; lpips.LPIPS(net='squeeze', spatial=True, version='0.1')\""
    )
    .add_local_python_source("dacn")
)


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=16384,
    timeout=2 * 60 * 60,
    volumes={str(DATA_ROOT): data_volume, str(RUN_ROOT): run_volume},
    secrets=[modal.Secret.from_name(HF_SECRET)],
)
def run_mask_rule_ablation_remote(
    evaluation_id: str, code_revision: str, working_tree_dirty: bool, mode: str
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
    from dacn.evaluation.ddim_ablation import ablation_pair_scores
    from dacn.evaluation.ddim_inpainting import ddim_indices, ddim_stitch_resample
    from dacn.evaluation.group_bootstrap import paired_group_bootstrap
    from dacn.evaluation.mask_rule_ablation import (
        CONTROL,
        MASK_RULES,
        SMOKE_RULES,
        check_control_replay,
        fit_percentile_threshold,
        mask_statistics,
        rule_mask,
        rule_name,
    )
    from dacn.evaluation.noise_quality import reconstruction_quality
    from dacn.evaluation.sweep_bootstrap import REFERENCE_RUN
    from dacn.evaluation.validation import evaluate_maps
    from dacn.training.ddpm_partial import (
        build_diffusion,
        derive_sampling_seed,
        load_ema_checkpoint,
    )

    if mode not in {"smoke", "full"} or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", evaluation_id
    ):
        raise ValueError("Invalid mask-rule ablation request")
    rules = SMOKE_RULES if mode == "smoke" else MASK_RULES
    if rules[0] != CONTROL:
        raise ValueError("The control rule must run first")
    output = RUN_ROOT / "evaluations" / evaluation_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "evaluation_id": evaluation_id,
        "method": "pseudo_mask_rule_ablation_final_lpips_ddim20_30_t300",
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
        "mask_rules": [rule_name(*rule) for rule in rules],
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
        labels, images, masks = cached["labels"], cached["images"], cached["masks"]
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
        config = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["config"]
        model = load_ema_checkpoint(checkpoint_path, config, "cuda")
        diffusion = build_diffusion(config)
        perceptual = lpips.LPIPS(net="squeeze", spatial=True, version="0.1").to("cuda")
        perceptual.eval().requires_grad_(False)
        if perceptual_provenance(perceptual)["state_sha256"] != LPIPS_STATE_SHA:
            raise ValueError("LPIPS weights differ from the accepted state")
        phase2_indices = ddim_indices(50, 30)
        calls = {"unet": 0}
        hook = model.register_forward_hook(lambda *_: calls.__setitem__("unet", calls["unet"] + 1))
        torch.cuda.reset_peak_memory_stats()
        heatmaps = cached["initial_product"]
        arms, replay, timings = {}, None, {}
        for percentile, dilation in rules:
            name = rule_name(percentile, dilation)
            threshold = fit_percentile_threshold(heatmaps, labels, calibration, percentile)
            predicted = np.stack([rule_mask(heatmaps[i], threshold, dilation) for i in selected])
            finals, lpips_maps, seconds = [], [], []
            for position, index in enumerate(selected):
                mask = torch.from_numpy(np.pad(predicted[position], 8))[None, None].to("cuda")
                image_t = torch.from_numpy(images[index])[None, None].to("cuda")
                initial_t = torch.from_numpy(cached["initials"][index])[None, None].to("cuda")
                before = calls["unet"]
                torch.cuda.synchronize()
                tick = time.perf_counter()
                final = ddim_stitch_resample(
                    model,
                    image_t,
                    initial_t,
                    mask,
                    diffusion=diffusion,
                    seed=derive_sampling_seed(17, sample_ids[index]),
                    indices=phase2_indices,
                    repeats=1,
                )
                torch.cuda.synchronize()
                seconds.append(time.perf_counter() - tick)
                if calls["unet"] - before != 30 or not torch.isfinite(final).all():
                    raise ValueError("Phase 2 requires exactly 30 finite U-Net evaluations")
                if float((final - image_t)[~mask].abs().max()) > 1e-7:
                    raise ValueError("Outside-mask context changed")
                final = final[0, 0].cpu().numpy()
                scores, _ = ablation_pair_scores(
                    perceptual, images[index], final, torch.device("cuda")
                )
                finals.append(final)
                lpips_maps.append(scores["lpips"])
            finals, lpips_maps = np.stack(finals), np.stack(lpips_maps)
            if (percentile, dilation) == CONTROL:
                subset = {
                    "mask_threshold": cached["mask_threshold"],
                    "predicted_masks": cached["predicted_masks"][selected],
                    "finals": cached["finals"][selected],
                    "autoddpm_lpips": cached["autoddpm_lpips"][selected],
                }
                replay = check_control_replay(threshold, predicted, finals, lpips_maps, subset)
            arms[name] = {
                "percentile": percentile,
                "dilation": dilation,
                "mask_threshold": threshold,
                "predicted": predicted,
                "finals": finals,
                "lpips": lpips_maps,
            }
            timings[name] = {
                "phase2_median_seconds": float(np.median(seconds)),
                "phase2_sum_seconds": float(np.sum(seconds)),
            }
            manifest["progress"] = {"completed_rules": list(arms), "control_replay": replay}
            persist()
        hook.remove()
        if calls["unet"] != 30 * len(selected) * len(rules):
            raise ValueError("Unexpected total U-Net count")
        report = {
            "method": manifest["method"],
            "mode": mode,
            "hypothesis": (
                "Tighter pseudo-masks (higher percentile and/or no dilation) re-sample less "
                "normal tissue, lowering normal-tissue LPIPS and FPR after phase 2; AP is "
                "retained only while lesion recall stays high."
            ),
            "fixed_components": (
                "Gaussian EMA30 checkpoint; cached DDIM20 t300 initials and "
                "LPIPS x clip(L1/P95) heatmaps; masked DDIM30 from t50, eta0, one pass; "
                "base17/UUID seeds; raw spatial LPIPS0.1 squeeze; full240 crop."
            ),
            "selection": "none; all arms reported; no test access",
            "control": rule_name(*CONTROL),
            "control_replay": replay,
            "producer_artifacts": {REFERENCE_RUN: verified},
            "samples": len(selected),
            "unet_calls_total": calls["unet"],
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "timings": timings,
            "arms": {},
        }
        if mode == "full":
            truth = masks[:, 8:-8, 8:-8]
            maps, thresholds = {}, {}
            for name, arm in arms.items():
                fitted = evaluate_maps(
                    arm["lpips"][calibration],
                    truth[calibration],
                    labels[calibration],
                    split="validation",
                )
                frozen = {level: fitted[level]["threshold"] for level in ("pixel", "image")}
                quality_arrays = {
                    "images": images,
                    "initials": cached["initials"],
                    "finals": arm["finals"],
                    "masks": masks,
                    "labels": labels,
                    "predicted_masks": arm["predicted"],
                    "initial_lpips_only": cached["initial_lpips_only"],
                    "autoddpm_lpips": arm["lpips"],
                }
                report["arms"][name] = {
                    "percentile": arm["percentile"],
                    "dilation": arm["dilation"],
                    "mask_threshold": arm["mask_threshold"],
                    "score_thresholds": frozen,
                    "calibration": fitted,
                    "group_holdout": evaluate_maps(
                        arm["lpips"][holdout],
                        truth[holdout],
                        labels[holdout],
                        split="validation",
                        thresholds=frozen,
                    ),
                    "mask_holdout": mask_statistics(
                        arm["predicted"], images, masks, labels, holdout
                    ),
                    "mask_all83": mask_statistics(
                        arm["predicted"], images, masks, labels, np.arange(len(labels))
                    ),
                    "final_quality_all83": reconstruction_quality(
                        quality_arrays, np.arange(len(labels))
                    )["phases"]["final"],
                }
                maps[name] = arm["lpips"][holdout]
                thresholds[name] = frozen["pixel"]
            report["paired_bootstrap_vs_control"] = paired_group_bootstrap(
                truth[holdout],
                maps,
                groups[holdout],
                thresholds,
                reference=rule_name(*CONTROL),
                replicates=BOOTSTRAP_REPLICATES,
                seed=17,
            )
            np.savez_compressed(
                output / "private_mask_rule_maps.npz",
                **{f"{n}_lpips": a["lpips"] for n, a in arms.items()},
                **{f"{n}_predicted_mask": a["predicted"] for n, a in arms.items()},
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
                commit_message=f"Store mask-rule ablation {evaluation_id}",
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
    evaluation_id: str, code_revision: str, mode: str = "smoke", working_tree_dirty: bool = True
):
    result = run_mask_rule_ablation_remote.remote(
        evaluation_id, code_revision, working_tree_dirty, mode
    )
    destination = Path("runs") / "mask-rule" / evaluation_id
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in result.items():
        (destination / f"{name}.json").write_bytes(
            (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        )
    print(json.dumps(result["manifest"], indent=2))

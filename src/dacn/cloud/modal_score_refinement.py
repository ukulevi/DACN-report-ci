"""CPU analysis: LPIPS backbone and normalised brightness gate (validation only).

Hypotheses, arms and rule: ``dacn.evaluation.score_refinement`` (fixed before
running). Recomputes spatial LPIPS with SqueezeNet, AlexNet and VGG16 on the cached
producer inputs and Final reconstructions of the t300 DDIM20+30 reference run. No
DDPM calls and no test access; only aggregates are uploaded.

    DACN_HF_SECRET=huggingface modal run -m dacn.cloud.modal_score_refinement \
        --evaluation-id refine-smoke-UNIQUE --code-revision GIT_SHA --mode smoke \
        --no-working-tree-dirty
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import modal

from dacn.cloud.modal_mask_rule_ablation import LPIPS_STATE_SHA
from dacn.cloud.modal_sweep_bootstrap import (
    DATA_ROOT,
    DATASET,
    HF_SECRET,
    INDEX_SHA,
    MODEL_REPO,
    MODEL_REVISION,
    PARTITION_SHA,
    REPLICATES,
    REVISION,
    RUN_ROOT,
    SCOPE,
    data_volume,
    run_volume,
)

app = modal.App("dacn-score-refinement")
LPIPS_INIT = "; ".join(
    f"lpips.LPIPS(net='{net}', spatial=True, version='0.1')" for net in ("squeeze", "alex", "vgg")
)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2", "torchvision>=0.17", index_url="https://download.pytorch.org/whl/cpu"
    )
    .pip_install(
        "numpy>=1.24,<3",
        "numba>=0.60,<1",
        "scipy>=1.10,<2",
        "scikit-learn>=1.3,<2",
        "huggingface_hub>=0.28,<3",
        "lpips==0.1.4",
    )
    .run_commands(f'python -c "import lpips; {LPIPS_INIT}"')
    .add_local_python_source("dacn")
)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=2 * 60 * 60,
    volumes={str(DATA_ROOT): data_volume, str(RUN_ROOT): run_volume},
    secrets=[modal.Secret.from_name(HF_SECRET)],
)
def run_refinement_remote(
    evaluation_id: str, code_revision: str, working_tree_dirty: bool, mode: str
):
    import hashlib
    import time

    import lpips
    import numpy as np
    import torch
    from huggingface_hub import HfApi

    from dacn.cloud.modal_sweep_bootstrap import (
        _evaluate_arms,
        _iter_verified_runs,
        _pinned_validation,
    )
    from dacn.cloud.modal_train import (
        _file_checksums,
        _json_safe,
        _reject_patient_identifiers,
        _write_run_manifest,
        _write_source_snapshot,
    )
    from dacn.evaluation import score_refinement
    from dacn.evaluation.postprocess_hybrid import HYBRID_REPLAY
    from dacn.evaluation.score_refinement import (
        BACKBONES,
        MAP_REPLAY_ATOL,
        REFERENCE,
        gates,
        improves,
        lpips_maps,
        refinement_arms,
    )
    from dacn.evaluation.sweep_bootstrap import REFERENCE_RUN

    if mode not in REPLICATES or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", evaluation_id
    ):
        raise ValueError("Invalid refinement request")
    output = RUN_ROOT / "evaluations" / evaluation_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "evaluation_id": evaluation_id,
        "method": "lpips_backbone_and_normalised_brightness_gate_t300",
        "mode": mode,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "split": "validation",
        "dataset_repo_id": DATASET,
        "dataset_revision": REVISION,
        "model_repo_revision": MODEL_REVISION,
        "index_sha256": INDEX_SHA,
        "validation_partition_sha256": PARTITION_SHA,
        "code_revision": code_revision,
        "working_tree_dirty": working_tree_dirty,
        "source_snapshot_sha256": _write_source_snapshot(output),
        "held_out_test_used": False,
        "new_ddpm_calls": 0,
        "gpu": None,
    }

    def persist():
        _write_run_manifest(output, manifest)
        run_volume.commit()

    persist()
    started = time.perf_counter()
    try:
        torch.set_num_threads(8)
        token = os.environ["HF_TOKEN"]
        api = HfApi(token=token)
        if not api.repo_info(MODEL_REPO).private:
            raise ValueError("Private artifact repository required")
        groups, holdout, dataset = _pinned_validation()
        [(_, _, arrays, results, verified)] = list(
            _iter_verified_runs(token, dataset, run_ids={REFERENCE_RUN}, keep_reconstructions=True)
        )
        labels, calibration = arrays["labels"], dataset["calibration"]
        truth = arrays["masks"][:, 8:-8, 8:-8]
        images, finals = arrays["images"], arrays["finals"]
        maps, provenance, seconds = {}, {}, {}
        for net in BACKBONES:
            model = lpips.LPIPS(net=net, spatial=True, version="0.1", verbose=False).eval()
            model.requires_grad_(False)
            digest = hashlib.sha256()
            for name, value in sorted(model.state_dict().items()):
                digest.update(name.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            provenance[net] = {
                "network": f"LPIPS0.1_{net}_spatial",
                "state_sha256": digest.hexdigest(),
            }
            if net == "squeeze" and digest.hexdigest() != LPIPS_STATE_SHA:
                raise ValueError("SqueezeNet LPIPS weights differ from the accepted state")
            tick = time.perf_counter()
            maps[net] = lpips_maps(model, images, finals)
            seconds[net] = time.perf_counter() - tick
            del model
        squeeze_error = float(np.max(np.abs(maps["squeeze"] - arrays["autoddpm_lpips"])))
        if squeeze_error > MAP_REPLAY_ATOL:
            raise ValueError(f"CPU SqueezeNet LPIPS does not replay producer maps: {squeeze_error}")
        gate_maps = gates(arrays["brightness"], images, labels, calibration)
        arms_maps = refinement_arms(maps, gate_maps, arrays["autoddpm_lpips"], arrays["brightness"])
        del maps, gate_maps, finals
        arms, summary, _, pixel, image_ci = _evaluate_arms(
            arms_maps,
            truth,
            labels,
            calibration,
            holdout,
            groups,
            results,
            mode,
            (REFERENCE,),
            replay_arms={},
        )
        held = arms[REFERENCE]["group_holdout"]
        hybrid_error = max(
            abs(held["pixel"]["auprc"] - HYBRID_REPLAY[0]),
            abs(held["pixel"]["f1"] - HYBRID_REPLAY[1]),
            abs(held["image"]["auroc"] - HYBRID_REPLAY[2]),
        )
        if hybrid_error > 1e-9:
            raise ValueError("Reference does not replay the frozen hybrid holdout metrics")
        improving = [
            name for name in arms if improves(name, summary, pixel[REFERENCE], image_ci[REFERENCE])
        ]
        report = {
            "method": manifest["method"],
            "hypothesis_and_rule": score_refinement.__doc__,
            "reference": REFERENCE,
            "reference_run": REFERENCE_RUN,
            "producer_artifacts": {REFERENCE_RUN: verified},
            "lpips_provenance": provenance,
            "lpips_seconds_83": seconds,
            "squeeze_cpu_map_max_abs_error": squeeze_error,
            "hybrid_replay_abs_error": hybrid_error,
            "selection": "none; all arms reported; rule fixed before running",
            "improving_arms": improving,
            "scope": SCOPE,
            "summary": summary,
            "arms": arms,
            "pixel_bootstrap": pixel[REFERENCE],
            "image_bootstrap": image_ci[REFERENCE],
        }
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
                allow_patterns=["results.json", "run_manifest.json", "source_snapshot.zip"],
                commit_message=f"Store score refinement {evaluation_id}",
            )
            manifest.update(artifact_sync_status="uploaded", hub_commit=commit.oid)
        except Exception as exc:  # noqa: BLE001 - upload failure is recorded; results stay on the Volume
            manifest.update(artifact_sync_status="failed", artifact_sync_error=type(exc).__name__)
        persist()
        return {"manifest": manifest, "results": report}
    except BaseException as exc:
        manifest.update(status="failed", error=type(exc).__name__)
        manifest["error_message"] = str(exc)[:300]
        manifest["total_seconds"] = time.perf_counter() - started
        persist()
        raise


@app.local_entrypoint()
def main(
    evaluation_id: str, code_revision: str, mode: str = "smoke", working_tree_dirty: bool = True
):
    result = run_refinement_remote.remote(evaluation_id, code_revision, working_tree_dirty, mode)
    destination = Path("runs") / "refinement" / evaluation_id
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in result.items():
        (destination / f"{name}.json").write_bytes(
            (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        )
    print(json.dumps(result["manifest"], indent=2))

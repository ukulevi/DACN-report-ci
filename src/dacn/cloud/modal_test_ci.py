"""T4 recomputation of the frozen four-arm test maps for patient-group CIs.

Uses the unchanged ``FrozenDDIMAdapter`` with the frozen sampler preset, mask
threshold and score thresholds. ``smoke`` runs two VALIDATION images and requires
exact agreement with cached producer maps (no test access). ``full`` processes all
3665 test slices in resumable chunks, requires the recomputed point metrics to
reproduce the published 10 Nov aggregate, and only then reports paired
patient-group bootstrap intervals. Nothing is fitted or selected on test.

    DACN_HF_SECRET=huggingface modal run --detach -m dacn.cloud.modal_test_ci \
        --evaluation-id test-ci-full-UNIQUE --code-revision GIT_SHA --mode full \
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
    MODEL_REPO,
    MODEL_REVISION,
    REVISION,
    RUN_ROOT,
    data_volume,
    run_volume,
)

app = modal.App("dacn-test-ci")
CHUNK = 250
# Published frozen four-arm test aggregate (docs/evidence/2026-11-10-fourarm-test.json).
PUBLISHED_TEST = "evaluations/final4-test-20261110-b/aggregate.json"
PUBLISHED_TEST_REVISION = "ddc715a7d17fbc00e176068c1d1ef91ae9e7835b"


@app.function(
    image=image,
    gpu="T4",
    cpu=8,
    memory=65536,
    timeout=6 * 60 * 60,
    volumes={str(DATA_ROOT): data_volume, str(RUN_ROOT): run_volume},
    secrets=[modal.Secret.from_name(HF_SECRET)],
)
def run_test_ci_remote(
    evaluation_id: str, code_revision: str, working_tree_dirty: bool, mode: str, resume: bool
):
    import csv
    import hashlib
    import io
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
    from dacn.data.shards import ShardDataset
    from dacn.evaluation.autoddpm_lpips import perceptual_provenance
    from dacn.evaluation.ddim_ablation import resolve_ablation_config
    from dacn.evaluation.ddim_inpainting import ddim_indices
    from dacn.evaluation.frozen_ddim import FrozenDDIMAdapter
    from dacn.evaluation.frozen_test_ci import (
        ARMS,
        FROZEN_THRESHOLDS,
        MASK_THRESHOLD,
        REFERENCE,
        TEST_SAMPLES,
        check_published,
        image_group_bootstrap,
        image_scores,
    )
    from dacn.evaluation.group_bootstrap import paired_group_bootstrap
    from dacn.evaluation.sweep_bootstrap import REFERENCE_RUN
    from dacn.training.ddpm_partial import build_diffusion, load_ema_checkpoint

    if mode not in {"smoke", "full"} or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", evaluation_id
    ):
        raise ValueError("Invalid test-CI request")
    output = RUN_ROOT / "evaluations" / evaluation_id
    output.mkdir(parents=True, exist_ok=resume)
    manifest_path = output / "run_manifest.json"
    if resume and not manifest_path.is_file():
        raise ValueError("Nothing to resume")
    manifest = json.loads(manifest_path.read_text()) if resume else {}
    if resume and (manifest.get("code_revision") != code_revision or manifest.get("mode") != mode):
        raise ValueError("Resume requires the same code revision and mode")
    manifest.update(
        evaluation_id=evaluation_id,
        method="frozen_four_arm_test_recompute_patient_group_ci",
        mode=mode,
        status="running",
        split="validation-smoke" if mode == "smoke" else "test",
        dataset_repo_id=DATASET,
        dataset_revision=REVISION,
        checkpoint_sha256=CHECKPOINT_SHA,
        frozen_mask_threshold=MASK_THRESHOLD,
        frozen_score_thresholds=FROZEN_THRESHOLDS,
        code_revision=code_revision,
        working_tree_dirty=working_tree_dirty,
        held_out_test_used=mode == "full",
        threshold_fitting=False,
        selection="none; frozen protocol recomputed for uncertainty only",
        gpu=torch.cuda.get_device_name(0),
    )
    manifest.setdefault("started_at", datetime.now(timezone.utc).isoformat())
    manifest["source_snapshot_sha256"] = _write_source_snapshot(output)
    manifest.setdefault("attempts", []).append(datetime.now(timezone.utc).isoformat())

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
        device = torch.device("cuda")
        # SHA-verified team checkpoint; same loader contract as the frozen adapter.
        train_config = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["config"]
        adapter = FrozenDDIMAdapter(device, include_controls=True)
        adapter.config = resolve_ablation_config({})
        adapter.mask_threshold = MASK_THRESHOLD
        adapter.indices = [ddim_indices(300, 20), ddim_indices(50, 30)]
        adapter.model = load_ema_checkpoint(checkpoint_path, train_config, device)
        adapter.diffusion = build_diffusion(train_config)
        perceptual = lpips.LPIPS(net="squeeze", spatial=True, version="0.1").to(device)
        perceptual.eval().requires_grad_(False)
        if perceptual_provenance(perceptual)["state_sha256"] != LPIPS_STATE_SHA:
            raise ValueError("LPIPS weights differ from the accepted state")
        adapter.perceptual = perceptual
        adapter.ready = True

        def infer(image_value, mask, sample_id):
            adapter({"image": image_value, "mask": mask, "sample_id": sample_id}, None)
            return {arm: np.asarray(adapter.last_score_maps[arm], np.float32) for arm in ARMS}

        if mode == "smoke":
            _, _, dataset = _pinned_validation()
            [(_, _, cached, _, verified)] = list(
                _iter_verified_runs(token, dataset, run_ids={REFERENCE_RUN})
            )
            labels = cached["labels"]
            compare = {
                "final_lpips": "autoddpm_lpips",
                "initial_lpips": "initial_lpips_only",
                "brightness": "brightness",
            }
            errors = {}
            for target in (0, 1):
                index = int(np.flatnonzero(labels == target)[0])
                maps = infer(
                    cached["images"][index],
                    cached["masks"][index].astype(np.uint8),
                    dataset["sample_ids"][index],
                )
                for arm, key in compare.items():
                    error = float(np.max(np.abs(maps[arm] - cached[key][index])))
                    errors[arm] = max(errors.get(arm, 0.0), error)
            if max(errors.values()) > 1e-5:
                raise ValueError(f"Frozen adapter does not replay validation maps: {errors}")
            report = {
                "mode": "smoke",
                "validation_replay_max_abs_error": errors,
                "producer_artifacts": {REFERENCE_RUN: verified},
                "test_accessed": False,
            }
        else:
            key = hashlib.sha256(f"{DATASET}@{REVISION}".encode()).hexdigest()[:20]
            test_dir = DATA_ROOT / key / "data" / "test"
            index_bytes = (test_dir / "index.csv").read_bytes()
            rows = list(csv.DictReader(io.StringIO(index_bytes.decode())))
            data = ShardDataset(test_dir, "test", include_masks=True)
            if len(data) != TEST_SAMPLES or [r["sample_id"] for r in rows] != data.sample_ids:
                raise ValueError("Pinned test split differs")
            manifest["test_index_sha256"] = hashlib.sha256(index_bytes).hexdigest()
            groups = np.array([r["patient_group_id"] for r in rows])
            chunk_dir = output / "chunks"
            chunk_dir.mkdir(exist_ok=True)
            seconds = []
            for first in range(0, TEST_SAMPLES, CHUNK):
                last = min(first + CHUNK, TEST_SAMPLES)
                path = chunk_dir / f"private_chunk_{first:05d}.npz"
                identity = hashlib.sha256(
                    "\n".join(data.sample_ids[first:last]).encode()
                ).hexdigest()
                if path.is_file():
                    with np.load(path, allow_pickle=False) as existing:
                        if str(existing["identity"]) == identity:
                            continue
                    raise ValueError("Existing chunk identity differs")
                chunk = {arm: [] for arm in ARMS}
                truths = []
                for index in range(first, last):
                    image_value, mask, _ = data[index]
                    mask = mask.numpy().astype(np.uint8)
                    tick = time.perf_counter()
                    maps = infer(image_value[0].numpy(), mask, data.sample_ids[index])
                    seconds.append(time.perf_counter() - tick)
                    for arm in ARMS:
                        chunk[arm].append(maps[arm])
                    truths.append(mask[8:-8, 8:-8])
                temporary = chunk_dir / f".chunk_{first:05d}.tmp.npz"
                np.savez_compressed(
                    temporary,
                    identity=np.array(identity),
                    truth=np.stack(truths),
                    **{arm: np.stack(values) for arm, values in chunk.items()},
                )
                temporary.replace(path)
                manifest["progress"] = {"completed_samples": last, "total": TEST_SAMPLES}
                persist()
                print(json.dumps(manifest["progress"]), flush=True)
            truth = np.empty((TEST_SAMPLES, 240, 240), dtype=np.uint8)
            maps = {arm: np.empty((TEST_SAMPLES, 240, 240), dtype=np.float32) for arm in ARMS}
            for first in range(0, TEST_SAMPLES, CHUNK):
                last = min(first + CHUNK, TEST_SAMPLES)
                with np.load(chunk_dir / f"private_chunk_{first:05d}.npz") as archive:
                    truth[first:last] = archive["truth"]
                    for arm in ARMS:
                        maps[arm][first:last] = archive[arm]
            labels = truth.reshape(TEST_SAMPLES, -1).any(axis=1).astype(int)
            image_values = {arm: image_scores(maps[arm]) for arm in ARMS}
            published = json.loads(
                Path(
                    hf_hub_download(
                        MODEL_REPO,
                        PUBLISHED_TEST,
                        revision=PUBLISHED_TEST_REVISION,
                        token=token,
                        cache_dir="/tmp/hf",
                    )
                ).read_text()
            )
            image_report = image_group_bootstrap(labels, image_values, groups, reference=REFERENCE)
            manifest["progress"] = {"stage": "pixel_bootstrap"}
            persist()
            pixel_report = paired_group_bootstrap(
                truth,
                maps,
                groups,
                {arm: FROZEN_THRESHOLDS[arm]["pixel"] for arm in ARMS},
                reference=REFERENCE,
                replicates=1000,
                seed=17,
            )
            replay = check_published(
                pixel_report["candidates"], image_report["candidates"], published
            )
            report = {
                "mode": "full",
                "test_accessed": True,
                "samples": TEST_SAMPLES,
                "normal_slices": int((labels == 0).sum()),
                "abnormal_slices": int(labels.sum()),
                "published_replay": replay,
                "pixel_bootstrap": pixel_report,
                "image_bootstrap": image_report,
                "adapter_seconds_median_this_attempt": (
                    float(np.median(seconds)) if seconds else None
                ),
                "scope": (
                    "Frozen four-arm protocol recomputed on the already inspected test split "
                    "for uncertainty only; thresholds and method unchanged; one training and "
                    "sampling seed; intervals conditional on this model and cohort."
                ),
            }
        report = _json_safe(report)
        _reject_patient_identifiers(report)
        (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        manifest.update(
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            total_seconds_last_attempt=time.perf_counter() - started,
            artifacts={
                k: v for k, v in _file_checksums(output).items() if not k.startswith("chunks/")
            },
        )
        persist()
        try:  # Aggregates only; private test chunks stay on this Workspace's Volume.
            commit = api.upload_folder(
                folder_path=str(output),
                path_in_repo=f"evaluations/{evaluation_id}",
                repo_id=MODEL_REPO,
                allow_patterns=["results.json", "run_manifest.json", "source_snapshot.zip"],
                commit_message=f"Store test CI {evaluation_id}",
            )
            manifest.update(artifact_sync_status="uploaded", hub_commit=commit.oid)
        except Exception as exc:  # noqa: BLE001 - upload failure is recorded; results stay on the Volume
            manifest.update(artifact_sync_status="failed", artifact_sync_error=type(exc).__name__)
        persist()
        return {"manifest": manifest, "results": report}
    except BaseException as exc:
        manifest.update(status="failed", error=type(exc).__name__)
        manifest["error_message"] = str(exc)[:300]
        persist()
        raise


@app.local_entrypoint()
def main(
    evaluation_id: str,
    code_revision: str,
    mode: str = "smoke",
    working_tree_dirty: bool = True,
    resume: bool = False,
):
    result = run_test_ci_remote.remote(
        evaluation_id, code_revision, working_tree_dirty, mode, resume
    )
    destination = Path("runs") / "test-ci" / evaluation_id
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in result.items():
        (destination / f"{name}.json").write_bytes(
            (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        )
    print(json.dumps(result["manifest"], indent=2))

"""Bounded trained-checkpoint CPU acceptance on two registered validation samples."""

import argparse
import hashlib
import io
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import try_to_load_from_cache

from dacn.data.shards import ShardDataset
from dacn.evaluation.dependency_snapshot import version_snapshot
from dacn.evaluation.frozen_ddim import FrozenDDIMAdapter
from dacn.evaluation.frozen_runner import run_frozen_evaluation


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    manifest = json.loads(Path("configs/final_evaluation_candidate.json").read_text())
    paths = {
        key: args.environment / filename
        for key, filename in {
            "checkpoint": "checkpoint.pt",
            "config": "config.json",
            "lpips_backbone": "lpips_backbone.pth",
            "lpips_calibration": "lpips_calibration.pth",
            "dependency_lock": "requirements.freeze.txt",
        }.items()
    }
    paths["validation_index"] = args.validation / "index.csv"
    paths["config"] = Path("configs/autoddpm_ddim20_30.json")
    prefix = "evaluations/autoddpm-ddim2030-full83-t4-20261106T074200Z/"
    paths["mask_parameters"] = (
        Path("runs/audit-2026-11-06/hf-latest") / prefix / "comparison/mask_parameters.json"
    )
    # Capture the source actually under acceptance, rather than claim the older snapshot runs it.
    paths["source_snapshot"] = args.output / "acceptance_source.zip"
    with zipfile.ZipFile(paths["source_snapshot"], "w") as archive:
        for path in sorted(Path("src").rglob("*.py")):
            archive.writestr(path.as_posix(), path.read_bytes())
        archive.writestr("scripts/accept_ddim_validation.py", Path(__file__).read_bytes())
    manifest["artifacts"]["source_snapshot"] = sha(paths["source_snapshot"].read_bytes())
    manifest["scope"] = "Bounded validation acceptance only; candidate remains unfrozen"
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--exclude-editable"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if version_snapshot(freeze) != version_snapshot(
        paths["dependency_lock"].read_text(encoding="utf-8")
    ):
        raise ValueError("Running dependency versions differ from exported environment")
    (args.output / "acceptance_manifest.json").write_text(json.dumps(manifest, indent=2))

    def samples():
        if args.validation.name != "val":
            raise ValueError("Only validation split allowed")
        cache_path = try_to_load_from_cache(
            "dacn-bmad-team/dacn-runs",
            prefix + "comparison/private_ablation.npz",
            revision=manifest["reference_artifact_revision"],
        )
        payload = Path(cache_path).read_bytes()
        if sha(payload) != "22656e1f63a0047d3dbe32f7833eea8e303988414e5942dec1ba5a1d9ed1ef19":
            raise ValueError("Reference cache differs")
        data = ShardDataset(args.validation, "val", include_masks=True)
        if len(data) != 83:
            raise ValueError("Validation83 required")
        with np.load(io.BytesIO(payload), allow_pickle=False) as cache:
            for index in range(83):
                image, mask, label = data[index]
                if not (
                    np.array_equal(image[0].numpy(), cache["images"][index])
                    and np.array_equal(mask.numpy(), cache["masks"][index])
                    and int(label) == int(cache["labels"][index])
                ):
                    raise ValueError("Validation content differs from registered cache")
        for target in (0, 1):
            index = int(np.flatnonzero(data.labels == target)[0])
            image, mask, _ = data[index]
            yield (
                data.sample_ids[index],
                {
                    "sample_id": data.sample_ids[index],
                    "image": image[0].numpy(),
                    "mask": mask.numpy(),
                },
            )

    kwargs = {
        "manifest": manifest,
        "artifact_paths": paths,
        "samples": samples,
        "output_dir": args.output / "private_shards",
        "validation_only": True,
    }
    adapter = FrozenDDIMAdapter("cpu")
    calls = 0

    def interrupted(sample, captured):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InterruptedError("Intentional acceptance interruption after one committed sample")
        return adapter(sample, captured)

    started = time.perf_counter()
    try:
        run_frozen_evaluation(**kwargs, infer=interrupted)
    except InterruptedError:
        pass
    first = sha((args.output / "private_shards/shard-000000.npz").read_bytes())
    print("First trained sample committed; resuming with fresh model loader", flush=True)
    result = run_frozen_evaluation(**kwargs, infer=FrozenDDIMAdapter("cpu"))
    assert first == sha((args.output / "private_shards/shard-000000.npz").read_bytes())

    def forbidden(*_):
        raise AssertionError("Completed resume must reuse verified shards")

    replay = run_frozen_evaluation(**kwargs, infer=forbidden)
    assert replay["metrics"] == result["metrics"]
    clean_kwargs = {**kwargs, "output_dir": args.output / "clean_private_shards"}
    clean = run_frozen_evaluation(**clean_kwargs, infer=FrozenDDIMAdapter("cpu"))
    for index in range(2):
        filename = f"shard-{index:06d}.npz"
        with (
            np.load(args.output / "private_shards" / filename, allow_pickle=False) as resumed,
            np.load(args.output / "clean_private_shards" / filename, allow_pickle=False) as full,
        ):
            for key in resumed.files:
                if not np.array_equal(resumed[key], full[key]):
                    raise ValueError("Resumed and uninterrupted score arrays differ")
    assert clean["metrics"] == result["metrics"]
    report = {
        "status": "completed",
        "samples": 2,
        "trained_unet_calls": 200,
        "interruption_resume": "passed",
        "completed_cache_resume": "passed",
        "first_shard_unchanged": True,
        "resume_matches_uninterrupted": True,
        "seconds_including_load_io": time.perf_counter() - started,
        "metrics": result["metrics"],
        "scope": manifest["scope"],
        "source_sha256": manifest["artifacts"]["source_snapshot"],
        "limitation": "Two samples only; CPU RNG differs from CUDA; no efficacy estimate or test",
    }
    (args.output / "aggregate_receipt.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

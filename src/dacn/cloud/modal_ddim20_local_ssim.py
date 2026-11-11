"""CPU-only evaluation of existing DDIM20 validation cache on the team Volume."""

import hashlib
import json
import os
import re
import sys
import time
import zipfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import modal

app = modal.App("dacn-ddim20-validation-ssim")
volume = modal.Volume.from_name("dacn-bmad-runs")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.4.6", "scipy==1.17.1", "scikit-learn==1.9.1", "huggingface_hub>=0.28,<3")
    .add_local_python_source("dacn")
)


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=1800,
    volumes={"/mnt/runs": volume},
    secrets=[modal.Secret.from_name("huggingface-secret-2")],
)
def evaluate(
    evaluation_id: str,
    code_revision: str,
    working_tree_dirty: bool = True,
    score_family: str = "local_ssim",
):
    from huggingface_hub import HfApi

    import dacn
    from dacn.evaluation.ddim20_local_ssim import evaluate_cached_validation, write_json

    if score_family not in {"local_ssim", "fusion"}:
        raise ValueError("Unsupported validation score family")
    runner = evaluate_cached_validation
    if score_family == "fusion":
        from dacn.evaluation.ddim20_score_fusion import evaluate_cached_fusions

        runner = evaluate_cached_fusions
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", evaluation_id):
        raise ValueError("Invalid evaluation ID")
    protocol = Path("/mnt/runs/final-evaluation/final-fourarm-20261110-b")
    output = Path("/mnt/runs/evaluations") / evaluation_id
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source_snapshot.zip"
    package = Path(dacn.__file__).resolve().parent
    with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package.rglob("*.py")):
            archive.writestr("dacn/" + path.relative_to(package).as_posix(), path.read_bytes())
    record = {
        "evaluation_id": evaluation_id,
        "status": "started",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": code_revision,
        "working_tree_dirty": working_tree_dirty,
        "source_snapshot_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scope": "Cached validation only; CPU scoring; no model inference, no test access",
        "score_family": score_family,
        "gpu": None,
        "cpu": 2,
        "memory_mib": 8192,
        "python": sys.version.split()[0],
        "versions": {
            key: version(key) for key in ("numpy", "scipy", "scikit-learn", "huggingface_hub")
        },
    }
    manifest_path = output / "run_manifest.json"

    def persist():
        write_json(manifest_path, record)
        volume.commit()

    persist()
    tick = time.perf_counter()
    try:
        manifest = json.loads((protocol / "manifest.json").read_bytes())
        result = runner(
            cache=protocol / "private_reference.npz",
            manifest=manifest,
            index=protocol / "artifacts/validation_index",
            partition=protocol / "artifacts/validation_partition",
            output=output,
            on_progress=lambda value: print(json.dumps(value), flush=True),
        )
        record.update(
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            total_seconds=time.perf_counter() - tick,
        )
        persist()
        api = HfApi(token=os.environ["HF_TOKEN"])
        repo = "dacn-bmad-team/dacn-runs"
        if not api.repo_info(repo).private:
            raise ValueError("Private artifact repository required")
        commit = api.upload_folder(
            repo_id=repo, folder_path=str(output), path_in_repo=f"evaluations/{evaluation_id}"
        )
        record.update(artifact_sync_status="uploaded", model_revision=commit.oid)
        persist()
        return {"manifest": record, "results": result}
    except BaseException as exc:
        record.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            exception_type=type(exc).__name__,
        )
        persist()
        raise


@app.local_entrypoint()
def main(evaluation_id: str, code_revision: str, output: str, score_family: str = "local_ssim"):
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    call = evaluate.spawn(evaluation_id, code_revision, True, score_family)
    submission = {"evaluation_id": evaluation_id, "call_id": call.object_id, "app_id": app.app_id}
    (folder / "submission.json").write_text(json.dumps(submission, indent=2) + "\n")
    print(json.dumps(submission), flush=True)
    result = call.get()
    for key, filename in (("manifest", "run_manifest.json"), ("results", "results.json")):
        (folder / filename).write_text(json.dumps(result[key], indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": result["manifest"]["status"],
                "revision": result["manifest"]["model_revision"],
                "holdout": result["results"].get(
                    "group_holdout", result["results"].get("holdout_results")
                ),
            }
        ),
        flush=True,
    )

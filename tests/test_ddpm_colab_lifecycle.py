"""Execute notebook orchestration with a CPU stand-in for the GPU boundary."""

import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dacn.evaluation import ddpm

NOTEBOOK = Path(__file__).parents[1] / "notebooks/ddpm_validation_colab.ipynb"


def run_cell(namespace, flag, enabled):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] != "code" or f"{flag} = False" not in source:
            continue
        tree = ast.parse(source)
        for statement in tree.body:
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == flag for target in statement.targets
            ):
                statement.value = ast.Constant(enabled)
        # Execute only this repository's notebook, with GPU/network boundaries replaced.
        exec(compile(ast.fix_missing_locations(tree), str(NOTEBOOK), "exec"), namespace)  # noqa: S102
        return
    pytest.fail(f"Missing independent execution gate: {flag}")


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    payload = b"synthetic checkpoint; no real model loaded"
    state = {"epoch": 20, "optimizer_steps": 18760, "config": {"seed": 17}}
    cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "CPU test double")
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=cuda, load=lambda *a, **k: state)
    )
    monkeypatch.setattr("importlib.metadata.version", lambda _: "test-double")
    calls = []

    def adapter(image, **kwargs):
        calls.append(kwargs["starting_timestep"])
        return image.copy()

    monkeypatch.setattr(ddpm, "_ddpm_backend", lambda *a: adapter)
    monkeypatch.setattr(
        ddpm,
        "produce_validation_artifact",
        lambda *a: pytest.fail("Smoke must not launch full-cohort inference"),
    )
    namespace = {
        "Path": lambda value: tmp_path / str(value).lstrip("/"),
        "json": json,
        "hashlib": hashlib,
        "checkpoint_bytes": payload,
        "CKPT_SHA": hashlib.sha256(payload).hexdigest(),
        "DATA_REPO": "synthetic/data",
        "DATA_REV": "a" * 40,
        "MODEL_REPO": "synthetic/model",
        "MODEL_REV": "b" * 40,
        "META_SHA": "c" * 64,
        "SOURCE_REVISION": "d" * 40,
        "DATA": tmp_path,
        "_validation_data": lambda *a: (
            {str(i): i for i in range(83)},
            np.zeros((83, 256, 256), dtype=np.float32),
            None,
            None,
        ),
    }
    return namespace, calls, state


def test_default_does_not_create_a_run(runtime, tmp_path):
    namespace, calls, _ = runtime
    with pytest.raises(RuntimeError):
        run_cell(namespace, "RUN_INFERENCE", False)
    assert calls == []
    assert not list(tmp_path.rglob("inference_manifest.json"))


def test_smoke_stops_after_four_short_reconstructions(runtime):
    namespace, calls, _ = runtime
    run_cell(namespace, "RUN_INFERENCE", True)
    manifest = json.loads((namespace["OUT"] / "inference_manifest.json").read_text())
    assert calls == [3, 3, 3, 3]
    assert manifest["status"] == "smoke_completed"
    assert not namespace["artifact"].exists()


def test_manual_interrupt_is_recorded(runtime, monkeypatch):
    namespace, _, _ = runtime

    def interrupted(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(ddpm, "_ddpm_backend", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_cell(namespace, "RUN_INFERENCE", True)
    manifest = json.loads((namespace["OUT"] / "inference_manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["error_class"] == "KeyboardInterrupt"


def test_bad_checkpoint_is_recorded_before_model_loading(runtime, tmp_path):
    namespace, calls, state = runtime
    state["epoch"] = 19
    with pytest.raises((ValueError, AssertionError)):
        run_cell(namespace, "RUN_INFERENCE", True)
    manifests = list(tmp_path.rglob("inference_manifest.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text())["status"] == "failed"
    assert calls == []


def test_full_run_requires_successful_smoke(runtime):
    namespace, calls, _ = runtime
    with pytest.raises(RuntimeError, match="smoke"):
        run_cell(namespace, "RUN_FULL_VALIDATION", True)
    assert calls == []


@pytest.mark.parametrize("interrupted", [False, True])
def test_full_run_records_completion_or_interruption(runtime, interrupted):
    namespace, _, _ = runtime
    run_cell(namespace, "RUN_INFERENCE", True)
    with pytest.raises(RuntimeError):
        run_cell(namespace, "RUN_FULL_VALIDATION", False)
    assert namespace["manifest"]["status"] == "smoke_completed"

    def produce(dataset, output, provenance, reconstruct):
        if interrupted:
            raise KeyboardInterrupt()
        output.write_bytes(b"synthetic full-cohort artifact")
        return {
            "samples": 83,
            "reconstruction_artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }

    namespace["produce_validation_artifact"] = produce
    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            run_cell(namespace, "RUN_FULL_VALIDATION", True)
    else:
        run_cell(namespace, "RUN_FULL_VALIDATION", True)
    manifest = json.loads((namespace["OUT"] / "inference_manifest.json").read_text())
    assert manifest["stage"] == "full_validation"
    assert manifest["status"] == ("interrupted" if interrupted else "completed")
    if not interrupted:
        assert manifest["receipt"]["samples"] == 83

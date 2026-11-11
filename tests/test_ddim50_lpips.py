import io
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from dacn.evaluation import ddim50_lpips as dd


def manifest():
    return {
        "schema_version": 1,
        "status": "prepared",
        "config": dd.CONFIG,
        "dataset_revision": "350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7",
        "artifacts": {key: "a" * 64 for key in dd.ARTIFACTS},
    }


def test_one_phase_fifty_calls_and_one_lpips_on_same_image(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.steps = []

        def forward(self, image, step):
            self.steps.append(int(step.item()))
            return torch.zeros_like(image)

    class Perceptual(torch.nn.Module):
        def forward(self, x, y, normalize=False):
            assert normalize is False and x.shape == y.shape == (1, 3, 240, 240)
            return (x[:, :1] - y[:, :1]).abs()

    model = Model()
    monkeypatch.setattr(dd, "load_ema_checkpoint", lambda *_: model)
    monkeypatch.setattr(dd, "load_perceptual", lambda *_: Perceptual())
    buffer = io.BytesIO()
    torch.save({"config": {}}, buffer)
    backend = dd.DDIM50Adapter("cpu")
    sample = {
        "image": np.full((256, 256), 0.2, dtype=np.float32),
        "sample_id": "00000000-0000-0000-0000-000000000001",
    }
    first = backend(sample, {"checkpoint": buffer.getvalue()})
    assert model.steps == list(dd.ddim_indices(300, 50))
    assert first["scores"].shape == (240, 240)
    second = backend(sample, {"checkpoint": buffer.getvalue()})
    np.testing.assert_array_equal(first["scores"], second["scores"])
    np.testing.assert_array_equal(first["reconstruction"], second["reconstruction"])


def test_candidate_test_rejected_before_dataset(tmp_path):
    with pytest.raises(ValueError, match="frozen"):
        dd.run_samples(
            manifest=manifest(),
            captured={},
            stage="test",
            samples=lambda: pytest.fail("Test opened before gate"),
            infer=None,
            output=tmp_path,
            commit=lambda: None,
        )


def test_test_requires_receipt_bound_to_thresholds():
    m = manifest()
    m.update(status="frozen", thresholds={"pixel": 0.3, "image": 0.4})
    receipt = dd.json_bytes(
        {
            "status": "completed",
            "stage": "validation",
            "sample_count": 83,
            "holdout_samples": 43,
            "method_contract": dd.method_contract(m),
            "thresholds": m["thresholds"],
        }
    )
    m["validation_acceptance_sha256"] = dd.digest(receipt)
    dd.check_stage(m, "test", receipt)
    m["thresholds"]["pixel"] = 0.5
    with pytest.raises(ValueError, match="acceptance differs"):
        dd.check_stage(m, "test", receipt)


def test_tampered_artifact_rejected(tmp_path):
    m = manifest()
    paths = {}
    for key in dd.ARTIFACTS:
        p = tmp_path / key
        p.write_bytes(dd.json_bytes(dd.CONFIG) if key == "config" else key.encode())
        m["artifacts"][key] = dd.digest(p.read_bytes())
        paths[key] = p
    dd.capture_inputs(m, paths)
    paths["checkpoint"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum differs: checkpoint"):
        dd.capture_inputs(m, paths)


def test_smoke_resume_reuses_outputs_and_logs_verification(tmp_path, capsys):
    samples = [
        (
            str(i),
            {
                "sample_id": str(i),
                "image": np.full((256, 256), 0.1 * i, dtype=np.float32),
                "mask": np.full((256, 256), i, dtype=np.uint8),
            },
        )
        for i in range(2)
    ]
    calls = []

    def infer(sample, _):
        calls.append(sample["sample_id"])
        if calls == ["0", "1"]:
            raise RuntimeError("deliberate interruption")
        return {
            "scores": np.full((240, 240), 0.2, dtype=np.float32),
            "reconstruction": sample["image"].copy(),
        }

    kwargs = {
        "manifest": manifest(),
        "captured": {},
        "stage": "smoke",
        "samples": lambda: samples,
        "infer": infer,
        "output": tmp_path,
        "commit": lambda: None,
    }
    with pytest.raises(RuntimeError):
        dd.run_samples(**kwargs)
    result = dd.run_samples(**kwargs)
    assert len(result["shards"]) == 2 and calls == ["0", "1", "1"]
    dd.run_samples(**kwargs)
    assert calls == ["0", "1", "1"]
    assert '"verified_cached_samples"' in capsys.readouterr().out
    samples[0][1]["image"][0, 0] = 0.7
    with pytest.raises(ValueError, match="checksum differs"):
        dd.run_samples(**kwargs)


def test_validation_writes_calibration_thresholds_before_holdout(tmp_path, monkeypatch):
    for i in range(83):
        label = i % 2
        np.savez(
            tmp_path / f"shard-{i:06d}.npz",
            scores=np.full((240, 240), 0.2 + 0.6 * label),
            mask=np.full((240, 240), label, dtype=np.uint8),
        )
    original = dd.evaluate_maps

    def checked(*args, **kwargs):
        if kwargs.get("thresholds") is not None:
            assert (
                json.loads((tmp_path / "frozen_thresholds.json").read_bytes())["thresholds"]
                == kwargs["thresholds"]
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(dd, "evaluate_maps", checked)
    report = dd.validation_metrics(tmp_path, np.arange(40), np.arange(40, 83))
    assert report["group_holdout"]["pixel"]["f1"] == 1

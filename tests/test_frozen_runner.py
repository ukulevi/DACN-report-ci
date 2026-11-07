import numpy as np
import pytest
from test_frozen_inputs import fixture

from dacn.evaluation.frozen_runner import run_frozen_evaluation


def test_candidate_cannot_open_dataset(tmp_path):
    manifest, paths = fixture(tmp_path)
    manifest["status"] = "candidate"

    def forbidden():
        pytest.fail("Dataset opened before freeze verification")

    with pytest.raises(ValueError, match="frozen"):
        run_frozen_evaluation(
            manifest=manifest,
            artifact_paths=paths,
            samples=forbidden,
            infer=forbidden,
            output_dir=tmp_path / "result",
        )


def test_interrupted_resume_uses_global_metrics_and_checks_identity(tmp_path):
    manifest, paths = fixture(tmp_path)
    output = tmp_path / "result"
    calls = []

    def infer(sample, captured):
        assert captured["checkpoint"] == b"checkpoint"
        calls.append(sample)
        if sample == 1 and len(calls) == 2:
            raise RuntimeError("interruption")
        return {
            "pixel_labels": np.array([sample]),
            "pixel_scores": np.array([sample]),
            "image_labels": np.array([sample]),
            "image_scores": np.array([sample]),
        }

    kwargs = {
        "manifest": manifest,
        "artifact_paths": paths,
        "samples": lambda: [("a", 0), ("b", 1)],
        "infer": infer,
        "output_dir": output,
    }
    with pytest.raises(RuntimeError):
        run_frozen_evaluation(**kwargs)
    result = run_frozen_evaluation(**kwargs)
    assert calls == [0, 1, 1]
    assert result["status"] == "completed"
    assert result["metrics"]["pixel"]["auroc"] == 1
    assert result["metrics"]["image"]["f1"] == 1
    assert result["metrics"]["pixel"]["dice_micro"] == 1
    assert result["metrics"]["pixel"]["macro_lesion_dice"] == 1
    assert result["metrics"]["pixel"]["normal_pixel_fpr"] == 0
    kwargs["samples"] = lambda: [("b", 1), ("a", 0)]
    with pytest.raises(ValueError, match="checksum"):
        run_frozen_evaluation(**kwargs)


def test_tampered_shard_cannot_resume(tmp_path):
    manifest, paths = fixture(tmp_path)
    output = tmp_path / "result"
    kwargs = {
        "manifest": manifest,
        "artifact_paths": paths,
        "samples": lambda: [("a", 0), ("b", 1)],
        "infer": lambda s, _: {
            "pixel_labels": [s],
            "pixel_scores": [s],
            "image_labels": [s],
            "image_scores": [s],
        },
        "output_dir": output,
    }
    run_frozen_evaluation(**kwargs)
    (output / "shard-000000.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        run_frozen_evaluation(**kwargs)

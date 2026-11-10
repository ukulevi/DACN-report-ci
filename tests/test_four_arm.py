import json

import numpy as np
import pytest
from test_frozen_inputs import fixture

from dacn.evaluation.four_arm import ARMS, digest, json_bytes, restore_pinned_text, run_four_arm
from dacn.evaluation.validation import evaluate_maps


class Backend:
    def __init__(self, fail_once=False):
        self.calls = 0
        self.fail_once = fail_once

    def __call__(self, sample, captured):
        self.calls += 1
        if self.fail_once and self.calls == 2:
            self.fail_once = False
            raise RuntimeError("interrupt after durable sample")
        assert captured["checkpoint"] == b"checkpoint"
        # Include ties and errors so global metric/threshold tests are meaningful.
        score = (sample["image"][8:-8, 8:-8] + 1) / 2
        self.last_score_maps = {arm: score.copy() for arm in ARMS}


def test_restore_text_preserves_historical_hash_and_rejects_other_changes():
    lf = b'{\n  "seed": 17\n}\n'
    crlf = lf.replace(b"\n", b"\r\n")
    assert restore_pinned_text(lf, digest(crlf)) == crlf
    assert restore_pinned_text(crlf, digest(lf)) == lf
    with pytest.raises(ValueError, match="historical checksum"):
        restore_pinned_text(lf.replace(b"17", b"18"), digest(crlf))


def setup(tmp_path):
    manifest, paths = fixture(tmp_path)
    controls = {
        "status": "candidate",
        "dataset_revision": manifest["dataset_revision"],
        "arms": {arm: {"pixel": 0.5, "image": 0.5} for arm in ARMS},
    }
    manifest.update(
        status="candidate",
        thresholds=controls["arms"]["final_lpips"],
        controls_sha256=digest(json_bytes(controls)),
        dataset_metadata_sha256="a" * 64,
    )
    samples = []
    for i in range(3):
        image = np.full((256, 256), -0.5 + 0.5 * i, dtype=np.float32)
        image[30:40, 30:40] = 0.5
        mask = np.zeros((256, 256), dtype=np.uint8)
        if i:
            mask[20:60, 20:60] = 1
        samples.append((str(i), {"sample_id": str(i), "image": image, "mask": mask}))
    kwargs = {
        "manifest": manifest,
        "controls": controls,
        "artifact_paths": paths,
        "samples": lambda: samples,
        "infer": Backend(),
        "output_dir": tmp_path / "output",
        "expected_samples": 3,
        "split": "validation",
        "runtime_check": lambda _: None,
    }
    return kwargs, samples


def test_shared_resume_matches_global_metrics_and_no_repeated_inference(tmp_path):
    kwargs, samples = setup(tmp_path)
    backend = Backend(fail_once=True)
    kwargs["infer"] = backend
    with pytest.raises(RuntimeError):
        run_four_arm(**kwargs)
    report = run_four_arm(**kwargs)
    assert backend.calls == 4  # sample0 once, sample1 failed+retry, sample2 once
    assert report["sample_count"] == 3
    maps = np.stack([(s["image"][8:-8, 8:-8] + 1) / 2 for _, s in samples])
    masks = np.stack([s["mask"][8:-8, 8:-8] for _, s in samples])
    expected = evaluate_maps(
        maps,
        masks,
        masks.any(axis=(1, 2)).astype(int),
        split="validation",
        thresholds={"pixel": 0.5, "image": 0.5},
    )
    for arm in ARMS:
        for level in ("pixel", "image"):
            for field in ("auroc", "auprc", "precision", "recall", "f1"):
                assert report["arms"][arm][level][field] == pytest.approx(expected[level][field])
    run_four_arm(**kwargs)
    assert backend.calls == 4
    curves = json.loads((kwargs["output_dir"] / "curves.json").read_bytes())
    assert set(curves) == set(ARMS)


def test_controls_rejected_before_dataset_or_inference(tmp_path):
    kwargs, _ = setup(tmp_path)
    kwargs["controls"]["arms"]["brightness"]["pixel"] = 0.9
    kwargs["samples"] = lambda: pytest.fail("opened dataset before controls gate")
    with pytest.raises(ValueError, match="checksum"):
        run_four_arm(**kwargs)


def test_candidate_cannot_open_test(tmp_path):
    kwargs, _ = setup(tmp_path)
    kwargs.update(
        split="test",
        expected_samples=3665,
        samples=lambda: pytest.fail("test opened before frozen gate"),
    )
    with pytest.raises(ValueError, match="frozen"):
        run_four_arm(**kwargs)


def test_runtime_failure_happens_before_dataset(tmp_path):
    kwargs, _ = setup(tmp_path)

    def wrong_runtime(_):
        raise ValueError("runtime mismatch")

    kwargs.update(
        runtime_check=wrong_runtime, samples=lambda: pytest.fail("opened before runtime check")
    )
    with pytest.raises(ValueError, match="runtime mismatch"):
        run_four_arm(**kwargs)


@pytest.mark.parametrize("tamper", ["input", "shard", "order", "shorten"])
def test_resume_rejects_changed_data_and_shards(tmp_path, tamper):
    kwargs, samples = setup(tmp_path)
    run_four_arm(**kwargs)
    if tamper == "input":
        samples[0][1]["image"][0, 0] = 0
    elif tamper == "shard":
        (kwargs["output_dir"] / "shard-000000.npz").write_bytes(b"bad")
    elif tamper == "order":
        samples.reverse()
    else:
        samples.pop()
    with pytest.raises(ValueError, match="checksum|shortened"):
        run_four_arm(**kwargs)


def test_final_requires_matching_real_validation_even_if_status_frozen(tmp_path):
    kwargs, _ = setup(tmp_path)
    kwargs["manifest"]["status"] = "frozen"
    kwargs["controls"]["status"] = "frozen"
    kwargs["manifest"]["controls_sha256"] = digest(json_bytes(kwargs["controls"]))
    kwargs.update(
        split="test",
        expected_samples=3665,
        samples=lambda: pytest.fail("test opened without validation receipt"),
    )
    with pytest.raises(ValueError, match="acceptance"):
        run_four_arm(**kwargs)

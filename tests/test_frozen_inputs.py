import hashlib

import pytest

from dacn.evaluation.frozen_inputs import ARTIFACTS, capture_frozen_inputs


def fixture(tmp_path):
    paths = {}
    hashes = {}
    for name in ARTIFACTS:
        path = tmp_path / name
        payload = name.encode()
        path.write_bytes(payload)
        paths[name] = path
        hashes[name] = hashlib.sha256(payload).hexdigest()
    return {
        "schema_version": 1,
        "status": "frozen",
        "dataset_revision": "a" * 40,
        "artifacts": hashes,
        "thresholds": {"pixel": 0.2, "image": 0.3},
    }, paths


@pytest.mark.parametrize(
    "change", ["candidate", "missing_pin", "mutable_revision", "nan_threshold"]
)
def test_incomplete_freeze_is_rejected(tmp_path, change):
    manifest, paths = fixture(tmp_path)
    if change == "candidate":
        manifest["status"] = "candidate"
    elif change == "missing_pin":
        manifest["artifacts"]["lpips_backbone"] = None
    elif change == "mutable_revision":
        manifest["dataset_revision"] = "main"
    else:
        manifest["thresholds"]["pixel"] = float("nan")
    with pytest.raises(ValueError):
        capture_frozen_inputs(manifest, paths)


def test_changed_artifact_is_rejected(tmp_path):
    manifest, paths = fixture(tmp_path)
    paths["checkpoint"].write_bytes(b"replacement")
    with pytest.raises(ValueError, match="checksum"):
        capture_frozen_inputs(manifest, paths)


def test_returns_verified_snapshot_not_mutable_paths(tmp_path):
    manifest, paths = fixture(tmp_path)
    captured = capture_frozen_inputs(manifest, paths)
    paths["checkpoint"].write_bytes(b"later replacement")
    assert captured["checkpoint"] == b"checkpoint"


def test_candidate_allowed_only_for_explicit_validation_acceptance(tmp_path):
    manifest, paths = fixture(tmp_path)
    manifest["status"] = "candidate"
    assert capture_frozen_inputs(manifest, paths, validation_only=True)["config"] == b"config"
    with pytest.raises(ValueError, match="frozen"):
        capture_frozen_inputs(manifest, paths)

import numpy as np
import pytest

from dacn.evaluation import handoff


def test_dataset_mismatch_rejected():
    manifest = {"dataset_repo_id": "team/data", "dataset_revision": "a" * 40}
    handoff.validate_dataset_binding(manifest, "team/data", "a" * 40)
    for repo, revision in [("other/data", "a" * 40), ("team/data", "b" * 40)]:
        with pytest.raises(ValueError):
            handoff.validate_dataset_binding(manifest, repo, revision)
    with pytest.raises(ValueError):
        handoff.validate_dataset_binding({}, "team/data", "a" * 40)


def test_private_pairs_roundtrip_and_collision(tmp_path):
    path = tmp_path / "pairs.npz"
    ids = ["12345678-1234-5678-1234-567812345678"]
    pairs = np.zeros((1, 256, 256), dtype=np.float32)
    handoff.write_pairs(path, ids, pairs)
    with np.load(path, allow_pickle=False) as data:
        assert data["sample_ids"].tolist() == ids
        assert data["reconstructions"].shape == (1, 1, 256, 256)
    with pytest.raises(FileExistsError):
        handoff.write_pairs(path, ids, pairs)


@pytest.mark.parametrize("kind", ["duplicate", "nan", "range", "count"])
def test_bad_pairs_rejected(tmp_path, kind):
    ids = ["12345678-1234-5678-1234-567812345678"]
    pairs = np.zeros((1, 256, 256), dtype=np.float32)
    if kind == "duplicate":
        ids *= 2
        pairs = np.repeat(pairs, 2, axis=0)
    elif kind == "nan":
        pairs[0, 0, 0] = np.nan
    elif kind == "range":
        pairs[0, 0, 0] = 2
    else:
        ids = []
    with pytest.raises(ValueError):
        handoff.write_pairs(tmp_path / "bad.npz", ids, pairs)
    assert not (tmp_path / "bad.npz").exists()

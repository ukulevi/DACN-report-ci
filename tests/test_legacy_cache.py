import hashlib

import numpy as np
import pytest
from test_validation_artifacts import case  # noqa: F401 -- shared synthetic dataset fixture

from dacn.evaluation.artifacts import _validation_data, load_validation_pairs
from dacn.evaluation.handoff import convert_legacy_cache


@pytest.mark.parametrize("corruption", [None, "checksum", "order", "nan", "range", "dtype"])
def test_legacy_conversion_binds_only_verified_arrays(case, tmp_path, corruption):  # noqa: F811 -- pytest fixture
    dataset, _, provenance, _, _ = case
    _, images, masks, labels = _validation_data(dataset, provenance["dataset_metadata_sha256"])
    recons = images.copy()
    if corruption == "order":
        images, masks, labels = images[::-1], masks[::-1], labels[::-1]
    elif corruption == "nan":
        recons[0, 0, 0] = np.nan
    elif corruption == "range":
        recons[0, 0, 0] = 2
    elif corruption == "dtype":
        recons = recons.astype(np.float64)
    cache = tmp_path / "legacy.npz"
    np.savez_compressed(cache, images=images, masks=masks, labels=labels, reconstructions=recons)
    expected = (
        "0" * 64 if corruption == "checksum" else hashlib.sha256(cache.read_bytes()).hexdigest()
    )
    output = tmp_path / "converted.npz"
    kwargs = {
        "expected_cache_sha256": expected,
        "expected_metadata_sha256": provenance["dataset_metadata_sha256"],
    }
    if corruption:
        with pytest.raises(ValueError):
            convert_legacy_cache(cache, dataset, output, **kwargs)
        assert not output.exists()
    else:
        receipt = convert_legacy_cache(cache, dataset, output, **kwargs)
        assert receipt["samples"] == 2
        assert receipt["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
        load_validation_pairs(output, dataset, provenance)
        with pytest.raises(FileExistsError):
            convert_legacy_cache(cache, dataset, output, **kwargs)

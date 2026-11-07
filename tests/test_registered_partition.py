import hashlib
import json

import numpy as np
import pytest

from dacn.evaluation.registered_partition import registered_partition


def fixture():
    return {
        "schema_version": 1,
        "index_sha256": hashlib.sha256(b"index").hexdigest(),
        "calibration_positions": list(range(40)),
        "holdout_positions": list(range(40, 83)),
    }


def test_exact_membership_does_not_call_splitter():
    cal, hold = registered_partition(json.dumps(fixture()), b"index", np.arange(83))
    assert cal.tolist() == list(range(40))
    assert hold.tolist() == list(range(40, 83))


@pytest.mark.parametrize("change", ["index", "overlap", "boolean", "group"])
def test_partition_changes_rejected(change):
    record = fixture()
    groups = np.arange(83)
    index = b"index"
    if change == "index":
        index = b"different"
    elif change == "overlap":
        record["holdout_positions"][0] = 0
    elif change == "boolean":
        record["calibration_positions"][0] = False
    else:
        groups[40] = groups[0]
    with pytest.raises(ValueError):
        registered_partition(json.dumps(record), index, groups)

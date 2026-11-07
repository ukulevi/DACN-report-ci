"""Load fixed validation membership instead of re-deriving it across sklearn versions."""

import hashlib
import json

import numpy as np


def registered_partition(payload, index_bytes, groups):
    record = json.loads(payload)
    if (
        record.get("schema_version") != 1
        or record.get("index_sha256") != hashlib.sha256(index_bytes).hexdigest()
    ):
        raise ValueError("Partition index binding differs")
    cal, hold = record["calibration_positions"], record["holdout_positions"]
    if (
        not isinstance(cal, list)
        or not isinstance(hold, list)
        or any(type(i) is not int for i in cal + hold)
        or (len(cal), len(hold)) != (40, 43)
        or sorted(cal + hold) != list(range(83))
    ):
        raise ValueError("Require the registered disjoint 40/43 membership")
    groups = np.asarray(groups)
    if groups.shape != (83,) or not set(groups[cal]).isdisjoint(groups[hold]):
        raise ValueError("Partition patient groups overlap or count differs")
    return np.array(cal), np.array(hold)

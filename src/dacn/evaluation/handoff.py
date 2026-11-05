"""Private reconstruction handoff checks, usable without Torch."""

import re
from pathlib import Path

import numpy as np

from dacn.evaluation.artifacts import _canonical_ids


def validate_dataset_binding(manifest, repo_id, revision):
    """Reject evaluation against a dataset other than the training export."""
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("evaluation requires an immutable dataset revision")
    if (manifest.get("dataset_repo_id"), manifest.get("dataset_revision")) != (repo_id, revision):
        raise ValueError("evaluation dataset differs from training manifest")


def write_pairs(path, sample_ids, reconstructions):
    """Persist private UUID-aligned tensors; caller must restrict artifact access."""
    ids = _canonical_ids(sample_ids)
    pairs = np.asarray(reconstructions)
    if (
        not ids
        or len(set(ids)) != len(ids)
        or pairs.shape != (len(ids), 256, 256)
        or pairs.dtype != np.float32
        or not np.isfinite(pairs).all()
        or np.any(np.abs(pairs) > 1)
    ):
        raise ValueError("invalid reconstruction cohort or tensors")
    with Path(path).open("xb") as stream:
        np.savez_compressed(
            stream, sample_ids=np.asarray(ids, dtype="U36"), reconstructions=pairs[:, None]
        )

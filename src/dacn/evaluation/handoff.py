"""Private reconstruction handoff checks, usable without Torch."""

import hashlib
import io
import re
from pathlib import Path

import numpy as np

from dacn.evaluation.artifacts import _canonical_ids, _validation_data


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


def convert_legacy_cache(
    cache_path, dataset_dir, output_path, *, expected_cache_sha256, expected_metadata_sha256
):
    """Bind a legacy ordered cache to UUIDs only after byte-verified full-array equality."""
    cache_path, output_path = Path(cache_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError("output exists; choose a fresh artifact path")
    payload = cache_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_cache_sha256:
        raise ValueError("legacy cache checksum mismatch")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != {"images", "reconstructions", "masks", "labels"}:
            raise ValueError("unexpected legacy cache fields")
        images, recons, masks, labels = (
            archive[k] for k in ("images", "reconstructions", "masks", "labels")
        )
    by_id, expected_images, expected_masks, expected_labels = _validation_data(
        dataset_dir, expected_metadata_sha256
    )
    if (
        not np.array_equal(images, expected_images)
        or not np.array_equal(masks, expected_masks)
        or not np.array_equal(labels, expected_labels)
    ):
        raise ValueError(
            "legacy cache order/content does not exactly match pinned validation export"
        )
    ids = [None] * len(by_id)
    for sample_id, position in by_id.items():
        ids[position] = sample_id
    if recons.shape != (len(ids), 256, 256) or recons.dtype != np.float32:
        raise ValueError("legacy reconstruction shape or dtype is invalid")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_pairs(output_path, ids, recons)
    return {
        "source_sha256": digest,
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "samples": len(ids),
        "dataset_metadata_sha256": expected_metadata_sha256,
    }

"""Validate and capture final-evaluation inputs without opening the test split."""

import hashlib
import math
import re
from pathlib import Path

ARTIFACTS = frozenset(
    {
        "source_snapshot",
        "config",
        "checkpoint",
        "lpips_backbone",
        "lpips_calibration",
        "dependency_lock",
        "validation_index",
        "mask_parameters",
    }
)


def capture_frozen_inputs(manifest, artifact_paths, *, validation_only=False):
    """Return the same bytes whose hashes were checked; callers must reuse them.

    This is an input boundary for a future runner, not proof of scientific
    acceptance or a standalone final-test runner. No path comes from the manifest.
    """
    allowed = {"candidate", "frozen"} if validation_only else {"frozen"}
    if manifest.get("schema_version") != 1 or manifest.get("status") not in allowed:
        raise ValueError("Final evaluation requires a frozen schema-1 manifest")
    if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("dataset_revision", ""))):
        raise ValueError("Dataset revision must be an immutable commit")
    hashes = manifest.get("artifacts", {})
    if not isinstance(hashes, dict) or set(hashes) != ARTIFACTS or set(artifact_paths) != ARTIFACTS:
        raise ValueError("Require the complete fixed artifact set")
    for value in hashes.values():
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("Every artifact requires a SHA256 pin")
    thresholds = manifest.get("thresholds", {})
    if not isinstance(thresholds, dict) or set(thresholds) != {"pixel", "image"}:
        raise ValueError("Require frozen image and pixel thresholds")
    if any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
        for x in thresholds.values()
    ):
        raise ValueError("Thresholds must be finite numbers")
    captured = {}
    for name in sorted(ARTIFACTS):
        path = Path(artifact_paths[name])
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or symlinked frozen artifact: {name}")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != hashes[name]:
            raise ValueError(f"Frozen artifact checksum differs: {name}")
        captured[name] = payload
    return captured

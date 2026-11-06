"""Capture verified continuation inputs before any child training starts."""

import hashlib
import os
from pathlib import Path

CHECKPOINTS = ("ddpm_last.pt", "ddpm_best.pt")


def capture_continuation(source, manifest):
    captured = {}
    for name in CHECKPOINTS:
        path = Path(source) / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Continuation checkpoint missing or symlinked: {name}")
        payload = path.read_bytes()
        expected = manifest.get("artifacts", {}).get(name)
        if not expected or hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Continuation checkpoint hash differs: {name}")
        captured[name] = payload
    return captured


def write_continuation(destination, captured):
    if set(captured) != set(CHECKPOINTS):
        raise ValueError("Both captured checkpoints are required")
    for name in CHECKPOINTS:
        path = Path(destination) / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(captured[name])
        os.replace(temporary, path)

"""Verify the executing source tree and installed package list before final data access."""

import io
import zipfile
from pathlib import Path

from dacn.evaluation.dependency_snapshot import version_snapshot


def verify_runtime_inputs(captured, source_root, dependency_freeze):
    root = Path(source_root).resolve()
    if version_snapshot(dependency_freeze) != version_snapshot(
        captured["dependency_lock"].decode()
    ):
        raise ValueError("Running dependencies differ from pinned environment")
    with zipfile.ZipFile(io.BytesIO(captured["source_snapshot"])) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.startswith(("src/", "scripts/")) and name.endswith(".py")
        ]
        if not names or len(names) != len(set(names)):
            raise ValueError("Source snapshot must have unique Python sources")
        expected = set(names)
        actual = {
            p.relative_to(root).as_posix()
            for folder in ("src", "scripts")
            for p in (root / folder).rglob("*.py")
        }
        if actual != expected:
            raise ValueError("Runtime source file set differs from snapshot")
        for name in names:
            path = root / name
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("Runtime source escapes root")
            if path.read_bytes() != archive.read(name):
                raise ValueError("Runtime source bytes differ from snapshot")
    return {"source_files_verified": len(names), "dependency_freeze_verified": True}

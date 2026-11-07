"""Version-only dependency inventory without exporting direct source URLs."""

import importlib.metadata
import re


def version_snapshot(freeze):
    versions = {}
    for raw in freeze.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if " @ " in line:
            name = line.split(" @ ", 1)[0]
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
                raise ValueError("Invalid dependency name")
            version = importlib.metadata.version(name)
        elif "==" in line:
            name, version = line.split("==", 1)
        else:
            raise ValueError("Dependency requires an installed version")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not re.fullmatch(
            r"[A-Za-z0-9_.!+\-]+", version
        ):
            raise ValueError("Invalid dependency name or version")
        name = re.sub(r"[-_.]+", "-", name).lower()
        if name in versions and versions[name] != version:
            raise ValueError("Conflicting installed dependency versions")
        versions[name] = version
    return "".join(f"{name}=={versions[name]}\n" for name in sorted(versions))

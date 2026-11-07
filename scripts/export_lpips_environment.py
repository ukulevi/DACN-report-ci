"""Export native SqueezeNet LPIPS states and an environment receipt (private output)."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import lpips
import torch

from dacn.evaluation.autoddpm_lpips import perceptual_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = lpips.LPIPS(net="squeeze", spatial=True, version="0.1").eval()
    state = model.state_dict()
    files = {
        "lpips_backbone": "lpips_backbone.pth",
        "lpips_calibration": "lpips_calibration.pth",
        "dependency_lock": "requirements.freeze.txt",
    }
    torch.save(model.net.state_dict(), args.output / files["lpips_backbone"])
    torch.save(
        {k: v for k, v in state.items() if not k.startswith("net.")},
        args.output / files["lpips_calibration"],
    )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--exclude-editable"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    # Keep local file URLs/credentials out of exported environment records.
    if " @ " in freeze or "://" in freeze:
        raise ValueError("Environment contains non-version package sources; sanitize privately")
    (args.output / files["dependency_lock"]).write_text(freeze, encoding="utf-8")
    receipt = {
        "scope": "New environment export; not evidence of historical runtime identity",
        "network": perceptual_provenance(model),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "lpips", "numpy")
        },
        "artifacts": {
            key: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
            for key, name in files.items()
        },
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

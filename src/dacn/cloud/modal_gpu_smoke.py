"""Minimal T4 startup probe: no Secrets, Volumes, Hugging Face or project data."""

import modal

app = modal.App("dacn-t4-minimal-check")


@app.function(image=modal.Image.debian_slim(python_version="3.11"), gpu="T4", timeout=60)
def check_gpu():
    import json
    import subprocess

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    result = {"status": "passed", "gpu": gpu, "test": "GPU startup and nvidia-smi"}
    print(json.dumps(result), flush=True)
    return result

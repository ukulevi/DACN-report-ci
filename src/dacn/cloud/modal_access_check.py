"""Short T4 startup/access probe for an explicitly selected Modal profile."""

import modal

app = modal.App("dacn-account-access-check")
image = modal.Image.debian_slim(python_version="3.11")
dataset_volume = modal.Volume.from_name("dacn-bmad-datasets", create_if_missing=False)
run_volume = modal.Volume.from_name("dacn-bmad-runs", create_if_missing=False)


@app.function(
    image=image,
    gpu="T4",
    timeout=120,
    volumes={"/mnt/datasets": dataset_volume, "/mnt/runs": run_volume},
    secrets=[modal.Secret.from_name("huggingface-secret-2")],
)
def check_remote():
    import hashlib
    import json
    import os
    import subprocess
    import urllib.error
    import urllib.request
    from pathlib import Path

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout.strip()
    dataset = "dacn-bmad-team/bmad-flair-patient-disjoint-v1"
    revision = "350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7"
    key = hashlib.sha256(f"{dataset}@{revision}".encode()).hexdigest()[:20]
    result = {
        "gpu_startup": "passed",
        "gpu": gpu,
        "cached_validation_index_exists": (
            Path("/mnt/datasets") / key / "data" / "val" / "index.csv"
        ).is_file(),
        "B_checkpoint_exists": (
            Path("/mnt/runs")
            / "feature-B-pretrain5-joint10-t4-20261105T053132Z"
            / "feature_condition_best.pt"
        ).is_file(),
        "training_started": False,
    }
    for name, route in (
        ("private_dataset", f"datasets/{dataset}"),
        ("private_model_repo", "models/dacn-bmad-team/dacn-runs"),
    ):
        request = urllib.request.Request(
            f"https://huggingface.co/api/{route}",
            headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                metadata = json.load(response)
                result[name] = {"http_status": response.status, "private": metadata["private"]}
        except urllib.error.HTTPError as exc:
            result[name] = {"http_status": exc.code}
    print(json.dumps(result), flush=True)
    return result

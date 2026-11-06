"""Inference-only interventions on B's fixed EMA checkpoint and validation cache."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
from torch import nn

from dacn.training.ddpm_partial import derive_sampling_seed

B_RUN = "feature-B-pretrain5-joint10-t4-20261105T053132Z"
B_SHA = "0c4f21705830f5dd6054174fc6b2c1591263779413737ce87b4407e3ca8b01b7"
B_RESULTS_SHA = "17cea7ce87afc15332835ffdafa514b2758883d92510339f86bb45b521ac247d"
INDEX_SHA = "57ad2e31df3ed4e625da00efaecd6fddc1e2a132329dfba6e60867caf1802db3"


class ConditionIntervention(nn.Module):
    """Zero means bypass the whole injection, including its learned bias.

    A donor changes only encoder input. Forward noising and reverse-chain input
    remain the evaluated image. No lesion masks are accepted by this wrapper.
    """

    def __init__(self, model, mode, donor=None):
        super().__init__()
        if mode not in {"zero", "normal_donor"} or (mode == "normal_donor" and donor is None):
            raise ValueError("Invalid condition intervention")
        self.model, self.mode, self.donor = model, mode, donor

    def encode(self, image):
        if self.mode == "zero":
            return image.new_zeros(len(image), self.model.condition_projection.in_features)
        return self.model.encode(self.donor)

    def forward(self, noisy, timesteps, condition):
        if self.mode == "zero":
            return self.model.unet(noisy, timesteps)
        return self.model(noisy, timesteps, condition)


def donor_index(sample_id, count):
    if count < 1:
        raise ValueError("Missing normal training donors")
    # Fixed UUID-derived mapping, independent of labels/masks/metrics. Keep IDs
    # and chosen indices private; donor inputs come only from normal training.
    return derive_sampling_seed(77, sample_id) % count


def verified_reference(reference_dir, checkpoint_path, validation_dir):
    reference_dir = Path(reference_dir)
    path = reference_dir / "results" / "results.json"
    result_bytes = path.read_bytes()
    if hashlib.sha256(result_bytes).hexdigest() != B_RESULTS_SHA:
        raise ValueError("Registered B result differs")
    if hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest() != B_SHA:
        raise ValueError("Registered B checkpoint differs")
    if hashlib.sha256((Path(validation_dir) / "index.csv").read_bytes()).hexdigest() != INDEX_SHA:
        raise ValueError("Validation index differs")
    manifest = json.loads((reference_dir / "run_manifest.json").read_text())
    cache_path = reference_dir / "results" / "private_reconstructions.npz"
    cache_bytes = cache_path.read_bytes()
    if (
        hashlib.sha256(cache_bytes).hexdigest()
        != manifest["artifacts"]["results/private_reconstructions.npz"]
    ):
        raise ValueError("Registered correct-condition cache differs")
    result = json.loads(result_bytes)
    if (
        result["checkpoint_sha256"] != B_SHA
        or result["samples"] != 83
        or result["held_out_test_used"]
    ):
        raise ValueError("Invalid B reference protocol")
    with np.load(io.BytesIO(cache_bytes), allow_pickle=False) as cache:
        arrays = {key: cache[key].copy() for key in cache.files}
    return result, arrays


def compare_reconstructions(reference, observed, labels):
    if reference.shape != observed.shape or not np.isfinite(observed).all():
        raise ValueError("Invalid intervention reconstruction")
    delta = np.abs(reference[:, 8:-8, 8:-8] - observed[:, 8:-8, 8:-8]) / 2
    return {
        "mean_absolute_change_01": float(delta.mean()),
        "max_absolute_change_01": float(delta.max()),
        "normal_mean_absolute_change_01": float(delta[labels == 0].mean()),
        "abnormal_mean_absolute_change_01": float(delta[labels == 1].mean()),
    }

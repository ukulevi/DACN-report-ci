"""Frozen grayscale features and bounded exact normal-neighbor search."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dacn.models.diffusion.spatial_condition import (
    WEIGHTS_PATH,
    WEIGHTS_SHA256,
    SpatialResNet18,
)


class MemoryFeatureEncoder(nn.Module):
    """Official ImageNet weights; raw DACN grayscale [-1,1], no RGB normalization."""

    def __init__(self, weights_path=WEIGHTS_PATH):
        super().__init__()
        self.backbone = SpatialResNet18()
        self.backbone.load_imagenet(weights_path)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # BatchNorm statistics must remain the official pretrained statistics.
        return super().train(False)

    @torch.inference_mode()
    def forward(self, images):
        maps = self.backbone(images)
        layer2, layer3 = maps[1:3]
        if layer2.shape[1:] != (128, 32, 32) or layer3.shape[1:] != (256, 16, 16):
            raise ValueError("Unexpected ResNet layer2/layer3 shapes")
        return torch.cat(
            [layer2, F.interpolate(layer3, (32, 32), mode="bilinear", align_corners=False)],
            dim=1,
        )


class RandomPriorityBank:
    """Uniform descriptor sampling without replacement; not a PatchCore coreset."""

    def __init__(self, budget=10000, seed=17):
        if budget < 1:
            raise ValueError("Bank budget must be positive")
        self.budget = budget
        self.rng = np.random.default_rng(seed)
        self.keys = np.empty(0)
        self.values = None
        self.seen = 0

    def update(self, descriptors):
        values = np.asarray(descriptors, dtype=np.float32)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Descriptors must be a finite matrix")
        if self.values is not None and values.shape[1] != self.values.shape[1]:
            raise ValueError("Descriptor dimensions differ")
        self.seen += len(values)
        keys = np.concatenate([self.keys, self.rng.random(len(values))])
        values = values if self.values is None else np.concatenate([self.values, values])
        keep = np.argsort(keys, kind="stable")[: self.budget]
        self.keys, self.values = keys[keep], values[keep].copy()


def foreground_descriptors(features, images):
    """Keep cells with >=50% input>-0.95 within their 8x8 footprint."""
    if images.shape[1:] != (1, 256, 256) or features.shape[2:] != (32, 32):
        raise ValueError("Expected padded256 images and32 feature grid")
    keep = F.adaptive_avg_pool2d((images > -0.95).float(), (32, 32))[:, 0] >= 0.5
    return features.permute(0, 2, 3, 1)[keep]


@torch.inference_mode()
def nearest_distances(queries, bank, query_chunk=256, bank_chunk=2048):
    if (
        queries.ndim != 2
        or bank.ndim != 2
        or not len(bank)
        or queries.shape[1] != bank.shape[1]
        or min(query_chunk, bank_chunk) < 1
        or not torch.isfinite(queries).all()
        or not torch.isfinite(bank).all()
    ):
        raise ValueError("Invalid nearest-neighbor inputs")
    values = []
    for query in queries.split(query_chunk):
        best = torch.full((len(query),), float("inf"), device=query.device)
        for candidates in bank.split(bank_chunk):
            distance = torch.cdist(query, candidates, p=2)
            best = torch.minimum(best, distance.min(dim=1).values)
        values.append(best)
    return torch.cat(values) if values else queries.new_empty(0)


def upsample_patch_scores(scores):
    if scores.ndim != 3 or scores.shape[1:] != (32, 32):
        raise ValueError("Expected32x32 patch score grids")
    return F.interpolate(scores[:, None], (256, 256), mode="bilinear", align_corners=False)[
        :, 0, 8:-8, 8:-8
    ]


@torch.inference_mode()
def memory_maps(encoder, images, bank):
    features = encoder(images)
    queries = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
    patch = nearest_distances(queries, bank).reshape(len(images), 32, 32)
    return upsample_patch_scores(patch)


def save_bank(path, values, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid bank")
    with path.open("wb") as stream:
        np.savez_compressed(
            stream, descriptors=values, metadata=json.dumps(metadata, sort_keys=True)
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bank(path, *, checksum, expected_metadata):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
        raise ValueError("Memory bank checksum differs")
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        values = archive["descriptors"].copy()
    if metadata != expected_metadata or metadata.get("encoder_sha256") != WEIGHTS_SHA256:
        raise ValueError("Memory bank provenance differs")
    if values.ndim != 2 or values.shape[1] != 384 or not len(values):
        raise ValueError("Memory bank descriptor shape differs")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite memory bank")
    return values

"""Single-sample DDIM20+30 adapter; never fits parameters on its inputs."""

import io
import json
import math

import numpy as np
import torch

from dacn.evaluation.autoddpm_lpips import pseudo_mask
from dacn.evaluation.ddim_ablation import ablation_pair_scores, resolve_ablation_config
from dacn.evaluation.ddim_inpainting import (
    ddim_indices,
    ddim_initial_reconstruction,
    ddim_stitch_resample,
)
from dacn.evaluation.frozen_runner import run_frozen_evaluation
from dacn.training.ddpm_partial import (
    build_diffusion,
    derive_sampling_seed,
    load_ema_checkpoint,
)


def load_perceptual(captured, device):
    """Load exported native LPIPS trunk/head states without network downloads."""
    import lpips

    model = lpips.LPIPS(net="squeeze", spatial=True, pnet_rand=True, pretrained=False)
    trunk = torch.load(io.BytesIO(captured["lpips_backbone"]), weights_only=True)
    head = torch.load(io.BytesIO(captured["lpips_calibration"]), weights_only=True)
    model.net.load_state_dict(trunk, strict=True)
    expected = {key for key in model.state_dict() if not key.startswith("net.")}
    if set(head) != expected:
        raise ValueError("LPIPS calibration state keys differ")
    model.load_state_dict({**model.state_dict(), **head}, strict=True)
    return model.to(device).eval().requires_grad_(False)


class FrozenDDIMAdapter:
    """Lazy model initialization from verified bytes supplied by frozen runner."""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.ready = False

    def initialize(self, captured):
        parameters = json.loads(captured["mask_parameters"])
        self.config = resolve_ablation_config(parameters["config"])
        if resolve_ablation_config(json.loads(captured["config"])) != self.config:
            raise ValueError("Sampler config and mask parameters differ")
        self.mask_threshold = parameters["mask_threshold"]
        if not math.isfinite(self.mask_threshold) or self.mask_threshold < 0:
            raise ValueError("Invalid frozen mask threshold")
        self.indices = [ddim_indices(300, 20), ddim_indices(50, 30)]
        for key, indices in zip(
            ("phase1_indices_zero_based", "phase2_indices_zero_based"),
            self.indices,
            strict=True,
        ):
            if parameters[key] != list(indices):
                raise ValueError("Frozen sampling indices differ")
        # The pinned config is the sampler preset, not the training architecture.
        # Architecture/schedule are bound by the hashed full-state checkpoint.
        train_config = torch.load(
            io.BytesIO(captured["checkpoint"]), map_location="cpu", weights_only=False
        )["config"]
        self.model = load_ema_checkpoint(
            io.BytesIO(captured["checkpoint"]), train_config, self.device
        )
        self.diffusion = build_diffusion(train_config)
        self.perceptual = load_perceptual(captured, self.device)
        self.ready = True

    @torch.inference_mode()
    def __call__(self, sample, captured):
        if not self.ready:
            self.initialize(captured)
        calls = []
        handle = self.model.register_forward_hook(lambda *_: calls.append(None))
        try:
            result = self.infer(sample)
            if len(calls) != 50:
                raise ValueError("Expected exactly 50 U-Net calls per sample")
            return result
        finally:
            handle.remove()

    def infer(self, sample):
        image, mask = np.asarray(sample["image"]), np.asarray(sample["mask"])
        if image.shape != (256, 256) or mask.shape != image.shape:
            raise ValueError("Expected padded 256x256 image and mask")
        if not np.isfinite(image).all() or (np.abs(image) > 1).any():
            raise ValueError("Expected finite [-1,1] image")
        if not np.isin(mask, [0, 1]).all():
            raise ValueError("Expected binary mask")
        seed = derive_sampling_seed(self.config["seed"], sample["sample_id"])
        tensor = torch.from_numpy(image.copy()).float()[None, None].to(self.device)
        initial = ddim_initial_reconstruction(
            self.model, self.diffusion, tensor, seed=seed, indices=self.indices[0]
        )
        initial_scores, _ = ablation_pair_scores(
            self.perceptual,
            image,
            initial[0, 0].cpu().numpy(),
            self.device,
        )
        predicted = pseudo_mask(initial_scores["product"], self.mask_threshold)
        region = torch.from_numpy(np.pad(predicted, 8))[None, None].to(self.device)
        final = ddim_stitch_resample(
            self.model,
            tensor,
            initial,
            region,
            diffusion=self.diffusion,
            seed=seed,
            indices=self.indices[1],
            repeats=1,
        )
        scores, _ = ablation_pair_scores(
            self.perceptual,
            image,
            final[0, 0].cpu().numpy(),
            self.device,
        )
        flat = scores["lpips"].reshape(-1)
        return {
            "pixel_labels": mask[8:-8, 8:-8].reshape(-1),
            "pixel_scores": flat,
            "image_labels": [int(mask.any())],
            "image_scores": [float(np.partition(flat, -576)[-576:].mean())],
        }


def run_frozen_ddim(*, manifest, artifact_paths, samples, output_dir, device="cpu"):
    """Connect the frozen-input gate/orchestrator to a fresh DDIM adapter.

    Caller must provide a revision/content-verified dataset factory. Merely
    pinning source and dependency files does not verify the running interpreter.
    """
    return run_frozen_evaluation(
        manifest=manifest,
        artifact_paths=artifact_paths,
        samples=samples,
        infer=FrozenDDIMAdapter(device),
        output_dir=output_dir,
    )

import io
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dacn.evaluation import frozen_ddim
from dacn.evaluation.ddim_ablation import DEFAULT_CONFIG
from dacn.evaluation.ddim_inpainting import ddim_indices
from dacn.training.ddpm_partial import build_diffusion


def test_adapter_runs_fifty_calls_with_frozen_mask(monkeypatch):
    class ZeroModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x, t):
            self.calls += 1
            return torch.zeros_like(x)

    model = ZeroModel()
    monkeypatch.setattr(frozen_ddim, "load_ema_checkpoint", lambda *args: model)
    monkeypatch.setattr(frozen_ddim, "load_perceptual", lambda *args: None)
    monkeypatch.setattr(
        frozen_ddim,
        "ablation_pair_scores",
        lambda *args: ({"product": np.zeros((240, 240)), "lpips": np.zeros((240, 240))}, {}),
    )
    params = {
        "config": DEFAULT_CONFIG,
        "mask_threshold": 0.13,
        "phase1_indices_zero_based": list(ddim_indices(300, 20)),
        "phase2_indices_zero_based": list(ddim_indices(50, 30)),
    }
    captured = {
        "config": b"{}",
        "checkpoint": b"checkpoint",
        "mask_parameters": json.dumps(params).encode(),
    }
    adapter = frozen_ddim.FrozenDDIMAdapter()
    monkeypatch.setattr(frozen_ddim, "build_diffusion", lambda _: build_diffusion({}))
    sample = {
        "image": np.zeros((256, 256), dtype=np.float32),
        "mask": np.zeros((256, 256), dtype=np.uint8),
        "sample_id": "00000000-0000-0000-0000-000000000001",
    }
    result = adapter(sample, captured)
    assert model.calls == 50
    assert result["pixel_scores"].shape == (57600,)
    assert result["image_scores"] == [0]
    assert result["image_labels"] == [0]


def test_perceptual_export_load_roundtrip():
    lpips = pytest.importorskip("lpips")
    original = lpips.LPIPS(
        net="squeeze", spatial=True, pnet_rand=True, pretrained=False, verbose=False
    )
    captured = {}
    for name, state in (
        ("lpips_backbone", original.net.state_dict()),
        (
            "lpips_calibration",
            {k: v for k, v in original.state_dict().items() if not k.startswith("net.")},
        ),
    ):
        buffer = io.BytesIO()
        torch.save(state, buffer)
        captured[name] = buffer.getvalue()
    restored = frozen_ddim.load_perceptual(captured, "cpu")
    for name, value in original.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
    assert not restored.training

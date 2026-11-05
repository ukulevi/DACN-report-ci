import csv
import hashlib
import uuid

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Sampler ablation requires the train extra")

from dacn.evaluation import sampler_ablation as ablation
from dacn.evaluation.patch_validation import gaussian_one_step
from dacn.training.ddpm_partial import _fixed_rng, build_diffusion


def test_ddim10_oracle_inversion_determinism_and_exact_model_calls():
    diffusion = build_diffusion({})
    clean = torch.full((1, 1, 16, 16), 0.2)
    noisy = diffusion.q_sample(clean, torch.tensor([299]), torch.randn_like(clean))
    calls = []

    class Oracle(torch.nn.Module):
        def forward(self, value, t):
            k = int(t.item())
            calls.append(k)
            alpha = float(diffusion.alphas_cumprod[k])
            return (value - alpha**0.5 * clean) / (1 - alpha) ** 0.5

    before = torch.random.get_rng_state().clone()
    output = ablation.deterministic_from_noisy(
        Oracle(), diffusion, noisy, indices=ablation.DDIM_INDICES
    )
    assert calls == [299, 269, 239, 209, 179, 149, 119, 89, 59, 29]
    assert torch.allclose(output, clean, atol=1e-6)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.allclose(
        ablation.deterministic_from_noisy(Oracle(), diffusion, noisy, indices=(299,)),
        clean,
        atol=1e-6,
    )


def test_ddim_eta0_transition_matches_independent_formula_and_one_step_control():
    diffusion = build_diffusion({})

    class Constant(torch.nn.Module):
        def forward(self, x, t):
            return torch.full_like(x, 0.02)

    model = Constant()
    noisy = torch.full((1, 1, 16, 16), 0.03)
    a, b = (float(diffusion.alphas_cumprod[k]) for k in (299, 149))
    x0 = (noisy - (1 - a) ** 0.5 * 0.02) / a**0.5
    intermediate = b**0.5 * x0 + (1 - b) ** 0.5 * 0.02
    expected = (intermediate - (1 - b) ** 0.5 * 0.02) / b**0.5
    result = ablation.deterministic_from_noisy(model, diffusion, noisy, indices=(299, 149))
    assert torch.allclose(result, expected, atol=1e-6)
    image = torch.zeros(1, 1, 256, 256)
    with _fixed_rng(17, image.device):
        t = torch.tensor([299])
        initial = diffusion.q_sample(image, t, diffusion.noise(image, t))
    assert torch.equal(
        ablation.deterministic_from_noisy(model, diffusion, initial, indices=(299,)),
        gaussian_one_step(model, diffusion, image, steps=300, seed=17),
    )
    with pytest.raises(ValueError):
        ablation.deterministic_from_noisy(model, diffusion, noisy, indices=(149, 299))


def test_all83_cache_alignment_shared_input_and_calibration_thresholds(tmp_path, monkeypatch):
    labels = np.asarray([i % 2 for i in range(83)], dtype=np.uint8)
    images = np.zeros((83, 256, 256), dtype=np.float32)
    masks = np.zeros_like(images, dtype=np.uint8)
    masks[labels == 1, 100:120, 100:120] = 1
    ids = [str(uuid.UUID(int=i)) for i in range(83)]
    validation_dir = tmp_path / "val"
    validation_dir.mkdir()
    with (validation_dir / "index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "patient_group_id"])
        writer.writeheader()
        writer.writerows(
            {"sample_id": value, "patient_group_id": str(i // 2)} for i, value in enumerate(ids)
        )
    checkpoint_path = tmp_path / "runs" / "train" / "source.pt"
    checkpoint_path.parent.mkdir(parents=True)
    cache_dir = checkpoint_path.parent.parent / "evaluations" / ablation.BASELINE_RUN
    cache_dir.mkdir(parents=True)
    cache_path = cache_dir / "ddpm_epoch_030_t0300_ancestral_reconstructions.npz"
    np.savez_compressed(
        cache_path, images=images, reconstructions=images, masks=masks, labels=labels
    )
    monkeypatch.setattr(
        ablation, "BASELINE_SHA", hashlib.sha256(cache_path.read_bytes()).hexdigest()
    )

    class Data:
        sample_ids = ids

        def __len__(self):
            return 83

        def __getitem__(self, i):
            return torch.from_numpy(images[i : i + 1]), torch.from_numpy(masks[i]), int(labels[i])

    data = Data()
    data.labels = labels
    monkeypatch.setattr(ablation, "ShardDataset", lambda *a, **kw: data)
    monkeypatch.setattr(
        ablation.torch,
        "load",
        lambda *a, **kw: {
            "epoch": 30,
            "method": "ddpm_partial_reconstruction_bmad_flair",
            "config": {},
        },
    )
    monkeypatch.setattr(ablation, "load_ema_checkpoint", lambda *a: None)
    monkeypatch.setattr(ablation, "partial_reconstruct", lambda model, image, **kw: image)
    seen = []

    def reconstruct(model, diffusion, noisy, *, indices):
        seen.append((noisy.data_ptr(), tuple(indices)))
        return torch.zeros_like(noisy)

    monkeypatch.setattr(ablation, "deterministic_from_noisy", reconstruct)
    monkeypatch.setattr(
        ablation,
        "paired_score_maps",
        lambda x, y: {"pixel_difference": np.zeros((240, 240)), "local_ssim": np.zeros((240, 240))},
    )
    thresholds = []

    def metrics(*a, **kw):
        thresholds.append(kw.get("thresholds"))
        return {"pixel": {"threshold": 0.1}, "image": {"threshold": 0.2}}

    monkeypatch.setattr(ablation, "evaluate_maps", metrics)
    progress = []
    result = ablation.evaluate_sampler_ablation(
        validation_dir=validation_dir,
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "out",
        device=torch.device("cpu"),
        on_progress=progress.append,
    )
    assert len(seen) == 166 and all(seen[i][0] == seen[i + 1][0] for i in range(0, 166, 2))
    assert thresholds == [None, {"pixel": 0.1, "image": 0.2}, None] * 7
    assert result["ancestral_control_max_abs_error"] == 0 and not result["held_out_test_used"]
    assert progress[-1]["completed_samples"] == 83
    assert result["samples"] == 83 and result["shared_noisy_input"]

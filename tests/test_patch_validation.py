import csv
import uuid

import numpy as np
import pytest
import torch

from dacn.evaluation import patch_validation


@pytest.mark.parametrize(
    "one_step,gaussian_baseline", [(False, False), (True, False), (True, True)]
)
def test_full_validation_freezes_calibration_thresholds_and_saves_prefix(
    tmp_path, monkeypatch, one_step, gaussian_baseline
):
    labels = np.array([0, 1] * 4, dtype=np.uint8)
    ids = [str(uuid.UUID(int=i)) for i in range(8)]

    class Data:
        sample_ids = ids

        def __len__(self):
            return len(labels)

        def __getitem__(self, i):
            x = torch.full((1, 256, 256), i / 20)
            mask = torch.zeros(256, 256, dtype=torch.uint8)
            if labels[i]:
                mask[100:120, 100:120] = 1
            return x, mask, torch.tensor(labels[i])

    data = Data()
    data.labels = labels
    with (tmp_path / "index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "patient_group_id"])
        writer.writeheader()
        writer.writerows(
            {"sample_id": value, "patient_group_id": str(i // 2)} for i, value in enumerate(ids)
        )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    monkeypatch.setattr(patch_validation, "ShardDataset", lambda *a, **kw: data)
    monkeypatch.setattr(patch_validation, "load_context_checkpoint", lambda *a: (None, None))
    monkeypatch.setattr(
        patch_validation.torch,
        "load",
        lambda *a, **kw: {
            "epoch": 30,
            "method": "ddpm_partial_reconstruction_bmad_flair",
            "config": {},
        },
    )
    monkeypatch.setattr(patch_validation, "load_ema_checkpoint", lambda *a: None)
    monkeypatch.setattr(patch_validation, "build_diffusion", lambda *a: None)
    seen = []

    def reconstruct(model, diffusion, image, *, steps, seed, **kwargs):
        assert steps == 100
        assert kwargs == ({"one_step": True} if one_step and not gaussian_baseline else {})
        seen.append(float(image[0, 0, 0, 0]))
        return image + 0.01

    monkeypatch.setattr(patch_validation, "sliding_reconstruct", reconstruct)
    monkeypatch.setattr(patch_validation, "gaussian_one_step", reconstruct)
    threshold_calls = []

    def measure(scores, masks, y, *, split, thresholds=None):
        assert split == "validation"
        threshold_calls.append(thresholds)
        return {"pixel": {"threshold": 0.12}, "image": {"threshold": 0.34}}

    monkeypatch.setattr(patch_validation, "evaluate_maps", measure)
    output = tmp_path / "results"
    progress_counts = []

    def progress(values):
        with np.load(output / "private_reconstructions.npz") as cache:
            assert len(cache["images"]) == values["completed_samples"]
        progress_counts.append(values["completed_samples"])

    result = patch_validation.evaluate_patch_validation(
        validation_dir=tmp_path,
        checkpoint_path=checkpoint,
        output_dir=output,
        device=torch.device("cpu"),
        on_progress=progress,
        one_step=one_step,
        gaussian_baseline=gaussian_baseline,
    )
    assert len(seen) == 8 and progress_counts == [1, 5, 8]
    assert threshold_calls == [None, {"pixel": 0.12, "image": 0.34}, None] * 3
    assert result["samples"] == 8 and result["held_out_test_used"] is False
    assert result["model_calls_per_image"] == (1 if gaussian_baseline else 25 if one_step else 2500)
    assert result["checkpoint_epoch"] == (30 if gaussian_baseline else 5)
    if gaussian_baseline:
        assert result["method"] == "gaussian_ddpm_full_image_one_step_x0"
        assert result["patch_size"] is None
    assert result["partition"]["calibration"]["groups"] == 2
    assert result["partition"]["holdout"]["groups"] == 2
    assert (output / "results.json").is_file()


def test_gaussian_one_step_calls_original_one_channel_model_once_and_inverts_noise():
    from dacn.training.ddpm_partial import _fixed_rng, build_diffusion

    image = torch.full((1, 1, 256, 256), 0.2)
    diffusion = build_diffusion({})
    with _fixed_rng(17, image.device):
        noise = diffusion.noise(image, torch.tensor([99]))

    class Oracle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x, t):
            self.calls += 1
            assert x.shape == image.shape and t.tolist() == [99]
            return noise

    model = Oracle()
    rng_before = torch.random.get_rng_state().clone()
    result = patch_validation.gaussian_one_step(model, diffusion, image, steps=100, seed=17)
    assert model.calls == 1 and torch.allclose(result, image, atol=1e-6)
    assert torch.equal(rng_before, torch.random.get_rng_state())

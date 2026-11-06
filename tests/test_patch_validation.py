import csv
import uuid

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Patch validation requires the train extra")

from dacn.evaluation import patch_validation


@pytest.mark.parametrize(
    "one_step,gaussian_baseline,feature_condition,condition_mode,unconditional_finetune",
    [
        (False, False, False, "self", False),
        (True, False, False, "self", False),
        (True, True, False, "self", False),
        (False, False, True, "self", False),
        (False, False, True, "zero", False),
        (False, False, True, "normal_donor", False),
        (False, False, False, "self", True),
    ],
)
def test_full_validation_freezes_calibration_thresholds_and_saves_prefix(
    tmp_path,
    monkeypatch,
    one_step,
    gaussian_baseline,
    feature_condition,
    condition_mode,
    unconditional_finetune,
):
    stores_noisy = feature_condition or unconditional_finetune
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

    class Donors:
        labels = np.zeros(2, dtype=np.uint8)

        def __len__(self):
            return 2

        def __getitem__(self, i):
            return torch.full((1, 256, 256), -0.4), torch.tensor(0)

    def dataset(path, split, *, include_masks):
        if split == "train":
            assert not include_masks
            return Donors()
        return data

    monkeypatch.setattr(patch_validation, "ShardDataset", dataset)
    monkeypatch.setattr(patch_validation, "load_context_checkpoint", lambda *a: (None, None))
    monkeypatch.setattr(
        patch_validation.torch,
        "load",
        lambda *a, **kw: {
            "epoch": 36 if unconditional_finetune else 10 if feature_condition else 30,
            "method": "ddpm_partial_reconstruction_bmad_flair",
            "config": {},
            "ema_initialization": {"source_epoch": 30, "encoder_added": False},
        },
    )
    monkeypatch.setattr(patch_validation, "load_ema_checkpoint", lambda *a: None)
    monkeypatch.setattr(patch_validation, "build_diffusion", lambda *a: None)
    seen = []

    def reconstruct(model, diffusion, image, *, steps, seed, **kwargs):
        if condition_mode != "self":
            assert model.mode == condition_mode
            if condition_mode == "normal_donor":
                assert torch.all(model.donor == -0.4)
        assert steps == (300 if stores_noisy else 100)
        assert kwargs == (
            {"return_noisy": True}
            if feature_condition
            else {"one_step": True}
            if one_step and not gaussian_baseline
            else {}
        )
        seen.append(float(image[0, 0, 0, 0]))
        return (image + 0.01, image + 0.2) if stores_noisy else image + 0.01

    monkeypatch.setattr(patch_validation, "sliding_reconstruct", reconstruct)
    monkeypatch.setattr(patch_validation, "gaussian_one_step", reconstruct)
    monkeypatch.setattr(patch_validation, "gaussian_ancestral_with_noisy", reconstruct)
    if feature_condition:
        from dacn.training import feature_condition as feature_training

        monkeypatch.setattr(feature_training, "load_feature_checkpoint", lambda *a: (None, None))
        monkeypatch.setattr(feature_training, "feature_reconstruct", reconstruct)
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
        feature_condition=feature_condition,
        unconditional_finetune=unconditional_finetune,
        condition_mode=condition_mode,
        donor_train_dir=tmp_path if condition_mode == "normal_donor" else None,
        starting_timestep=300 if stores_noisy else 100,
    )
    assert len(seen) == 8 and progress_counts == [1, 5, 8]
    assert threshold_calls == [None, {"pixel": 0.12, "image": 0.34}, None] * 3
    assert result["samples"] == 8 and result["held_out_test_used"] is False
    assert result["model_calls_per_image"] == (
        300 if stores_noisy else 1 if gaussian_baseline else 25 if one_step else 2500
    )
    assert result["checkpoint_epoch"] == (
        36
        if unconditional_finetune
        else 10
        if feature_condition
        else 30
        if gaussian_baseline
        else 5
    )
    if gaussian_baseline:
        assert result["method"] == "gaussian_ddpm_full_image_one_step_x0"
        assert result["patch_size"] is None
    assert result["partition"]["calibration"]["groups"] == 2
    assert result["partition"]["holdout"]["groups"] == 2
    assert (output / "results.json").is_file()
    if unconditional_finetune:
        assert result["condition_mode"] == "none"
        assert result["method"] == "unconditional_ddpm_EMA30_finetune"
    if stores_noisy:
        with np.load(output / "private_reconstructions.npz") as cache:
            prefix = {key: cache[key][:3].copy() for key in cache.files}
        np.savez_compressed(output / "private_reconstructions.npz", **prefix)
        seen.clear()
        patch_validation.evaluate_patch_validation(
            validation_dir=tmp_path,
            checkpoint_path=checkpoint,
            output_dir=output,
            device=torch.device("cpu"),
            feature_condition=feature_condition,
            unconditional_finetune=unconditional_finetune,
            condition_mode=condition_mode,
            donor_train_dir=tmp_path if condition_mode == "normal_donor" else None,
            starting_timestep=300,
            resume_partial=True,
        )
        assert len(seen) == 5


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


def test_unconditional_ancestral_returns_same_chain_as_original_and_keeps_rng():
    from dacn.training.ddpm_partial import build_diffusion, partial_reconstruct

    class Zero(torch.nn.Module):
        def forward(self, image, timestep):
            return torch.zeros_like(image)

    model, diffusion = Zero(), build_diffusion({})
    image = torch.full((1, 1, 256, 256), 0.2)
    rng = torch.random.get_rng_state().clone()
    expected = partial_reconstruct(model, image, starting_timestep=3, seed=17, diffusion=diffusion)
    observed, noisy = patch_validation.gaussian_ancestral_with_noisy(
        model, diffusion, image, steps=3, seed=17
    )
    assert torch.equal(observed, expected)
    assert noisy.shape == image.shape and not torch.equal(noisy, image)
    assert torch.equal(rng, torch.random.get_rng_state())

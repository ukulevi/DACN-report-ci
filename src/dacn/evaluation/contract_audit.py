"""Read-only checkpoint/data audit and controlled reconstruction diagnostics."""

import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from dacn.data.shards import ShardDataset
from dacn.evaluation.scores import paired_score_maps
from dacn.training.ddpm_partial import (
    _fixed_rng,
    build_diffusion,
    build_model,
    derive_sampling_seed,
    load_ema_checkpoint,
)


@torch.inference_mode()
def controlled_chain(model, image, *, diffusion, seed, steps, variance):
    """Same initial Gaussian noise and same reverse draws across variance choices."""
    if variance not in {"large", "small", "mean"}:
        raise ValueError("Unknown variance mode")
    with _fixed_rng(seed, image.device):
        t = torch.tensor([steps - 1], device=image.device)
        value = diffusion.q_sample(image, t, torch.randn_like(image))
        for k in range(steps - 1, -1, -1):
            t.fill_(k)
            mean, log_variance = diffusion._p_mean_variance(model, value, t)
            noise = torch.randn_like(value)
            if k == 0 or variance == "mean":
                value = mean
            elif variance == "large":
                value = mean + torch.exp(0.5 * log_variance) * noise
            else:
                value = mean + np.sqrt(diffusion.posterior_variance[k]) * noise
        return value.clamp(-1, 1)


def evaluate_contract_audit(
    *, validation_dir: Path, checkpoint_path: Path, output_dir: Path, device
):
    import time

    if output_dir.exists():
        raise FileExistsError("Use a fresh output directory")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 30 and checkpoint["optimizer_steps"] == 28140
    ema = load_ema_checkpoint(checkpoint_path, checkpoint["config"], device)
    assert all(
        torch.equal(value.cpu(), checkpoint["ema"][key]) for key, value in ema.state_dict().items()
    )
    online = build_model(checkpoint["config"], device)
    online.load_state_dict(checkpoint["model"], strict=True)
    online.eval()
    squared_difference = sum(
        float((checkpoint["ema"][k].double() - v.double()).square().sum())
        for k, v in checkpoint["model"].items()
    )
    squared_online = sum(float(v.double().square().sum()) for v in checkpoint["model"].values())
    diffusion = build_diffusion(checkpoint["config"])
    assert diffusion.noise_type == "gaussian" and diffusion.timesteps == 1000
    betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
    abar = np.cumprod(1 - betas)
    previous = np.r_[1.0, abar[:-1]]
    small = betas * (1 - previous) / (1 - abar)
    assert np.array_equal(diffusion.betas, betas)
    assert np.allclose(diffusion.posterior_variance, small, atol=1e-15)
    assert np.allclose(np.exp(diffusion.model_log_variance), np.r_[small[1], betas[1:]], atol=1e-15)
    source_equivalence = {}
    with zipfile.ZipFile(checkpoint_path.parent / "source_snapshot.zip") as archive:
        import dacn

        root = Path(dacn.__file__).parent
        for relative in ("models/diffusion/unet.py", "data/shards.py", "data/bmad.py"):
            source_equivalence[relative] = (
                archive.read(f"dacn/{relative}") == (root / relative).read_bytes()
            )
    data = ShardDataset(validation_dir, "val", include_masks=True)
    normal = np.flatnonzero(data.labels == 0)
    selected = np.sort(np.random.default_rng(17).choice(normal, size=4, replace=False))
    train_data = ShardDataset(validation_dir.parent / "train", "train", include_masks=False)

    def valid_input(image):
        assert image.shape == (1, 256, 256) and image.dtype == torch.float32
        assert image.min() >= -1 and image.max() <= 1
        assert torch.all(image[:, :8] == -1) and torch.all(image[:, -8:] == -1)
        assert torch.all(image[:, :, :8] == -1) and torch.all(image[:, :, -8:] == -1)
        assert float(((image + 1) * 127.5 - ((image + 1) * 127.5).round()).abs().max()) < 1e-4

    for i in range(4):
        valid_input(train_data[i][0])
    summary = {
        "split": "validation_normal_only",
        "held_out_test_used": False,
        "normal_samples": 4,
        "selection": "seed17 uniform without replacement among39 normal validation slices",
        "checkpoint_epoch": 30,
        "ema_exactly_loaded": True,
        "online_ema_relative_l2": (squared_difference / max(squared_online, 1e-30)) ** 0.5,
        "source_files_identical_to_training": source_equivalence,
        "linear_schedule_and_variance_verified": True,
        "input_contract_checks_passed": True,
        "timesteps_one_based": [100, 300],
        "seed": 17,
        "batch_size": 1,
        "scope": "one-step clean prediction; matching-noise large/small/mean reverse; no anomaly AP experiment",
        "metrics": {},
        "math_checks": {},
        "partition": {"diagnostic_normal": {"samples": 4}},
    }
    records = {}
    saved = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for count, index in enumerate(selected):
        image, mask, _ = data[int(index)]
        valid_input(image)
        assert not mask.any()
        image = image[None].to(device)
        foreground = image[0, 0, 8:-8, 8:-8].cpu().numpy() > -0.95
        interior = distance_transform_edt(foreground) > 5
        x = image[0, 0].cpu().numpy()
        seed = derive_sampling_seed(17, data.sample_ids[int(index)])
        for steps in (100, 300):
            k = steps - 1
            t = torch.tensor([k], device=device)
            with _fixed_rng(seed, device):
                noise = torch.randn_like(image)
            noisy = diffusion.q_sample(image, t, noise)
            oracle = noisy / np.sqrt(abar[k]) - np.sqrt((1 - abar[k]) / abar[k]) * noise
            oracle_error = float((oracle - image).abs().max())
            assert oracle_error < 3e-6
            with torch.inference_mode():
                eps_ema = ema(noisy, t)
                eps_online = online(noisy, t)
                single_ema = diffusion._predict_clean(noisy, t, eps_ema)
                single_online = diffusion._predict_clean(noisy, t, eps_online)
                mean, _ = diffusion._p_mean_variance(ema, noisy, t)
                direct = (betas[k] * np.sqrt(previous[k]) / (1 - abar[k])) * single_ema + (
                    (1 - previous[k]) * np.sqrt(1 - betas[k]) / (1 - abar[k])
                ) * noisy
                mu_error = float((mean - direct).abs().max())
                assert mu_error < 3e-6
            outputs = {
                "noisy_input": noisy.clamp(-1, 1),
                "single_ema": single_ema,
                "single_online": single_online,
            }
            for mode in ("large", "small", "mean"):
                outputs[f"chain_{mode}"] = controlled_chain(
                    ema, image, diffusion=diffusion, seed=seed, steps=steps, variance=mode
                )
            summary["math_checks"][str(steps)] = {
                "last_oracle_max_error": oracle_error,
                "last_mu_formula_max_error": mu_error,
                "noise_error_amplification_to_x0": float(np.sqrt((1 - abar[k]) / abar[k])),
            }
            for name, value in outputs.items():
                y = value[0, 0].detach().cpu().numpy()
                paired = paired_score_maps(x, y)
                mae = paired["pixel_difference"] / 2
                sq = ((x[8:-8, 8:-8] - y[8:-8, 8:-8]) / 2) ** 2
                for region, roi in (("foreground", foreground), ("interior", interior)):
                    records.setdefault(f"t{steps}/{name}/{region}", []).append(
                        {
                            "mae": float(mae[roi].mean()),
                            "rmse": float(np.sqrt(sq[roi].mean())),
                            "ssim": float((1 - paired["local_ssim"])[roi].mean()),
                        }
                    )
                saved.setdefault(f"t{steps}_{name}", []).append(y)
            summary["math_checks"][str(steps)]["last_ema_noise_mse_fg"] = float(
                (eps_ema - noise)
                .square()[0, 0, 8:-8, 8:-8][torch.from_numpy(foreground).to(device)]
                .mean()
            )
            print(f"contract audit sample={count + 1}/4 t={steps}", flush=True)
        saved.setdefault("images", []).append(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    summary["inference_seconds"] = time.perf_counter() - started
    summary["peak_vram_bytes"] = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    for key, values in records.items():
        summary["metrics"][key] = {
            metric: float(np.mean([r[metric] for r in values]))
            for metric in ("mae", "rmse", "ssim")
        }
    output_dir.mkdir(parents=True)
    np.savez_compressed(
        output_dir / "private_comparison.npz", **{k: np.stack(v) for k, v in saved.items()}
    )
    (output_dir / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary

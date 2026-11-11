import csv

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from dacn.models.conv_autoencoder import ConvAutoencoder, loss_terms
from dacn.training import conv_autoencoder as trainer


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_matched_backbone_latent256_and_vae_sampling():
    torch.manual_seed(17)
    ae = ConvAutoencoder("conv_ae", width=8)
    torch.manual_seed(17)
    vae = ConvAutoencoder("conv_vae", width=8)
    for name, value in ae.state_dict().items():
        assert torch.equal(value, vae.state_dict()[name])
    x = torch.randn(2, 1, 256, 256).clamp(-1, 1)
    reconstructed, mean, logvar = ae(x)
    assert reconstructed.shape == x.shape and mean.shape == logvar.shape == (2, 256)
    assert reconstructed.abs().max() <= 1
    assert torch.equal(ae(x)[0], reconstructed)
    assert not torch.equal(vae(x)[0], vae(x)[0])
    vae.eval()
    assert torch.equal(vae(x)[0], vae(x)[0])
    assert sum(p.numel() for p in vae.parameters()) > sum(p.numel() for p in ae.parameters())


def test_vae_objective_has_analytic_kl_scale_and_ae_no_kl():
    x = torch.zeros(2, 1, 256, 256)
    mean = torch.ones(2, 256)
    logvar = torch.zeros_like(mean)
    objective, mse, kl = loss_terms(torch.ones_like(x), x, mean, logvar, model_type="conv_vae")
    assert float(mse) == 1 and float(kl) == 128
    assert float(objective) == pytest.approx(1 + 128 / 57600)
    ae_loss, _, ae_kl = loss_terms(torch.ones_like(x), x, mean, logvar, model_type="conv_ae")
    assert float(ae_loss) == 1 and float(ae_kl) == 0
    border = torch.ones_like(x)
    border[:, :, 8:-8, 8:-8] = 0
    assert float(loss_terms(border, x, mean, logvar, model_type="conv_ae")[0]) == 0


def data_fixture(root, split, labels):
    folder = root / split
    folder.mkdir(parents=True)
    rng = np.random.default_rng(3)
    images = rng.uniform(-0.7, 0.7, size=(len(labels), 1, 256, 256)).astype(np.float32)
    masks = np.stack([np.full((256, 256), v, dtype=np.uint8) for v in labels])
    np.savez_compressed(
        folder / "shard-00000.npz",
        images=images,
        masks=masks,
        labels=np.array(labels, dtype=np.uint8),
    )
    with (folder / "index.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "patient_group_id", "split", "label", "shard", "offset"])
        for i, label in enumerate(labels):
            writer.writerow(
                [
                    f"{split}-{i}",
                    f"group-{split}-{i}",
                    split,
                    label,
                    f"data/{split}/shard-00000.npz",
                    i,
                ]
            )
    return folder


@pytest.mark.parametrize("model_type", ["conv_ae", "conv_vae"])
def test_training_normal_only_checkpoint_and_exact_resume(tmp_path, model_type):
    train = data_fixture(tmp_path / "data", "train", [0, 0, 0, 0])
    val = data_fixture(tmp_path / "data", "val", [0, 1, 0, 1])
    config = {"model_type": model_type, "width": 8, "epochs": 2, "batch_size": 2}
    kwargs = {"train_dir": train, "validation_dir": val, "config": config, "device": "cpu"}
    full = trainer.train(**kwargs, output_dir=tmp_path / "full")
    assert full["epochs_completed"] == 2 and full["normal_validation_samples"] == 2
    assert full["optimizer_steps"] == 4 and full["latent_dim"] == 256
    assert full["early_stopping"] is False and full["test_used"] is False

    def interrupt(progress):
        if progress["epoch"] == 1:
            raise RuntimeError("intentional epoch-boundary interruption")

    interrupted = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="intentional"):
        trainer.train(**kwargs, output_dir=interrupted, on_checkpoint=interrupt)
    checkpoint = interrupted / "autoencoder_last.pt"
    resumed = trainer.train(**kwargs, output_dir=interrupted, resume_from_checkpoint=checkpoint)
    assert resumed["epochs_completed"] == 2 and resumed["optimizer_steps"] == 4
    a = torch.load(tmp_path / "full/autoencoder_last.pt", weights_only=False)
    b = torch.load(checkpoint, weights_only=False)
    for name, value in a["model"].items():
        assert torch.equal(value, b["model"][name])
    with pytest.raises(ValueError, match="config/trainer differs"):
        trainer.train(
            **{**kwargs, "config": {**config, "learning_rate": 2e-4}},
            output_dir=interrupted,
            resume_from_checkpoint=checkpoint,
        )


def test_anomalous_train_rejected_before_model(tmp_path, monkeypatch):
    train = data_fixture(tmp_path / "data", "train", [0, 1])
    val = data_fixture(tmp_path / "data", "val", [0, 1])
    monkeypatch.setattr(
        trainer, "ConvAutoencoder", lambda *_a, **_kw: pytest.fail("Model before data gate")
    )
    with pytest.raises(ValueError, match="normal-only"):
        trainer.train(
            train_dir=train,
            validation_dir=val,
            output_dir=tmp_path / "out",
            config={},
            device="cpu",
        )

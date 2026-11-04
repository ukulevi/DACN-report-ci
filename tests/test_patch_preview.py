import json
import uuid

import numpy as np
import torch

from dacn.evaluation import patch_preview


def test_diagnostic_selects_both_classes_without_passing_lesion_masks(tmp_path, monkeypatch):
    class Data:
        labels = np.array([0, 0, 1, 1])
        sample_ids = tuple(str(uuid.UUID(int=i)) for i in range(4))

        def __getitem__(self, i):
            image = torch.zeros(1, 256, 256)
            mask = torch.zeros(256, 256, dtype=torch.uint8)
            if self.labels[i]:
                mask[100:120, 100:120] = 1
            return image, mask, torch.tensor(self.labels[i])

    calls = []

    def reconstruct(model, diffusion, image, *, steps, seed):
        calls.append((image.shape, steps))
        return image + 0.2

    monkeypatch.setattr(patch_preview, "ShardDataset", lambda *a, **kw: Data())
    monkeypatch.setattr(patch_preview, "load_context_checkpoint", lambda *a: (None, None))
    monkeypatch.setattr(patch_preview, "sliding_reconstruct", reconstruct)
    output = tmp_path / "output"
    result = patch_preview.evaluate_patch_preview(
        validation_dir=tmp_path,
        checkpoint_path=tmp_path / "weights.pt",
        output_dir=output,
        device=torch.device("cpu"),
    )
    assert calls == [(torch.Size([1, 1, 256, 256]), 100)] * 4
    assert result["held_out_test_used"] is False
    records = result["metrics"]["per_example"]
    assert [r["label"] for r in records] == [0, 0, 1, 1]
    assert records[0]["lesion_mae_01"] is None
    assert abs(records[2]["lesion_mae_01"] - 0.1) < 1e-6
    with np.load(output / "private_reconstructions.npz") as saved:
        assert saved["masks"].shape == (4, 256, 256)
        assert saved["difference"].shape == (4, 240, 240)
    assert json.loads((output / "results.json").read_text())["checkpoint_epoch"] == 5

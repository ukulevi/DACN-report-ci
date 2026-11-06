import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dacn.evaluation.rad_dino import prepare_pair, token_distances


def test_processor_gets_full_unpadded_rgb_and_same_quantization():
    image = np.full((256, 256), -1, dtype=np.float32)
    image[8:-8, 8:-8] = 0
    image[8, 8] = 1
    observed = []

    def processor(*, images, return_tensors):
        observed.extend(np.asarray(v) for v in images)
        assert return_tensors == "pt"
        return {"pixel_values": torch.zeros(2, 3, 518, 518)}

    prepare_pair(processor, image, image)
    assert observed[0].shape == (240, 240, 3)
    assert (observed[0][0, 0] == 255).all()
    assert (observed[0][100, 100] == 128).all()
    np.testing.assert_array_equal(observed[0], observed[1])
    with pytest.raises(ValueError, match="finite"):
        prepare_pair(processor, image[:10], image)


def test_spatial_distance_excludes_cls_and_retains_patch_coordinates():
    features = torch.zeros(2, 1370, 768)
    features[:, 1:, 0] = 1
    features[1, 0] = 100  # CLS cannot enter spatial scores.
    states = [features] * 13
    assert np.max(token_distances(states, torch.nn.Identity())) == 0
    # Opposite unit vector at top-left patch gives squared distance4/768.
    features[1, 1, 0] = -1
    distances = token_distances(states, torch.nn.Identity())
    assert distances.shape == (4, 240, 240)
    np.testing.assert_allclose(distances[:, 0, 0], 4 / 768, rtol=1e-6)
    assert (distances[:, -1, -1] == 0).all()
    assert (distances >= 0).all()


def test_selected_taps_are_fixed_and_channel_scale_invariant():
    states = [torch.zeros(2, 1370, 768) for _ in range(13)]
    for i in (3, 6, 9, 12):
        states[i][0, 1:, 0] = 2
        states[i][1, 1:, 0] = 5
    values = token_distances(states, torch.nn.Identity())
    np.testing.assert_allclose(values, 0, atol=1e-12)
    with pytest.raises(ValueError, match="patch tokens"):
        token_distances([torch.zeros(2, 10, 768)] * 13, torch.nn.Identity())

"""Four-scale ImageNet ResNet-18 conditioning with zero-initialized spatial fusion."""

import hashlib
from pathlib import Path

import torch
from torch import nn

from dacn.models.diffusion.feature_condition import MRIResNet18

METHOD = "ddpm_spatial_resnet18_imagenet_v1"
WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
WEIGHTS_PATH = "/opt/dacn/pretrained/resnet18-f37072fd.pth"
# Full SHA256 is checked at training startup; the official filename embeds its prefix.
WEIGHTS_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


class SpatialResNet18(MRIResNet18):
    def __init__(self):
        super().__init__()
        del self.pool

    def load_imagenet(self, path=WEIGHTS_PATH):
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != WEIGHTS_SHA256:
            raise ValueError("ImageNet ResNet18 weights checksum differs")
        pretrained = torch.load(path, map_location="cpu", weights_only=True)
        remapped = {}
        for key, value in pretrained.items():
            if key.startswith("fc."):
                continue
            if key.startswith("conv1."):
                target = key.replace("conv1.", "stem.0.", 1)
                value = value.mean(dim=1, keepdim=True)
            elif key.startswith("bn1."):
                target = key.replace("bn1.", "stem.1.", 1)
            elif key.startswith("layer") and key[5].isdigit():
                target = f"layers.{int(key[5]) - 1}." + key[7:]
            else:
                raise ValueError("Unexpected official ImageNet parameter")
            remapped[target] = value
        self.load_state_dict(remapped, strict=True)

    def forward(self, original):
        if original.ndim != 4 or original.shape[1:] != (1, 256, 256):
            raise ValueError("Spatial encoder expects grayscale256 input")
        h = self.stem(original)
        maps = []
        for stage in self.layers:
            h = stage(h)
            maps.append(h)
        return tuple(maps)


class SpatialConditionedDDPM(nn.Module):
    def __init__(self, unet, *, pretrained=False, weights_path=WEIGHTS_PATH):
        super().__init__()
        self.unet = unet
        self.encoder = SpatialResNet18()
        if pretrained:
            self.encoder.load_imagenet(weights_path)
        self.fusion_layers = nn.ModuleDict()
        self.fusion_indices = []
        for size, incoming in ((64, 64), (32, 128), (16, 256), (8, 512)):
            level = unet.image_size.bit_length() - size.bit_length()
            if level < 0 or level >= len(unet.channel_mult):
                raise ValueError("U-Net must have all four spatial scales")
            index = level * (unet.num_res_blocks + 1) + unet.num_res_blocks
            outgoing = unet.model_channels * unet.channel_mult[level]
            fusion = nn.Conv2d(incoming, outgoing, 1)
            nn.init.zeros_(fusion.weight)
            nn.init.zeros_(fusion.bias)
            self.fusion_layers[str(index)] = fusion
            self.fusion_indices.append(index)

    def encode(self, original):
        return self.encoder(original)

    def forward(self, noisy, timesteps, condition):
        if len(condition) != 4:
            raise ValueError("Expected four spatial feature maps")
        residuals = {
            index: self.fusion_layers[str(index)](feature)
            for index, feature in zip(self.fusion_indices, condition, strict=True)
        }
        return self.unet(noisy, timesteps, spatial_residuals=residuals)

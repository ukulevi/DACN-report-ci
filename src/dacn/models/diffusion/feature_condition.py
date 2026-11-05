"""Random ResNet-18 global feature conditioning, with zero initial injection."""

import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outgoing)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outgoing)
        self.downsample = (
            nn.Identity()
            if incoming == outgoing and stride == 1
            else nn.Sequential(
                nn.Conv2d(incoming, outgoing, 1, stride, bias=False), nn.BatchNorm2d(outgoing)
            )
        )

    def forward(self, x):
        identity = self.downsample(x)
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + identity)


class MRIResNet18(nn.Module):
    """Standard [2,2,2,2] BasicBlock ResNet-18; grayscale stem, no classifier.

    Implemented directly in torch; no external weights or torchvision downloads.
    Adaptive pooling yields512 features. BatchNorm buffers are part of checkpoints.
    """

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layers = nn.Sequential(
            self.stage(64, 64, 1),
            self.stage(64, 128, 2),
            self.stage(128, 256, 2),
            self.stage(256, 512, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    @staticmethod
    def stage(incoming, outgoing, stride):
        return nn.Sequential(BasicBlock(incoming, outgoing, stride), BasicBlock(outgoing, outgoing))

    def forward(self, image):
        return self.pool(self.layers(self.stem(image))).flatten(1)


class FeatureConditionedDDPM(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.encoder = MRIResNet18()
        self.feature_projection = nn.Sequential(nn.Linear(512, 128), nn.SiLU())
        embedding_dim = unet.time_embedding[-1].out_features
        self.condition_projection = nn.Linear(128, embedding_dim)
        nn.init.zeros_(self.condition_projection.weight)
        nn.init.zeros_(self.condition_projection.bias)

    def encode(self, original):
        return self.feature_projection(self.encoder(original))

    def forward(self, noisy, timesteps, condition):
        return self.unet(noisy, timesteps, condition_embedding=self.condition_projection(condition))


class BoundCondition(nn.Module):
    """Bind one clean-image feature vector for all reverse steps; accepts no masks."""

    def __init__(self, model, condition):
        super().__init__()
        self.model, self.condition = model, condition

    def forward(self, noisy, timesteps):
        return self.model(noisy, timesteps, self.condition)


@torch.no_grad()
def update_feature_ema(target, source, decay):
    source_parameters = dict(source.named_parameters())
    for name, value in target.named_parameters():
        value.mul_(decay).add_(source_parameters[name], alpha=1 - decay)
    # EMA helper used by the original U-Net only averages parameters. ResNet also
    # requires valid BatchNorm statistics/counters, copied from the online model.
    source_buffers = dict(source.named_buffers())
    for name, value in target.named_buffers():
        value.copy_(source_buffers[name])

"""
ResNet encoder for Monodepth2.

Wraps torchvision ResNet as a multi-scale feature extractor.
Supports multi-image input (for the pose encoder) by replicating conv1 weights.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class ResNetMultiImageInput(models.ResNet):
    """ResNet variant that accepts multiple concatenated images as input."""

    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super().__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Construct a ResNet model that accepts multiple images concatenated along channels.

    Args:
        num_layers: 18 or 50
        pretrained: whether to load ImageNet-pretrained weights
        num_input_images: number of images concatenated channel-wise
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).state_dict() \
            if num_layers == 18 else \
            models.resnet50(weights=models.ResNet50_Weights.DEFAULT).state_dict()
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded, strict=False)
    return model


class ResnetEncoder(nn.Module):
    """Encoder based on a ResNet backbone.

    Produces 5 feature maps at progressively lower spatial resolutions:
        [H/2, H/4, H/8, H/16, H/32]

    Args:
        num_layers: ResNet variant (18, 34, 50, 101, 152)
        pretrained: load ImageNet weights
        num_input_images: number of images stacked channel-wise (>1 for pose encoder)
    """

    def __init__(self, num_layers, pretrained, num_input_images=1):
        super().__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152,
        }

        if num_layers not in resnets:
            raise ValueError(f"{num_layers} is not a valid number of resnet layers")

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            if pretrained:
                weights_map = {
                    18: models.ResNet18_Weights.DEFAULT,
                    34: models.ResNet34_Weights.DEFAULT,
                    50: models.ResNet50_Weights.DEFAULT,
                    101: models.ResNet101_Weights.DEFAULT,
                    152: models.ResNet152_Weights.DEFAULT,
                }
                self.encoder = resnets[num_layers](weights=weights_map[num_layers])
            else:
                self.encoder = resnets[num_layers](weights=None)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        """Extract multi-scale features.

        Args:
            input_image: (B, C, H, W) where C = 3 * num_input_images

        Returns:
            List of 5 feature tensors at decreasing spatial resolution.
        """
        features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        features.append(self.encoder.relu(x))
        features.append(self.encoder.layer1(self.encoder.maxpool(features[-1])))
        features.append(self.encoder.layer2(features[-1]))
        features.append(self.encoder.layer3(features[-1]))
        features.append(self.encoder.layer4(features[-1]))

        return features

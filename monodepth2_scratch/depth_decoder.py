"""
Depth decoder for Monodepth2.

U-Net style decoder with skip connections from the encoder.
Produces disparity maps at multiple scales (0-3).
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from monodepth2_scratch.layers import ConvBlock, Conv3x3, upsample


class DepthDecoder(nn.Module):
    """Multi-scale depth decoder with skip connections.

    Takes encoder features and produces sigmoid disparity maps at `scales`.

    Architecture per decoder level i (4 → 0):
        upconv_0: ConvBlock(ch_in, ch_dec[i])
        upsample 2×
        concatenate with encoder skip connection (if i > 0)
        upconv_1: ConvBlock(ch_in + skip, ch_dec[i])
        dispconv at selected scales: Conv3x3 → sigmoid
    """

    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        """
        Args:
            input_features: list of 5 encoder feature tensors

        Returns:
            dict with keys ("disp", scale) → sigmoid disparity at each scale
        """
        outputs = {}

        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return outputs

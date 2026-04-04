"""
Pose decoder for Monodepth2.

Predicts 6-DoF relative camera pose (axis-angle rotation + translation)
from encoder features.
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict


class PoseDecoder(nn.Module):
    """Predict relative camera pose from encoder features.

    Architecture:
        1. Squeeze last encoder features to 256 channels via 1×1 conv
        2. Concatenate squeezed features from all input frames
        3. Three conv layers: 3×3, 3×3, 1×1 → 6 × num_frames output
        4. Global average pooling → scale by 0.01

    Output: (axisangle, translation), each (B, num_frames, 1, 3)
    """

    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super().__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = max(1, num_input_features - 1)
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs["squeeze"] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()
        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        """
        Args:
            input_features: list of feature lists, one per input frame.
                Each feature list has 5 tensors from the encoder.

        Returns:
            axisangle: (B, num_frames, 1, 3) rotation as axis-angle
            translation: (B, num_frames, 1, 3) translation vector
        """
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)
        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation

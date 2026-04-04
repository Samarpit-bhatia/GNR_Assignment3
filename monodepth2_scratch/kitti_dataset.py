"""
KITTI monocular dataset for Monodepth2 self-supervised training.

Loads triplets of consecutive frames (t-1, t, t+1) from KITTI raw sequences
along with multi-scale versions and camera intrinsics.
"""

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class KITTIDataset(data.Dataset):
    """KITTI monocular training dataset.

    Each sample provides:
        - ("color", frame_id, scale): RGB image at given temporal offset and scale
        - ("color_aug", frame_id, 0): Color-augmented image (training only)
        - ("K", scale), ("inv_K", scale): Camera intrinsics at each scale
    """

    def __init__(self, data_path, filenames, height, width, frame_ids, num_scales,
                 is_train=True, img_ext='.jpg'):
        super().__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.frame_ids = frame_ids
        self.num_scales = num_scales
        self.is_train = is_train
        self.img_ext = img_ext

        self.interp = Image.LANCZOS
        self.loader = self._pil_loader

        # Augmentation transforms
        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize(
                (self.height // s, self.width // s), interpolation=self.interp)

        self.to_tensor = transforms.ToTensor()

        # KITTI camera intrinsics (normalised to image dimensions)
        # These are approximate values from KITTI that work across most sequences
        self.K = np.array([
            [0.58, 0, 0.5, 0],
            [0, 1.92, 0.5, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=np.float32)

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def _pil_loader(path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def _get_image_path(self, folder, frame_index, side):
        """Construct the path to a KITTI image."""
        f_str = f"{frame_index:010d}{self.img_ext}"
        image_path = os.path.join(
            self.data_path,
            folder,
            f"image_0{side}/data",
            f_str)
        return image_path

    def _get_color(self, folder, frame_index, side, do_flip):
        """Load an RGB image from disk."""
        path = self._get_image_path(folder, frame_index, side)
        color = self.loader(path)
        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)
        return color

    def __getitem__(self, index):
        """Return a single training sample.

        Each line in the split file has format:
            <folder> <frame_index> <side>
        where side is 'l' (left=image_02) or 'r' (right=image_03)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
            side = {"l": 2, "r": 3}[line[2]]
        else:
            frame_index = 0
            side = None

        # Load target and source frames
        for i in self.frame_ids:
            if i == "s":
                other_side = {"l": "r", "r": "l"}[line[2]]
                other_side_num = {"l": 2, "r": 3}[other_side]
                inputs[("color", i, -1)] = self._get_color(
                    folder, frame_index, other_side_num, do_flip)
            else:
                try:
                    inputs[("color", i, -1)] = self._get_color(
                        folder, frame_index + i, side, do_flip)
                except FileNotFoundError:
                    # If adjacent frame doesn't exist, use the current frame
                    inputs[("color", i, -1)] = self._get_color(
                        folder, frame_index, side, do_flip)

        # Color augmentation
        if do_color_aug:
            color_aug = transforms.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue)
        else:
            color_aug = lambda x: x

        # Resize to multi-scale and convert to tensors
        for scale in range(self.num_scales):
            for frame_id in self.frame_ids:
                color = inputs[("color", frame_id, -1)]
                color = self.resize[scale](color)
                inputs[("color", frame_id, scale)] = self.to_tensor(color)

        # Augmented images (only at scale 0, used for encoder input)
        for frame_id in self.frame_ids:
            color = inputs[("color", frame_id, -1)]
            color = self.resize[0](color)
            inputs[("color_aug", frame_id, 0)] = self.to_tensor(color_aug(color))

        # Remove raw PIL images
        for frame_id in self.frame_ids:
            del inputs[("color", frame_id, -1)]

        # Camera intrinsics at each scale
        K = self.K.copy()
        K[0, :] *= self.width
        K[1, :] *= self.height

        for scale in range(self.num_scales):
            Ks = K.copy()
            Ks[0, :] /= (2 ** scale)
            Ks[1, :] /= (2 ** scale)
            inv_Ks = np.linalg.pinv(Ks)

            inputs[("K", scale)] = torch.from_numpy(Ks)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_Ks)

        if do_flip:
            # Adjust principal point for horizontal flip
            for scale in range(self.num_scales):
                K_flip = inputs[("K", scale)].clone()
                K_flip[0, 2] = (self.width // (2 ** scale)) - K_flip[0, 2]
                inputs[("K", scale)] = K_flip
                inputs[("inv_K", scale)] = torch.from_numpy(
                    np.linalg.pinv(K_flip.numpy()))

        return inputs

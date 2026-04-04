"""
Self-supervised monocular depth trainer for Monodepth2.

Implements the full training pipeline:
    - Depth prediction via encoder-decoder
    - Pose prediction via separate encoder + pose decoder
    - Image warping via differentiable reprojection
    - Photometric loss (0.85 × SSIM + 0.15 × L1)
    - Minimum reprojection loss across source frames
    - Auto-masking of static pixels
    - Multi-scale edge-aware smoothness
"""

from __future__ import absolute_import, division, print_function

import os
import time
import json
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from monodepth2_scratch.resnet_encoder import ResnetEncoder
from monodepth2_scratch.depth_decoder import DepthDecoder
from monodepth2_scratch.pose_decoder import PoseDecoder
from monodepth2_scratch.kitti_dataset import KITTIDataset
from monodepth2_scratch.layers import (
    SSIM, BackprojectDepth, Project3D,
    disp_to_depth, transformation_from_parameters,
    get_smooth_loss, compute_depth_errors,
)
from monodepth2_scratch.utils import sec_to_hm_str, readlines


class Trainer:
    """Full self-supervised training loop for Monodepth2."""

    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        # Device selection: CUDA > MPS > CPU
        if not self.opt.no_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif not self.opt.no_cuda and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        # ----- Build models -----

        # Depth encoder + decoder
        self.models["encoder"] = ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        # Pose encoder + decoder (separate ResNet)
        self.models["pose_encoder"] = ResnetEncoder(
            self.opt.num_layers,
            self.opt.weights_init == "pretrained",
            num_input_images=self.num_pose_frames)
        self.models["pose_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["pose_encoder"].parameters())

        self.models["pose"] = PoseDecoder(
            self.models["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2)
        self.models["pose"].to(self.device)
        self.parameters_to_train += list(self.models["pose"].parameters())

        # ----- Optimiser -----
        self.model_optimizer = optim.Adam(
            self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        print(f"Training model: {self.opt.model_name}")
        print(f"Saving to: {self.opt.log_dir}")
        print(f"Device: {self.device}")

        # ----- Data -----
        fpath = os.path.join(self.opt.split_path, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = KITTIDataset(
            self.opt.data_path, train_filenames,
            self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, shuffle=True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        val_dataset = KITTIDataset(
            self.opt.data_path, val_filenames,
            self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, shuffle=True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        # ----- SSIM loss -----
        self.ssim = SSIM()
        self.ssim.to(self.device)

        # ----- Geometry layers -----
        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms",
            "da/a1", "da/a2", "da/a3"]

        print(f"Using split: {self.opt.split_path}")
        print(f"Training items: {len(train_dataset)}, Validation items: {len(val_dataset)}")

        self.save_opts()

    def save_opts(self):
        """Save training options to JSON."""
        os.makedirs(self.log_path, exist_ok=True)
        opts_dict = vars(self.opt)
        with open(os.path.join(self.log_path, "opt.json"), 'w') as f:
            json.dump(opts_dict, f, indent=2, default=str)

    def set_train(self):
        for m in self.models.values():
            m.train()

    def set_eval(self):
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the full training pipeline."""
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training."""
        print(f"\n=== Epoch {self.epoch + 1}/{self.opt.num_epochs} ===")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            # Optional early stop for debugging
            if hasattr(self.opt, 'max_batches') and self.opt.max_batches > 0:
                if batch_idx >= self.opt.max_batches:
                    break

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # Log periodically
            if batch_idx % self.opt.log_frequency == 0:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

            self.step += 1

        self.model_lr_scheduler.step()

    def process_batch(self, inputs):
        """Forward pass through depth and pose networks, then compute losses."""
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        # Depth prediction
        features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs = self.models["depth"](features)

        # Pose prediction
        outputs.update(self.predict_poses(inputs, features))

        # Image warping
        self.generate_images_pred(inputs, outputs)

        # Loss
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict relative poses between the target and each source frame."""
        outputs = {}

        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

        for f_i in self.opt.frame_ids[1:]:
            # Always pass frames in temporal order
            if f_i < 0:
                pose_inputs = [pose_feats[f_i], pose_feats[0]]
            else:
                pose_inputs = [pose_feats[0], pose_feats[f_i]]

            pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

            axisangle, translation = self.models["pose"](pose_inputs)
            outputs[("axisangle", 0, f_i)] = axisangle
            outputs[("translation", 0, f_i)] = translation

            # Build 4×4 transformation matrix
            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        return outputs

    def generate_images_pred(self, inputs, outputs):
        """Warp source images into the target frame using predicted depth and poses."""
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]

            # Full-resolution multi-scale: upsample disparity to full resolution
            disp = F.interpolate(
                disp, [self.opt.height, self.opt.width],
                mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="zeros",
                    align_corners=True)

                # Identity images for auto-masking
                outputs[("color_identity", frame_id, scale)] = \
                    inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Photometric reprojection loss: 0.85 × SSIM + 0.15 × L1."""
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute all losses for a minibatch.

        Key components:
            1. Minimum reprojection loss (key paper contribution)
            2. Auto-masking of stationary pixels
            3. Edge-aware smoothness loss
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            # Reprojection loss for each source frame
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            # ----- Auto-masking -----
            # Compute identity reprojection losses (what you'd get with no motion)
            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            # Keep both for per-pixel minimum below
            identity_reprojection_loss = identity_reprojection_losses

            # Add small random noise to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape,
                device=self.device) * 0.00001

            # ----- Minimum reprojection -----
            # Concatenate identity and warped reprojection losses
            combined = torch.cat(
                (identity_reprojection_loss, reprojection_losses), dim=1)

            # Per-pixel minimum across all candidates
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            loss += to_optimise.mean()

            # ----- Edge-aware smoothness -----
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses[f"loss/{scale}"] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def log_time(self, batch_idx, duration, loss):
        """Print training progress."""
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print(f"  epoch {self.epoch + 1:>3} | batch {batch_idx:>6} | "
              f"examples/s: {samples_per_sec:5.1f} | loss: {loss:.5f} | "
              f"elapsed: {sec_to_hm_str(time_sofar)} | "
              f"remaining: {sec_to_hm_str(training_time_left)}")

    def save_model(self):
        """Save all model weights."""
        save_folder = os.path.join(self.log_path, f"weights_epoch_{self.epoch + 1}")
        os.makedirs(save_folder, exist_ok=True)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, f"{model_name}.pth")
            to_save = model.state_dict()
            if model_name == "encoder":
                to_save["height"] = self.opt.height
                to_save["width"] = self.opt.width
            torch.save(to_save, save_path)

        # Save optimizer
        save_path = os.path.join(save_folder, "adam.pth")
        torch.save(self.model_optimizer.state_dict(), save_path)

        print(f"  → Saved model to {save_folder}")

    def load_model(self, weights_folder):
        """Load model weights from disk."""
        print(f"Loading model from {weights_folder}")
        for model_name in self.models:
            path = os.path.join(weights_folder, f"{model_name}.pth")
            if os.path.isfile(path):
                model_dict = self.models[model_name].state_dict()
                pretrained_dict = torch.load(path, map_location=self.device)
                pretrained_dict = {k: v for k, v in pretrained_dict.items()
                                   if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[model_name].load_state_dict(model_dict)
                print(f"  Loaded {model_name}")

"""
Core layers for Monodepth2 from-scratch implementation.
Implements geometric projection, loss functions, and basic conv building blocks.

Reference: Godard et al., "Digging into Self-Supervised Monocular Depth Estimation", ICCV 2019
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Basic conv building blocks
# ---------------------------------------------------------------------------

class Conv3x3(nn.Module):
    """3x3 convolution with reflection padding."""

    def __init__(self, in_channels, out_channels, use_refl=True):
        super().__init__()
        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        return self.conv(self.pad(x))


class ConvBlock(nn.Module):
    """Conv3x3 followed by ELU activation."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        return self.nonlin(self.conv(x))


# ---------------------------------------------------------------------------
# Disparity ↔ Depth conversion
# ---------------------------------------------------------------------------

def disp_to_depth(disp, min_depth, max_depth):
    """Convert sigmoid disparity output to depth.

    Uses the formula from Section 3.4 of the paper:
        scaled_disp = 1/max_depth + (1/min_depth - 1/max_depth) * disp
        depth = 1 / scaled_disp
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


# ---------------------------------------------------------------------------
# Rotation / transformation helpers
# ---------------------------------------------------------------------------

def rot_from_axisangle(vec):
    """Convert axis-angle vector to a 4×4 rotation matrix.

    Args:
        vec: (B, 1, 3) axis-angle vector

    Returns:
        (B, 4, 4) rotation matrix
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4), device=vec.device)
    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


def get_translation_matrix(translation_vector):
    """Convert translation vector (B, 1, 3) → 4×4 transformation matrix."""
    T = torch.zeros(translation_vector.shape[0], 4, 4, device=translation_vector.device)
    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert (axisangle, translation) network output to a 4×4 rigid transform.

    Args:
        axisangle: (B, 1, 3) rotation as axis-angle
        translation: (B, 1, 3) translation vector
        invert: if True, return the inverse transformation

    Returns:
        (B, 4, 4) transformation matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


# ---------------------------------------------------------------------------
# Geometric projection layers
# ---------------------------------------------------------------------------

class BackprojectDepth(nn.Module):
    """Unproject a depth map into a 3D point cloud using camera intrinsics.

    Given a depth map D and inverse intrinsics K^{-1}, computes:
        P = D * K^{-1} * [u, v, 1]^T
    """

    def __init__(self, batch_size, height, width):
        super().__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(
            torch.from_numpy(self.id_coords), requires_grad=False)

        self.ones = nn.Parameter(
            torch.ones(self.batch_size, 1, self.height * self.width),
            requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(
            torch.cat([self.pix_coords, self.ones], 1), requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)
        return cam_points


class Project3D(nn.Module):
    """Project 3D points into a camera with intrinsics K and pose T.

    Returns normalised pixel coordinates in [-1, 1] for use with grid_sample.
    """

    def __init__(self, batch_size, height, width, eps=1e-7):
        super().__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


# ---------------------------------------------------------------------------
# Upsampling helper
# ---------------------------------------------------------------------------

def upsample(x):
    """Upsample input tensor by a factor of 2 (nearest neighbour)."""
    return F.interpolate(x, scale_factor=2, mode="nearest")


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def get_smooth_loss(disp, img):
    """Edge-aware smoothness loss for disparity.

    Penalises disparity gradients, weighted by image gradients
    so that edges in the image allow sharp depth discontinuities.
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(
        torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(
        torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Compute SSIM loss between two images.

    Uses 3×3 average pooling as the windowing function.
    """

    def __init__(self):
        super().__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_depth_errors(gt, pred):
    """Compute standard depth evaluation metrics.

    Returns:
        abs_rel, sq_rel, rmse, rmse_log, a1 (δ<1.25), a2 (δ<1.25²), a3 (δ<1.25³)
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

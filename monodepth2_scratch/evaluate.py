#!/usr/bin/env python3
"""
Evaluate depth predictions on KITTI Eigen test split.

Loads a trained from-scratch Monodepth2 model, predicts depth for each test image,
and computes standard depth metrics (with median scaling).

Usage:
    python3 -m monodepth2_scratch.evaluate \
        --data_path ./kitti_data \
        --load_weights_folder ./logs/monodepth2_scratch/weights_epoch_10
"""

from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from monodepth2_scratch.resnet_encoder import ResnetEncoder
from monodepth2_scratch.depth_decoder import DepthDecoder
from monodepth2_scratch.layers import disp_to_depth, compute_depth_errors
from monodepth2_scratch.utils import readlines


MIN_DEPTH = 1e-3
MAX_DEPTH = 80


def evaluate(opt):
    """Run evaluation and print metrics."""

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Evaluating on {device}")

    # Load model
    encoder = ResnetEncoder(18, False)
    depth_decoder = DepthDecoder(encoder.num_ch_enc)

    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path, map_location=device)
    height = encoder_dict.get("height", 192)
    width = encoder_dict.get("width", 640)

    # Filter out non-model keys
    model_dict = encoder.state_dict()
    encoder_dict = {k: v for k, v in encoder_dict.items() if k in model_dict}
    encoder.load_state_dict(encoder_dict, strict=False)
    encoder.to(device)
    encoder.eval()

    depth_decoder.load_state_dict(
        torch.load(decoder_path, map_location=device), strict=False)
    depth_decoder.to(device)
    depth_decoder.eval()

    # Load test split
    test_filenames = readlines(
        os.path.join(opt.split_path, "test_files.txt"))
    print(f"Evaluating {len(test_filenames)} images")

    # Load GT depths
    gt_path = os.path.join(opt.split_path, "gt_depths.npz")
    if os.path.isfile(gt_path):
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)
        gt_depths = gt_depths["data"]
    else:
        print("No gt_depths.npz found. Computing GT from velodyne data...")
        gt_depths = export_gt_depths(opt.data_path, test_filenames)
        np.savez_compressed(gt_path, data=gt_depths)

    pred_depths = []

    to_tensor = transforms.ToTensor()

    print("Running predictions...")
    with torch.no_grad():
        for idx, line in enumerate(test_filenames):
            parts = line.split()
            folder = parts[0]
            frame_idx = int(parts[1])
            side = {"l": 2, "r": 3}.get(parts[2], 2) if len(parts) == 3 else 2

            img_path = os.path.join(
                opt.data_path, folder,
                f"image_0{side}/data",
                f"{frame_idx:010d}.jpg")

            if not os.path.isfile(img_path):
                img_path = img_path.replace('.jpg', '.png')

            if not os.path.isfile(img_path):
                print(f"  Warning: {img_path} not found, skipping")
                pred_depths.append(np.zeros((375, 1242)))
                continue

            img = Image.open(img_path).convert('RGB')
            original_width, original_height = img.size
            img = img.resize((width, height), Image.LANCZOS)
            img = to_tensor(img).unsqueeze(0).to(device)

            features = encoder(img)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = F.interpolate(
                disp, (original_height, original_width),
                mode="bilinear", align_corners=False)

            _, depth = disp_to_depth(disp_resized, 0.1, 100)
            depth = depth.squeeze().cpu().numpy()

            pred_depths.append(depth)

            if (idx + 1) % 100 == 0:
                print(f"  {idx + 1}/{len(test_filenames)}")

    # Compute metrics
    print("\nComputing metrics...")
    errors = []
    for i in range(len(pred_depths)):
        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]

        gt_height, gt_width = gt_depth.shape[:2]
        pred_depth = np.array(
            Image.fromarray(pred_depth).resize(
                (gt_width, gt_height), Image.NEAREST))

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        # Garg/Eigen crop
        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                         0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape, dtype=bool)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = True
        mask = np.logical_and(mask, crop_mask)

        pred_depth = pred_depth[mask]
        gt_depth_masked = gt_depth[mask]

        if len(pred_depth) == 0:
            continue

        # Median scaling
        ratio = np.median(gt_depth_masked) / np.median(pred_depth)
        pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        err = compute_depth_errors(
            torch.from_numpy(gt_depth_masked).float(),
            torch.from_numpy(pred_depth).float())
        errors.append([e.item() for e in err])

    mean_errors = np.array(errors).mean(0)

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS (Our From-Scratch Implementation)")
    print("=" * 80)
    metrics = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    header = "  ".join([f"{m:>10}" for m in metrics])
    values = "  ".join([f"{e:>10.4f}" for e in mean_errors])
    print(header)
    print(values)
    print("=" * 80)

    # Save results
    results = {m: float(v) for m, v in zip(metrics, mean_errors)}
    results_path = os.path.join(opt.load_weights_folder, "eval_results.npy")
    np.save(results_path, results)
    print(f"\nResults saved to {results_path}")

    return results


def export_gt_depths(data_path, test_filenames):
    """Export ground truth depths from KITTI velodyne data.

    If velodyne data is not available, create placeholder GT depths.
    """
    gt_depths = []
    for line in test_filenames:
        parts = line.split()
        folder = parts[0]
        frame_idx = int(parts[1])

        velo_path = os.path.join(
            data_path, folder,
            "velodyne_points/data",
            f"{frame_idx:010d}.bin")

        if os.path.isfile(velo_path):
            # Load velodyne pointcloud and project to image plane
            velo = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)

            # Load calibration
            calib_path = os.path.join(data_path, folder.split('/')[0])
            gt_depth = velo_to_depth(velo, calib_path)
            gt_depths.append(gt_depth)
        else:
            # Placeholder — will be filled by official GT
            gt_depths.append(np.zeros((375, 1242)))

    return gt_depths


def velo_to_depth(velo, calib_path):
    """Project velodyne points to a depth map (simplified)."""
    # This is a simplified version; for full accuracy use the official KITTI devkit
    H, W = 375, 1242
    depth = np.zeros((H, W), dtype=np.float32)

    # Keep only forward-facing points
    mask = velo[:, 0] > 0
    velo = velo[mask]

    # Simplified projection (uses approximate calibration)
    fx, fy = 721.5377, 721.5377
    cx, cy = 609.5593, 172.854
    baseline_shift = 0.0

    x = velo[:, 0]
    y = velo[:, 1]
    z = velo[:, 2]

    u = np.round(fx * y / x + cx).astype(int)
    v = np.round(fy * z / x + cy).astype(int)
    d = x

    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (d > 0)
    u, v, d = u[valid], v[valid], d[valid]

    # Use closest point when multiple project to same pixel
    order = np.argsort(d)[::-1]
    u, v, d = u[order], v[order], d[order]
    depth[v, u] = d

    return depth


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_path", type=str, default="./splits/eigen_lite")
    parser.add_argument("--load_weights_folder", type=str, required=True)

    evaluate(parser.parse_args())

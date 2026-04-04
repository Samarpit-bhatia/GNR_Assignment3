#!/usr/bin/env python3
"""
Compare from-scratch vs official Monodepth2 results.

Generates:
    1. Side-by-side metrics table
    2. Qualitative depth map comparisons (if available)

Usage:
    python3 compare/compare_results.py \
        --ours_weights ./logs/monodepth2_scratch/weights_epoch_10 \
        --data_path ./kitti_data
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms

from monodepth2_scratch.resnet_encoder import ResnetEncoder
from monodepth2_scratch.depth_decoder import DepthDecoder
from monodepth2_scratch.layers import disp_to_depth
from monodepth2_scratch.utils import readlines


def load_model(weights_folder, device):
    """Load a trained depth model."""
    encoder = ResnetEncoder(18, False)
    depth_decoder = DepthDecoder(encoder.num_ch_enc)

    enc_dict = torch.load(
        os.path.join(weights_folder, "encoder.pth"), map_location=device)
    model_dict = encoder.state_dict()
    enc_dict_filtered = {k: v for k, v in enc_dict.items() if k in model_dict}
    encoder.load_state_dict(enc_dict_filtered, strict=False)
    encoder.to(device)
    encoder.eval()

    dec_dict = torch.load(
        os.path.join(weights_folder, "depth.pth"), map_location=device)
    depth_decoder.load_state_dict(dec_dict, strict=False)
    depth_decoder.to(device)
    depth_decoder.eval()

    height = enc_dict.get("height", 192)
    width = enc_dict.get("width", 640)

    return encoder, depth_decoder, height, width


def predict_depth(encoder, depth_decoder, img_path, height, width, device):
    """Predict depth for a single image."""
    to_tensor = transforms.ToTensor()

    img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img.size
    img_resized = img.resize((width, height), Image.LANCZOS)
    img_tensor = to_tensor(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(img_tensor)
        outputs = depth_decoder(features)
        disp = outputs[("disp", 0)]
        disp = F.interpolate(disp, (orig_h, orig_w),
                              mode="bilinear", align_corners=False)

    disp_np = disp.squeeze().cpu().numpy()
    return img, disp_np


def compare(opt):
    """Generate comparison table and visualisations."""
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # ----- Load metrics -----
    metrics_names = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]

    # Our results
    ours_path = os.path.join(opt.ours_weights, "eval_results.npy")
    if os.path.isfile(ours_path):
        ours_results = np.load(ours_path, allow_pickle=True).item()
    else:
        print("Warning: No eval_results.npy for our model. Run evaluate.py first.")
        ours_results = {m: 0.0 for m in metrics_names}

    # Official results
    official_path = os.path.join(results_dir, "official_results.npy")
    if os.path.isfile(official_path):
        official_results = np.load(official_path, allow_pickle=True).item()
    else:
        print("Warning: No official_results.npy. Run run_official.py first.")
        # Use paper-reported values for mono_640x192
        official_results = {
            "abs_rel": 0.115, "sq_rel": 0.903, "rmse": 4.863,
            "rmse_log": 0.193, "a1": 0.877, "a2": 0.959, "a3": 0.981
        }
        print("Using paper-reported official results instead.")

    # ----- Print table -----
    print("\n" + "=" * 100)
    print("COMPARISON: From-Scratch vs Official Monodepth2")
    print("=" * 100)
    print(f"{'Metric':>12}  {'Ours':>12}  {'Official':>12}  {'Δ':>12}  {'Note':>20}")
    print("-" * 100)

    for m in metrics_names:
        ours_val = ours_results.get(m, 0.0)
        off_val = official_results.get(m, 0.0)
        delta = ours_val - off_val

        # For error metrics, lower is better; for accuracy metrics, higher is better
        if m in ["a1", "a2", "a3"]:
            note = "✅ better" if delta > 0 else "⚠️  worse" if delta < 0 else "equal"
        else:
            note = "✅ better" if delta < 0 else "⚠️  worse" if delta > 0 else "equal"

        print(f"{m:>12}  {ours_val:>12.4f}  {off_val:>12.4f}  {delta:>+12.4f}  {note:>20}")

    print("=" * 100)

    # ----- Generate qualitative comparison -----
    if opt.data_path and opt.ours_weights:
        print("\nGenerating qualitative depth comparisons...")

        test_file = os.path.join(opt.split_path, "test_files.txt")
        if not os.path.isfile(test_file):
            print("No test_files.txt found, skipping qualitative comparison.")
            return

        test_filenames = readlines(test_file)

        # Pick a few sample images
        sample_indices = [0, len(test_filenames) // 4,
                          len(test_filenames) // 2,
                          3 * len(test_filenames) // 4]
        sample_indices = [i for i in sample_indices if i < len(test_filenames)]

        # Load our model
        ours_enc, ours_dec, h, w = load_model(opt.ours_weights, device)

        # Try loading official model
        official_weights = os.path.join(
            os.path.dirname(__file__), '..', 'monodepth2_official',
            'models', 'mono_640x192')
        has_official = os.path.isdir(official_weights)
        if has_official:
            off_enc, off_dec, _, _ = load_model(official_weights, device)

        for idx in sample_indices:
            parts = test_filenames[idx].split()
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
                continue

            img, ours_disp = predict_depth(ours_enc, ours_dec, img_path, h, w, device)

            ncols = 3 if has_official else 2
            fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))

            axes[0].imshow(img)
            axes[0].set_title("Input RGB")
            axes[0].axis("off")

            axes[1].imshow(ours_disp, cmap="magma")
            axes[1].set_title("Our Depth (from scratch)")
            axes[1].axis("off")

            if has_official:
                _, off_disp = predict_depth(off_enc, off_dec, img_path, h, w, device)
                axes[2].imshow(off_disp, cmap="magma")
                axes[2].set_title("Official Pretrained")
                axes[2].axis("off")

            plt.tight_layout()
            save_path = os.path.join(results_dir, f"comparison_{idx}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved {save_path}")

    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours_weights", type=str, required=True,
                        help="Path to our from-scratch model weights")
    parser.add_argument("--data_path", type=str, default="./kitti_data")
    parser.add_argument("--split_path", type=str, default="./splits/eigen_lite")
    compare(parser.parse_args())

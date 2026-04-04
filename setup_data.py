#!/usr/bin/env python3
"""
Setup script for Monodepth2 from-scratch implementation.

This script:
1. Creates a small synthetic dataset for smoke testing
2. Downloads a few real KITTI sequences (if --download_kitti is set)
3. Downloads official pretrained Monodepth2 weights
4. Creates the Eigen-lite split files

Usage:
    python3 setup_data.py                          # Synthetic data only (for smoke test)
    python3 setup_data.py --download_kitti          # Also download a few KITTI sequences
    python3 setup_data.py --download_official_weights  # Also download official weights
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))


def create_synthetic_dataset(data_path, num_sequences=3, frames_per_seq=20):
    """Create a synthetic KITTI-like dataset for smoke testing.

    Generates random images with simple geometric patterns that provide
    enough texture for the photometric loss to work.
    """
    print("Creating synthetic KITTI-like dataset...")

    np.random.seed(42)
    h, w = 375, 1242

    for seq_idx in range(num_sequences):
        seq_name = f"2011_09_26/2011_09_26_drive_{seq_idx:04d}_sync"
        img_dir = os.path.join(data_path, seq_name, "image_02/data")
        os.makedirs(img_dir, exist_ok=True)

        # Also create image_03 (right camera) for completeness
        img_dir_r = os.path.join(data_path, seq_name, "image_03/data")
        os.makedirs(img_dir_r, exist_ok=True)

        # Create a base scene
        base_color = np.random.randint(50, 200, size=(3,))

        for frame_idx in range(frames_per_seq):
            # Generate an image with gradients, shapes, and noise to simulate texture
            img = np.zeros((h, w, 3), dtype=np.uint8)

            # Sky gradient (top)
            sky_h = h // 3
            for y in range(sky_h):
                ratio = y / sky_h
                img[y, :] = [int(180 - 60 * ratio), int(200 - 40 * ratio), int(230 - 30 * ratio)]

            # Ground with texture
            for y in range(sky_h, h):
                ratio = (y - sky_h) / (h - sky_h)
                base = np.array([80 + 40 * ratio, 90 + 30 * ratio, 70 + 20 * ratio])
                noise = np.random.randint(-15, 15, size=(w, 3))
                img[y, :] = np.clip(base + noise, 0, 255).astype(np.uint8)

            # Add some "objects" (rectangles simulating buildings/cars)
            num_objects = np.random.randint(3, 8)
            for _ in range(num_objects):
                ox = np.random.randint(0, w - 100)
                oy = np.random.randint(sky_h // 2, h - 50)
                ow = np.random.randint(30, 150)
                oh = np.random.randint(30, 120)
                color = np.random.randint(40, 220, size=(3,))
                img[oy:oy + oh, ox:ox + ow] = color

            # Simulate slight horizontal motion between frames
            shift = frame_idx * 2
            img = np.roll(img, shift, axis=1)

            # Save as both jpg and png
            pil_img = Image.fromarray(img)
            pil_img.save(os.path.join(img_dir, f"{frame_idx:010d}.jpg"), quality=95)
            pil_img.save(os.path.join(img_dir, f"{frame_idx:010d}.png"))

            # Right image (slightly shifted)
            img_r = np.roll(img, 5, axis=1)
            pil_img_r = Image.fromarray(img_r)
            pil_img_r.save(os.path.join(img_dir_r, f"{frame_idx:010d}.jpg"), quality=95)
            pil_img_r.save(os.path.join(img_dir_r, f"{frame_idx:010d}.png"))

        print(f"  Created sequence {seq_name} ({frames_per_seq} frames)")

    print(f"Synthetic dataset created at {data_path}")
    return data_path


def create_split_files(data_path, split_path):
    """Create train/val/test split files for the synthetic or real data."""
    os.makedirs(split_path, exist_ok=True)

    # Discover all sequences
    train_lines = []
    val_lines = []
    test_lines = []

    for date_dir in sorted(os.listdir(data_path)):
        date_path = os.path.join(data_path, date_dir)
        if not os.path.isdir(date_path):
            continue

        for drive_dir in sorted(os.listdir(date_path)):
            seq_path = os.path.join(date_path, drive_dir)
            img_dir = os.path.join(seq_path, "image_02/data")
            if not os.path.isdir(img_dir):
                continue

            frames = sorted([f for f in os.listdir(img_dir)
                             if f.endswith(('.jpg', '.png'))])
            # Deduplicate (jpg and png may coexist)
            frame_indices = sorted(set(
                int(os.path.splitext(f)[0]) for f in frames))

            folder = f"{date_dir}/{drive_dir}"

            # Use frames with ±1 neighbours available
            valid_frames = frame_indices[1:-1]
            n = len(valid_frames)

            # 70% train, 15% val, 15% test
            n_train = max(1, int(0.7 * n))
            n_val = max(1, int(0.15 * n))

            for i, idx in enumerate(valid_frames):
                line = f"{folder} {idx} l"
                if i < n_train:
                    train_lines.append(line)
                elif i < n_train + n_val:
                    val_lines.append(line)
                else:
                    test_lines.append(line)

    for name, lines in [("train", train_lines), ("val", val_lines), ("test", test_lines)]:
        path = os.path.join(split_path, f"{name}_files.txt")
        with open(path, 'w') as f:
            f.write("\n".join(lines) + "\n")
        print(f"  {name}: {len(lines)} samples → {path}")


def download_official_weights(target_dir):
    """Download official Monodepth2 pretrained weights."""
    weights_dir = os.path.join(target_dir, "monodepth2_official", "models", "mono_640x192")
    os.makedirs(weights_dir, exist_ok=True)

    base_url = "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192"
    for fname in ["encoder.pth", "depth.pth"]:
        out_path = os.path.join(weights_dir, fname)
        if os.path.isfile(out_path):
            print(f"  {fname} already exists, skipping")
            continue
        url = f"{base_url}/{fname}"
        print(f"  Downloading {url}...")
        os.system(f"curl -L -o '{out_path}' '{url}'")

    print(f"Official weights saved to {weights_dir}")
    return weights_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./kitti_data")
    parser.add_argument("--split_path", type=str, default="./splits/eigen_lite")
    parser.add_argument("--download_kitti", action="store_true",
                        help="Download a few real KITTI sequences")
    parser.add_argument("--download_official_weights", action="store_true",
                        help="Download official pretrained weights")
    opt = parser.parse_args()

    os.chdir(ROOT)

    # Step 1: Create synthetic dataset
    data_path = os.path.join(ROOT, opt.data_path)
    create_synthetic_dataset(data_path)

    # Step 2: Create split files
    split_path = os.path.join(ROOT, opt.split_path)
    print("\nCreating split files...")
    create_split_files(data_path, split_path)

    # Step 3: Download official weights
    if opt.download_official_weights:
        print("\nDownloading official pretrained weights...")
        download_official_weights(ROOT)

    print("\n✅ Setup complete!")
    print(f"\nTo train: python3 train.py --data_path {opt.data_path}")
    print(f"To smoke test: python3 train.py --data_path {opt.data_path} --num_epochs 1 --max_batches 2")


if __name__ == "__main__":
    main()

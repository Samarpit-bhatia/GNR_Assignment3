#!/usr/bin/env python3
"""
Train Monodepth2 from scratch.

Usage:
    python3 train.py --data_path ./kitti_data --num_epochs 10
    python3 train.py --data_path ./kitti_data --num_epochs 1 --max_batches 2  # smoke test
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
from monodepth2_scratch.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Monodepth2 from-scratch training")

    # Paths
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to KITTI raw data")
    parser.add_argument("--split_path", type=str, default="./splits/eigen_lite",
                        help="Path to directory with train/val split files")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Directory to save model weights")
    parser.add_argument("--model_name", type=str, default="monodepth2_scratch",
                        help="Name of the model run")

    # Architecture
    parser.add_argument("--num_layers", type=int, default=18, choices=[18, 50],
                        help="ResNet variant")
    parser.add_argument("--weights_init", type=str, default="pretrained",
                        choices=["pretrained", "scratch"],
                        help="Whether to use ImageNet-pretrained encoder")

    # Input
    parser.add_argument("--height", type=int, default=192,
                        help="Input image height (must be multiple of 32)")
    parser.add_argument("--width", type=int, default=640,
                        help="Input image width (must be multiple of 32)")
    parser.add_argument("--scales", nargs="+", type=int, default=[0, 1, 2, 3],
                        help="Scales used in the loss")
    parser.add_argument("--frame_ids", nargs="+", type=int, default=[0, -1, 1],
                        help="Frames to load (0 = target, -1 = prev, 1 = next)")

    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--scheduler_step_size", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_frequency", type=int, default=50)
    parser.add_argument("--save_frequency", type=int, default=5)

    # Loss
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=100.0)
    parser.add_argument("--disparity_smoothness", type=float, default=1e-3)

    # Misc
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--png", action="store_true",
                        help="Use .png instead of .jpg")
    parser.add_argument("--max_batches", type=int, default=0,
                        help="Max batches per epoch (0 = all). For debugging.")

    return parser.parse_args()


if __name__ == "__main__":
    opts = parse_args()
    trainer = Trainer(opts)
    trainer.train()
    print("\n✅ Training complete!")

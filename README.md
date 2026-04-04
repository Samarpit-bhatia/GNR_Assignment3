# Monodepth2 From-Scratch Implementation

**GNR638 Assignment 3** — Implement a paper from scratch and compare with official results.

**Paper:** *Digging into Self-Supervised Monocular Depth Estimation* (Godard et al., ICCV 2019)  
**Official repo:** [nianticlabs/monodepth2](https://github.com/nianticlabs/monodepth2)

## Quick Start

```bash
# 1. Setup synthetic data (for testing)
python3 setup_data.py

# 2. Train from scratch (10 epochs)
python3 train.py --data_path ./kitti_data --num_epochs 10 --weights_init scratch

# 3. Generate results & comparisons
python3 run_all.py
```

## Project Structure

```
monodepth2_scratch/          # From-scratch implementation
├── layers.py                # Conv blocks, SSIM, projections, losses, metrics
├── resnet_encoder.py        # ResNet-18 multi-scale encoder
├── depth_decoder.py         # U-Net decoder with skip connections
├── pose_decoder.py          # 6-DoF pose prediction network
├── kitti_dataset.py         # KITTI monocular triplet loader
├── trainer.py               # Self-supervised training loop
├── evaluate.py              # Depth evaluation with standard metrics
└── utils.py                 # Helpers

compare/                     # Official model comparison
├── run_official.py          # Run official pretrained model
└── compare_results.py       # Side-by-side comparison

train.py                     # Training entry point
run_all.py                   # End-to-end pipeline
setup_data.py                # Data setup (synthetic + splits)
```

## Key Components Implemented

| Component | Description |
|-----------|-------------|
| **ResNet Encoder** | 5-level feature extractor (H/2 → H/32) |
| **Depth Decoder** | U-Net with skip connections, 4-scale disparity output |
| **Pose Decoder** | 6-DoF relative camera pose (axis-angle + translation) |
| **Photometric Loss** | 0.85 × SSIM + 0.15 × L1 |
| **Min Reprojection** | Per-pixel minimum across source frames |
| **Auto-masking** | Ignores static/textureless pixels |
| **Smoothness Loss** | Edge-aware disparity regularisation |

## Architecture

- **Total parameters:** 27.9M
- **Depth network:** ResNet-18 encoder (11.7M) + U-Net decoder (3.2M)
- **Pose network:** ResNet-18 encoder (11.7M) + pose decoder (1.3M)
- **Input:** 192 × 640 RGB images
- **Output:** Dense disparity maps at 4 scales

## Training on Full KITTI

For paper-level results, train on the full KITTI raw dataset:

```bash
python3 train.py \
    --data_path /path/to/kitti_raw \
    --split_path ./splits/eigen_full \
    --num_epochs 20 \
    --batch_size 12 \
    --weights_init pretrained
```

## Official Paper Results (mono_640×192)

| abs_rel | sq_rel | rmse  | rmse_log | δ<1.25 | δ<1.25² | δ<1.25³ |
|---------|--------|-------|----------|--------|---------|---------|
| 0.115   | 0.903  | 4.863 | 0.193    | 0.877  | 0.959   | 0.981   |

## References

1. Godard, C., Mac Aodha, O., Firman, M., & Brostow, G. J. (2019). *Digging into Self-Supervised Monocular Depth Estimation.* ICCV. [arXiv:1806.01260](https://arxiv.org/abs/1806.01260)
2. [Official Implementation](https://github.com/nianticlabs/monodepth2)

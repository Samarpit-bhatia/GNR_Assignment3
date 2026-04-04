#!/usr/bin/env python3
"""
Run the official Monodepth2 pretrained model on KITTI test images.

Downloads pretrained weights (mono_640x192) and evaluates on the same
images used for our from-scratch model.

Usage:
    python3 compare/run_official.py --data_path ./kitti_data
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

# We clone the official repo and import from it
OFFICIAL_REPO = os.path.join(os.path.dirname(__file__), '..', 'monodepth2_official')


def setup_official_repo():
    """Clone official Monodepth2 repo and download pretrained weights."""
    if not os.path.isdir(OFFICIAL_REPO):
        print("Cloning official Monodepth2 repository...")
        os.system(f"git clone https://github.com/nianticlabs/monodepth2.git {OFFICIAL_REPO}")

    # Download pretrained weights
    weights_dir = os.path.join(OFFICIAL_REPO, "models", "mono_640x192")
    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir, exist_ok=True)
        print("Downloading pretrained mono_640x192 weights...")
        base_url = "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192"
        for fname in ["encoder.pth", "depth.pth"]:
            url = f"{base_url}/{fname}"
            out = os.path.join(weights_dir, fname)
            os.system(f"curl -L -o {out} {url}")

    return weights_dir


def evaluate_official(opt):
    """Evaluate using official pretrained model."""
    weights_dir = setup_official_repo()

    # Import official networks
    sys.path.insert(0, OFFICIAL_REPO)

    # We reuse our own encoder/decoder since the architecture is identical
    from monodepth2_scratch.resnet_encoder import ResnetEncoder
    from monodepth2_scratch.depth_decoder import DepthDecoder
    from monodepth2_scratch.layers import disp_to_depth, compute_depth_errors
    from monodepth2_scratch.utils import readlines

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load pretrained model
    encoder = ResnetEncoder(18, False)
    depth_decoder = DepthDecoder(encoder.num_ch_enc)

    encoder_path = os.path.join(weights_dir, "encoder.pth")
    decoder_path = os.path.join(weights_dir, "depth.pth")

    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    height = loaded_dict_enc.get("height", 192)
    width = loaded_dict_enc.get("width", 640)

    model_dict = encoder.state_dict()
    filtered_dict = {k: v for k, v in loaded_dict_enc.items() if k in model_dict}
    encoder.load_state_dict(filtered_dict, strict=False)
    encoder.to(device)
    encoder.eval()

    loaded_dict_dec = torch.load(decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict_dec, strict=False)
    depth_decoder.to(device)
    depth_decoder.eval()

    # Test split
    test_filenames = readlines(
        os.path.join(opt.split_path, "test_files.txt"))
    print(f"Evaluating {len(test_filenames)} images with official model")

    # GT depths
    gt_path = os.path.join(opt.split_path, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    pred_depths = []
    to_tensor = transforms.ToTensor()

    print("Running official model predictions...")
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
                pred_depths.append(np.zeros((375, 1242)))
                continue

            img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = img.size
            img = img.resize((width, height), Image.LANCZOS)
            img = to_tensor(img).unsqueeze(0).to(device)

            features = encoder(img)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = F.interpolate(
                disp, (orig_h, orig_w),
                mode="bilinear", align_corners=False)
            _, depth = disp_to_depth(disp_resized, 0.1, 100)
            pred_depths.append(depth.squeeze().cpu().numpy())

            if (idx + 1) % 100 == 0:
                print(f"  {idx + 1}/{len(test_filenames)}")

    # Compute metrics
    MIN_DEPTH, MAX_DEPTH = 1e-3, 80
    errors = []
    for i in range(len(pred_depths)):
        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]

        gt_h, gt_w = gt_depth.shape[:2]
        pred_depth = np.array(
            Image.fromarray(pred_depth).resize((gt_w, gt_h), Image.NEAREST))

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        crop = np.array([0.40810811 * gt_h, 0.99189189 * gt_h,
                         0.03594771 * gt_w, 0.96405229 * gt_w]).astype(np.int32)
        crop_mask = np.zeros(mask.shape, dtype=bool)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = True
        mask = np.logical_and(mask, crop_mask)

        pred_depth = pred_depth[mask]
        gt_masked = gt_depth[mask]

        if len(pred_depth) == 0:
            continue

        ratio = np.median(gt_masked) / np.median(pred_depth)
        pred_depth *= ratio
        pred_depth = np.clip(pred_depth, MIN_DEPTH, MAX_DEPTH)

        err = compute_depth_errors(
            torch.from_numpy(gt_masked).float(),
            torch.from_numpy(pred_depth).float())
        errors.append([e.item() for e in err])

    mean_errors = np.array(errors).mean(0)

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS (Official Monodepth2 Pretrained)")
    print("=" * 80)
    metrics = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    header = "  ".join([f"{m:>10}" for m in metrics])
    values = "  ".join([f"{e:>10.4f}" for e in mean_errors])
    print(header)
    print(values)
    print("=" * 80)

    results = {m: float(v) for m, v in zip(metrics, mean_errors)}
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, "official_results.npy"), results)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_path", type=str, default="./splits/eigen_lite")
    evaluate_official(parser.parse_args())

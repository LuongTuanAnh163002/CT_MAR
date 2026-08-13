"""
sweep_exit_threshold.py
=========================
Sau khi da train xong ExitHead, script nay:
1. Chay inference CO EARLY-EXIT tren tap test, voi NHIEU gia tri threshold khac nhau
2. Do PSNR trung binh + thoi gian inference trung binh cho tung threshold
3. In ra bang de chon "diem goi" (knee point) -- threshold can bang tot nhat
   giua toc do va chat luong

CACH DUNG:
    python sweep_exit_threshold.py \
        -backbone_checkpoint checkpoint/stage2_baseline/XXXXX_ckpt \
        -exithead_checkpoint checkpoint/exit_heads_baseline/5000_exitheads.pth \
        -test_data_dir /path/to/data_CT_MAR_split/test \
        -n_test_samples 100
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import cv2

import sys
sys.path.append('.')
from model.mamba_exithead import MambaFormerWithExit, ExitHead
from utils.metrics import calculate_psnr, calculate_ssim, calculate_rmse
from utils.aapm_dataset import hu_to_unit, load_raw, _find_anatomy_folders, _index_one_anatomy


def load_backbone_and_exitheads(backbone_ckpt, exithead_ckpt, device):
    net = MambaFormerWithExit(in_channels=1).to(device)
    net = nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())) or [0])
    net.load_state_dict(torch.load(backbone_ckpt, map_location=device))
    net.eval()
    for p in net.parameters():
        p.requires_grad = False

    # tao ExitHead khop dung shape (giong luc train)
    with torch.no_grad():
        dummy = torch.zeros(1, 1, 336, 336).to(device)
        sample_intermediates, _, _, _ = net.module.forward_get_intermediate(dummy)
    exit_heads = nn.ModuleList([
        ExitHead(in_channels=feat.shape[1], downsample_factor=factor)
        for feat, factor in sample_intermediates
    ]).to(device)
    exit_heads.load_state_dict(torch.load(exithead_ckpt, map_location=device))
    exit_heads.eval()

    return net, exit_heads


def inference_with_early_exit(net, exit_heads, x, threshold):
    """Chay 1 anh, thoat som neu confidence tai 1 diem thoat > threshold.
    Tra ve (output, exit_point_used) -- exit_point_used: 1/2/3, hoac 'full' neu di het."""
    with torch.no_grad():
        feat = x.clone()
        x_ori = x.clone()
        _, _, ori_h, ori_w = x.shape
        feat = net.module.pad_to_multiple_of_eight(feat)
        feat = net.module.embedder(feat)

        stages = [
            (net.module.down1_stage, net.module.down1, 1),
            (net.module.down2_stage, net.module.down2, 2),
            (net.module.down3_stage, net.module.down3, 4),
        ]

        for i, (stage_fn, downsample_fn, factor) in enumerate(stages):
            feat = stage_fn(feat)
            early_output, confidence = exit_heads[i](feat, x_ori, ori_h, ori_w)
            if confidence.item() > threshold:
                return early_output, i + 1
            if i < len(stages) - 1:
                feat = downsample_fn(feat)

        # khong thoat som o dau -- chay full model binh thuong
        full_output = net(x)
        return full_output, 'full'


def to_uint8_bgr3(img_float01):
    img_u8 = np.clip(img_float01 * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)


def get_test_subset(test_data_dir, n_samples, seed=42):
    anatomy_dirs = _find_anatomy_folders(test_data_dir)
    all_samples = []
    for d in anatomy_dirs:
        all_samples.extend(_index_one_anatomy(d))
    rng = np.random.RandomState(seed)
    if n_samples < len(all_samples):
        idx = rng.choice(len(all_samples), size=n_samples, replace=False)
        return [all_samples[i] for i in idx]
    return all_samples


def evaluate_threshold(net, exit_heads, samples, threshold, device):
    psnrs = []
    times = []
    exit_point_counts = {1: 0, 2: 0, 3: 0, 'full': 0}

    for s in samples:
        baseline_hu = load_raw(s["baseline"])
        target_hu = load_raw(s["target"])
        Xma = hu_to_unit(baseline_hu)
        Xgt = hu_to_unit(target_hu)
        input_t = torch.from_numpy((Xma - 0.5) / 0.5).unsqueeze(0).unsqueeze(0).float().to(device)

        torch.cuda.synchronize()
        start = time.time()
        output, exit_point = inference_with_early_exit(net, exit_heads, input_t, threshold)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        pred = np.clip(output.squeeze().cpu().numpy(), 0.0, 1.0)
        pred_bgr = to_uint8_bgr3(pred)
        gt_bgr = to_uint8_bgr3(Xgt)
        psnr = calculate_psnr(pred_bgr, gt_bgr, test_y_channel=True)

        psnrs.append(psnr)
        times.append(elapsed)
        exit_point_counts[exit_point] += 1

    return {
        "threshold": threshold,
        "psnr_mean": np.mean(psnrs),
        "psnr_std": np.std(psnrs),
        "time_mean_ms": np.mean(times) * 1000,
        "exit_distribution": exit_point_counts,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-backbone_checkpoint", required=True)
    parser.add_argument("-exithead_checkpoint", default=None,
                         help="Duong dan 1 file cu the. Bo qua neu dung -exithead_dir de tu dong tim best.")
    parser.add_argument("-exithead_dir", default=None,
                         help="Thu muc chua nhieu file XXXX_exitheads.pth -- se tu dong quet tim best truoc")
    parser.add_argument("-test_data_dir", required=True)
    parser.add_argument("-n_test_samples", type=int, default=100)
    parser.add_argument("-fixed_threshold_for_ckpt_search", type=float, default=0.7,
                         help="Threshold CO DINH dung khi so sanh CAC CHECKPOINT ExitHead voi nhau")
    parser.add_argument("-device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    # --- Buoc 0 (MOI): neu dung -exithead_dir, tu dong tim checkpoint ExitHead TOT NHAT truoc ---
    exithead_checkpoint = args.exithead_checkpoint
    if args.exithead_dir is not None:
        import glob, re
        ckpt_files = sorted(
            glob.glob(os.path.join(args.exithead_dir, "*_exitheads.pth")),
            key=lambda f: int(re.search(r"(\d+)_exitheads", f).group(1))
        )
        print(f"Tim thay {len(ckpt_files)} checkpoint ExitHead, dang so sanh (threshold co dinh={args.fixed_threshold_for_ckpt_search})...")

        samples_for_search = get_test_subset(args.test_data_dir, min(30, args.n_test_samples))  # tap nho hon de nhanh
        best_psnr = -1
        for ckpt in ckpt_files:
            net, exit_heads = load_backbone_and_exitheads(args.backbone_checkpoint, ckpt, device)
            r = evaluate_threshold(net, exit_heads, samples_for_search,
                                     threshold=args.fixed_threshold_for_ckpt_search, device=device)
            step = re.search(r"(\d+)_exitheads", ckpt).group(1)
            print(f"  step {step}: PSNR={r['psnr_mean']:.3f}")
            if r['psnr_mean'] > best_psnr:
                best_psnr = r['psnr_mean']
                exithead_checkpoint = ckpt

        print(f"\n>>> Checkpoint ExitHead TOT NHAT: {exithead_checkpoint} (PSNR={best_psnr:.3f})\n")

    if exithead_checkpoint is None:
        raise ValueError("Can chi dinh -exithead_checkpoint HOAC -exithead_dir")

    net, exit_heads = load_backbone_and_exitheads(args.backbone_checkpoint, exithead_checkpoint, device)
    samples = get_test_subset(args.test_data_dir, args.n_test_samples)
    print(f"Dung {len(samples)} anh test (co dinh seed=42)")

    # Baseline: KHONG early-exit, luon di het (threshold = 1.1, khong bao gio thoat)
    print("\n=== Baseline (khong early-exit, luon di full model) ===")
    baseline_result = evaluate_threshold(net, exit_heads, samples, threshold=1.1, device=device)
    print(f"  PSNR: {baseline_result['psnr_mean']:.3f} +- {baseline_result['psnr_std']:.3f}")
    print(f"  Thoi gian: {baseline_result['time_mean_ms']:.2f} ms/anh")

    print("\n=== Quet threshold ===")
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    results = []
    for t in thresholds:
        r = evaluate_threshold(net, exit_heads, samples, threshold=t, device=device)
        speedup = baseline_result['time_mean_ms'] / r['time_mean_ms']
        psnr_drop = baseline_result['psnr_mean'] - r['psnr_mean']
        print(f"threshold={t:.2f}: PSNR={r['psnr_mean']:.3f} (giam {psnr_drop:.3f}dB), "
              f"Time={r['time_mean_ms']:.2f}ms (nhanh hon {speedup:.2f}x), "
              f"phan bo thoat={r['exit_distribution']}")
        results.append(r)

    print("\n=== Goi y: chon threshold co PSNR giam < 0.3dB nhung speedup cao nhat ===")

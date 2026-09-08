#!/usr/bin/env python3
"""
Evaluate Metal-Geometry Guided Dynamic FMB on AAPM CT-MAR.

This file is intentionally separate from test/test_aapm.py so the existing
baseline / previous-improvement evaluation code is not modified.

Expected project files:
    model/mamba_exp1.py
    utils/aapm_dataset.py

The Dynamic FMB model requires TWO inputs:
    prediction = model(artifact_image, metal_mask)

Outputs:
    <output_dir>/
        output/                 # predicted PNG images
        gt/                     # target PNG images
        mask/                   # binary metal masks
        per_case_metrics.csv
        summary.txt

The CSV also stores the learned guidance weights w0/w1/w2 for each case.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as tvu
from torchvision.transforms import Compose, Normalize, ToTensor

# Allow this file to live under MARMamba/test/ and still import model/ and utils/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.mamba_exp1 import MetalGuidedMambaFormer
from utils.aapm_dataset import (
    _find_anatomy_folders,
    _index_one_anatomy,
    hu_to_unit,
    load_raw,
)
from utils.metrics import calculate_psnr, calculate_rmse, calculate_ssim


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def save_image(tensor: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tvu.save_image(tensor, path)


def unwrap_checkpoint(checkpoint_obj):
    """
    Support both:
      1) old checkpoint: torch.save(model.state_dict(), path)
      2) new checkpoint:
         {
             "step": ...,
             "model": model.state_dict(),
             "optimizer": ...,
             "scheduler": ...
         }
    """
    if isinstance(checkpoint_obj, dict) and "model" in checkpoint_obj:
        return checkpoint_obj["model"]
    return checkpoint_obj


def normalize_state_dict_for_model(state_dict, model):
    """
    Make checkpoints saved with/without nn.DataParallel interchangeable.
    """
    model_is_dp = isinstance(model, nn.DataParallel)
    keys = list(state_dict.keys())
    state_has_module = bool(keys) and all(k.startswith("module.") for k in keys)

    if model_is_dp and not state_has_module:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}
    elif (not model_is_dp) and state_has_module:
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    return state_dict


def load_checkpoint(model, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = unwrap_checkpoint(ckpt)
    state = normalize_state_dict_for_model(state, model)
    model.load_state_dict(state, strict=True)


def tensor_to_uint8_gray(x: torch.Tensor) -> np.ndarray:
    """
    [1,1,H,W] or [1,H,W] in [0,1] -> uint8 [H,W].
    """
    x = x.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3:
        x = x[0]
    x = x.clamp(0.0, 1.0).numpy()
    return np.round(x * 255.0).astype(np.uint8)


def gray_to_bgr(x: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)


def count_metals(binary_mask: np.ndarray) -> int:
    """
    Count connected components in a binary metal mask.
    AAPM masks are clean, so no aggressive component filtering is applied.
    """
    m = (binary_mask > 0).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(m, connectivity=8)
    return max(0, int(num_labels) - 1)


def detect_size_group(sample: dict) -> str:
    """
    Do NOT invent Large/Medium/Small/Tiny thresholds.

    If the current AAPM indexing code or directory structure already contains
    an explicit size label, preserve it. Otherwise return "Unknown".

    This keeps the Dynamic-FMB evaluator from silently using a grouping rule
    different from test/test_aapm.py.
    """
    candidate_keys = (
        "size_group",
        "metal_size_group",
        "metal_size",
        "group",
    )

    for key in candidate_keys:
        value = sample.get(key)
        if value is not None:
            text = str(value).strip().lower()
            for g in ("large", "medium", "small", "tiny"):
                if g in text:
                    return g.capitalize()

    paths = [
        sample.get("baseline"),
        sample.get("target"),
        sample.get("mask"),
        sample.get("id"),
    ]
    joined = " ".join(str(x).lower() for x in paths if x is not None)
    for g in ("large", "medium", "small", "tiny"):
        if g in joined:
            return g.capitalize()

    return "Unknown"


def metrics_from_uint8(pred_u8: np.ndarray, gt_u8: np.ndarray):
    """
    Reuse the project's metric implementations exactly as train/test code does.
    utils.metrics is called on 3-channel uint8 images.
    """
    pred_bgr = gray_to_bgr(pred_u8)
    gt_bgr = gray_to_bgr(gt_u8)

    psnr = calculate_psnr(pred_bgr, gt_bgr, test_y_channel=True)
    ssim = calculate_ssim(pred_bgr, gt_bgr, test_y_channel=True)
    rmse = calculate_rmse(pred_bgr, gt_bgr)

    return float(psnr), float(ssim), float(rmse)


def lpips_gray(
    lpips_model,
    pred_01: torch.Tensor,
    gt_01: torch.Tensor,
) -> float:
    """
    LPIPS expects 3 channels in [-1,1].
    """
    pred = pred_01.clamp(0, 1)
    gt = gt_01.clamp(0, 1)

    if pred.shape[1] == 1:
        pred = pred.repeat(1, 3, 1, 1)
    if gt.shape[1] == 1:
        gt = gt.repeat(1, 3, 1, 1)

    pred = pred * 2.0 - 1.0
    gt = gt * 2.0 - 1.0

    return float(lpips_model(pred, gt).mean().item())


def non_metal_prediction(
    pred_01: torch.Tensor,
    gt_01: torch.Tensor,
    mask_01: torch.Tensor,
) -> torch.Tensor:
    """
    Ignore reconstruction error INSIDE the metal itself by replacing predicted
    metal pixels with GT metal pixels. The surrounding artifact region is left
    untouched.

    This gives a consistent "non-metallic area" version while preserving the
    full image shape needed by SSIM and LPIPS.
    """
    metal = mask_01 > 0.5
    return torch.where(metal, gt_01, pred_01)


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


def format_metric(values, digits=4):
    mean, std = mean_std(values)
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def summarize_rows(rows, indices, prefix="included"):
    metrics = ["psnr", "ssim", "rmse", "lpips"]
    out = {}
    for metric in metrics:
        key = f"{prefix}_{metric}"
        vals = [float(rows[i][key]) for i in indices]
        out[metric] = vals
    return out


def print_summary_block(title, rows, indices, prefix="included"):
    if not indices:
        return

    s = summarize_rows(rows, indices, prefix=prefix)

    print(f"\n{title} (n={len(indices)})")
    print(
        "  PSNR : "
        + format_metric(s["psnr"], 4)
        + "\n  SSIM : "
        + format_metric(s["ssim"], 4)
        + "\n  RMSE : "
        + format_metric(s["rmse"], 4)
        + "\n  LPIPS: "
        + format_metric(s["lpips"], 4)
    )


# -------------------------------------------------------------------------
# Main evaluation
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AAPM CT-MAR Dynamic FMB checkpoint."
    )
    parser.add_argument(
        "-checkpoint",
        "--checkpoint",
        required=True,
        type=str,
        help="Dynamic FMB checkpoint, e.g. checkpoint/exp1_dynamic_fmb/233000_ckpt",
    )
    parser.add_argument(
        "-test_data_dir",
        "--test_data_dir",
        required=True,
        type=str,
        help="AAPM test dataset root.",
    )
    parser.add_argument(
        "-output_dir",
        "--output_dir",
        default="./eval_dynamic_fmb",
        type=str,
        help="Directory for predictions and evaluation outputs.",
    )
    parser.add_argument(
        "-guidance_temperature",
        "--guidance_temperature",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "-device",
        "--device",
        default="cuda",
        type=str,
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--no_save_images",
        action="store_true",
        help="Do not save output/gt/mask PNG files.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if not os.path.isdir(args.test_data_dir):
        raise FileNotFoundError(f"Test data directory not found: {args.test_data_dir}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Same architecture used in train_exp1.py.
    model = MetalGuidedMambaFormer(
        in_channels=1,
        guidance_temperature=args.guidance_temperature,
    ).to(device)

    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Test data  : {args.test_data_dir}")
    print(f"Output dir : {args.output_dir}")
    print(f"Parameters : {total_params / 1e6:.4f} M")
    print(f"Device     : {device}")

    # LPIPS is used in the previous AAPM analysis.
    lpips_model = lpips.LPIPS(net="vgg", spatial=False).to(device)
    lpips_model.eval()

    transform_input = Compose(
        [
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    transform_gt = Compose([ToTensor()])
    transform_mask = Compose([ToTensor()])

    output_png_dir = os.path.join(args.output_dir, "output")
    gt_png_dir = os.path.join(args.output_dir, "gt")
    mask_png_dir = os.path.join(args.output_dir, "mask")

    if not args.no_save_images:
        os.makedirs(output_png_dir, exist_ok=True)
        os.makedirs(gt_png_dir, exist_ok=True)
        os.makedirs(mask_png_dir, exist_ok=True)

    anatomy_dirs = _find_anatomy_folders(args.test_data_dir)
    if not anatomy_dirs:
        raise RuntimeError(
            f"No AAPM anatomy directories found under {args.test_data_dir}"
        )

    rows = []
    total_inference_time = 0.0

    with torch.inference_mode():
        for anatomy_dir in anatomy_dirs:
            samples = _index_one_anatomy(anatomy_dir)

            for sample in samples:
                if sample.get("mask") is None:
                    raise FileNotFoundError(
                        "Dynamic FMB requires Mask data, but no mask was indexed "
                        f"for sample {sample.get('id', '<unknown>')}"
                    )

                # ---------------------------------------------------------
                # Load AAPM raw data
                # ---------------------------------------------------------
                input_img = hu_to_unit(load_raw(sample["baseline"]))
                gt_img = hu_to_unit(load_raw(sample["target"]))

                mask_img = load_raw(sample["mask"]).astype(np.float32)
                mask_img = np.squeeze(mask_img)
                mask_img = (mask_img != 0).astype(np.float32)

                input_t = transform_input(input_img).unsqueeze(0).to(device)
                gt_t = transform_gt(gt_img).unsqueeze(0).float().to(device)
                mask_t = transform_mask(mask_img).unsqueeze(0).float().to(device)

                # ---------------------------------------------------------
                # Dynamic FMB inference
                # return_guidance=True gives the learned [w0,w1,w2].
                # ---------------------------------------------------------
                if device.type == "cuda":
                    torch.cuda.synchronize()

                start = time.perf_counter()
                pred_t, guidance = model(
                    input_t,
                    mask_t,
                    return_guidance=True,
                )

                if device.type == "cuda":
                    torch.cuda.synchronize()

                elapsed = time.perf_counter() - start
                total_inference_time += elapsed

                pred_t = pred_t.clamp(0.0, 1.0)
                gt_t = gt_t.clamp(0.0, 1.0)

                # ---------------------------------------------------------
                # 1) Metal-included metrics = full image
                # ---------------------------------------------------------
                pred_u8 = tensor_to_uint8_gray(pred_t)
                gt_u8 = tensor_to_uint8_gray(gt_t)

                inc_psnr, inc_ssim, inc_rmse = metrics_from_uint8(
                    pred_u8,
                    gt_u8,
                )
                inc_lpips = lpips_gray(lpips_model, pred_t, gt_t)

                # ---------------------------------------------------------
                # 2) Non-metallic-area metrics
                # Ignore only pixels occupied by the physical metal.
                # ---------------------------------------------------------
                pred_nonmetal_t = non_metal_prediction(
                    pred_t,
                    gt_t,
                    mask_t,
                )
                pred_nonmetal_u8 = tensor_to_uint8_gray(pred_nonmetal_t)

                nm_psnr, nm_ssim, nm_rmse = metrics_from_uint8(
                    pred_nonmetal_u8,
                    gt_u8,
                )
                nm_lpips = lpips_gray(
                    lpips_model,
                    pred_nonmetal_t,
                    gt_t,
                )

                # ---------------------------------------------------------
                # Metal geometry diagnostics
                # ---------------------------------------------------------
                mask_u8 = (mask_img > 0).astype(np.uint8) * 255
                metal_count = count_metals(mask_img)
                metal_pixels = int((mask_img > 0).sum())
                metal_fraction = float((mask_img > 0).mean())
                size_group = detect_size_group(sample)

                w = guidance[0].detach().cpu().numpy().astype(float)

                sample_id = str(sample.get("id", len(rows)))
                anatomy_name = os.path.basename(os.path.normpath(anatomy_dir))

                row = {
                    "sample_id": sample_id,
                    "anatomy": anatomy_name,
                    "size_group": size_group,
                    "metal_count": metal_count,
                    "metal_pixels": metal_pixels,
                    "metal_fraction": metal_fraction,
                    "w0": float(w[0]),
                    "w1": float(w[1]),
                    "w2": float(w[2]),
                    "inference_time_s": elapsed,
                    "included_psnr": inc_psnr,
                    "included_ssim": inc_ssim,
                    "included_rmse": inc_rmse,
                    "included_lpips": inc_lpips,
                    "nonmetal_psnr": nm_psnr,
                    "nonmetal_ssim": nm_ssim,
                    "nonmetal_rmse": nm_rmse,
                    "nonmetal_lpips": nm_lpips,
                }
                rows.append(row)

                if not args.no_save_images:
                    save_image(
                        pred_t,
                        os.path.join(output_png_dir, f"{sample_id}.png"),
                    )
                    save_image(
                        gt_t,
                        os.path.join(gt_png_dir, f"{sample_id}.png"),
                    )

                    mask_save = (
                        torch.from_numpy(mask_img)
                        .unsqueeze(0)
                        .float()
                    )
                    save_image(
                        mask_save,
                        os.path.join(mask_png_dir, f"{sample_id}.png"),
                    )

                if len(rows) % 50 == 0:
                    print(f"Evaluated {len(rows)} cases...")

    if not rows:
        raise RuntimeError("No test cases were evaluated.")

    # ---------------------------------------------------------------------
    # Save per-case CSV
    # ---------------------------------------------------------------------
    csv_path = os.path.join(args.output_dir, "per_case_metrics.csv")
    fieldnames = list(rows[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ---------------------------------------------------------------------
    # Summaries
    # ---------------------------------------------------------------------
    all_indices = list(range(len(rows)))

    included = summarize_rows(rows, all_indices, prefix="included")
    nonmetal = summarize_rows(rows, all_indices, prefix="nonmetal")

    avg_time = total_inference_time / len(rows)

    summary_lines = []

    def emit(line=""):
        print(line)
        summary_lines.append(line)

    emit("\n" + "=" * 72)
    emit("DYNAMIC FMB - AAPM CT-MAR EVALUATION")
    emit("=" * 72)
    emit(f"Checkpoint: {args.checkpoint}")
    emit(f"Number of cases: {len(rows)}")
    emit(f"Mean inference time: {avg_time:.6f} s/image")
    emit(f"Model parameters: {total_params / 1e6:.4f} M")

    emit("\n[Metal-included / full image]")
    emit(f"PSNR : {format_metric(included['psnr'], 4)}")
    emit(f"SSIM : {format_metric(included['ssim'], 4)}")
    emit(f"RMSE : {format_metric(included['rmse'], 4)}")
    emit(f"LPIPS: {format_metric(included['lpips'], 4)}")

    emit("\n[Non-metallic area]")
    emit(f"PSNR : {format_metric(nonmetal['psnr'], 4)}")
    emit(f"SSIM : {format_metric(nonmetal['ssim'], 4)}")
    emit(f"RMSE : {format_metric(nonmetal['rmse'], 4)}")
    emit(f"LPIPS: {format_metric(nonmetal['lpips'], 4)}")

    # Single vs multi metal.
    single_idx = [i for i, r in enumerate(rows) if int(r["metal_count"]) == 1]
    multi_idx = [i for i, r in enumerate(rows) if int(r["metal_count"]) >= 2]

    for label, idx in (("Single metal", single_idx), ("Multi metal", multi_idx)):
        if idx:
            s_inc = summarize_rows(rows, idx, "included")
            s_nm = summarize_rows(rows, idx, "nonmetal")

            emit(f"\n[{label}] n={len(idx)}")
            emit(
                "Included : "
                f"PSNR {format_metric(s_inc['psnr'])}, "
                f"SSIM {format_metric(s_inc['ssim'])}, "
                f"RMSE {format_metric(s_inc['rmse'])}, "
                f"LPIPS {format_metric(s_inc['lpips'])}"
            )
            emit(
                "Nonmetal : "
                f"PSNR {format_metric(s_nm['psnr'])}, "
                f"SSIM {format_metric(s_nm['ssim'])}, "
                f"RMSE {format_metric(s_nm['rmse'])}, "
                f"LPIPS {format_metric(s_nm['lpips'])}"
            )

    # 1/2/3/... metal count.
    unique_counts = sorted(set(int(r["metal_count"]) for r in rows))
    emit("\n[By metal count]")
    for count in unique_counts:
        idx = [
            i for i, r in enumerate(rows)
            if int(r["metal_count"]) == count
        ]
        s = summarize_rows(rows, idx, "nonmetal")
        emit(
            f"{count} metal(s), n={len(idx)}: "
            f"PSNR {format_metric(s['psnr'])}, "
            f"SSIM {format_metric(s['ssim'])}, "
            f"RMSE {format_metric(s['rmse'])}, "
            f"LPIPS {format_metric(s['lpips'])}"
        )

    # Large / Medium / Small / Tiny only when the dataset/index already
    # contains a reliable label.
    known_size_groups = ("Large", "Medium", "Small", "Tiny")
    has_explicit_size = any(r["size_group"] != "Unknown" for r in rows)

    if has_explicit_size:
        emit("\n[By explicit metal size group]")
        for group in known_size_groups:
            idx = [
                i for i, r in enumerate(rows)
                if r["size_group"] == group
            ]
            if not idx:
                continue

            s = summarize_rows(rows, idx, "nonmetal")
            emit(
                f"{group}, n={len(idx)}: "
                f"PSNR {format_metric(s['psnr'])}, "
                f"SSIM {format_metric(s['ssim'])}, "
                f"RMSE {format_metric(s['rmse'])}, "
                f"LPIPS {format_metric(s['lpips'])}"
            )
    else:
        emit(
            "\n[Size groups] Not computed: no explicit Large/Medium/Small/Tiny "
            "label was found in the indexed samples or paths. Use the exact "
            "grouping rule from test/test_aapm.py rather than inventing new "
            "thresholds."
        )

    # Guidance diagnostics.
    w_all = np.asarray(
        [[r["w0"], r["w1"], r["w2"]] for r in rows],
        dtype=np.float64,
    )
    emit("\n[Dynamic FMB guidance weights]")
    for j, name in enumerate(("w0-normal", "w1-flip1", "w2-flip2")):
        emit(
            f"{name}: mean={w_all[:, j].mean():.6f}, "
            f"std={w_all[:, j].std():.6f}, "
            f"min={w_all[:, j].min():.6f}, "
            f"max={w_all[:, j].max():.6f}"
        )

    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"\nPer-case CSV : {csv_path}")
    print(f"Summary      : {summary_path}")


if __name__ == "__main__":
    main()

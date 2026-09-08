#!/usr/bin/env python3
"""
analyze_frequency_residual.py

Kiểm chứng 3 giả thuyết cho hướng Frequency-Aware MARMamba:

1) Metal count tăng -> tỷ lệ high-frequency trong residual output tăng?
2) Metal count tăng -> MARMamba giữ lại nhiều high-frequency artifact hơn?
3) HF retention tăng -> PSNR non-metal giảm?

Không train model. Chỉ chạy inference baseline trên AAPM test set.

Output:
- frequency_per_case.csv
- summary_by_metal_count.csv
- correlations.csv
- hypothesis_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.mamba import MambaFormer
from utils.metrics import calculate_psnr

try:
    from scipy.stats import pearsonr, spearmanr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


HU_MIN, HU_MAX = -1000.0, 3000.0
FNAME_ID_RE = re.compile(r"img(\d+)")
DIMS_RE = re.compile(r"(\d+)x(\d+)x(\d+)")
EPS = 1e-12


# ---------------------------------------------------------------------
# AAPM loading
# ---------------------------------------------------------------------

def parse_dims(filename: str):
    m = DIMS_RE.search(filename)
    if not m:
        raise ValueError(f"Cannot parse dimensions from: {filename}")
    w, h, d = (int(x) for x in m.groups())
    return h, w, d


def load_raw(path: str, dtype=np.float32):
    rows, cols, slices = parse_dims(os.path.basename(path))
    arr = np.fromfile(path, dtype=dtype)
    expected = rows * cols * slices
    if arr.size != expected:
        raise ValueError(f"{path}: found {arr.size}, expected {expected}")
    return arr.reshape(rows, cols) if slices == 1 else arr.reshape(slices, rows, cols)


def hu_to_unit(img):
    img = np.clip(img, HU_MIN, HU_MAX)
    return ((img - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def find_test_samples(test_data_dir: str):
    def index_anatomy(anatomy_dir):
        anatomy = os.path.basename(os.path.normpath(anatomy_dir))
        baseline_dir = os.path.join(anatomy_dir, "Baseline")
        target_dir = os.path.join(anatomy_dir, "Target")
        mask_dir = os.path.join(anatomy_dir, "Mask")

        if not (os.path.isdir(baseline_dir) and os.path.isdir(target_dir)):
            return []

        target_map = {}
        for f in glob.glob(os.path.join(target_dir, "*.raw")):
            m = FNAME_ID_RE.search(os.path.basename(f))
            if m:
                target_map[m.group(1)] = f

        mask_map = {}
        n_materials_map = {}

        if os.path.isdir(mask_dir):
            for f in glob.glob(os.path.join(mask_dir, "*.raw")):
                m = FNAME_ID_RE.search(os.path.basename(f))
                if m:
                    mask_map[m.group(1)] = f

            for f in glob.glob(os.path.join(mask_dir, "*.json")):
                m = re.search(r"metalinfo(\d+)", os.path.basename(f))
                if not m:
                    continue
                try:
                    with open(f, "r", encoding="utf-8") as jf:
                        n_materials_map[m.group(1)] = json.load(jf).get("n_materials")
                except Exception:
                    pass

        samples = []
        for bf in glob.glob(os.path.join(baseline_dir, "*.raw")):
            m = FNAME_ID_RE.search(os.path.basename(bf))
            if not m:
                continue
            image_id = m.group(1)
            if image_id not in target_map:
                continue

            samples.append({
                "id": f"{anatomy}_{image_id}",
                "anatomy": anatomy,
                "baseline": bf,
                "target": target_map[image_id],
                "mask": mask_map.get(image_id),
                "n_materials": n_materials_map.get(image_id),
            })

        return samples

    if os.path.isdir(os.path.join(test_data_dir, "Baseline")):
        return index_anatomy(test_data_dir)

    samples = []
    for name in sorted(os.listdir(test_data_dir)):
        sub = os.path.join(test_data_dir, name)
        if os.path.isdir(sub) and os.path.isdir(os.path.join(sub, "Baseline")):
            samples.extend(index_anatomy(sub))
    return samples


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def load_model(checkpoint_path: str, device):
    net = MambaFormer(in_channels=1).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    if any(k.startswith("module.") for k in state):
        state = {
            k[len("module."):]: v
            for k, v in state.items()
        }

    net.load_state_dict(state, strict=True)
    net.eval()
    return net


# ---------------------------------------------------------------------
# Haar DWT
# ---------------------------------------------------------------------

def haar_dwt2(x: np.ndarray):
    """
    One-level orthonormal 2D Haar DWT.
    x: H x W, H/W even.

    Returns LL, LH, HL, HH with shape H/2 x W/2.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {x.shape}")

    h, w = x.shape
    if h % 2 != 0 or w % 2 != 0:
        x = x[:h - (h % 2), :w - (w % 2)]

    a = x[0::2, 0::2]
    b = x[0::2, 1::2]
    c = x[1::2, 0::2]
    d = x[1::2, 1::2]

    ll = (a + b + c + d) / 2.0
    lh = (-a + b - c + d) / 2.0
    hl = (-a - b + c + d) / 2.0
    hh = (a - b - c + d) / 2.0

    return ll, lh, hl, hh


def downsample_mask_any(mask: np.ndarray):
    """
    Map HxW metal mask to DWT coefficient grid.
    A coefficient is excluded if ANY pixel in its corresponding 2x2 block is metal.
    """
    if mask is None:
        return None

    h, w = mask.shape
    if h % 2 != 0 or w % 2 != 0:
        mask = mask[:h - (h % 2), :w - (w % 2)]

    return (
        mask[0::2, 0::2]
        | mask[0::2, 1::2]
        | mask[1::2, 0::2]
        | mask[1::2, 1::2]
    )


def band_energy(band: np.ndarray, valid_mask: np.ndarray | None):
    if valid_mask is None:
        vals = band.reshape(-1)
    else:
        vals = band[valid_mask]

    if vals.size == 0:
        return float("nan")

    return float(np.mean(np.square(vals, dtype=np.float64)))


def frequency_stats(residual: np.ndarray, metal_mask: np.ndarray | None):
    ll, lh, hl, hh = haar_dwt2(residual)

    metal_dwt = downsample_mask_any(metal_mask)
    valid = None if metal_dwt is None else ~metal_dwt

    e_ll = band_energy(ll, valid)
    e_lh = band_energy(lh, valid)
    e_hl = band_energy(hl, valid)
    e_hh = band_energy(hh, valid)

    hf = e_lh + e_hl + e_hh
    total = e_ll + hf
    hf_ratio = hf / (total + EPS)

    return {
        "ll_energy": e_ll,
        "lh_energy": e_lh,
        "hl_energy": e_hl,
        "hh_energy": e_hh,
        "hf_energy": hf,
        "hf_ratio": hf_ratio,
    }


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def to_uint8_bgr3(img_float01):
    img_u8 = np.clip(img_float01 * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)


def calculate_nonmetal_psnr(pred, gt, metal_mask):
    pred_eval = pred.copy()
    gt_eval = gt.copy()

    if metal_mask is not None:
        pred_eval[metal_mask] = 0.0
        gt_eval[metal_mask] = 0.0

    return float(
        calculate_psnr(
            to_uint8_bgr3(pred_eval),
            to_uint8_bgr3(gt_eval),
            test_y_channel=True,
        )
    )


def corr(xs, ys):
    xy = []
    for x, y in zip(xs, ys):
        try:
            x = float(x)
            y = float(y)
        except Exception:
            continue
        if np.isfinite(x) and np.isfinite(y):
            xy.append((x, y))

    if len(xy) < 3:
        return {
            "n": len(xy),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
        }

    x = np.asarray([v[0] for v in xy])
    y = np.asarray([v[1] for v in xy])

    if HAVE_SCIPY:
        p = pearsonr(x, y)
        s = spearmanr(x, y)
        return {
            "n": len(x),
            "pearson_r": float(p.statistic),
            "pearson_p": float(p.pvalue),
            "spearman_rho": float(s.statistic),
            "spearman_p": float(s.pvalue),
        }

    return {
        "n": len(x),
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "pearson_p": float("nan"),
        "spearman_rho": float("nan"),
        "spearman_p": float("nan"),
    }


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="./frequency_residual_analysis")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    device = torch.device(args.device)

    samples = find_test_samples(args.test_data_dir)
    if not samples:
        raise RuntimeError(f"No AAPM test samples found in {args.test_data_dir}")

    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    print(f"Found {len(samples)} cases.")
    print(f"Loading checkpoint: {args.checkpoint}")
    net = load_model(args.checkpoint, device)

    rows = []

    with torch.inference_mode():
        for idx, s in enumerate(samples):
            input_hu = load_raw(s["baseline"])
            gt_hu = load_raw(s["target"])

            input_img = hu_to_unit(input_hu)
            gt_img = hu_to_unit(gt_hu)

            metal_mask = None
            mask_size = 0
            if s["mask"] is not None:
                metal_mask = load_raw(s["mask"], dtype=np.float32) > 0.5
                mask_size = int(metal_mask.sum())

            input_t = (
                torch.from_numpy((input_img - 0.5) / 0.5)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
                .to(device)
            )

            pred = net(input_t).squeeze().detach().cpu().numpy()
            pred = np.clip(pred, 0.0, 1.0)

            # Residual before and after MARMamba
            residual_in = input_img - gt_img
            residual_out = pred - gt_img

            fin = frequency_stats(residual_in, metal_mask)
            fout = frequency_stats(residual_out, metal_mask)

            hf_retention = fout["hf_energy"] / (fin["hf_energy"] + EPS)
            lh_retention = fout["lh_energy"] / (fin["lh_energy"] + EPS)
            hl_retention = fout["hl_energy"] / (fin["hl_energy"] + EPS)
            hh_retention = fout["hh_energy"] / (fin["hh_energy"] + EPS)
            ll_retention = fout["ll_energy"] / (fin["ll_energy"] + EPS)

            psnr_nm = calculate_nonmetal_psnr(pred, gt_img, metal_mask)

            rows.append({
                "id": s["id"],
                "anatomy": s["anatomy"],
                "n_materials": int(s["n_materials"]) if s["n_materials"] is not None else None,
                "mask_size": mask_size,
                "psnr_nonmetal": psnr_nm,

                "hf_ratio_input": fin["hf_ratio"],
                "hf_ratio_output": fout["hf_ratio"],

                "ll_energy_input": fin["ll_energy"],
                "lh_energy_input": fin["lh_energy"],
                "hl_energy_input": fin["hl_energy"],
                "hh_energy_input": fin["hh_energy"],
                "hf_energy_input": fin["hf_energy"],

                "ll_energy_output": fout["ll_energy"],
                "lh_energy_output": fout["lh_energy"],
                "hl_energy_output": fout["hl_energy"],
                "hh_energy_output": fout["hh_energy"],
                "hf_energy_output": fout["hf_energy"],

                "ll_retention": ll_retention,
                "lh_retention": lh_retention,
                "hl_retention": hl_retention,
                "hh_retention": hh_retention,
                "hf_retention": hf_retention,
            })

            if (idx + 1) % 50 == 0 or idx == 0:
                print(
                    f"Processed {idx + 1}/{len(samples)} | "
                    f"metal={rows[-1]['n_materials']} | "
                    f"HF_ret={hf_retention:.4f} | "
                    f"PSNR={psnr_nm:.3f}"
                )

    # -------------------------------------------------------------
    # Summary by metal count
    # -------------------------------------------------------------
    grouped = defaultdict(list)
    for r in rows:
        if r["n_materials"] is not None:
            grouped[int(r["n_materials"])].append(r)

    summary_metrics = [
        "hf_ratio_input",
        "hf_ratio_output",
        "hf_retention",
        "lh_retention",
        "hl_retention",
        "hh_retention",
        "ll_retention",
        "psnr_nonmetal",
    ]

    summary_rows = []
    for n_materials in sorted(grouped):
        group = grouped[n_materials]
        rec = {
            "n_materials": n_materials,
            "n": len(group),
        }

        for metric in summary_metrics:
            mean, std = mean_std([r[metric] for r in group])
            rec[f"{metric}_mean"] = mean
            rec[f"{metric}_std"] = std

        summary_rows.append(rec)

    # -------------------------------------------------------------
    # Correlations for exactly the three questions
    # -------------------------------------------------------------
    corr_pairs = [
        ("n_materials", "hf_ratio_output"),
        ("n_materials", "hf_retention"),
        ("hf_retention", "psnr_nonmetal"),
        # extra directional detail, still useful and compact
        ("n_materials", "lh_retention"),
        ("n_materials", "hl_retention"),
        ("n_materials", "hh_retention"),
        ("n_materials", "ll_retention"),
    ]

    corr_rows = []
    for x_key, y_key in corr_pairs:
        c = corr(
            [r[x_key] for r in rows],
            [r[y_key] for r in rows],
        )
        corr_rows.append({
            "x": x_key,
            "y": y_key,
            **c,
        })

    per_case_path = os.path.join(args.output_dir, "frequency_per_case.csv")
    summary_path = os.path.join(args.output_dir, "summary_by_metal_count.csv")
    corr_path = os.path.join(args.output_dir, "correlations.csv")
    txt_path = os.path.join(args.output_dir, "hypothesis_summary.txt")

    write_csv(per_case_path, rows)
    write_csv(summary_path, summary_rows)
    write_csv(corr_path, corr_rows)

    # -------------------------------------------------------------
    # Console + text summary
    # -------------------------------------------------------------
    lines = []
    lines.append("=" * 92)
    lines.append("FREQUENCY RESIDUAL DIAGNOSTIC — SUMMARY BY METAL COUNT")
    lines.append("=" * 92)
    lines.append(
        f"{'Metal':>5} {'n':>5} {'HF-ratio-out':>14} "
        f"{'HF-ret':>12} {'LH-ret':>12} {'HL-ret':>12} "
        f"{'HH-ret':>12} {'LL-ret':>12} {'PSNR':>10}"
    )

    for r in summary_rows:
        lines.append(
            f"{int(r['n_materials']):>5d} "
            f"{int(r['n']):>5d} "
            f"{r['hf_ratio_output_mean']:>14.6f} "
            f"{r['hf_retention_mean']:>12.6f} "
            f"{r['lh_retention_mean']:>12.6f} "
            f"{r['hl_retention_mean']:>12.6f} "
            f"{r['hh_retention_mean']:>12.6f} "
            f"{r['ll_retention_mean']:>12.6f} "
            f"{r['psnr_nonmetal_mean']:>10.3f}"
        )

    lines.append("")
    lines.append("KEY CORRELATIONS")
    lines.append("-" * 92)

    for r in corr_rows:
        lines.append(
            f"{r['x']} vs {r['y']}: "
            f"Pearson r={r['pearson_r']:+.4f}, p={r['pearson_p']:.3e}; "
            f"Spearman rho={r['spearman_rho']:+.4f}, p={r['spearman_p']:.3e}"
        )

    lines.append("")
    lines.append("PATTERN NEEDED TO SUPPORT FREQUENCY-AWARE MARMAMBA")
    lines.append("  metal count ↑ -> HF ratio output ↑")
    lines.append("  metal count ↑ -> HF retention ↑")
    lines.append("  HF retention ↑ -> PSNR ↓")
    lines.append("")
    lines.append(
        "If these trends are weak/flat/inconsistent, do NOT proceed with "
        "frequency-aware architecture based on this hypothesis."
    )

    text = "\n".join(lines)
    print("\n" + text)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print("\nSaved:")
    print(f"  {per_case_path}")
    print(f"  {summary_path}")
    print(f"  {corr_path}")
    print(f"  {txt_path}")


if __name__ == "__main__":
    main()

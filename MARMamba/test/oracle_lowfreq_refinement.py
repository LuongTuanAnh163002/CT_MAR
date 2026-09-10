#!/usr/bin/env python3
"""
oracle_lowfreq_refinement.py

Oracle diagnostic for an output-space low-frequency refinement idea.

Goal
----
Before training a new refinement head, measure the MAXIMUM potential benefit
of correcting only coarse/low-frequency residuals.

For each AAPM test case:

    baseline_pred = MARMamba(input)
    residual      = GT - baseline_pred

For each spatial scale s (default: 2, 4, 8):

    coarse_residual = AvgPool(residual, factor=s)
    oracle_delta    = nearest-upsample(coarse_residual)
    oracle_pred     = baseline_pred + oracle_delta

This is an ORACLE because it uses GT to construct the correction.
It is NOT a deployable method. It only answers:

    "If a model could predict the coarse residual perfectly,
     how much PSNR headroom would be available?"

Why nearest upsampling?
-----------------------
For scale=2, AvgPool2d + nearest upsample is equivalent to reconstructing
only the one-level Haar LL component as a piecewise-constant image.
For larger scales it gives progressively coarser low-frequency corrections.

Outputs
-------
- oracle_lowfreq_per_case.csv
- oracle_lowfreq_summary.json
- compact console summary by:
    * metal count 1..5
    * single / multi
    * overall

Recommended interpretation
--------------------------
If oracle PSNR improves only trivially (e.g. <~0.1 dB), do NOT build a
low-frequency refinement head.

If oracle gains are large, especially for 3-5 metal / multi-metal cases,
then an output-space coarse refinement head has meaningful headroom.
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
import torch.nn.functional as F

# -------------------------------------------------------------------------
# Robust repo-root discovery: works if this file is placed in repo root
# OR in repo/test/.
# -------------------------------------------------------------------------

HERE = Path(__file__).resolve()
_repo_candidates = [HERE.parent, HERE.parent.parent]

PROJECT_ROOT = None
for candidate in _repo_candidates:
    if (candidate / "model" / "mamba.py").exists():
        PROJECT_ROOT = candidate
        break

if PROJECT_ROOT is None:
    raise RuntimeError(
        "Cannot locate repository root containing model/mamba.py. "
        "Place this script in the repo root or test/ directory."
    )

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.mamba import MambaFormer
from utils.metrics import calculate_psnr, calculate_rmse


HU_MIN, HU_MAX = -1000.0, 3000.0
FNAME_ID_RE = re.compile(r"img(\d+)")
DIMS_RE = re.compile(r"(\d+)x(\d+)x(\d+)")


# -------------------------------------------------------------------------
# AAPM loading
# -------------------------------------------------------------------------

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
        raise ValueError(
            f"{path}: found {arr.size} values, expected {expected}"
        )

    if slices == 1:
        return arr.reshape(rows, cols)

    return arr.reshape(slices, rows, cols)


def hu_to_unit(img):
    img = np.clip(img, HU_MIN, HU_MAX)
    return ((img - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def find_test_samples(test_data_dir: str):
    def index_one_anatomy(anatomy_dir):
        anatomy = os.path.basename(os.path.normpath(anatomy_dir))

        baseline_dir = os.path.join(anatomy_dir, "Baseline")
        target_dir = os.path.join(anatomy_dir, "Target")
        mask_dir = os.path.join(anatomy_dir, "Mask")

        if not (
            os.path.isdir(baseline_dir)
            and os.path.isdir(target_dir)
        ):
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
                        n_materials_map[m.group(1)] = json.load(jf).get(
                            "n_materials"
                        )
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

    # Case 1: test_data_dir itself is one anatomy folder.
    if os.path.isdir(os.path.join(test_data_dir, "Baseline")):
        return index_one_anatomy(test_data_dir)

    # Case 2: root contains anatomy subfolders.
    samples = []
    for name in sorted(os.listdir(test_data_dir)):
        sub = os.path.join(test_data_dir, name)
        if (
            os.path.isdir(sub)
            and os.path.isdir(os.path.join(sub, "Baseline"))
        ):
            samples.extend(index_one_anatomy(sub))

    return samples


# -------------------------------------------------------------------------
# Baseline model
# -------------------------------------------------------------------------

def extract_state_dict(obj):
    if isinstance(obj, dict) and "model" in obj:
        obj = obj["model"]

    if not isinstance(obj, dict):
        raise TypeError("Checkpoint does not contain a valid state_dict.")

    if any(k.startswith("module.") for k in obj):
        obj = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in obj.items()
        }

    return obj


def load_baseline_model(checkpoint_path: str, device):
    net = MambaFormer(in_channels=1).to(device)

    obj = torch.load(checkpoint_path, map_location=device)
    state = extract_state_dict(obj)

    net.load_state_dict(state, strict=True)
    net.eval()

    return net


# -------------------------------------------------------------------------
# Oracle low-frequency correction
# -------------------------------------------------------------------------

def oracle_coarse_correction(
    pred: np.ndarray,
    gt: np.ndarray,
    scale: int,
):
    """
    Build an oracle low-frequency residual correction.

    residual = gt - pred

    coarse residual is obtained by non-overlapping average pooling with
    factor=scale and nearest upsampling back to original resolution.

    For scale=2, this corresponds to reconstruction using only Haar-LL.
    """
    if scale < 2:
        raise ValueError("scale must be >= 2")

    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch: pred={pred.shape}, gt={gt.shape}"
        )

    h, w = pred.shape

    if h % scale != 0 or w % scale != 0:
        raise ValueError(
            f"Image shape {pred.shape} is not divisible by scale={scale}"
        )

    residual = gt - pred

    residual_t = (
        torch.from_numpy(residual)
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
    )

    coarse = F.avg_pool2d(
        residual_t,
        kernel_size=scale,
        stride=scale,
    )

    correction = F.interpolate(
        coarse,
        size=(h, w),
        mode="nearest",
    )

    correction = correction.squeeze().numpy()

    oracle_pred = np.clip(
        pred + correction,
        0.0,
        1.0,
    )

    # Fraction of float-domain residual energy captured by this coarse
    # projection. With block-average/nearest reconstruction this is a
    # meaningful diagnostic of low-frequency headroom.
    residual_energy = float(np.mean(residual.astype(np.float64) ** 2))
    correction_energy = float(
        np.mean(correction.astype(np.float64) ** 2)
    )

    capture_ratio = correction_energy / (residual_energy + 1e-12)

    return oracle_pred, correction, capture_ratio


# -------------------------------------------------------------------------
# Metric protocol
# -------------------------------------------------------------------------

def to_uint8_bgr3(img_float01):
    img_u8 = np.clip(
        img_float01 * 255.0,
        0,
        255,
    ).astype(np.uint8)

    return cv2.cvtColor(
        img_u8,
        cv2.COLOR_GRAY2BGR,
    )


def evaluate_nonmetal(pred, gt, mask):
    """
    Same non-metal protocol used previously:
    zero BOTH prediction and GT inside the metal mask, then evaluate the
    whole image shape.
    """
    pred_eval = pred.copy()
    gt_eval = gt.copy()

    if mask is not None:
        pred_eval[mask] = 0.0
        gt_eval[mask] = 0.0

    pred_bgr = to_uint8_bgr3(pred_eval)
    gt_bgr = to_uint8_bgr3(gt_eval)

    return {
        "psnr": float(
            calculate_psnr(
                pred_bgr,
                gt_bgr,
                test_y_channel=True,
            )
        ),
        "rmse": float(
            calculate_rmse(
                pred_bgr,
                gt_bgr,
            )
        ),
    }


# -------------------------------------------------------------------------
# Aggregation
# -------------------------------------------------------------------------

def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return float("nan"), float("nan")

    return float(arr.mean()), float(arr.std())


def summarize_rows(rows, scales):
    if not rows:
        return None

    out = {
        "n": len(rows),
    }

    for metric in ("baseline_psnr", "baseline_rmse"):
        m, s = mean_std([r[metric] for r in rows])
        out[f"{metric}_mean"] = m
        out[f"{metric}_std"] = s

    for scale in scales:
        for metric in (
            f"oracle_s{scale}_psnr",
            f"oracle_s{scale}_rmse",
            f"oracle_s{scale}_delta_psnr",
            f"oracle_s{scale}_capture_ratio",
        ):
            m, s = mean_std([r[metric] for r in rows])
            out[f"{metric}_mean"] = m
            out[f"{metric}_std"] = s

    return out


def print_group_summary(title, grouped, scales):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    header = (
        f"{'Group':>10} {'n':>5} {'Baseline':>10}"
    )

    for scale in scales:
        header += (
            f" | {'Oracle@'+str(scale):>10} "
            f"{'ΔPSNR':>9} {'Capture':>9}"
        )

    print(header)

    for group_name, rows in grouped:
        s = summarize_rows(rows, scales)
        if s is None:
            continue

        line = (
            f"{str(group_name):>10} "
            f"{s['n']:>5d} "
            f"{s['baseline_psnr_mean']:>10.3f}"
        )

        for scale in scales:
            line += (
                f" | "
                f"{s[f'oracle_s{scale}_psnr_mean']:>10.3f} "
                f"{s[f'oracle_s{scale}_delta_psnr_mean']:>+9.3f} "
                f"{s[f'oracle_s{scale}_capture_ratio_mean']:>9.3f}"
            )

        print(line)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Oracle low-frequency output-refinement diagnostic."
    )

    parser.add_argument(
        "--test_data_dir",
        required=True,
        help="AAPM test root.",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Best BASELINE MARMamba checkpoint.",
    )

    parser.add_argument(
        "--output_dir",
        default="./oracle_lowfreq_refinement",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--scales",
        nargs="+",
        type=int,
        default=[2, 4, 8],
        help=(
            "Coarse block sizes to test. "
            "scale=2 corresponds to one-level Haar-LL reconstruction."
        ),
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional quick test, e.g. --max_samples 20.",
    )

    args = parser.parse_args()

    scales = sorted(set(args.scales))

    if any(s < 2 for s in scales):
        raise ValueError("All scales must be >= 2.")

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False."
        )

    device = torch.device(args.device)

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    samples = find_test_samples(
        args.test_data_dir
    )

    if not samples:
        raise RuntimeError(
            f"No AAPM test samples found in {args.test_data_dir}"
        )

    samples = sorted(
        samples,
        key=lambda x: x["id"],
    )

    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    print(f"Found {len(samples)} cases.")
    print(f"Scales: {scales}")
    print(f"Loading baseline checkpoint: {args.checkpoint}")

    net = load_baseline_model(
        args.checkpoint,
        device,
    )

    rows = []

    with torch.inference_mode():
        for idx, sample in enumerate(samples):
            input_img = hu_to_unit(
                load_raw(sample["baseline"])
            )

            gt_img = hu_to_unit(
                load_raw(sample["target"])
            )

            mask = None
            if sample["mask"] is not None:
                mask = (
                    load_raw(
                        sample["mask"],
                        dtype=np.float32,
                    )
                    > 0.5
                )

            n_materials = sample["n_materials"]

            if n_materials is None:
                raise RuntimeError(
                    f"Missing n_materials for {sample['id']}. "
                    "Check metalinfo*.json indexing."
                )

            n_materials = int(n_materials)

            input_t = (
                torch.from_numpy(
                    (input_img - 0.5) / 0.5
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
                .to(device)
            )

            pred = (
                net(input_t)
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )

            pred = np.clip(
                pred,
                0.0,
                1.0,
            )

            base_metrics = evaluate_nonmetal(
                pred,
                gt_img,
                mask,
            )

            row = {
                "id": sample["id"],
                "anatomy": sample["anatomy"],
                "n_materials": n_materials,
                "group": (
                    "multi"
                    if n_materials >= 2
                    else "single"
                ),
                "baseline_psnr": base_metrics["psnr"],
                "baseline_rmse": base_metrics["rmse"],
            }

            for scale in scales:
                oracle_pred, correction, capture_ratio = (
                    oracle_coarse_correction(
                        pred,
                        gt_img,
                        scale,
                    )
                )

                oracle_metrics = evaluate_nonmetal(
                    oracle_pred,
                    gt_img,
                    mask,
                )

                row[f"oracle_s{scale}_psnr"] = (
                    oracle_metrics["psnr"]
                )

                row[f"oracle_s{scale}_rmse"] = (
                    oracle_metrics["rmse"]
                )

                row[f"oracle_s{scale}_delta_psnr"] = (
                    oracle_metrics["psnr"]
                    - base_metrics["psnr"]
                )

                row[f"oracle_s{scale}_capture_ratio"] = (
                    capture_ratio
                )

            rows.append(row)

            if (idx + 1) % 50 == 0 or idx == 0:
                msg = (
                    f"Processed {idx+1}/{len(samples)} | "
                    f"metal={n_materials} | "
                    f"baseline={base_metrics['psnr']:.3f}"
                )

                for scale in scales:
                    msg += (
                        f" | s{scale} "
                        f"Δ={row[f'oracle_s{scale}_delta_psnr']:+.3f}"
                    )

                print(msg)

    # ------------------------------------------------------------------
    # Save per-case CSV
    # ------------------------------------------------------------------

    csv_path = os.path.join(
        args.output_dir,
        "oracle_lowfreq_per_case.csv",
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------------------
    # Group summaries
    # ------------------------------------------------------------------

    by_count = []
    for n in range(1, 6):
        group_rows = [
            r for r in rows
            if r["n_materials"] == n
        ]
        by_count.append(
            (f"{n} metal", group_rows)
        )

    single_rows = [
        r for r in rows
        if r["group"] == "single"
    ]

    multi_rows = [
        r for r in rows
        if r["group"] == "multi"
    ]

    overall_rows = rows

    print_group_summary(
        "ORACLE LOW-FREQUENCY HEADROOM — BY METAL COUNT",
        by_count,
        scales,
    )

    print_group_summary(
        "ORACLE LOW-FREQUENCY HEADROOM — SINGLE / MULTI",
        [
            ("single", single_rows),
            ("multi", multi_rows),
            ("overall", overall_rows),
        ],
        scales,
    )

    # ------------------------------------------------------------------
    # Save JSON summary
    # ------------------------------------------------------------------

    summary = {
        "scales": scales,
        "by_metal_count": {
            str(n): summarize_rows(
                [
                    r for r in rows
                    if r["n_materials"] == n
                ],
                scales,
            )
            for n in range(1, 6)
        },
        "single": summarize_rows(
            single_rows,
            scales,
        ),
        "multi": summarize_rows(
            multi_rows,
            scales,
        ),
        "overall": summarize_rows(
            overall_rows,
            scales,
        ),
    }

    json_path = os.path.join(
        args.output_dir,
        "oracle_lowfreq_summary.json",
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {json_path}")

    print(
        "\nInterpretation:"
        "\n  Large positive ΔPSNR, especially for 3-5 metals / multi, "
        "means coarse output correction has real headroom."
        "\n  Tiny ΔPSNR means the output-space low-frequency refinement "
        "idea is not worth training."
    )


if __name__ == "__main__":
    main()

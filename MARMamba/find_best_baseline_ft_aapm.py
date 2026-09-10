"""
find_best_baseline_ft_aapm.py
=============================

Chọn best checkpoint cho Baseline-FT bằng CÙNG protocol validation
đã dùng cho Coarse Refine:

- AAPM raw HU -> [0, 1]
- input model normalize -> [-1, 1]
- NON-METAL evaluation:
    pred[metal_mask] = 0
    gt[metal_mask]   = 0
- PSNR / SSIM / RMSE tính trên uint8 BGR
- PSNR dùng test_y_channel=True
- Best checkpoint = checkpoint có OVERALL NON-METAL PSNR cao nhất

Script sẽ:
1. Tìm các checkpoint dạng "<step>_ckpt" trong checkpoint_dir.
2. Chỉ evaluate các checkpoint trong khoảng start_step..end_step.
3. Tính Overall + Single/Multi + 1..5 metal.
4. Xếp hạng checkpoint theo Overall Non-metal PSNR.
5. Lưu toàn bộ kết quả ra JSON.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.mamba import MambaFormer
from utils.metrics import calculate_psnr, calculate_ssim, calculate_rmse


HU_MIN, HU_MAX = -1000.0, 3000.0
FNAME_ID_RE = re.compile(r"img(\d+)")
DIMS_RE = re.compile(r"(\d+)x(\d+)x(\d+)")
STEP_RE = re.compile(r"^(\d+)_ckpt$")


# ============================================================================
# AAPM data helpers
# ============================================================================

def parse_dims(filename):
    m = DIMS_RE.search(filename)
    if not m:
        raise ValueError(f"Cannot parse dimensions from: {filename}")

    w, h, d = (int(x) for x in m.groups())
    return h, w, d


def load_raw(path, dtype=np.float32):
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


def find_eval_samples(root_dir):
    def index_one(anatomy_dir):
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

            samples.append(
                {
                    "id": f"{anatomy}_{image_id}",
                    "baseline": bf,
                    "target": target_map[image_id],
                    "mask": mask_map.get(image_id),
                    "n_materials": n_materials_map.get(image_id),
                }
            )

        return samples

    if os.path.isdir(os.path.join(root_dir, "Baseline")):
        samples = index_one(root_dir)
    else:
        samples = []
        for name in sorted(os.listdir(root_dir)):
            sub = os.path.join(root_dir, name)
            if (
                os.path.isdir(sub)
                and os.path.isdir(os.path.join(sub, "Baseline"))
            ):
                samples.extend(index_one(sub))

    return sorted(samples, key=lambda x: x["id"])


# ============================================================================
# Metric helpers
# ============================================================================

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


def metric_one_non_metal(pred, gt, mask):
    """
    Same NON-METAL protocol as Coarse Refine validation:
    zero BOTH prediction and GT inside metal mask.
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
        "ssim": float(
            calculate_ssim(
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


def summarize(records):
    if not records:
        return None

    result = {"n": len(records)}

    for metric in ("psnr", "ssim", "rmse"):
        arr = np.asarray(
            [r["metrics"][metric] for r in records],
            dtype=np.float64,
        )
        result[metric] = float(arr.mean())
        result[f"{metric}_std"] = float(arr.std())

    return result


# ============================================================================
# Checkpoint helpers
# ============================================================================

def find_checkpoints(
    checkpoint_dir,
    start_step=None,
    end_step=None,
    step_interval=None,
):
    found = []

    for path in glob.glob(os.path.join(checkpoint_dir, "*_ckpt")):
        name = os.path.basename(path)
        m = STEP_RE.match(name)
        if not m:
            continue

        step = int(m.group(1))

        if start_step is not None and step < start_step:
            continue

        if end_step is not None and step > end_step:
            continue

        if (
            step_interval is not None
            and start_step is not None
            and (step - start_step) % step_interval != 0
        ):
            continue

        found.append((step, path))

    found.sort(key=lambda x: x[0])

    if not found:
        raise RuntimeError(
            f"No checkpoints found in {checkpoint_dir} "
            "for the requested step range."
        )

    return found


def normalize_state_dict(state):
    # Support training-state dicts if needed.
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    # Baseline training used nn.DataParallel, so checkpoints usually contain
    # keys beginning with "module.".
    if any(k.startswith("module.") for k in state.keys()):
        state = {
            (
                k[len("module."):]
                if k.startswith("module.")
                else k
            ): v
            for k, v in state.items()
        }

    return state


def load_checkpoint_into_model(model, checkpoint, device):
    state = torch.load(
        checkpoint,
        map_location=device,
    )
    state = normalize_state_dict(state)

    model.load_state_dict(
        state,
        strict=True,
    )
    model.eval()


# ============================================================================
# Validation
# ============================================================================

@torch.inference_mode()
def evaluate_checkpoint(
    model,
    checkpoint,
    samples,
    device,
):
    load_checkpoint_into_model(
        model,
        checkpoint,
        device,
    )

    records = []
    start = time.time()

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

        input_t = (
            torch.from_numpy(
                (input_img - 0.5) / 0.5
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(device)
        )

        pred_t = model(input_t)

        pred = np.clip(
            pred_t.squeeze().cpu().numpy(),
            0.0,
            1.0,
        )

        n_materials = int(sample["n_materials"])

        records.append(
            {
                "id": sample["id"],
                "n_materials": n_materials,
                "group": (
                    "multi"
                    if n_materials >= 2
                    else "single"
                ),
                "metrics": metric_one_non_metal(
                    pred,
                    gt_img,
                    mask,
                ),
            }
        )

        if (idx + 1) % 100 == 0:
            print(
                f"    processed {idx + 1}/{len(samples)}"
            )

    overall = summarize(records)

    single_records = [
        r for r in records
        if r["group"] == "single"
    ]
    multi_records = [
        r for r in records
        if r["group"] == "multi"
    ]

    result = {
        "overall": overall,
        "single": summarize(single_records),
        "multi": summarize(multi_records),
        "by_metal_count": {},
        "seconds": float(time.time() - start),
    }

    for n in range(1, 6):
        group_records = [
            r for r in records
            if r["n_materials"] == n
        ]
        result["by_metal_count"][str(n)] = summarize(
            group_records
        )

    return result


def print_result(step, result):
    o = result["overall"]
    m = result["multi"]

    print(
        f"step {step:>7} | "
        f"Overall PSNR={o['psnr']:.6f} "
        f"SSIM={o['ssim']:.6f} "
        f"RMSE={o['rmse']:.6f} | "
        f"Multi PSNR={m['psnr']:.6f} | "
        f"time={result['seconds']:.1f}s"
    )


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Select best Baseline-FT checkpoint by "
            "Overall Non-metal Validation PSNR."
        )
    )

    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Directory containing checkpoints such as 297500_ckpt, 298000_ckpt, ...",
    )
    parser.add_argument(
        "--val_data_dir",
        required=True,
        help="AAPM validation directory; must be the same validation set used for Coarse Refine.",
    )
    parser.add_argument(
        "--output_dir",
        default="./baseline_ft_checkpoint_selection",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
    )
    parser.add_argument(
        "--start_step",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--end_step",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--step_interval",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    device = torch.device(args.device)

    samples = find_eval_samples(
        args.val_data_dir
    )

    if not samples:
        raise RuntimeError(
            f"No validation samples found in {args.val_data_dir}"
        )

    missing_n = [
        s["id"]
        for s in samples
        if s["n_materials"] is None
    ]

    if missing_n:
        raise RuntimeError(
            "Validation metadata is missing n_materials for "
            f"{len(missing_n)} cases. First examples: {missing_n[:5]}"
        )

    count_hist = {
        n: 0
        for n in range(1, 6)
    }

    for s in samples:
        n = int(s["n_materials"])
        if n not in count_hist:
            raise RuntimeError(
                f"Unexpected n_materials={n} for {s['id']}"
            )
        count_hist[n] += 1

    checkpoints = find_checkpoints(
        args.checkpoint_dir,
        start_step=args.start_step,
        end_step=args.end_step,
        step_interval=args.step_interval,
    )

    print(
        f"Validation samples: {len(samples)}"
    )
    print(
        f"Metal-count histogram: {count_hist}"
    )
    print(
        f"Checkpoints found: {len(checkpoints)}"
    )

    # Single model allocation; only weights are reloaded per checkpoint.
    model = MambaFormer(
        in_channels=1,
    ).to(device)

    all_results = []

    for idx, (step, checkpoint) in enumerate(
        checkpoints,
        start=1,
    ):
        print(
            f"\n[{idx}/{len(checkpoints)}] "
            f"Evaluating step {step}: {checkpoint}"
        )

        result = evaluate_checkpoint(
            model,
            checkpoint,
            samples,
            device,
        )

        print_result(
            step,
            result,
        )

        all_results.append(
            {
                "step": step,
                "checkpoint": checkpoint,
                **result,
            }
        )

        # Save incrementally in case evaluation is interrupted.
        partial_path = os.path.join(
            args.output_dir,
            "checkpoint_results.json",
        )

        with open(
            partial_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "selection_rule": (
                        "highest overall non-metal validation PSNR"
                    ),
                    "n_samples": len(samples),
                    "metal_count_histogram": {
                        str(k): int(v)
                        for k, v in count_hist.items()
                    },
                    "results": all_results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    # ------------------------------------------------------------------------
    # Select best by SAME criterion we want to use for Coarse Refine:
    # OVERALL NON-METAL PSNR.
    # ------------------------------------------------------------------------
    ranking = sorted(
        all_results,
        key=lambda x: x["overall"]["psnr"],
        reverse=True,
    )

    best = ranking[0]

    print(
        "\n"
        + "=" * 100
    )
    print(
        "BEST BASELINE-FT CHECKPOINT "
        "(Overall Non-metal Validation PSNR)"
    )
    print(
        "=" * 100
    )

    print(
        f"Step:       {best['step']}"
    )
    print(
        f"Checkpoint: {best['checkpoint']}"
    )
    print(
        f"PSNR:       {best['overall']['psnr']:.8f}"
    )
    print(
        f"SSIM:       {best['overall']['ssim']:.8f}"
    )
    print(
        f"RMSE:       {best['overall']['rmse']:.8f}"
    )
    print(
        f"Multi PSNR: {best['multi']['psnr']:.8f}"
    )

    print(
        "\nTop checkpoints:"
    )

    for rank, item in enumerate(
        ranking[: min(10, len(ranking))],
        start=1,
    ):
        print(
            f"{rank:>2}. "
            f"step={item['step']:>7} | "
            f"Overall PSNR={item['overall']['psnr']:.6f} | "
            f"SSIM={item['overall']['ssim']:.6f} | "
            f"RMSE={item['overall']['rmse']:.6f} | "
            f"Multi PSNR={item['multi']['psnr']:.6f}"
        )

    final_path = os.path.join(
        args.output_dir,
        "checkpoint_ranking.json",
    )

    with open(
        final_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "selection_rule": (
                    "highest overall non-metal validation PSNR"
                ),
                "best": best,
                "ranking": ranking,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    best_txt = os.path.join(
        args.output_dir,
        "best_checkpoint.txt",
    )

    with open(
        best_txt,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            f"selection_rule=highest overall non-metal validation PSNR\n"
            f"step={best['step']}\n"
            f"checkpoint={best['checkpoint']}\n"
            f"psnr={best['overall']['psnr']}\n"
            f"ssim={best['overall']['ssim']}\n"
            f"rmse={best['overall']['rmse']}\n"
            f"multi_psnr={best['multi']['psnr']}\n"
        )

    print(
        f"\nSaved ranking: {final_path}"
    )
    print(
        f"Saved best info: {best_txt}"
    )


if __name__ == "__main__":
    main()

"""Paired inference, per-image evaluation, and case recommendation.

This script evaluates an independently trained vanilla MARMamba checkpoint and
a Dynamic-Gated Multi-Scale Coarse checkpoint on exactly the same AAPM samples.
It writes per-image metrics/deltas, saves qualitative previews, and recommends
representative cases for the paper.

Delta convention (positive always means that Gate is better):
    delta_psnr  = gate_psnr  - vanilla_psnr
    delta_ssim  = gate_ssim  - vanilla_ssim
    delta_rmse  = vanilla_rmse  - gate_rmse
    delta_lpips = vanilla_lpips - gate_lpips
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (
    SCRIPT_DIR
    if (SCRIPT_DIR / "model").is_dir()
    else SCRIPT_DIR.parent
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.mamba import MambaFormer
from model.mamba_multiscale_coarse_gate import (
    GatedMultiScaleCoarseRefinedMambaFormer,
)
from utils.metrics import calculate_psnr, calculate_rmse, calculate_ssim


HU_MIN = -1000.0
HU_MAX = 3000.0
FNAME_ID_RE = re.compile(r"img(\d+)")
DIMS_RE = re.compile(r"(\d+)x(\d+)x(\d+)")
REGIONS = ("non_metal", "metal_included")
METRICS = ("psnr", "ssim", "rmse", "lpips")
SIZE_GROUPS = ("Large", "Medium", "Small", "Tiny")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run paired Vanilla/Gate inference, compute per-image deltas, "
            "and recommend qualitative examples."
        )
    )
    parser.add_argument("--test_data_dir", required=True)
    parser.add_argument("--vanilla_checkpoint", required=True)
    parser.add_argument("--gate_checkpoint", required=True)
    parser.add_argument("--output_dir", default="./paired_vanilla_gate")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden_channels", default=32, type=int)
    parser.add_argument("--gate_hidden", default=32, type=int)
    parser.add_argument("--gate_range", default=0.5, type=float)
    parser.add_argument(
        "--ranking_region",
        choices=REGIONS,
        default="non_metal",
        help="Region used to rank and recommend images.",
    )
    parser.add_argument(
        "--top_k",
        default=3,
        type=int,
        help="Number of samples listed for each automatic ranking.",
    )
    parser.add_argument(
        "--save_npy",
        action="store_true",
        help=(
            "Also save input/GT/predictions as float16 NPY files. This may "
            "require roughly 2 GB for 1,000 images at 512x512."
        ),
    )
    parser.add_argument(
        "--error_max",
        default=0.10,
        type=float,
        help="Fixed maximum absolute error represented by the heatmap.",
    )
    return parser.parse_args()


def parse_dims(filename):
    match = DIMS_RE.search(filename)
    if not match:
        raise ValueError(f"Cannot parse dimensions from: {filename}")
    width, height, depth = [int(value) for value in match.groups()]
    return height, width, depth


def load_raw(path, dtype=np.float32):
    rows, cols, slices = parse_dims(os.path.basename(path))
    array = np.fromfile(path, dtype=dtype)
    expected = rows * cols * slices
    if array.size != expected:
        raise ValueError(f"{path}: found {array.size}, expected {expected}")
    if slices == 1:
        return array.reshape(rows, cols)
    return array.reshape(slices, rows, cols)


def hu_to_unit(image):
    image = np.clip(image, HU_MIN, HU_MAX)
    return ((image - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def find_test_samples(test_data_dir):
    def index_anatomy(anatomy_dir):
        anatomy = os.path.basename(os.path.normpath(anatomy_dir))
        baseline_dir = os.path.join(anatomy_dir, "Baseline")
        target_dir = os.path.join(anatomy_dir, "Target")
        mask_dir = os.path.join(anatomy_dir, "Mask")

        if not os.path.isdir(baseline_dir) or not os.path.isdir(target_dir):
            return []

        target_map = {}
        for path in glob.glob(os.path.join(target_dir, "*.raw")):
            match = FNAME_ID_RE.search(os.path.basename(path))
            if match:
                target_map[match.group(1)] = path

        mask_map = {}
        metal_count_map = {}
        if os.path.isdir(mask_dir):
            for path in glob.glob(os.path.join(mask_dir, "*.raw")):
                match = FNAME_ID_RE.search(os.path.basename(path))
                if match:
                    mask_map[match.group(1)] = path

            for path in glob.glob(os.path.join(mask_dir, "*.json")):
                match = re.search(r"metalinfo(\d+)", os.path.basename(path))
                if not match:
                    continue
                try:
                    with open(path, "r", encoding="utf-8") as file:
                        metal_count_map[match.group(1)] = json.load(file).get(
                            "n_materials"
                        )
                except (OSError, json.JSONDecodeError):
                    pass

        samples = []
        for baseline_path in glob.glob(os.path.join(baseline_dir, "*.raw")):
            match = FNAME_ID_RE.search(os.path.basename(baseline_path))
            if not match:
                continue
            image_id = match.group(1)
            if image_id not in target_map:
                continue
            samples.append(
                {
                    "id": f"{anatomy}_{image_id}",
                    "anatomy": anatomy,
                    "image_id": image_id,
                    "baseline": baseline_path,
                    "target": target_map[image_id],
                    "mask": mask_map.get(image_id),
                    "n_materials": metal_count_map.get(image_id),
                }
            )
        return samples

    if os.path.isdir(os.path.join(test_data_dir, "Baseline")):
        samples = index_anatomy(test_data_dir)
    else:
        samples = []
        for name in sorted(os.listdir(test_data_dir)):
            anatomy_dir = os.path.join(test_data_dir, name)
            if os.path.isdir(os.path.join(anatomy_dir, "Baseline")):
                samples.extend(index_anatomy(anatomy_dir))

    # A stable order is essential for a paired comparison.
    return sorted(samples, key=lambda item: item["id"])


def size_to_group(mask_size, q25, q50, q75):
    if mask_size >= q75:
        return "Large"
    if mask_size >= q50:
        return "Medium"
    if mask_size >= q25:
        return "Small"
    return "Tiny"


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a state_dict.")
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in checkpoint.items()
    }


def load_vanilla(checkpoint_path, device):
    model = MambaFormer(in_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    model.eval()
    return model


def load_gate(
    checkpoint_path,
    device,
    hidden_channels,
    gate_hidden,
    gate_range,
):
    model = GatedMultiScaleCoarseRefinedMambaFormer(
        in_channels=1,
        hidden_channels=hidden_channels,
        gate_hidden=gate_hidden,
        gate_range=gate_range,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    model.eval()
    return model


def to_uint8_gray(image_float01):
    return np.clip(image_float01 * 255.0, 0, 255).astype(np.uint8)


def to_uint8_bgr3(image_float01):
    return cv2.cvtColor(to_uint8_gray(image_float01), cv2.COLOR_GRAY2BGR)


def to_lpips_tensor(image_bgr_uint8, device):
    image_rgb = cv2.cvtColor(image_bgr_uint8, cv2.COLOR_BGR2RGB)
    image = image_rgb.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)


def evaluate_prediction(prediction, target, mask, lpips_fn, device, exclude_metal):
    prediction_eval = prediction.copy()
    target_eval = target.copy()
    if exclude_metal and mask is not None:
        prediction_eval[mask] = 0.0
        target_eval[mask] = 0.0

    prediction_bgr = to_uint8_bgr3(prediction_eval)
    target_bgr = to_uint8_bgr3(target_eval)

    return {
        "psnr": float(
            calculate_psnr(prediction_bgr, target_bgr, test_y_channel=True)
        ),
        "ssim": float(
            calculate_ssim(prediction_bgr, target_bgr, test_y_channel=True)
        ),
        "rmse": float(calculate_rmse(prediction_bgr, target_bgr)),
        "lpips": float(
            lpips_fn(
                to_lpips_tensor(prediction_bgr, device),
                to_lpips_tensor(target_bgr, device),
            ).item()
        ),
    }


def improvement_delta(vanilla_metrics, gate_metrics):
    return {
        "psnr": gate_metrics["psnr"] - vanilla_metrics["psnr"],
        "ssim": gate_metrics["ssim"] - vanilla_metrics["ssim"],
        "rmse": vanilla_metrics["rmse"] - gate_metrics["rmse"],
        "lpips": vanilla_metrics["lpips"] - gate_metrics["lpips"],
    }


def ensure_output_directories(output_dir, save_npy):
    output_dir = Path(output_dir)
    subdirectories = (
        "input",
        "ground_truth",
        "vanilla",
        "coarse_gate",
        "metal_mask",
        "error_vanilla",
        "error_gate",
        "improvement_map",
    )
    for subdirectory in subdirectories:
        (output_dir / "previews" / subdirectory).mkdir(parents=True, exist_ok=True)

    if save_npy:
        for subdirectory in ("input", "ground_truth", "vanilla", "coarse_gate"):
            (output_dir / "arrays" / subdirectory).mkdir(parents=True, exist_ok=True)
    return output_dir


def save_heatmap(error, path, error_max):
    normalized = np.clip(error / error_max, 0.0, 1.0)
    heatmap = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(path), heatmap)


def save_improvement_map(vanilla_error, gate_error, path, error_max):
    # Positive (Gate improves) is red; negative (Gate worsens) is blue.
    improvement = vanilla_error - gate_error
    normalized = np.clip(improvement / error_max, -1.0, 1.0)
    canvas = np.zeros((*normalized.shape, 3), dtype=np.uint8)
    positive = np.clip(normalized, 0.0, 1.0)
    negative = np.clip(-normalized, 0.0, 1.0)
    canvas[..., 2] = (positive * 255).astype(np.uint8)
    canvas[..., 0] = (negative * 255).astype(np.uint8)
    cv2.imwrite(str(path), canvas)


def save_case_outputs(
    output_dir,
    sample_id,
    input_image,
    target,
    vanilla,
    gate,
    mask,
    error_max,
    save_npy,
):
    preview_dir = output_dir / "previews"
    cv2.imwrite(str(preview_dir / "input" / f"{sample_id}.png"), to_uint8_gray(input_image))
    cv2.imwrite(
        str(preview_dir / "ground_truth" / f"{sample_id}.png"),
        to_uint8_gray(target),
    )
    cv2.imwrite(str(preview_dir / "vanilla" / f"{sample_id}.png"), to_uint8_gray(vanilla))
    cv2.imwrite(
        str(preview_dir / "coarse_gate" / f"{sample_id}.png"),
        to_uint8_gray(gate),
    )

    mask_image = np.zeros_like(target, dtype=np.uint8)
    if mask is not None:
        mask_image[mask] = 255
    cv2.imwrite(str(preview_dir / "metal_mask" / f"{sample_id}.png"), mask_image)

    vanilla_error = np.abs(vanilla - target)
    gate_error = np.abs(gate - target)
    save_heatmap(
        vanilla_error,
        preview_dir / "error_vanilla" / f"{sample_id}.png",
        error_max,
    )
    save_heatmap(
        gate_error,
        preview_dir / "error_gate" / f"{sample_id}.png",
        error_max,
    )
    save_improvement_map(
        vanilla_error,
        gate_error,
        preview_dir / "improvement_map" / f"{sample_id}.png",
        error_max,
    )

    if save_npy:
        array_dir = output_dir / "arrays"
        np.save(array_dir / "input" / f"{sample_id}.npy", input_image.astype(np.float16))
        np.save(array_dir / "ground_truth" / f"{sample_id}.npy", target.astype(np.float16))
        np.save(array_dir / "vanilla" / f"{sample_id}.npy", vanilla.astype(np.float16))
        np.save(array_dir / "coarse_gate" / f"{sample_id}.npy", gate.astype(np.float16))


def flatten_entry(entry):
    row = {
        "sample_id": entry["sample_id"],
        "anatomy": entry["anatomy"],
        "image_id": entry["image_id"],
        "n_materials": entry["n_materials"],
        "mask_size": entry["mask_size"],
        "size_group": entry["size_group"],
        "alpha2": entry["alpha2"],
        "alpha4": entry["alpha4"],
        "refinement_abs_mean": entry["refinement_abs_mean"],
        "prediction_abs_difference": entry["prediction_abs_difference"],
    }
    for region in REGIONS:
        for metric in METRICS:
            row[f"{region}_vanilla_{metric}"] = entry[region]["vanilla"][metric]
            row[f"{region}_gate_{metric}"] = entry[region]["gate"][metric]
            row[f"{region}_delta_{metric}"] = entry[region]["delta"][metric]
    return row


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def z_scores(values):
    values = np.asarray(values, dtype=np.float64)
    std = values.std()
    if std < 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def add_ranking_scores(rows, region):
    for metric in METRICS:
        key = f"{region}_delta_{metric}"
        normalized = z_scores([row[key] for row in rows])
        for row, score in zip(rows, normalized):
            row[f"z_delta_{metric}"] = float(score)

    for row in rows:
        row["fidelity_score"] = float(
            np.mean(
                [
                    row["z_delta_psnr"],
                    row["z_delta_ssim"],
                    row["z_delta_rmse"],
                ]
            )
        )
        row["all_metrics_score"] = float(
            np.mean([row[f"z_delta_{metric}"] for metric in METRICS])
        )
        row["change_magnitude_score"] = float(
            np.mean([abs(row[f"z_delta_{metric}"]) for metric in METRICS])
        )

        d_psnr = row[f"{region}_delta_psnr"]
        d_ssim = row[f"{region}_delta_ssim"]
        d_rmse = row[f"{region}_delta_rmse"]
        d_lpips = row[f"{region}_delta_lpips"]
        row["is_pixel_perceptual_tradeoff"] = bool(
            d_psnr > 0 and d_ssim > 0 and d_rmse > 0 and d_lpips < 0
        )
        row["is_reverse_tradeoff"] = bool(
            d_psnr < 0 and d_ssim < 0 and d_rmse < 0 and d_lpips > 0
        )
        row["is_true_failure"] = bool(
            d_psnr < 0 and d_ssim < 0 and d_rmse < 0 and d_lpips < 0
        )

    alpha2_median = float(np.median([row["alpha2"] for row in rows]))
    alpha4_median = float(np.median([row["alpha4"] for row in rows]))
    alpha2_scale = max(float(np.std([row["alpha2"] for row in rows])), 1e-12)
    alpha4_scale = max(float(np.std([row["alpha4"] for row in rows])), 1e-12)
    for row in rows:
        row["alpha_deviation_score"] = float(
            0.5
            * (
                abs(row["alpha2"] - alpha2_median) / alpha2_scale
                + abs(row["alpha4"] - alpha4_median) / alpha4_scale
            )
        )


def compact_case(row, region):
    return {
        "sample_id": row["sample_id"],
        "anatomy": row["anatomy"],
        "n_materials": int(row["n_materials"]),
        "size_group": row["size_group"],
        "delta_psnr": row[f"{region}_delta_psnr"],
        "delta_ssim": row[f"{region}_delta_ssim"],
        "delta_rmse": row[f"{region}_delta_rmse"],
        "delta_lpips": row[f"{region}_delta_lpips"],
        "fidelity_score": row["fidelity_score"],
        "all_metrics_score": row["all_metrics_score"],
        "change_magnitude_score": row["change_magnitude_score"],
        "alpha2": row["alpha2"],
        "alpha4": row["alpha4"],
        "alpha_deviation_score": row["alpha_deviation_score"],
        "is_pixel_perceptual_tradeoff": row["is_pixel_perceptual_tradeoff"],
        "is_reverse_tradeoff": row["is_reverse_tradeoff"],
        "is_true_failure": row["is_true_failure"],
    }


def top_rows(rows, key, top_k, reverse=True):
    return sorted(rows, key=lambda row: row[key], reverse=reverse)[:top_k]


def nearest_rows(rows, key_function, top_k):
    return sorted(rows, key=key_function)[:top_k]


def first_unique(candidates, used_ids):
    for row in candidates:
        if row["sample_id"] not in used_ids:
            used_ids.add(row["sample_id"])
            return row
    return None


def build_recommendations(rows, region, top_k):
    delta_psnr_key = f"{region}_delta_psnr"
    delta_ssim_key = f"{region}_delta_ssim"
    delta_lpips_key = f"{region}_delta_lpips"

    tradeoff_rows = [row for row in rows if row["is_pixel_perceptual_tradeoff"]]
    reverse_tradeoff_rows = [row for row in rows if row["is_reverse_tradeoff"]]
    true_failure_rows = [row for row in rows if row["is_true_failure"]]

    # More negative delta LPIPS means a stronger perceptual deterioration.
    # Fidelity score is used as a tie breaker so the selected examples still
    # exhibit a meaningful pixel-level improvement.
    strongest_tradeoffs = sorted(
        tradeoff_rows,
        key=lambda row: (
            row[delta_lpips_key],
            -row["fidelity_score"],
        ),
    )[:top_k]

    rankings = {
        "top_delta_psnr": top_rows(rows, delta_psnr_key, top_k),
        "top_delta_ssim": top_rows(rows, delta_ssim_key, top_k),
        "top_fidelity_score": top_rows(rows, "fidelity_score", top_k),
        "top_all_metrics_score": top_rows(rows, "all_metrics_score", top_k),
        "limited_improvement_cases": top_rows(
            rows, "change_magnitude_score", top_k, reverse=False
        ),
        "largest_lpips_improvement": top_rows(rows, delta_lpips_key, top_k),
        "strongest_pixel_perceptual_tradeoffs": strongest_tradeoffs,
        "reverse_tradeoffs": top_rows(
            reverse_tradeoff_rows, delta_lpips_key, top_k
        ),
        "true_failures_all_metrics": top_rows(
            true_failure_rows, "all_metrics_score", top_k, reverse=False
        ),
        "worst_cases_regardless_of_sign": top_rows(
            rows, "all_metrics_score", top_k, reverse=False
        ),
        "largest_gate_weight_deviation": top_rows(
            rows, "alpha_deviation_score", top_k
        ),
        "smallest_gate_weight_deviation": top_rows(
            rows, "alpha_deviation_score", top_k, reverse=False
        ),
    }

    median_score = float(np.median([row["all_metrics_score"] for row in rows]))
    rankings["typical_cases"] = nearest_rows(
        rows,
        lambda row: abs(row["all_metrics_score"] - median_score),
        top_k,
    )

    # A six-case, non-duplicated proposal for the qualitative figures.
    used_ids = set()
    selections = []

    def choose(label, candidates, reason):
        row = first_unique(candidates, used_ids)
        if row is not None:
            selections.append({"role": label, "reason": reason, "case": row})

    choose(
        "best_fidelity_improvement",
        top_rows(rows, "fidelity_score", len(rows)),
        "Strongest joint improvement in PSNR, SSIM, and RMSE.",
    )
    choose(
        "large_metal",
        top_rows([row for row in rows if row["size_group"] == "Large"], "fidelity_score", len(rows)),
        "Representative difficult case with a large metal mask.",
    )
    choose(
        "five_metals",
        top_rows([row for row in rows if int(row["n_materials"]) == 5], "fidelity_score", len(rows)),
        "Representative case containing five metal objects.",
    )
    choose(
        "pixel_perceptual_tradeoff",
        sorted(
            tradeoff_rows,
            key=lambda row: (row[delta_lpips_key], -row["fidelity_score"]),
        ),
        "Pixel metrics improve while LPIPS deteriorates; use as the main limitation case.",
    )
    choose(
        "limited_improvement",
        top_rows(rows, "change_magnitude_score", len(rows), reverse=False),
        "Gate and Vanilla are nearly equivalent across the four metrics.",
    )

    challenging_candidates = top_rows(
        true_failure_rows, "all_metrics_score", len(true_failure_rows), reverse=False
    )
    challenging_label = "true_failure"
    challenging_reason = "Gate is worse than Vanilla in all four metrics."
    if not challenging_candidates:
        challenging_candidates = top_rows(
            rows, "all_metrics_score", len(rows), reverse=False
        )
        challenging_label = "worst_observed_case"
        challenging_reason = (
            "No strict all-metric failure exists; this is the worst observed "
            "case by the balanced score."
        )
    choose(
        challenging_label,
        challenging_candidates,
        challenging_reason,
    )

    alpha2_values = np.asarray([row["alpha2"] for row in rows], dtype=np.float64)
    alpha4_values = np.asarray([row["alpha4"] for row in rows], dtype=np.float64)

    category_counts = {
        "pixel_perceptual_tradeoff": len(tradeoff_rows),
        "reverse_tradeoff": len(reverse_tradeoff_rows),
        "true_failure_all_metrics": len(true_failure_rows),
    }

    return {
        "ranking_region": region,
        "delta_convention": "Positive means Gate is better for all four metrics.",
        "limitation_case_counts": {
            name: {
                "count": count,
                "percentage": float(100.0 * count / len(rows)),
            }
            for name, count in category_counts.items()
        },
        "gate_weight_variation": {
            "alpha2_mean": float(alpha2_values.mean()),
            "alpha2_std": float(alpha2_values.std()),
            "alpha2_min": float(alpha2_values.min()),
            "alpha2_max": float(alpha2_values.max()),
            "alpha4_mean": float(alpha4_values.mean()),
            "alpha4_std": float(alpha4_values.std()),
            "alpha4_min": float(alpha4_values.min()),
            "alpha4_max": float(alpha4_values.max()),
            "interpretation": (
                "Small standard deviations and narrow ranges indicate limited "
                "sample-wise dynamic adaptation."
            ),
        },
        "rankings": {
            name: [compact_case(row, region) for row in selected]
            for name, selected in rankings.items()
        },
        "recommended_six_cases": [
            {
                "role": item["role"],
                "reason": item["reason"],
                **compact_case(item["case"], region),
            }
            for item in selections
        ],
    }


def summarize_overall(rows):
    summary = {}
    for region in REGIONS:
        summary[region] = {}
        for model in ("vanilla", "gate"):
            summary[region][model] = {}
            for metric in METRICS:
                values = np.asarray(
                    [row[f"{region}_{model}_{metric}"] for row in rows],
                    dtype=np.float64,
                )
                summary[region][model][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                }
        summary[region]["delta"] = {}
        for metric in METRICS:
            values = np.asarray(
                [row[f"{region}_delta_{metric}"] for row in rows],
                dtype=np.float64,
            )
            summary[region]["delta"][metric] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std()),
                "win_rate_percent": float((values > 0).mean() * 100.0),
            }
    return summary


def print_recommendations(recommendations):
    region = recommendations["ranking_region"]
    print(f"\n=== RECOMMENDED CASES ({region}) ===")
    for item in recommendations["recommended_six_cases"]:
        print(
            f"{item['role']:>14}: {item['sample_id']} | "
            f"count={item['n_materials']} size={item['size_group']} | "
            f"dPSNR={item['delta_psnr']:+.4f} "
            f"dSSIM={item['delta_ssim']:+.6f} "
            f"dRMSE={item['delta_rmse']:+.4f} "
            f"dLPIPS={item['delta_lpips']:+.5f}"
        )

    print("\n=== TOP DELTA PSNR ===")
    for rank, item in enumerate(recommendations["rankings"]["top_delta_psnr"], 1):
        print(
            f"{rank}. {item['sample_id']} | "
            f"dPSNR={item['delta_psnr']:+.4f} dSSIM={item['delta_ssim']:+.6f}"
        )

    print("\n=== LIMITATION CASE COUNTS ===")
    for name, stats in recommendations["limitation_case_counts"].items():
        print(
            f"{name}: {stats['count']} "
            f"({stats['percentage']:.2f}%)"
        )

    print("\n=== STRONGEST PIXEL-PERCEPTUAL TRADE-OFFS ===")
    tradeoffs = recommendations["rankings"][
        "strongest_pixel_perceptual_tradeoffs"
    ]
    if not tradeoffs:
        print("No case satisfies: dPSNR>0, dSSIM>0, dRMSE>0, dLPIPS<0.")
    for rank, item in enumerate(tradeoffs, 1):
        print(
            f"{rank}. {item['sample_id']} | "
            f"dPSNR={item['delta_psnr']:+.4f} "
            f"dSSIM={item['delta_ssim']:+.6f} "
            f"dRMSE={item['delta_rmse']:+.4f} "
            f"dLPIPS={item['delta_lpips']:+.5f}"
        )


def main():
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top_k must be positive.")
    if args.error_max <= 0:
        raise ValueError("--error_max must be positive.")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    output_dir = ensure_output_directories(args.output_dir, args.save_npy)
    samples = find_test_samples(args.test_data_dir)
    if not samples:
        raise RuntimeError(f"No AAPM samples found in {args.test_data_dir}")

    print(f"Found {len(samples)} paired samples.")
    for sample in samples:
        if sample["mask"] is None:
            sample["mask_size"] = 0
        else:
            mask = load_raw(sample["mask"], dtype=np.float32) > 0.5
            sample["mask_size"] = int(mask.sum())

        if sample["n_materials"] is None:
            raise RuntimeError(f"Missing n_materials metadata for {sample['id']}")
        sample["n_materials"] = int(sample["n_materials"])

    mask_sizes = np.asarray([sample["mask_size"] for sample in samples])
    q25, q50, q75 = np.percentile(mask_sizes, [25, 50, 75])
    for sample in samples:
        sample["size_group"] = size_to_group(
            sample["mask_size"], q25, q50, q75
        )

    print(
        f"Mask quartiles: q25={q25:.0f}, q50={q50:.0f}, q75={q75:.0f} pixels"
    )
    print("Loading Vanilla checkpoint...")
    vanilla_model = load_vanilla(args.vanilla_checkpoint, device)
    print("Loading Coarse Gate checkpoint...")
    gate_model = load_gate(
        args.gate_checkpoint,
        device,
        args.hidden_channels,
        args.gate_hidden,
        args.gate_range,
    )

    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    for parameter in lpips_fn.parameters():
        parameter.requires_grad_(False)

    entries = []
    with torch.inference_mode():
        for sample in tqdm(samples, desc="Paired inference", unit="image"):
            input_image = hu_to_unit(load_raw(sample["baseline"]))
            target = hu_to_unit(load_raw(sample["target"]))
            mask = None
            if sample["mask"] is not None:
                mask = load_raw(sample["mask"], dtype=np.float32) > 0.5

            input_tensor = (
                torch.from_numpy((input_image - 0.5) / 0.5)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
                .to(device)
            )

            vanilla_tensor = vanilla_model(input_tensor)
            gate_output = gate_model(input_tensor, return_refinement=True)
            gate_tensor, _, refinement_tensor, gate_stats = gate_output

            vanilla = np.clip(
                vanilla_tensor.squeeze().detach().cpu().numpy(), 0.0, 1.0
            )
            gate = np.clip(gate_tensor.squeeze().detach().cpu().numpy(), 0.0, 1.0)

            entry = {
                "sample_id": sample["id"],
                "anatomy": sample["anatomy"],
                "image_id": sample["image_id"],
                "n_materials": sample["n_materials"],
                "mask_size": sample["mask_size"],
                "size_group": sample["size_group"],
                "alpha2": float(gate_stats["alpha2_mean"].item()),
                "alpha4": float(gate_stats["alpha4_mean"].item()),
                "refinement_abs_mean": float(refinement_tensor.abs().mean().item()),
                "prediction_abs_difference": float(np.abs(gate - vanilla).mean()),
            }

            for exclude_metal, region in (
                (True, "non_metal"),
                (False, "metal_included"),
            ):
                vanilla_metrics = evaluate_prediction(
                    vanilla, target, mask, lpips_fn, device, exclude_metal
                )
                gate_metrics = evaluate_prediction(
                    gate, target, mask, lpips_fn, device, exclude_metal
                )
                entry[region] = {
                    "vanilla": vanilla_metrics,
                    "gate": gate_metrics,
                    "delta": improvement_delta(vanilla_metrics, gate_metrics),
                }

            entries.append(entry)
            save_case_outputs(
                output_dir,
                sample["id"],
                input_image,
                target,
                vanilla,
                gate,
                mask,
                args.error_max,
                args.save_npy,
            )

    rows = [flatten_entry(entry) for entry in entries]
    add_ranking_scores(rows, args.ranking_region)
    recommendations = build_recommendations(
        rows, args.ranking_region, min(args.top_k, len(rows))
    )

    write_csv(rows, output_dir / "per_image_metrics_and_deltas.csv")
    with open(output_dir / "per_image_details.json", "w", encoding="utf-8") as file:
        json.dump(entries, file, indent=2, ensure_ascii=False)
    with open(output_dir / "recommendations.json", "w", encoding="utf-8") as file:
        json.dump(recommendations, file, indent=2, ensure_ascii=False)

    summary = {
        "n_samples": len(rows),
        "vanilla_checkpoint": args.vanilla_checkpoint,
        "gate_checkpoint": args.gate_checkpoint,
        "mask_size_quartiles": {
            "q25": float(q25),
            "q50": float(q50),
            "q75": float(q75),
        },
        "delta_convention": "Positive means Gate is better for every metric.",
        "overall": summarize_overall(rows),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print_recommendations(recommendations)
    print(f"\nSaved CSV: {output_dir / 'per_image_metrics_and_deltas.csv'}")
    print(f"Saved recommendations: {output_dir / 'recommendations.json'}")
    print(f"Saved previews: {output_dir / 'previews'}")


if __name__ == "__main__":
    main()

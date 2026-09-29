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
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
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
        default=10,
        type=int,
        help="Number of candidates retained in auxiliary rankings.",
    )
    parser.add_argument(
        "--roi_size",
        default=96,
        type=int,
        help="Square ROI size used to locate and magnify visible differences.",
    )
    parser.add_argument("--window_center", default=40.0, type=float)
    parser.add_argument("--window_width", default=400.0, type=float)
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

    for case_name in (
        "clear_improvement",
        "pixel_perceptual_tradeoff",
        "true_failure",
    ):
        case_dir = output_dir / "qualitative_cases" / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        # Remove panels from a previous run so obsolete selections are not
        # mistaken for current recommendations.
        for old_panel in case_dir.glob("*.png"):
            old_panel.unlink()

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


def anatomical_valid_mask(target_unit):
    """Approximate the patient cross-section and exclude surrounding air."""
    target_hu = target_unit * (HU_MAX - HU_MIN) + HU_MIN
    foreground = (target_hu > -950.0).astype(np.uint8)
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), dtype=np.uint8),
        iterations=2,
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground, connectivity=8
    )
    if count <= 1:
        return foreground.astype(bool)

    # The patient cross-section is normally the largest non-air component.
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    patient = (labels == largest_label).astype(np.uint8)

    # Fill internal holes (e.g. lungs) so valid ROIs may cover all anatomy.
    contours, _ = cv2.findContours(
        patient, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros_like(patient)
    if contours:
        cv2.drawContours(filled, contours, -1, color=1, thickness=cv2.FILLED)
    return filled.astype(bool)


def local_extreme_score(
    value_map,
    roi_size,
    find_max=True,
    valid_mask=None,
    min_valid_fraction=0.60,
):
    """Return the strongest square-ROI mean and its center coordinate."""
    height, width = value_map.shape
    kernel = int(max(8, min(roi_size, height, width)))
    local_mean = cv2.boxFilter(
        value_map.astype(np.float32),
        ddepth=cv2.CV_32F,
        ksize=(kernel, kernel),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )

    if valid_mask is not None:
        valid_fraction = cv2.boxFilter(
            valid_mask.astype(np.float32),
            ddepth=cv2.CV_32F,
            ksize=(kernel, kernel),
            normalize=True,
            borderType=cv2.BORDER_CONSTANT,
        )
        allowed = valid_fraction >= float(min_valid_fraction)
        if not np.any(allowed):
            # Small head/body cross-sections, anatomy near an image boundary,
            # or an imperfect HU foreground mask may make it impossible for a
            # fixed-size ROI to reach the requested coverage. Do not abort the
            # complete evaluation for one such sample. Instead, retain ROIs
            # whose anatomical coverage is close to the best coverage that is
            # actually attainable for this image.
            maximum_valid_fraction = float(np.max(valid_fraction))
            if maximum_valid_fraction > 0.0:
                adaptive_fraction = max(
                    0.05,
                    min(
                        float(min_valid_fraction),
                        0.90 * maximum_valid_fraction,
                    ),
                )
                allowed = valid_fraction >= adaptive_fraction
            else:
                # Last-resort fallback for a completely empty/invalid anatomy
                # mask. This keeps evaluation running; the extremum is then
                # selected from the complete image.
                allowed = np.ones_like(valid_fraction, dtype=bool)
        local_mean = local_mean.copy()
        local_mean[~allowed] = -np.inf if find_max else np.inf

    minimum, maximum, minimum_location, maximum_location = cv2.minMaxLoc(local_mean)
    if find_max:
        return float(maximum), (int(maximum_location[0]), int(maximum_location[1]))
    return float(minimum), (int(minimum_location[0]), int(minimum_location[1]))


def pixel_visual_scores(vanilla, gate, target, roi_size):
    """Positive map means Gate has lower absolute pixel error."""
    advantage = np.abs(vanilla - target) - np.abs(gate - target)
    valid_mask = anatomical_valid_mask(target)
    improvement, improvement_center = local_extreme_score(
        advantage,
        roi_size,
        find_max=True,
        valid_mask=valid_mask,
    )
    minimum, worsening_center = local_extreme_score(
        advantage,
        roi_size,
        find_max=False,
        valid_mask=valid_mask,
    )
    return {
        "visual_pixel_improvement": improvement,
        "visual_pixel_worsening": float(-minimum),
        "pixel_improvement_center": improvement_center,
        "pixel_worsening_center": worsening_center,
    }


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
        "visual_pixel_improvement": entry["visual_pixel_improvement"],
        "visual_pixel_worsening": entry["visual_pixel_worsening"],
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

    visual_improvement_z = z_scores(
        [row["visual_pixel_improvement"] for row in rows]
    )
    visual_worsening_z = z_scores(
        [row["visual_pixel_worsening"] for row in rows]
    )
    for row, improvement_z, worsening_z in zip(
        rows, visual_improvement_z, visual_worsening_z
    ):
        row["z_visual_pixel_improvement"] = float(improvement_z)
        row["z_visual_pixel_worsening"] = float(worsening_z)

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
        row["clear_visual_score"] = float(
            row["z_delta_psnr"]
            + 0.50 * row["fidelity_score"]
            + 0.75 * row["z_visual_pixel_improvement"]
        )
        row["tradeoff_visual_score"] = float(
            -row["z_delta_lpips"]
            + 0.50 * row["fidelity_score"]
            + 0.75 * row["z_visual_pixel_improvement"]
        )
        row["failure_visual_score"] = float(
            -row["all_metrics_score"]
            + 0.75 * row["z_visual_pixel_worsening"]
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
        "visual_pixel_improvement": row["visual_pixel_improvement"],
        "visual_pixel_worsening": row["visual_pixel_worsening"],
        "clear_visual_score": row["clear_visual_score"],
        "tradeoff_visual_score": row["tradeoff_visual_score"],
        "failure_visual_score": row["failure_visual_score"],
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


def select_one_head_two_body(candidates):
    """Select one head and two body cases, preferring distinct body groups."""
    heads = [
        row for row in candidates
        if str(row["anatomy"]).lower().startswith("head")
    ]
    bodies = [
        row for row in candidates
        if str(row["anatomy"]).lower().startswith("body")
    ]

    selected = []
    if heads:
        selected.append(heads[0])

    used_anatomies = set()
    for row in bodies:
        if row["anatomy"] in used_anatomies:
            continue
        selected.append(row)
        used_anatomies.add(row["anatomy"])
        if len([item for item in selected if str(item["anatomy"]).lower().startswith("body")]) == 2:
            break

    selected_body_ids = {
        row["sample_id"] for row in selected
        if str(row["anatomy"]).lower().startswith("body")
    }
    while len(selected_body_ids) < 2:
        next_body = next(
            (
                row for row in bodies
                if row["sample_id"] not in selected_body_ids
            ),
            None,
        )
        if next_body is None:
            break
        selected.append(next_body)
        selected_body_ids.add(next_body["sample_id"])
    return selected


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

    # Three qualitative categories. Each category contains exactly one head
    # and two body images whenever the dataset provides enough candidates.
    # Metric strength and local visual saliency are both used for ranking.
    clear_candidates = [
        row for row in rows
        if row[delta_psnr_key] > 0
        and row[f"{region}_delta_ssim"] > 0
        and row[f"{region}_delta_rmse"] > 0
    ]
    clear_candidates = sorted(
        clear_candidates,
        key=lambda row: row["clear_visual_score"],
        reverse=True,
    )

    tradeoff_candidates = sorted(
        tradeoff_rows,
        key=lambda row: row["tradeoff_visual_score"],
        reverse=True,
    )

    failure_candidates = sorted(
        true_failure_rows,
        key=lambda row: row["failure_visual_score"],
        reverse=True,
    )
    failure_is_strict = True
    if not failure_candidates:
        failure_is_strict = False
        failure_candidates = sorted(
            rows,
            key=lambda row: row["failure_visual_score"],
            reverse=True,
        )

    qualitative_cases = {
        "clear_improvement": {
            "definition": (
                "Gate improves PSNR, SSIM, and RMSE; selected cases also "
                "contain a spatially concentrated reduction in pixel error."
            ),
            "cases": [
                compact_case(row, region)
                for row in select_one_head_two_body(clear_candidates)
            ],
        },
        "pixel_perceptual_tradeoff": {
            "definition": (
                "Gate improves PSNR, SSIM, and RMSE while LPIPS worsens."
            ),
            "cases": [
                compact_case(row, region)
                for row in select_one_head_two_body(tradeoff_candidates)
            ],
        },
        "true_failure": {
            "definition": (
                "Gate is worse in all four metrics."
                if failure_is_strict
                else "No strict all-metric failures exist; these are the worst observed cases."
            ),
            "strict_definition_satisfied": failure_is_strict,
            "cases": [
                compact_case(row, region)
                for row in select_one_head_two_body(failure_candidates)
            ],
        },
    }

    incomplete = {
        name: len(group["cases"])
        for name, group in qualitative_cases.items()
        if len(group["cases"]) != 3
    }
    if incomplete:
        raise RuntimeError(
            "Cannot construct exactly 1 head + 2 body samples for every "
            f"qualitative category. Incomplete groups: {incomplete}"
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
        "qualitative_cases": qualitative_cases,
    }


ADVANTAGE_CMAP = LinearSegmentedColormap.from_list(
    "gate_advantage", ["#d73027", "#ffffff", "#1a9850"]
)


def read_preview_float(output_dir, kind, sample_id):
    path = Path(output_dir) / "previews" / kind / f"{sample_id}.png"
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read preview: {path}")
    return image.astype(np.float32) / 255.0


def ct_window(image_unit, center, width):
    hu = image_unit * (HU_MAX - HU_MIN) + HU_MIN
    lower = center - width / 2.0
    upper = center + width / 2.0
    return np.clip((hu - lower) / max(upper - lower, 1e-6), 0.0, 1.0)


def roi_bounds(center, roi_size, shape):
    height, width = shape
    size = int(max(8, min(roi_size, height, width)))
    x_center, y_center = center
    x0 = int(np.clip(x_center - size // 2, 0, width - size))
    y0 = int(np.clip(y_center - size // 2, 0, height - size))
    return x0, y0, x0 + size, y0 + size


def crop_roi(image, bounds):
    x0, y0, x1, y1 = bounds
    return image[y0:y1, x0:x1]


def normalize_for_joint_display(*maps, percentile=99.0):
    values = np.concatenate([np.abs(value).ravel() for value in maps])
    limit = float(np.percentile(values, percentile))
    return max(limit, 1e-8)


@torch.inference_mode()
def spatial_lpips_advantage(vanilla, gate, target, lpips_spatial_fn, device):
    target_bgr = to_uint8_bgr3(target)
    vanilla_bgr = to_uint8_bgr3(vanilla)
    gate_bgr = to_uint8_bgr3(gate)
    target_tensor = to_lpips_tensor(target_bgr, device)
    vanilla_map = lpips_spatial_fn(
        to_lpips_tensor(vanilla_bgr, device), target_tensor
    )
    gate_map = lpips_spatial_fn(
        to_lpips_tensor(gate_bgr, device), target_tensor
    )
    advantage = vanilla_map - gate_map
    advantage = torch.nn.functional.interpolate(
        advantage,
        size=target.shape,
        mode="bilinear",
        align_corners=False,
    )
    return advantage.squeeze().detach().cpu().numpy().astype(np.float32)


def add_roi_rectangle(axis, bounds, color, label=None):
    x0, y0, x1, y1 = bounds
    axis.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor=color,
            linewidth=2.0,
        )
    )
    if label:
        axis.text(
            x0,
            max(0, y0 - 4),
            label,
            color=color,
            fontsize=8,
            weight="bold",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1},
        )


def render_qualitative_panel(
    output_dir,
    case_name,
    case,
    lpips_spatial_fn,
    device,
    roi_size,
    window_center,
    window_width,
):
    sample_id = case["sample_id"]
    input_image = read_preview_float(output_dir, "input", sample_id)
    target = read_preview_float(output_dir, "ground_truth", sample_id)
    vanilla = read_preview_float(output_dir, "vanilla", sample_id)
    gate = read_preview_float(output_dir, "coarse_gate", sample_id)

    vanilla_error = np.abs(vanilla - target)
    gate_error = np.abs(gate - target)
    pixel_advantage = vanilla_error - gate_error
    valid_mask = anatomical_valid_mask(target)
    lpips_advantage = spatial_lpips_advantage(
        vanilla, gate, target, lpips_spatial_fn, device
    )

    _, pixel_good_center = local_extreme_score(
        pixel_advantage,
        roi_size,
        find_max=True,
        valid_mask=valid_mask,
    )
    _, pixel_bad_center = local_extreme_score(
        pixel_advantage,
        roi_size,
        find_max=False,
        valid_mask=valid_mask,
    )
    _, perceptual_bad_center = local_extreme_score(
        lpips_advantage,
        roi_size,
        find_max=False,
        valid_mask=valid_mask,
    )

    if case_name == "true_failure":
        main_center = pixel_bad_center
        main_color = "#d73027"
        main_label = "Gate worse"
    else:
        main_center = pixel_good_center
        main_color = "#1a9850"
        main_label = "Gate better"

    main_bounds = roi_bounds(main_center, roi_size, target.shape)
    perceptual_bounds = roi_bounds(perceptual_bad_center, roi_size, target.shape)

    display_images = [
        ct_window(image, window_center, window_width)
        for image in (input_image, target, vanilla, gate)
    ]
    input_display, target_display, vanilla_display, gate_display = display_images

    fig, axes = plt.subplots(2, 6, figsize=(18, 7.4), constrained_layout=True)
    whole_images = [input_display, target_display, vanilla_display, gate_display]
    whole_titles = ["Corrupted input", "Ground truth", "Vanilla", "Gate"]
    for axis, image, title in zip(axes[0, :4], whole_images, whole_titles):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        add_roi_rectangle(axis, main_bounds, main_color, "A")
        if case_name == "pixel_perceptual_tradeoff":
            add_roi_rectangle(axis, perceptual_bounds, "#d73027", "B")
        axis.set_title(title)
        axis.axis("off")

    pixel_advantage_display = cv2.GaussianBlur(
        pixel_advantage.astype(np.float32), (0, 0), sigmaX=1.2
    )
    pixel_limit = normalize_for_joint_display(pixel_advantage_display)
    lpips_limit = normalize_for_joint_display(lpips_advantage)

    if case_name == "pixel_perceptual_tradeoff":
        # For a trade-off example, focus the two diagnostic maps on the
        # regions that define the trade-off instead of showing whole slices:
        # A is the strongest pixel-fidelity gain and B is the strongest
        # perceptual degradation.
        pixel_map_to_show = crop_roi(pixel_advantage_display, main_bounds)
        lpips_map_to_show = crop_roi(lpips_advantage, perceptual_bounds)
        pixel_map_limit = normalize_for_joint_display(pixel_map_to_show)
        lpips_map_limit = normalize_for_joint_display(lpips_map_to_show)
        pixel_map_title = "Pixel-error advantage crop A\nGreen: Gate better"
        lpips_map_title = "Spatial-LPIPS advantage crop B\nRed: Gate worse"
    else:
        pixel_map_to_show = pixel_advantage_display
        lpips_map_to_show = lpips_advantage
        pixel_map_limit = pixel_limit
        lpips_map_limit = lpips_limit
        pixel_map_title = "Pixel-error advantage (smoothed)\nGreen: Gate better"
        lpips_map_title = "Spatial-LPIPS advantage\nRed: Gate worse"

    axes[0, 4].imshow(
        pixel_map_to_show,
        cmap=ADVANTAGE_CMAP,
        vmin=-pixel_map_limit,
        vmax=pixel_map_limit,
    )
    axes[0, 4].set_title(pixel_map_title)
    axes[0, 4].axis("off")

    axes[0, 5].imshow(
        lpips_map_to_show,
        cmap=ADVANTAGE_CMAP,
        vmin=-lpips_map_limit,
        vmax=lpips_map_limit,
    )
    axes[0, 5].set_title(lpips_map_title)
    axes[0, 5].axis("off")

    for axis, image, title in zip(
        axes[1, :3],
        (target_display, vanilla_display, gate_display),
        ("GT crop A", "Vanilla crop A", "Gate crop A"),
    ):
        axis.imshow(crop_roi(image, main_bounds), cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")

    if case_name == "pixel_perceptual_tradeoff":
        for axis, image, title in zip(
            axes[1, 3:],
            (target_display, vanilla_display, gate_display),
            ("GT crop B", "Vanilla crop B", "Gate crop B"),
        ):
            axis.imshow(
                crop_roi(image, perceptual_bounds), cmap="gray", vmin=0, vmax=1
            )
            axis.set_title(title)
            axis.axis("off")
    else:
        error_limit = normalize_for_joint_display(
            crop_roi(vanilla_error, main_bounds),
            crop_roi(gate_error, main_bounds),
        )
        axes[1, 3].imshow(
            crop_roi(vanilla_error, main_bounds),
            cmap="inferno",
            vmin=0,
            vmax=error_limit,
        )
        axes[1, 3].set_title("Vanilla error crop")
        axes[1, 4].imshow(
            crop_roi(gate_error, main_bounds),
            cmap="inferno",
            vmin=0,
            vmax=error_limit,
        )
        axes[1, 4].set_title("Gate error crop")
        axes[1, 5].imshow(
            crop_roi(pixel_advantage_display, main_bounds),
            cmap=ADVANTAGE_CMAP,
            vmin=-pixel_limit,
            vmax=pixel_limit,
        )
        axes[1, 5].set_title("Pixel advantage crop")
        for axis in axes[1, 3:]:
            axis.axis("off")

    metric_text = (
        f"dPSNR={case['delta_psnr']:+.4f} dB   "
        f"dSSIM={case['delta_ssim']:+.6f}   "
        f"dRMSE={case['delta_rmse']:+.4f}   "
        f"dLPIPS={case['delta_lpips']:+.5f}   "
        "(positive = Gate better)"
    )
    fig.suptitle(
        f"{case_name.replace('_', ' ').title()} — {sample_id}\n{metric_text}",
        fontsize=13,
        weight="bold",
    )
    destination = (
        Path(output_dir) / "qualitative_cases" / case_name / f"{sample_id}.png"
    )
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "sample_id": sample_id,
        "panel": str(destination),
        "main_roi_xyxy": list(main_bounds),
        "perceptual_roi_xyxy": (
            list(perceptual_bounds)
            if case_name == "pixel_perceptual_tradeoff"
            else None
        ),
    }


def render_selected_cases(
    output_dir,
    recommendations,
    device,
    roi_size,
    window_center,
    window_width,
):
    print("Loading spatial LPIPS for qualitative panels...")
    lpips_spatial_fn = lpips.LPIPS(net="vgg", spatial=True).to(device).eval()
    for parameter in lpips_spatial_fn.parameters():
        parameter.requires_grad_(False)

    rendered = {}
    for case_name, group in recommendations["qualitative_cases"].items():
        rendered[case_name] = []
        for case in group["cases"]:
            rendered[case_name].append(
                render_qualitative_panel(
                    output_dir,
                    case_name,
                    case,
                    lpips_spatial_fn,
                    device,
                    roi_size,
                    window_center,
                    window_width,
                )
            )
    return rendered


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
    print(f"\n=== THREE QUALITATIVE CASE GROUPS ({region}) ===")
    for case_name, group in recommendations["qualitative_cases"].items():
        print(f"\n[{case_name}] {group['definition']}")
        if not group["cases"]:
            print("  No eligible cases.")
        for index, item in enumerate(group["cases"], 1):
            print(
                f"  {index}. {item['sample_id']} ({item['anatomy']}) | "
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
    if args.roi_size <= 0:
        raise ValueError("--roi_size must be positive.")
    if args.window_width <= 0:
        raise ValueError("--window_width must be positive.")

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
            entry.update(
                pixel_visual_scores(
                    vanilla,
                    gate,
                    target,
                    args.roi_size,
                )
            )

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

    del lpips_fn
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rendered_panels = render_selected_cases(
        output_dir,
        recommendations,
        device,
        args.roi_size,
        args.window_center,
        args.window_width,
    )
    recommendations["rendered_panels"] = rendered_panels

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
    print(f"Saved qualitative panels: {output_dir / 'qualitative_cases'}")


if __name__ == "__main__":
    main()

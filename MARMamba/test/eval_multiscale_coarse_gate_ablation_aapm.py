"""Evaluate fixed, scale-2-only, and scale-4-only Multi-Coarse ablations.

The metric pipeline intentionally matches the dynamic-gate evaluator:
PSNR/SSIM/RMSE/LPIPS, non-metal and metal-included regions, mask-size
quartiles, metal-count groups, single/multi groups, and overall results.
"""

from __future__ import annotations

import argparse
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


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    (
        candidate
        for candidate in (SCRIPT_DIR, SCRIPT_DIR.parent)
        if (candidate / "model").is_dir() and (candidate / "utils").is_dir()
    ),
    SCRIPT_DIR.parent,
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.mamba_multiscale_coarse_gate import (  # noqa: E402
    GatedMultiScaleCoarseRefinedMambaFormer,
)
from utils.metrics import (  # noqa: E402
    calculate_psnr,
    calculate_rmse,
    calculate_ssim,
)


HU_MIN, HU_MAX = -1000.0, 3000.0
FNAME_ID_RE = re.compile(r"img(\d+)")
DIMS_RE = re.compile(r"(\d+)x(\d+)x(\d+)")
METRICS = ("psnr", "ssim", "rmse", "lpips")
REGIONS = ("non_metal", "metal_included")
SIZE_GROUPS = ("Large", "Medium", "Small", "Tiny")
FUSION_MODES = ("fixed", "scale2", "scale4")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full AAPM evaluator for Multi-Coarse ablations."
    )
    parser.add_argument("--test_data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fusion_mode", required=True, choices=FUSION_MODES)
    parser.add_argument("--output_dir", default="./eval_multiscale_ablation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden_channels", default=32, type=int)
    parser.add_argument("--gate_hidden", default=32, type=int)
    parser.add_argument("--gate_range", default=0.5, type=float)
    parser.add_argument(
        "--max_samples",
        default=None,
        type=int,
        help="Optional smoke-test limit. Omit for the official full evaluation.",
    )
    return parser.parse_args()


def parse_dims(filename):
    match = DIMS_RE.search(filename)
    if not match:
        raise ValueError(f"Cannot parse dimensions from: {filename}")
    width, height, depth = (int(x) for x in match.groups())
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
    def index_one(anatomy_dir):
        anatomy = os.path.basename(os.path.normpath(anatomy_dir))
        baseline_dir = os.path.join(anatomy_dir, "Baseline")
        target_dir = os.path.join(anatomy_dir, "Target")
        mask_dir = os.path.join(anatomy_dir, "Mask")
        if not os.path.isdir(baseline_dir) or not os.path.isdir(target_dir):
            return []

        target_map = {}
        for path in sorted(glob.glob(os.path.join(target_dir, "*.raw"))):
            match = FNAME_ID_RE.search(os.path.basename(path))
            if match:
                target_map[match.group(1)] = path

        mask_map = {}
        count_map = {}
        if os.path.isdir(mask_dir):
            for path in sorted(glob.glob(os.path.join(mask_dir, "*.raw"))):
                match = FNAME_ID_RE.search(os.path.basename(path))
                if match:
                    mask_map[match.group(1)] = path
            for path in sorted(glob.glob(os.path.join(mask_dir, "*.json"))):
                match = re.search(r"metalinfo(\d+)", os.path.basename(path))
                if not match:
                    continue
                try:
                    with open(path, "r", encoding="utf-8") as file:
                        count_map[match.group(1)] = json.load(file).get(
                            "n_materials"
                        )
                except (OSError, json.JSONDecodeError):
                    pass

        samples = []
        for baseline_path in sorted(
            glob.glob(os.path.join(baseline_dir, "*.raw"))
        ):
            match = FNAME_ID_RE.search(os.path.basename(baseline_path))
            if not match:
                continue
            image_id = match.group(1)
            if image_id not in target_map:
                continue
            samples.append(
                {
                    "id": f"{anatomy}_{image_id}",
                    "baseline": baseline_path,
                    "target": target_map[image_id],
                    "mask": mask_map.get(image_id),
                    "n_materials": count_map.get(image_id),
                }
            )
        return samples

    if os.path.isdir(os.path.join(test_data_dir, "Baseline")):
        return index_one(test_data_dir)

    samples = []
    for name in sorted(os.listdir(test_data_dir)):
        subdirectory = os.path.join(test_data_dir, name)
        if os.path.isdir(os.path.join(subdirectory, "Baseline")):
            samples.extend(index_one(subdirectory))
    return samples


def size_to_group(size, q25, q50, q75):
    if size >= q75:
        return "Large"
    if size >= q50:
        return "Medium"
    if size >= q25:
        return "Small"
    return "Tiny"


def to_uint8_bgr3(image_float01):
    image_u8 = np.clip(image_float01 * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)


def to_lpips_tensor(image_bgr_uint8, device):
    image_rgb = cv2.cvtColor(image_bgr_uint8, cv2.COLOR_BGR2RGB)
    image = image_rgb.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)


def evaluate_image(prediction, target, mask, lpips_fn, device, exclude_metal):
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


def extract_state_dict(checkpoint, requested_mode, device):
    checkpoint_object = torch.load(checkpoint, map_location=device)
    saved_mode = None

    if isinstance(checkpoint_object, dict):
        saved_args = checkpoint_object.get("args")
        if isinstance(saved_args, dict):
            saved_mode = saved_args.get("fusion_mode")

        state = checkpoint_object
        for key in ("model", "state_dict", "net"):
            if key in checkpoint_object and isinstance(checkpoint_object[key], dict):
                state = checkpoint_object[key]
                break
    else:
        state = checkpoint_object

    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a state_dict.")
    if saved_mode is not None and saved_mode != requested_mode:
        raise ValueError(
            "Fusion-mode mismatch: "
            f"checkpoint={saved_mode}, command={requested_mode}."
        )

    if any(key.startswith("module.") for key in state):
        state = {
            (key[len("module."):] if key.startswith("module.") else key): value
            for key, value in state.items()
        }
    return state, saved_mode


def load_model(args, device):
    model = GatedMultiScaleCoarseRefinedMambaFormer(
        in_channels=1,
        hidden_channels=args.hidden_channels,
        gate_hidden=args.gate_hidden,
        gate_range=args.gate_range,
    ).to(device)
    state, saved_mode = extract_state_dict(
        args.checkpoint, args.fusion_mode, device
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, saved_mode


def compose_refinement(outputs, fusion_mode):
    baseline = outputs["baseline"]
    prediction_scale2 = outputs["pred_detail2"]
    prediction_scale4 = outputs["pred_coarse4"]
    weight_shape = (baseline.shape[0], 1, 1, 1)

    if fusion_mode == "fixed":
        alpha2 = baseline.new_ones(weight_shape)
        alpha4 = baseline.new_ones(weight_shape)
        delta = prediction_scale2 + prediction_scale4
    elif fusion_mode == "scale2":
        alpha2 = baseline.new_ones(weight_shape)
        alpha4 = baseline.new_zeros(weight_shape)
        delta = prediction_scale2
    elif fusion_mode == "scale4":
        alpha2 = baseline.new_zeros(weight_shape)
        alpha4 = baseline.new_ones(weight_shape)
        delta = prediction_scale4
    else:
        raise ValueError(f"Unknown fusion mode: {fusion_mode}")

    return {
        **outputs,
        "alpha2": alpha2,
        "alpha4": alpha4,
        "delta": delta,
        "refined": baseline + delta,
    }


def summarize(entries, region_key, model_key):
    if not entries:
        return None
    output = {"n": len(entries)}
    for metric in METRICS:
        values = np.asarray(
            [entry["metrics"][region_key][model_key][metric] for entry in entries],
            dtype=np.float64,
        )
        output[metric] = float(values.mean())
        output[f"{metric}_std"] = float(values.std())
    return output


def metric_delta(baseline, refined):
    return {
        metric: float(refined[metric] - baseline[metric])
        for metric in METRICS
    }


def summarize_pair(entries, region_key):
    if not entries:
        return None
    baseline = summarize(entries, region_key, "baseline")
    refined = summarize(entries, region_key, "refined")
    return {
        "baseline": baseline,
        "refined": refined,
        "delta": metric_delta(baseline, refined),
    }


def summarize_weights(entries):
    if not entries:
        return None
    output = {"n": len(entries)}
    for key in ("alpha2", "alpha4"):
        values = np.asarray([entry[key] for entry in entries], dtype=np.float64)
        output[f"{key}_mean"] = float(values.mean())
        output[f"{key}_std"] = float(values.std())
        output[f"{key}_min"] = float(values.min())
        output[f"{key}_max"] = float(values.max())
    return output


def summarize_refinement(entries):
    if not entries:
        return None
    keys = (
        "delta_abs_mean",
        "delta_rms",
        "detail2_abs_mean",
        "coarse4_abs_mean",
    )
    output = {"n": len(entries)}
    for key in keys:
        values = np.asarray([entry[key] for entry in entries], dtype=np.float64)
        output[f"{key}_mean"] = float(values.mean())
        output[f"{key}_std"] = float(values.std())
    return output


def print_group(label, entries, region_key):
    if not entries:
        return
    pair = summarize_pair(entries, region_key)
    baseline, refined, delta = pair["baseline"], pair["refined"], pair["delta"]
    print(
        f"{label:>10} | n={refined['n']:4d} | "
        f"PSNR {baseline['psnr']:.4f}->{refined['psnr']:.4f} "
        f"({delta['psnr']:+.4f}) | "
        f"SSIM {baseline['ssim']:.6f}->{refined['ssim']:.6f} "
        f"({delta['ssim']:+.6f}) | "
        f"RMSE {baseline['rmse']:.4f}->{refined['rmse']:.4f} "
        f"({delta['rmse']:+.4f}) | "
        f"LPIPS {baseline['lpips']:.5f}->{refined['lpips']:.5f} "
        f"({delta['lpips']:+.5f})"
    )


def grouped_entries(entries, field, value):
    return [entry for entry in entries if entry[field] == value]


def build_summary(entries, args, count_hist, quartiles, saved_mode):
    q25, q50, q75 = quartiles
    summary = {
        "metadata": {
            "fusion_mode": args.fusion_mode,
            "checkpoint": os.path.abspath(args.checkpoint),
            "checkpoint_fusion_mode": saved_mode,
            "n_samples": len(entries),
            "metal_count_histogram": {
                str(key): int(value) for key, value in count_hist.items()
            },
            "mask_size_quartiles": {
                "q25": float(q25),
                "q50": float(q50),
                "q75": float(q75),
            },
            "fusion_weights": {
                "fixed": {"alpha2": 1.0, "alpha4": 1.0},
                "scale2": {"alpha2": 1.0, "alpha4": 0.0},
                "scale4": {"alpha2": 0.0, "alpha4": 1.0},
            }[args.fusion_mode],
            "regions": {
                "non_metal": (
                    "Prediction and GT are both zeroed inside the metal mask "
                    "before metrics."
                ),
                "metal_included": "Metrics use the full image without masking.",
            },
        },
        "overall": {
            region: summarize_pair(entries, region) for region in REGIONS
        },
        "by_size": {},
        "by_metal_count": {},
        "by_multiplicity": {},
        "fusion_weight_stats": {
            "overall": summarize_weights(entries),
            "by_size": {},
            "by_metal_count": {},
            "by_multiplicity": {},
        },
        "refinement_stats": {
            "overall": summarize_refinement(entries),
            "by_size": {},
            "by_metal_count": {},
            "by_multiplicity": {},
        },
    }

    for group in SIZE_GROUPS:
        subset = grouped_entries(entries, "size_group", group)
        summary["by_size"][group] = {
            region: summarize_pair(subset, region) for region in REGIONS
        }
        summary["fusion_weight_stats"]["by_size"][group] = summarize_weights(
            subset
        )
        summary["refinement_stats"]["by_size"][group] = summarize_refinement(
            subset
        )

    for count in range(1, 6):
        subset = grouped_entries(entries, "n_materials", count)
        key = str(count)
        summary["by_metal_count"][key] = {
            region: summarize_pair(subset, region) for region in REGIONS
        }
        summary["fusion_weight_stats"]["by_metal_count"][key] = (
            summarize_weights(subset)
        )
        summary["refinement_stats"]["by_metal_count"][key] = (
            summarize_refinement(subset)
        )

    for group in ("single", "multi"):
        subset = grouped_entries(entries, "multiplicity", group)
        summary["by_multiplicity"][group] = {
            region: summarize_pair(subset, region) for region in REGIONS
        }
        summary["fusion_weight_stats"]["by_multiplicity"][group] = (
            summarize_weights(subset)
        )
        summary["refinement_stats"]["by_multiplicity"][group] = (
            summarize_refinement(subset)
        )
    return summary


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    samples = find_test_samples(args.test_data_dir)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError("No AAPM test samples found.")

    count_hist = {count: 0 for count in range(1, 6)}
    missing_masks = []
    for sample in samples:
        if sample["n_materials"] is None:
            raise RuntimeError(f"Missing n_materials for {sample['id']}")
        count = int(sample["n_materials"])
        if count not in count_hist:
            raise RuntimeError(
                f"Unexpected n_materials={count} for {sample['id']}"
            )
        sample["n_materials"] = count
        count_hist[count] += 1

        if sample["mask"] is None:
            missing_masks.append(sample["id"])
            sample["mask_size"] = 0
        else:
            mask = load_raw(sample["mask"], dtype=np.float32) > 0.5
            sample["mask_size"] = int(mask.sum())

    sizes = np.asarray([sample["mask_size"] for sample in samples], dtype=np.float64)
    q25, q50, q75 = np.percentile(sizes, [25, 50, 75])
    for sample in samples:
        sample["size_group"] = size_to_group(
            sample["mask_size"], q25, q50, q75
        )

    print(f"Fusion mode: {args.fusion_mode}")
    print(f"Found {len(samples)} test cases.")
    print(f"Mask quartiles: q25={q25:.0f}, q50={q50:.0f}, q75={q75:.0f}")
    print(f"Metal-count histogram: {count_hist}")
    if missing_masks:
        print(f"WARNING: {len(missing_masks)} samples have no mask.")

    model, saved_mode = load_model(args, device)
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    entries = []

    with torch.inference_mode():
        for index, sample in enumerate(samples):
            input_image = hu_to_unit(load_raw(sample["baseline"]))
            target_image = hu_to_unit(load_raw(sample["target"]))
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
            outputs = model(input_tensor, return_aux=True)
            outputs = compose_refinement(outputs, args.fusion_mode)

            refined = np.clip(
                outputs["refined"].squeeze().cpu().numpy(), 0.0, 1.0
            )
            baseline = np.clip(
                outputs["baseline"].squeeze().cpu().numpy(), 0.0, 1.0
            )
            delta = outputs["delta"].detach()

            metrics = {}
            for exclude_metal, region_key in (
                (True, "non_metal"),
                (False, "metal_included"),
            ):
                metrics[region_key] = {
                    "baseline": evaluate_image(
                        baseline,
                        target_image,
                        mask,
                        lpips_fn,
                        device,
                        exclude_metal,
                    ),
                    "refined": evaluate_image(
                        refined,
                        target_image,
                        mask,
                        lpips_fn,
                        device,
                        exclude_metal,
                    ),
                }

            count = sample["n_materials"]
            entries.append(
                {
                    "id": sample["id"],
                    "n_materials": count,
                    "mask_size": sample["mask_size"],
                    "size_group": sample["size_group"],
                    "multiplicity": "multi" if count >= 2 else "single",
                    "metrics": metrics,
                    "alpha2": float(outputs["alpha2"].mean().item()),
                    "alpha4": float(outputs["alpha4"].mean().item()),
                    "delta_abs_mean": float(delta.abs().mean().item()),
                    "delta_rms": float(
                        torch.sqrt(torch.mean(delta.float() ** 2)).item()
                    ),
                    "detail2_abs_mean": float(
                        outputs["pred_detail2"].detach().abs().mean().item()
                    ),
                    "coarse4_abs_mean": float(
                        outputs["pred_coarse4"].detach().abs().mean().item()
                    ),
                }
            )
            if (index + 1) % 50 == 0 or index == 0:
                print(f"Processed {index + 1}/{len(samples)}")

    for region_key, title in (
        ("non_metal", "NON-METAL"),
        ("metal_included", "METAL-INCLUDED"),
    ):
        print(f"\n=== {title} BY SIZE ===")
        for group in SIZE_GROUPS:
            print_group(
                group, grouped_entries(entries, "size_group", group), region_key
            )
        print(f"\n=== {title} BY METAL COUNT ===")
        for count in range(1, 6):
            print_group(
                f"{count} metal",
                grouped_entries(entries, "n_materials", count),
                region_key,
            )
        print(f"\n=== {title} SINGLE / MULTI / OVERALL ===")
        for group in ("single", "multi"):
            print_group(
                group,
                grouped_entries(entries, "multiplicity", group),
                region_key,
            )
        print_group("overall", entries, region_key)

    summary = build_summary(
        entries, args, count_hist, (q25, q50, q75), saved_mode
    )
    summary_path = os.path.join(args.output_dir, "eval_summary.json")
    per_image_path = os.path.join(args.output_dir, "per_image_results.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    with open(per_image_path, "w", encoding="utf-8") as file:
        json.dump(entries, file, indent=2, ensure_ascii=False)

    print(f"\nSaved summary:   {summary_path}")
    print(f"Saved per-image: {per_image_path}")


if __name__ == "__main__":
    main()

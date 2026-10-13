#!/usr/bin/env python3
"""Empirical analysis for the preserve-and-refine hypothesis.

This script compares a frozen Stage-2 checkpoint (b) with vanilla Stage-3
fine-tuning (v) against the same ground truth (y) on the validation set.

It measures:
  1. image-level PSNR/SSIM/RMSE changes from Stage 2 to Stage 3;
  2. how often Stage 3 increases pixel error;
  3. whether Stage 3 damages pixels that Stage 2 already reconstructed well;
  4. representative error maps for paper figures.

No Gate output is used. All three folders must contain aligned images. By
default, samples are matched by filename stem (e.g. 0001.png -> 0001).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import wilcoxon


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare Stage-2 and vanilla Stage-3 predictions on validation data."
    )
    p.add_argument("--stage2-dir", type=Path)
    p.add_argument("--stage3-dir", type=Path)
    p.add_argument("--gt-dir", type=Path)
    p.add_argument(
        "--stage2-checkpoint",
        type=Path,
        help="Stage-2 MARMamba checkpoint; enables integrated inference mode.",
    )
    p.add_argument(
        "--stage3-checkpoint",
        type=Path,
        help="Vanilla Stage-3 checkpoint; enables integrated inference mode.",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        help="AAPM validation root containing Baseline/Target folders.",
    )
    p.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device used for integrated inference (default: cuda:0).",
    )
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--analysis-mask-dir",
        type=Path,
        default=None,
        help="Optional binary valid-region masks. White/non-zero pixels are analyzed.",
    )
    p.add_argument(
        "--metal-mask-dir",
        type=Path,
        default=None,
        help="Optional metal masks. Use with --exclude-metal to analyze non-metal pixels.",
    )
    p.add_argument("--exclude-metal", action="store_true")
    p.add_argument(
        "--key-regex",
        default=None,
        help="Optional regex with one capture group used as sample ID from each stem.",
    )
    p.add_argument(
        "--good-quantile",
        type=float,
        default=0.25,
        help="Lowest Stage-2-error fraction regarded as already well reconstructed.",
    )
    p.add_argument(
        "--gt-threshold",
        type=float,
        default=None,
        help="Optional normalized GT threshold; only pixels above it are analyzed.",
    )
    p.add_argument(
        "--epsilon",
        type=float,
        default=1.0 / 255.0,
        help="Minimum normalized error increase counted as meaningful degradation.",
    )
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=19)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument(
        "--allow-resize",
        action="store_true",
        help="Resize predictions to GT size. Avoid this unless scientifically justified.",
    )
    args = p.parse_args()
    if not 0.0 < args.good_quantile < 1.0:
        p.error("--good-quantile must be between 0 and 1")
    if args.exclude_metal and args.metal_mask_dir is None:
        p.error("--exclude-metal requires --metal-mask-dir")
    inference_values = (args.stage2_checkpoint, args.stage3_checkpoint, args.data_dir)
    using_inference = any(value is not None for value in inference_values)
    if using_inference and not all(value is not None for value in inference_values):
        p.error(
            "Integrated inference requires --stage2-checkpoint, "
            "--stage3-checkpoint, and --data-dir together"
        )
    if not using_inference and not all(
        value is not None for value in (args.stage2_dir, args.stage3_dir, args.gt_dir)
    ):
        p.error(
            "Provide either the three inference arguments or all of "
            "--stage2-dir, --stage3-dir, and --gt-dir"
        )
    return args


def save_unit_png16(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = np.rint(np.clip(array, 0.0, 1.0) * 65535.0).astype(np.uint16)
    Image.fromarray(encoded, mode="I;16").save(path)


def load_marmamba(checkpoint: Path, device: str):
    """Load a vanilla MARMamba checkpoint saved with or without DataParallel."""
    import torch
    import torch.nn as nn
    from model.mamba import MambaFormer

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    net = MambaFormer(in_channels=1).to(device)
    raw = torch.load(checkpoint, map_location=device)
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint}")

    has_module_prefix = bool(raw) and next(iter(raw)).startswith("module.")
    if has_module_prefix:
        net = nn.DataParallel(net)
    try:
        net.load_state_dict(raw, strict=True)
    except RuntimeError:
        # Handle the inverse case: a non-DataParallel state dict loaded in a
        # DataParallel environment is not wrapped automatically.
        if has_module_prefix:
            stripped = {key.removeprefix("module."): value for key, value in raw.items()}
            net = MambaFormer(in_channels=1).to(device)
            net.load_state_dict(stripped, strict=True)
        else:
            raise
    net.eval()
    return net


def run_integrated_inference(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Run both vanilla checkpoints on the same full-resolution AAPM validation set."""
    import torch
    from utils.aapm_dataset import (
        _find_anatomy_folders,
        _index_one_anatomy,
        hu_to_unit,
        load_raw,
    )

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    anatomy_dirs = _find_anatomy_folders(str(args.data_dir))
    if not anatomy_dirs:
        raise FileNotFoundError(
            f"No AAPM Baseline/Target structure found under: {args.data_dir}"
        )
    samples = []
    for anatomy_dir in anatomy_dirs:
        samples.extend(_index_one_anatomy(anatomy_dir))
    samples = sorted(samples, key=lambda item: item["id"])
    if not samples:
        raise RuntimeError(f"No paired AAPM validation images found under: {args.data_dir}")

    inference_root = args.output_dir / "inference_outputs"
    stage2_dir = inference_root / "stage2"
    stage3_dir = inference_root / "stage3"
    gt_dir = inference_root / "gt"
    print(f"Validation samples for integrated inference: {len(samples)}")

    checkpoints = (
        ("Stage 2", args.stage2_checkpoint, stage2_dir),
        ("Vanilla Stage 3", args.stage3_checkpoint, stage3_dir),
    )
    for model_name, checkpoint, prediction_dir in checkpoints:
        print(f"\nLoading {model_name}: {checkpoint}")
        net = load_marmamba(checkpoint, args.device)
        prediction_dir.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode():
            for index, sample in enumerate(samples, start=1):
                x_unit = hu_to_unit(load_raw(sample["baseline"]))
                target = hu_to_unit(load_raw(sample["target"]))
                if x_unit.ndim != 2 or target.ndim != 2:
                    raise ValueError(
                        f"Expected one 2-D slice for {sample['id']}, got "
                        f"input={x_unit.shape}, target={target.shape}"
                    )
                # Identical to Compose([ToTensor(), Normalize([0.5], [0.5])]).
                input_tensor = torch.from_numpy(x_unit).unsqueeze(0).unsqueeze(0)
                input_tensor = ((input_tensor - 0.5) / 0.5).to(
                    args.device, dtype=torch.float32, non_blocking=True
                )
                prediction = net(input_tensor).detach().float().cpu().numpy()[0, 0]
                save_unit_png16(prediction, prediction_dir / f"{sample['id']}.png")
                if model_name == "Stage 2":
                    save_unit_png16(target, gt_dir / f"{sample['id']}.png")
                if index == 1 or index % 50 == 0 or index == len(samples):
                    print(f"{model_name}: {index}/{len(samples)}")
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nInference outputs saved under: {inference_root.resolve()}")
    return stage2_dir, stage3_dir, gt_dir


def sample_key(path: Path, pattern: re.Pattern[str] | None) -> str:
    if pattern is None:
        return path.stem
    match = pattern.search(path.stem)
    if match is None or match.lastindex is None:
        raise ValueError(f"Cannot extract sample ID from {path.name!r} using regex")
    return match.group(1)


def index_images(folder: Path, pattern: re.Pattern[str] | None) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {folder}")
    result: dict[str, Path] = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            key = sample_key(path, pattern)
            if key in result:
                raise ValueError(
                    f"Duplicate sample ID {key!r} in {folder}: "
                    f"{result[key].name} and {path.name}"
                )
            result[key] = path
    if not result:
        raise RuntimeError(f"No supported images found in {folder}")
    return result


def read_gray(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            # Preserve scalar precision for grayscale images; convert RGB exports
            # to luminance only when necessary.
            if image.mode not in ("1", "L", "I", "I;16", "F"):
                image = image.convert("L")
            return np.asarray(image).copy()
    except Exception as exc:
        raise RuntimeError(f"Cannot read image: {path}") from exc


def common_normalize(arrays: list[np.ndarray]) -> list[np.ndarray]:
    """Convert aligned images to [0,1] using one common intensity range."""
    max_value = max(float(np.nanmax(a)) for a in arrays)
    min_value = min(float(np.nanmin(a)) for a in arrays)
    if min_value < 0:
        raise ValueError(
            "Negative-valued images detected. Export all predictions and GT with the "
            "same CT window/normalization before using this script."
        )
    if max_value <= 1.0 + 1e-6:
        scale = 1.0
    elif max_value <= 255.0:
        scale = 255.0
    elif max_value <= 65535.0:
        scale = 65535.0
    else:
        raise ValueError(
            f"Unsupported intensity range [0, {max_value}]. Export consistently scaled images."
        )
    return [np.clip(a.astype(np.float64) / scale, 0.0, 1.0) for a in arrays]


def match_shape(arr: np.ndarray, shape: tuple[int, int], allow_resize: bool) -> np.ndarray:
    if arr.shape == shape:
        return arr
    if not allow_resize:
        raise ValueError(
            f"Prediction shape {arr.shape} differs from GT shape {shape}. "
            "Run both checkpoints on the same 512x512 inputs, or explicitly use --allow-resize."
        )
    image = Image.fromarray(arr)
    return np.asarray(image.resize((shape[1], shape[0]), Image.Resampling.BICUBIC))


def load_mask(path: Path, shape: tuple[int, int], allow_resize: bool) -> np.ndarray:
    mask = read_gray(path)
    if mask.shape != shape:
        if not allow_resize:
            raise ValueError(f"Mask {path} has shape {mask.shape}, expected {shape}")
        image = Image.fromarray(mask)
        mask = np.asarray(image.resize((shape[1], shape[0]), Image.Resampling.NEAREST))
    return mask > 0


def masked_values(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = arr[mask]
    if values.size == 0:
        raise ValueError("Analysis mask contains no valid pixels")
    return values


def rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred[mask] - gt[mask]) ** 2)))


def psnr(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    mse = float(np.mean((pred[mask] - gt[mask]) ** 2))
    return float("inf") if mse == 0 else float(10.0 * math.log10(1.0 / mse))


def ssim(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """Single-channel SSIM with the conventional 11x11 Gaussian window."""
    # SSIM is computed on the full rectangular image, consistent across stages.
    # For irregular masks, use masked PSNR/RMSE and pixel-error analyses.
    del mask
    sigma = 1.5
    mu_x = gaussian_filter(pred, sigma=sigma, truncate=3.5, mode="reflect")
    mu_y = gaussian_filter(gt, sigma=sigma, truncate=3.5, mode="reflect")
    sigma_x = gaussian_filter(pred * pred, sigma=sigma, truncate=3.5, mode="reflect") - mu_x**2
    sigma_y = gaussian_filter(gt * gt, sigma=sigma, truncate=3.5, mode="reflect") - mu_y**2
    sigma_xy = gaussian_filter(pred * gt, sigma=sigma, truncate=3.5, mode="reflect") - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )
    return float(np.mean(score))


def bootstrap_mean_ci(
    values: np.ndarray, n_resamples: int, rng: np.random.Generator
) -> tuple[float, float]:
    if values.size == 1:
        return float(values[0]), float(values[0])
    means = np.empty(n_resamples, dtype=np.float64)
    chunk = 1000
    for start in range(0, n_resamples, chunk):
        count = min(chunk, n_resamples - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means[start : start + count] = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def finite_wilcoxon(values: np.ndarray) -> tuple[float | None, float | None]:
    values = values[np.isfinite(values)]
    if values.size == 0 or np.allclose(values, 0.0):
        return None, None
    result = wilcoxon(values, zero_method="wilcox", alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def summarize_column(
    rows: list[dict[str, float | str]],
    column: str,
    rng: np.random.Generator,
    bootstrap: int,
) -> dict[str, float | int | None]:
    values = np.asarray([float(row[column]) for row in rows], dtype=np.float64)
    finite = values[np.isfinite(values)]
    ci_low, ci_high = bootstrap_mean_ci(finite, bootstrap, rng)
    stat, p_value = finite_wilcoxon(finite)
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "sd": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "median": float(np.median(finite)),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "win_rate": float(np.mean(finite > 0.0)),
        "tie_rate": float(np.mean(np.isclose(finite, 0.0))),
        "wilcoxon_statistic": stat,
        "wilcoxon_p": p_value,
    }


def save_representative_figure(
    records: list[dict], output_path: Path, top_k: int
) -> None:
    chosen = sorted(records, key=lambda r: r["good_mean_error_increase"], reverse=True)[:top_k]
    if not chosen:
        return
    fig, axes = plt.subplots(len(chosen), 7, figsize=(18, 2.8 * len(chosen)), squeeze=False)
    titles = ["Ground truth", "Stage 2", "Vanilla Stage 3", "$E_2$", "$E_3$", "$\\Delta E$", "Worsened good pixels"]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=10)
    for row_idx, rec in enumerate(chosen):
        gt, b, v = rec["gt"], rec["stage2"], rec["stage3"]
        e2, e3, delta = rec["e2"], rec["e3"], rec["delta"]
        display_limit = max(float(np.percentile(e2, 99)), float(np.percentile(e3, 99)), 1e-6)
        signed_limit = max(float(np.percentile(np.abs(delta), 99)), 1e-6)
        panels = [gt, b, v, e2, e3, delta, rec["worsened_good"]]
        cmaps = ["gray", "gray", "gray", "magma", "magma", "coolwarm", "Reds"]
        ranges = [(0, 1), (0, 1), (0, 1), (0, display_limit), (0, display_limit), (-signed_limit, signed_limit), (0, 1)]
        for col, (panel, cmap, (vmin, vmax)) in enumerate(zip(panels, cmaps, ranges)):
            axes[row_idx, col].imshow(panel, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row_idx, col].axis("off")
        axes[row_idx, 0].set_ylabel(str(rec["sample_id"]), fontsize=9)
    fig.suptitle("Representative validation cases with the largest Stage-3 degradation in Stage-2-good regions", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_distribution_figure(rows: list[dict], output_path: Path) -> None:
    psnr_diff = np.asarray([float(r["psnr_difference"]) for r in rows])
    good_harm = np.asarray([float(r["good_mean_error_increase"]) for r in rows])
    worsened = np.asarray([float(r["good_worsened_fraction"]) for r in rows]) * 100.0
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    plots = [
        (psnr_diff, "Stage 3 - Stage 2 PSNR (dB)"),
        (good_harm, "Mean positive error increase\nin Stage-2-good pixels"),
        (worsened, "Worsened Stage-2-good pixels (%)"),
    ]
    for ax, (values, label) in zip(axes, plots):
        ax.hist(values, bins=30, color="#377eb8", edgecolor="white", alpha=0.9)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.axvline(float(np.mean(values)), color="#e41a1c", linewidth=1.5, label="Mean")
        ax.set_xlabel(label)
        ax.set_ylabel("Number of images")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage2_checkpoint is not None:
        args.stage2_dir, args.stage3_dir, args.gt_dir = run_integrated_inference(args)
    pattern = re.compile(args.key_regex) if args.key_regex else None

    stage2_files = index_images(args.stage2_dir, pattern)
    stage3_files = index_images(args.stage3_dir, pattern)
    gt_files = index_images(args.gt_dir, pattern)
    mask_files = index_images(args.analysis_mask_dir, pattern) if args.analysis_mask_dir else None
    metal_files = index_images(args.metal_mask_dir, pattern) if args.metal_mask_dir else None

    common = sorted(set(stage2_files) & set(stage3_files) & set(gt_files))
    if not common:
        raise RuntimeError("No common sample IDs across Stage 2, Stage 3, and GT folders")
    for name, mapping in (("Stage 2", stage2_files), ("Stage 3", stage3_files), ("GT", gt_files)):
        missing = set(common) - set(mapping)
        if missing:
            raise RuntimeError(f"{name} is missing {len(missing)} matched samples")
    if mask_files is not None and not set(common).issubset(mask_files):
        raise RuntimeError("Analysis-mask folder does not cover all matched samples")
    if metal_files is not None and not set(common).issubset(metal_files):
        raise RuntimeError("Metal-mask folder does not cover all matched samples")

    print(f"Matched validation images: {len(common)}")
    rows: list[dict[str, float | str]] = []
    visual_records: list[dict] = []

    for index, key in enumerate(common, start=1):
        gt_raw = read_gray(gt_files[key])
        b_raw = match_shape(read_gray(stage2_files[key]), gt_raw.shape, args.allow_resize)
        v_raw = match_shape(read_gray(stage3_files[key]), gt_raw.shape, args.allow_resize)
        b, v, gt = common_normalize([b_raw, v_raw, gt_raw])

        valid = np.ones(gt.shape, dtype=bool)
        if mask_files is not None:
            valid &= load_mask(mask_files[key], gt.shape, args.allow_resize)
        if args.gt_threshold is not None:
            valid &= gt > args.gt_threshold
        if args.exclude_metal:
            valid &= ~load_mask(metal_files[key], gt.shape, args.allow_resize)
        if not np.any(valid):
            raise ValueError(f"No valid pixels remain for sample {key}")

        e2 = np.abs(gt - b)
        e3 = np.abs(gt - v)
        delta = e3 - e2  # positive means Stage 3 is worse
        threshold = float(np.quantile(e2[valid], args.good_quantile))
        good = valid & (e2 <= threshold)
        worsened = valid & (delta > args.epsilon)
        worsened_good = good & (delta > args.epsilon)

        row = {
            "sample_id": key,
            "stage2_psnr": psnr(b, gt, valid),
            "stage3_psnr": psnr(v, gt, valid),
            "psnr_difference": psnr(v, gt, valid) - psnr(b, gt, valid),
            "stage2_ssim": ssim(b, gt, valid),
            "stage3_ssim": ssim(v, gt, valid),
            "ssim_difference": ssim(v, gt, valid) - ssim(b, gt, valid),
            "stage2_rmse": rmse(b, gt, valid),
            "stage3_rmse": rmse(v, gt, valid),
            "rmse_improvement": rmse(b, gt, valid) - rmse(v, gt, valid),
            "mean_error_difference": float(np.mean(delta[valid])),
            "worsened_pixel_fraction": float(np.mean(worsened[valid])),
            "good_error_threshold": threshold,
            "good_worsened_fraction": float(np.mean(worsened_good[good])),
            "good_mean_error_difference": float(np.mean(delta[good])),
            "good_mean_error_increase": float(np.mean(np.maximum(delta[good], 0.0))),
        }
        rows.append(row)
        visual_records.append(
            {
                "sample_id": key,
                "gt": gt,
                "stage2": b,
                "stage3": v,
                "e2": e2,
                "e3": e3,
                "delta": delta,
                "worsened_good": worsened_good.astype(np.float32),
                "good_mean_error_increase": row["good_mean_error_increase"],
            }
        )
        if index == 1 or index % 50 == 0 or index == len(common):
            print(f"Processed {index}/{len(common)}")

    csv_path = args.output_dir / "stage2_vs_vanilla_stage3_per_image.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    rng = np.random.default_rng(args.seed)
    summary = {
        "interpretation": {
            "metric_differences": "Positive PSNR/SSIM/RMSE-improvement means Stage 3 is better.",
            "error_differences": "Positive error difference means Stage 3 is worse.",
            "good_region": f"Lowest {args.good_quantile:.0%} of Stage-2 absolute errors within each image.",
            "meaningful_degradation_epsilon": args.epsilon,
        },
        "dataset": {
            "n": len(rows),
            "analysis_mask_used": args.analysis_mask_dir is not None,
            "metal_excluded": bool(args.exclude_metal),
            "gt_threshold": args.gt_threshold,
        },
        "paired_image_metrics": {
            key: summarize_column(rows, key, rng, args.bootstrap)
            for key in ("psnr_difference", "ssim_difference", "rmse_improvement")
        },
        "pixel_error_analysis": {
            key: summarize_column(rows, key, rng, args.bootstrap)
            for key in (
                "mean_error_difference",
                "worsened_pixel_fraction",
                "good_worsened_fraction",
                "good_mean_error_difference",
                "good_mean_error_increase",
            )
        },
    }
    json_path = args.output_dir / "stage2_preservation_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    save_representative_figure(
        visual_records,
        args.output_dir / "stage2_vs_stage3_error_maps.png",
        args.top_k,
    )
    save_distribution_figure(rows, args.output_dir / "stage2_vs_stage3_distributions.png")

    print("\n=== PRESERVATION ANALYSIS ===")
    for key, stats in summary["paired_image_metrics"].items():
        print(
            f"{key:24s}: {stats['mean']:+.6f} "
            f"(95% CI {stats['ci95_low']:+.6f}, {stats['ci95_high']:+.6f}), "
            f"win rate={stats['win_rate']:.3f}, p={stats['wilcoxon_p']}"
        )
    print("\nPositive values below mean that Vanilla Stage 3 increases error:")
    for key, stats in summary["pixel_error_analysis"].items():
        print(
            f"{key:28s}: {stats['mean']:+.6f} "
            f"(95% CI {stats['ci95_low']:+.6f}, {stats['ci95_high']:+.6f})"
        )
    print(f"\nSaved results to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate Vanilla MARMamba and MARMamba-Gate checkpoints, then run paired tests.

The script uses the same AAPM preprocessing and metric implementation as the
provided Vanilla/Gate evaluators. Checkpoints are evaluated sequentially to
keep GPU memory bounded. For each method, per-image metrics are averaged across
its supplied seeds before paired Gate-vs-Vanilla statistics are computed.

Positive paired differences always favor Gate:
  PSNR/SSIM: Gate - Vanilla
  RMSE/LPIPS: Vanilla - Gate
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

import cv2
import lpips
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats


def find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent, Path.cwd(), Path.cwd().parent):
        if (candidate / "model").is_dir() and (candidate / "utils").is_dir():
            return candidate
    raise RuntimeError(
        "Cannot locate the project root containing model/ and utils/. "
        "Place this script in the repository root or its test/ directory."
    )


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.mamba import MambaFormer
from model.mamba_multiscale_coarse_gate import GatedMultiScaleCoarseRefinedMambaFormer
from utils.metrics import calculate_psnr, calculate_rmse, calculate_ssim


HU_MIN, HU_MAX = -1000.0, 3000.0
FNAME_ID_RE = re.compile(r"img(\d+)")
DIMS_RE = re.compile(r"(\d+)x(\d+)x(\d+)")
METRICS = ("psnr", "ssim", "rmse", "lpips")
REGIONS = ("non_metal", "metal_included")
HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "rmse": False, "lpips": False}
DISPLAY_NAMES = {"psnr": "PSNR (dB)", "ssim": "SSIM", "rmse": "RMSE", "lpips": "LPIPS"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_dir", required=True)
    parser.add_argument("--vanilla_checkpoints", nargs="+", required=True)
    parser.add_argument("--gate_checkpoints", nargs="+", required=True)
    parser.add_argument("--output_dir", default="./paired_model_evaluation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden_channels", default=32, type=int)
    parser.add_argument("--gate_hidden", default=32, type=int)
    parser.add_argument("--gate_range", default=0.5, type=float)
    parser.add_argument("--bootstrap", default=10_000, type=int)
    parser.add_argument("--confidence", default=0.95, type=float)
    parser.add_argument("--statistics_seed", default=3407, type=int)
    parser.add_argument("--tie_tolerance", default=0.0, type=float)
    parser.add_argument("--dpi", default=300, type=int)
    return parser.parse_args()


def parse_dims(filename: str) -> tuple[int, int, int]:
    match = DIMS_RE.search(filename)
    if not match:
        raise ValueError(f"Cannot parse dimensions from: {filename}")
    width, height, depth = (int(x) for x in match.groups())
    return height, width, depth


def load_raw(path: str, dtype=np.float32) -> np.ndarray:
    rows, cols, slices = parse_dims(os.path.basename(path))
    array = np.fromfile(path, dtype=dtype)
    expected = rows * cols * slices
    if array.size != expected:
        raise ValueError(f"{path}: found {array.size} elements, expected {expected}")
    return array.reshape(rows, cols) if slices == 1 else array.reshape(slices, rows, cols)


def hu_to_unit(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, HU_MIN, HU_MAX)
    return ((image - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def find_test_samples(test_data_dir: str) -> list[dict]:
    def index_one(anatomy_dir: str) -> list[dict]:
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

        mask_map, count_map = {}, {}
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
                    with open(path, "r", encoding="utf-8") as handle:
                        count_map[match.group(1)] = json.load(handle).get("n_materials")
                except Exception:
                    pass

        samples = []
        for baseline_path in sorted(glob.glob(os.path.join(baseline_dir, "*.raw"))):
            match = FNAME_ID_RE.search(os.path.basename(baseline_path))
            if not match or match.group(1) not in target_map:
                continue
            image_id = match.group(1)
            samples.append({
                "id": f"{anatomy}_{image_id}",
                "baseline": baseline_path,
                "target": target_map[image_id],
                "mask": mask_map.get(image_id),
                "n_materials": count_map.get(image_id),
            })
        return samples

    if os.path.isdir(os.path.join(test_data_dir, "Baseline")):
        samples = index_one(test_data_dir)
    else:
        samples = []
        for name in sorted(os.listdir(test_data_dir)):
            subdir = os.path.join(test_data_dir, name)
            if os.path.isdir(os.path.join(subdir, "Baseline")):
                samples.extend(index_one(subdir))
    return sorted(samples, key=lambda item: item["id"])


def prepare_sample_metadata(samples: list[dict]) -> tuple[float, float, float]:
    for sample in samples:
        if sample["mask"] is None:
            sample["mask_size"] = 0
        else:
            sample["mask_size"] = int((load_raw(sample["mask"]) > 0.5).sum())
    q25, q50, q75 = np.percentile([sample["mask_size"] for sample in samples], [25, 50, 75])
    for sample in samples:
        size = sample["mask_size"]
        sample["size_group"] = (
            "Large" if size >= q75 else
            "Medium" if size >= q50 else
            "Small" if size >= q25 else "Tiny"
        )
    return float(q25), float(q50), float(q75)


def to_uint8_bgr3(image: np.ndarray) -> np.ndarray:
    image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)


def to_lpips_tensor(image_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_np = image_rgb.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)


def evaluate_image(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None,
    lpips_fn: torch.nn.Module,
    device: torch.device,
    exclude_metal: bool,
) -> dict[str, float]:
    pred_eval, gt_eval = prediction.copy(), target.copy()
    if exclude_metal and mask is not None:
        pred_eval[mask] = 0.0
        gt_eval[mask] = 0.0
    pred_bgr, gt_bgr = to_uint8_bgr3(pred_eval), to_uint8_bgr3(gt_eval)
    return {
        "psnr": float(calculate_psnr(pred_bgr, gt_bgr, test_y_channel=True)),
        "ssim": float(calculate_ssim(pred_bgr, gt_bgr, test_y_channel=True)),
        "rmse": float(calculate_rmse(pred_bgr, gt_bgr)),
        "lpips": float(lpips_fn(
            to_lpips_tensor(pred_bgr, device), to_lpips_tensor(gt_bgr, device)
        ).item()),
    }


def extract_state(checkpoint: str, device: torch.device) -> dict:
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint object in {checkpoint}")
    return state


def load_vanilla(checkpoint: str, device: torch.device) -> torch.nn.Module:
    model = MambaFormer(in_channels=1).to(device)
    state = extract_state(checkpoint, device)
    clean_state = {
        (key[len("module."):] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    model.load_state_dict(clean_state, strict=True)
    return model.eval()


def load_gate(checkpoint: str, device: torch.device, args: argparse.Namespace) -> torch.nn.Module:
    model = GatedMultiScaleCoarseRefinedMambaFormer(
        in_channels=1,
        hidden_channels=args.hidden_channels,
        gate_hidden=args.gate_hidden,
        gate_range=args.gate_range,
    ).to(device)
    state = extract_state(checkpoint, device)
    clean_state = {
        (key[len("module."):] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    model.load_state_dict(clean_state, strict=True)
    return model.eval()


def infer_seed_label(checkpoint: str, index: int) -> str:
    name = Path(checkpoint).name
    match = re.search(r"(?:seed[_-]?)?(19|42|3407)(?:\D|$)", name, flags=re.IGNORECASE)
    return match.group(1) if match else f"run_{index + 1}"


def evaluate_checkpoint(
    method: str,
    checkpoint: str,
    run_index: int,
    samples: list[dict],
    lpips_fn: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict]:
    model = load_vanilla(checkpoint, device) if method == "vanilla" else load_gate(checkpoint, device, args)
    seed = infer_seed_label(checkpoint, run_index)
    rows = []
    print(f"\n[{method.upper()} seed={seed}] {checkpoint}")

    with torch.inference_mode():
        for index, sample in enumerate(samples):
            input_image = hu_to_unit(load_raw(sample["baseline"]))
            target = hu_to_unit(load_raw(sample["target"]))
            mask = load_raw(sample["mask"]) > 0.5 if sample["mask"] else None
            input_tensor = torch.from_numpy((input_image - 0.5) / 0.5).unsqueeze(0).unsqueeze(0).float().to(device)
            if method == "vanilla":
                output = model(input_tensor)
            else:
                output, _, _, _ = model(input_tensor, return_refinement=True)
            prediction = np.clip(output.squeeze().cpu().numpy(), 0.0, 1.0)

            base = {
                "image_id": sample["id"],
                "method": method,
                "seed": seed,
                "n_materials": sample["n_materials"],
                "mask_size": sample["mask_size"],
                "size_group": sample["size_group"],
            }
            for exclude_metal, region in ((True, "non_metal"), (False, "metal_included")):
                metrics = evaluate_image(prediction, target, mask, lpips_fn, device, exclude_metal)
                rows.append({**base, "region": region, **metrics})
            if (index + 1) % 50 == 0 or index == 0:
                print(f"  Processed {index + 1}/{len(samples)}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def bootstrap_mean_ci(values, n_bootstrap, confidence, rng):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    means = np.empty(n_bootstrap, dtype=np.float64)
    batch_size = max(1, min(1000, 2_000_000 // max(n, 1)))
    offset = 0
    while offset < n_bootstrap:
        current = min(batch_size, n_bootstrap - offset)
        indices = rng.integers(0, n, size=(current, n))
        means[offset:offset + current] = values[indices].mean(axis=1)
        offset += current
    alpha = 1.0 - confidence
    return tuple(float(x) for x in np.quantile(means, [alpha / 2, 1 - alpha / 2]))


def rank_biserial(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[values != 0]
    if not len(values):
        return 0.0
    ranks = stats.rankdata(np.abs(values), method="average")
    positive = float(ranks[values > 0].sum())
    negative = float(ranks[values < 0].sum())
    return (positive - negative) / (positive + negative)


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running, total = 0.0, len(p_values)
    for rank, original_index in enumerate(order):
        running = max(running, (total - rank) * p_values[original_index])
        adjusted[original_index] = min(running, 1.0)
    return adjusted


def build_paired_tables(raw: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_keys = ["image_id", "method", "region"]
    metadata = raw.groupby(group_keys, as_index=False)[["n_materials", "mask_size"]].first()
    averages = raw.groupby(group_keys, as_index=False)[list(METRICS)].mean()
    averages = averages.merge(metadata, on=group_keys, validate="one_to_one")
    counts = raw.groupby(["method", "image_id", "region"])["seed"].nunique()
    expected = raw.groupby("method")["seed"].nunique().to_dict()
    for method, expected_count in expected.items():
        method_counts = counts.loc[method]
        if not (method_counts == expected_count).all():
            raise RuntimeError(f"Some {method} images do not have all {expected_count} seed results.")

    vanilla = averages[averages.method == "vanilla"].drop(columns="method")
    gate = averages[averages.method == "gate"].drop(columns="method")
    paired = vanilla.merge(
        gate, on=["image_id", "region"], suffixes=("_vanilla", "_gate"), validate="one_to_one"
    )
    if len(paired) != len(vanilla) or len(paired) != len(gate):
        raise RuntimeError("Vanilla and Gate do not contain identical image/region pairs.")
    for metric in METRICS:
        if HIGHER_IS_BETTER[metric]:
            paired[f"delta_{metric}"] = paired[f"{metric}_gate"] - paired[f"{metric}_vanilla"]
        else:
            paired[f"delta_{metric}"] = paired[f"{metric}_vanilla"] - paired[f"{metric}_gate"]

    rng = np.random.default_rng(args.statistics_seed)
    rows = []
    for region in REGIONS:
        region_frame = paired[paired.region == region]
        region_p_values, region_rows = [], []
        for metric in METRICS:
            values = region_frame[f"delta_{metric}"].to_numpy(dtype=np.float64)
            ci_low, ci_high = bootstrap_mean_ci(values, args.bootstrap, args.confidence, rng)
            try:
                result = stats.wilcoxon(values, zero_method="wilcox", alternative="two-sided", method="auto")
                statistic, p_value = float(result.statistic), float(result.pvalue)
            except ValueError:
                statistic, p_value = 0.0, 1.0
            wins = int((values > args.tie_tolerance).sum())
            losses = int((values < -args.tie_tolerance).sum())
            ties = len(values) - wins - losses
            sd = float(values.std(ddof=1)) if len(values) > 1 else math.nan
            region_rows.append({
                "region": region,
                "metric": metric.upper(),
                "n": len(values),
                "vanilla_seeds": expected["vanilla"],
                "gate_seeds": expected["gate"],
                "mean_difference": float(values.mean()),
                "median_difference": float(np.median(values)),
                "sd_difference": sd,
                "ci_level": args.confidence,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "win_rate": wins / len(values),
                "tie_rate": ties / len(values),
                "loss_rate": losses / len(values),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_raw": p_value,
                "rank_biserial": rank_biserial(values),
                "cohen_dz": float(values.mean() / sd) if sd and np.isfinite(sd) else math.nan,
            })
            region_p_values.append(p_value)
        adjusted = holm_adjust(region_p_values)
        for row, adjusted_p in zip(region_rows, adjusted):
            row["wilcoxon_p_holm"] = float(adjusted_p)
        rows.extend(region_rows)
    return paired, pd.DataFrame(rows)


def plot_region(paired, summary, region, output_dir, dpi):
    frame = paired[paired.region == region]
    region_summary = summary[summary.region == region]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    colors = {"psnr": "#2878B5", "ssim": "#3A923A", "rmse": "#E07B39", "lpips": "#8E5AA9"}
    for panel, (axis, metric) in enumerate(zip(axes.ravel(), METRICS)):
        values = frame[f"delta_{metric}"].to_numpy(dtype=np.float64)
        color = colors[metric]
        axis.hist(values, bins=min(40, max(15, int(np.sqrt(len(values))))), density=True,
                  color=color, alpha=0.62, edgecolor="white", linewidth=0.5)
        if len(np.unique(values)) > 1 and values.std() > 0:
            kde = stats.gaussian_kde(values)
            padding = 0.08 * (values.max() - values.min())
            x_axis = np.linspace(values.min() - padding, values.max() + padding, 400)
            axis.plot(x_axis, kde(x_axis), color=color, linewidth=2)
        row = region_summary[region_summary.metric == metric.upper()].iloc[0]
        axis.axvline(0, color="black", linestyle="--", linewidth=1.2)
        axis.axvline(row.mean_difference, color="#C62828", linewidth=1.8)
        axis.axvspan(row.ci_low, row.ci_high, color="#C62828", alpha=0.12)
        axis.set_title(f"({chr(97 + panel)}) {DISPLAY_NAMES[metric]}")
        axis.set_xlabel(f"Oriented paired difference in {metric.upper()}\n(positive = Gate better)")
        axis.set_ylabel("Density")
        axis.grid(axis="y", alpha=0.22)
        axis.text(
            0.98, 0.96,
            f"Mean = {row.mean_difference:.6g}\n"
            f"CI = [{row.ci_low:.6g}, {row.ci_high:.6g}]\n"
            f"Win rate = {100 * row.win_rate:.1f}%\n"
            f"Holm p = {row.wilcoxon_p_holm:.3g}",
            transform=axis.transAxes, ha="right", va="top",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88,
                  "edgecolor": "#BBBBBB"},
        )
    label = region.replace("_", " ")
    fig.suptitle(f"Per-image paired differences: {label}", fontsize=14)
    fig.savefig(output_dir / f"paired_distributions_{region}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"paired_distributions_{region}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_latex_tables(summary: pd.DataFrame, output_dir: Path) -> None:
    for region in REGIONS:
        lines = [
            r"\begin{tabular}{lrrrrr}", r"\toprule",
            r"Metric & Mean diff. & 95\% CI & Win rate (\%) & Holm $p$ & Rank-biserial \\",
            r"\midrule",
        ]
        for _, row in summary[summary.region == region].iterrows():
            p_text = "$<0.001$" if row.wilcoxon_p_holm < 0.001 else f"{row.wilcoxon_p_holm:.3f}"
            lines.append(
                f"{row.metric} & {row.mean_difference:.6f} & "
                f"[{row.ci_low:.6f}, {row.ci_high:.6f}] & "
                f"{100 * row.win_rate:.1f} & {p_text} & {row.rank_biserial:.3f} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        (output_dir / f"paired_statistics_{region}.tex").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    samples = find_test_samples(args.test_data_dir)
    if not samples:
        raise RuntimeError("No AAPM test samples found.")
    q25, q50, q75 = prepare_sample_metadata(samples)
    print(f"Found {len(samples)} cases; mask quartiles: {q25:.0f}, {q50:.1f}, {q75:.0f}")

    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    all_rows = []
    for method, checkpoints in (
        ("vanilla", args.vanilla_checkpoints),
        ("gate", args.gate_checkpoints),
    ):
        for run_index, checkpoint in enumerate(checkpoints):
            all_rows.extend(evaluate_checkpoint(
                method, checkpoint, run_index, samples, lpips_fn, device, args
            ))

    raw = pd.DataFrame(all_rows)
    raw.to_csv(output_dir / "per_image_per_seed_metrics.csv", index=False)
    paired, summary = build_paired_tables(raw, args)
    paired.to_csv(output_dir / "paired_per_image_metrics.csv", index=False)
    summary.to_csv(output_dir / "paired_statistics.csv", index=False)
    with (output_dir / "paired_statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(orient="records"), handle, indent=2, ensure_ascii=False)
    write_latex_tables(summary, output_dir)
    for region in REGIONS:
        plot_region(paired, summary, region, output_dir, args.dpi)

    print("\n=== PAIRED STATISTICS (positive = Gate better) ===")
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

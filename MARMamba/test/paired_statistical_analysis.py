"""Paired statistical comparison of separately trained Vanilla and Gate models.

This evaluator follows the same AAPM loading and metric pipeline as the
provided Vanilla/Gate test scripts, but it evaluates both checkpoints on each
sample in one pass and retains per-image results for paired inference.

Recommended primary run (when anatomy folders represent patients/volumes):

python test/paired_statistical_analysis.py \
    --test_data_dir /path/to/data_CT_MAR_train_test_8k/test \
    --vanilla_checkpoint /path/to/vanilla_512_best.pth \
    --gate_checkpoint /path/to/gate_512_best.pth \
    --output_dir ./paired_stats_512_best \
    --inference_unit cluster

Positive improvement always means that Gate is better:
  PSNR/SSIM: Gate - Vanilla
  RMSE/LPIPS: Vanilla - Gate
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
import pandas as pd
import torch
from scipy.stats import rankdata, wilcoxon
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


HU_MIN, HU_MAX = -1000.0, 3000.0
FNAME_ID_RE = re.compile(r"img(\d+)")
DIMS_RE = re.compile(r"(\d+)x(\d+)x(\d+)")
METRICS = ("psnr", "ssim", "rmse", "lpips")
REGIONS = ("non_metal", "metal_included")


def parse_dims(filename):
    match = DIMS_RE.search(filename)
    if not match:
        raise ValueError(f"Cannot parse dimensions from: {filename}")
    width, height, depth = (int(value) for value in match.groups())
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
        cluster_id = os.path.basename(os.path.normpath(anatomy_dir))
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
        metal_count_map = {}
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
                        metal_count_map[match.group(1)] = json.load(file).get(
                            "n_materials"
                        )
                except Exception:
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
                    "id": f"{cluster_id}_{image_id}",
                    "cluster_id": cluster_id,
                    "baseline": baseline_path,
                    "target": target_map[image_id],
                    "mask": mask_map.get(image_id),
                    "n_materials": metal_count_map.get(image_id),
                }
            )
        return samples

    test_data_dir = str(test_data_dir)
    if os.path.isdir(os.path.join(test_data_dir, "Baseline")):
        return index_anatomy(test_data_dir)

    samples = []
    for name in sorted(os.listdir(test_data_dir)):
        subdir = os.path.join(test_data_dir, name)
        if os.path.isdir(os.path.join(subdir, "Baseline")):
            samples.extend(index_anatomy(subdir))
    return samples


def to_uint8_bgr3(image_float01):
    image_u8 = np.clip(image_float01 * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)


def to_lpips_tensor(image_bgr_uint8, device):
    image_rgb = cv2.cvtColor(image_bgr_uint8, cv2.COLOR_BGR2RGB)
    image_np = image_rgb.astype(np.float32) / 127.5 - 1.0
    return (
        torch.from_numpy(image_np)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )


def evaluate_image(prediction, target, mask, lpips_fn, device, exclude_metal):
    prediction = prediction.copy()
    target = target.copy()
    if exclude_metal and mask is not None:
        prediction[mask] = 0.0
        target[mask] = 0.0

    prediction_bgr = to_uint8_bgr3(prediction)
    target_bgr = to_uint8_bgr3(target)
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


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a state_dict.")
    return {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in checkpoint.items()
    }


def load_models(args, device):
    vanilla = MambaFormer(in_channels=1).to(device)
    vanilla_state = extract_state_dict(
        torch.load(args.vanilla_checkpoint, map_location=device)
    )
    vanilla.load_state_dict(vanilla_state, strict=True)
    vanilla.eval()

    gate = GatedMultiScaleCoarseRefinedMambaFormer(
        in_channels=1,
        hidden_channels=args.hidden_channels,
        gate_hidden=args.gate_hidden,
        gate_range=args.gate_range,
    ).to(device)
    gate_state = extract_state_dict(
        torch.load(args.gate_checkpoint, map_location=device)
    )
    gate.load_state_dict(gate_state, strict=True)
    gate.eval()
    return vanilla, gate


def oriented_delta(metric, vanilla_values, gate_values):
    """Return improvement where positive always means Gate is better."""
    if metric in ("psnr", "ssim"):
        return gate_values - vanilla_values
    return vanilla_values - gate_values


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adjusted_sorted = np.empty_like(sorted_p)
    running_max = 0.0
    total = len(sorted_p)
    for index, p_value in enumerate(sorted_p):
        candidate = (total - index) * p_value
        running_max = max(running_max, candidate)
        adjusted_sorted[index] = min(running_max, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def rank_biserial_effect(deltas):
    deltas = np.asarray(deltas, dtype=np.float64)
    deltas = deltas[np.isfinite(deltas) & (deltas != 0)]
    if deltas.size == 0:
        return 0.0
    ranks = rankdata(np.abs(deltas), method="average")
    positive = ranks[deltas > 0].sum()
    negative = ranks[deltas < 0].sum()
    return float((positive - negative) / (positive + negative))


def bootstrap_mean_ci(deltas, clusters, unit, n_bootstrap, seed):
    deltas = np.asarray(deltas, dtype=np.float64)
    clusters = np.asarray(clusters)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)

    if unit == "sample":
        for index in range(n_bootstrap):
            indices = rng.integers(0, len(deltas), size=len(deltas))
            boot_means[index] = deltas[indices].mean()
    else:
        unique_clusters = np.unique(clusters)
        cluster_arrays = {
            cluster: deltas[clusters == cluster]
            for cluster in unique_clusters
        }
        for index in range(n_bootstrap):
            sampled_clusters = rng.choice(
                unique_clusters,
                size=len(unique_clusters),
                replace=True,
            )
            sampled_values = np.concatenate(
                [cluster_arrays[cluster] for cluster in sampled_clusters]
            )
            boot_means[index] = sampled_values.mean()

    lower, upper = np.percentile(boot_means, [2.5, 97.5])
    return float(lower), float(upper)


def inferential_values(deltas, clusters, inference_unit):
    """Use image deltas or one mean delta per anatomy cluster."""
    deltas = np.asarray(deltas, dtype=np.float64)
    clusters = np.asarray(clusters)
    if inference_unit == "sample":
        return deltas
    return np.asarray(
        [deltas[clusters == cluster].mean() for cluster in np.unique(clusters)],
        dtype=np.float64,
    )


def paired_statistics(frame, region, args):
    rows = []
    clusters = frame["cluster_id"].to_numpy()

    for metric in METRICS:
        vanilla = frame[f"{region}_vanilla_{metric}"].to_numpy(np.float64)
        gate = frame[f"{region}_gate_{metric}"].to_numpy(np.float64)
        deltas = oriented_delta(metric, vanilla, gate)
        test_values = inferential_values(
            deltas, clusters, args.inference_unit
        )

        if np.allclose(test_values, 0.0):
            statistic, raw_p = 0.0, 1.0
        else:
            test = wilcoxon(
                test_values,
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            )
            statistic, raw_p = float(test.statistic), float(test.pvalue)

        ci_low, ci_high = bootstrap_mean_ci(
            deltas,
            clusters,
            args.inference_unit,
            args.n_bootstrap,
            args.seed,
        )
        tolerance = args.tie_tolerance
        wins = int(np.sum(deltas > tolerance))
        losses = int(np.sum(deltas < -tolerance))
        ties = int(len(deltas) - wins - losses)

        rows.append(
            {
                "region": region,
                "metric": metric.upper(),
                "n_images": int(len(deltas)),
                "n_inference_units": int(len(test_values)),
                "vanilla_mean": float(vanilla.mean()),
                "gate_mean": float(gate.mean()),
                "mean_improvement": float(deltas.mean()),
                "median_improvement": float(np.median(deltas)),
                "improvement_std": float(deltas.std(ddof=1)),
                "bootstrap_95_ci_low": ci_low,
                "bootstrap_95_ci_high": ci_high,
                "win_rate_percent": float(100.0 * wins / len(deltas)),
                "tie_rate_percent": float(100.0 * ties / len(deltas)),
                "loss_rate_percent": float(100.0 * losses / len(deltas)),
                "wilcoxon_statistic": statistic,
                "raw_p_value": raw_p,
                "rank_biserial": rank_biserial_effect(test_values),
            }
        )

    adjusted = holm_adjust([row["raw_p_value"] for row in rows])
    for row, adjusted_p in zip(rows, adjusted):
        row["holm_adjusted_p_value"] = float(adjusted_p)
        row["significant_after_holm_0.05"] = bool(adjusted_p < 0.05)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_dir", required=True)
    parser.add_argument("--vanilla_checkpoint", required=True)
    parser.add_argument("--gate_checkpoint", required=True)
    parser.add_argument("--output_dir", default="./paired_stats")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--gate_hidden", type=int, default=32)
    parser.add_argument("--gate_range", type=float, default=0.5)
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--inference_unit",
        choices=("sample", "cluster"),
        default="cluster",
        help=(
            "cluster groups images by the top-level anatomy folder "
            "(e.g. body10/head1); use it only when those folders represent "
            "patients or source volumes."
        ),
    )
    parser.add_argument(
        "--tie_tolerance",
        type=float,
        default=0.0,
        help="Absolute improvement treated as a tie; default is exact tie.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    samples = find_test_samples(args.test_data_dir)
    if not samples:
        raise RuntimeError("No AAPM evaluation samples were found.")
    if len({sample["id"] for sample in samples}) != len(samples):
        raise RuntimeError("Duplicate sample IDs were detected.")

    print(f"Found {len(samples)} paired samples.")
    print(
        f"Clusters ({len(set(s['cluster_id'] for s in samples))}): "
        f"{sorted(set(s['cluster_id'] for s in samples))}"
    )

    vanilla_model, gate_model = load_models(args, device)
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    records = []

    with torch.inference_mode():
        for sample in tqdm(samples, desc="Paired evaluation"):
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
            vanilla_prediction = np.clip(
                vanilla_model(input_tensor).squeeze().cpu().numpy(), 0.0, 1.0
            )
            gate_prediction = np.clip(
                gate_model(input_tensor).squeeze().cpu().numpy(), 0.0, 1.0
            )

            record = {
                "sample_id": sample["id"],
                "cluster_id": sample["cluster_id"],
                "n_materials": sample["n_materials"],
            }
            for region in REGIONS:
                exclude_metal = region == "non_metal"
                vanilla_metrics = evaluate_image(
                    vanilla_prediction,
                    target_image,
                    mask,
                    lpips_fn,
                    device,
                    exclude_metal,
                )
                gate_metrics = evaluate_image(
                    gate_prediction,
                    target_image,
                    mask,
                    lpips_fn,
                    device,
                    exclude_metal,
                )
                for metric in METRICS:
                    record[f"{region}_vanilla_{metric}"] = vanilla_metrics[metric]
                    record[f"{region}_gate_{metric}"] = gate_metrics[metric]
                    record[f"{region}_improvement_{metric}"] = float(
                        oriented_delta(
                            metric,
                            np.asarray(vanilla_metrics[metric]),
                            np.asarray(gate_metrics[metric]),
                        )
                    )
            records.append(record)

    per_image = pd.DataFrame(records).sort_values("sample_id").reset_index(drop=True)
    if len(per_image) != len(samples):
        raise RuntimeError("The number of output rows does not match input samples.")

    all_statistics = []
    for region in REGIONS:
        all_statistics.extend(paired_statistics(per_image, region, args))
    statistics = pd.DataFrame(all_statistics)

    per_image.to_csv(output_dir / "paired_per_image_metrics.csv", index=False)
    statistics.to_csv(output_dir / "paired_statistics.csv", index=False)
    with open(output_dir / "paired_statistics.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "metadata": {
                    "n_images": len(per_image),
                    "n_clusters": int(per_image["cluster_id"].nunique()),
                    "clusters": sorted(per_image["cluster_id"].unique().tolist()),
                    "inference_unit": args.inference_unit,
                    "bootstrap_repetitions": args.n_bootstrap,
                    "seed": args.seed,
                    "positive_improvement_definition": {
                        "PSNR": "Gate - Vanilla",
                        "SSIM": "Gate - Vanilla",
                        "RMSE": "Vanilla - Gate",
                        "LPIPS": "Vanilla - Gate",
                    },
                },
                "results": all_statistics,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    columns = [
        "region",
        "metric",
        "vanilla_mean",
        "gate_mean",
        "mean_improvement",
        "median_improvement",
        "bootstrap_95_ci_low",
        "bootstrap_95_ci_high",
        "win_rate_percent",
        "holm_adjusted_p_value",
        "rank_biserial",
    ]
    print("\n=== PAIRED STATISTICAL ANALYSIS ===")
    print(statistics[columns].to_string(index=False))
    print(f"\nSaved results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

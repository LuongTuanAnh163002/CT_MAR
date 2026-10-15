#!/usr/bin/env python3
"""Test whether Stage-2 residual scale composition varies across images.

This is a pre-method analysis.  It uses only the per-image residual energies
computed from ground truth and frozen Stage-2 predictions by
``analyze_multiscale_residual.py``.  It does not import, load, or evaluate the
proposed Gate model.

For every image i, define the relative shares within the two modeled scales:

    q2_i = E2_i / (E2_i + E4_i)
    q4_i = E4_i / (E2_i + E4_i) = 1 - q2_i

E2 and E4 are normalized component energies from the additive residual
pyramid.  q2/q4 are descriptive relative shares, not orthogonal explained
variance.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


REQUIRED_COLUMNS = {
    "sample_id",
    "energy_ratio_scale2",
    "energy_ratio_scale4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze image-to-image variation in the relative scale-2 and "
            "scale-4 composition of frozen Stage-2 residuals."
        )
    )
    parser.add_argument(
        "--per-image-csv",
        required=True,
        type=Path,
        help="multiscale_residual_per_image.csv produced from GT and Stage 2.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", default=10000, type=int)
    parser.add_argument("--seed", default=19, type=int)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    return args


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise RuntimeError("Input CSV contains no image records")
    return rows


def bootstrap_mean_ci(
    values: np.ndarray, repeats: int, rng: np.random.Generator
) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), float(values[0])
    means = np.empty(repeats, dtype=np.float64)
    chunk_size = 1000
    for start in range(0, repeats, chunk_size):
        count = min(chunk_size, repeats - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def describe(
    values: np.ndarray, repeats: int, rng: np.random.Generator
) -> dict[str, float | int]:
    ci_low, ci_high = bootstrap_mean_ci(values, repeats, rng)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        "n": int(len(values)),
        "mean": mean,
        "sd": sd,
        "coefficient_of_variation_abs_mean": (
            float(sd / abs(mean)) if abs(mean) > 1e-20 else 0.0
        ),
        "minimum": float(values.min()),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
        "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
        "p90_range": float(np.quantile(values, 0.95) - np.quantile(values, 0.05)),
        "ci95_mean_low": ci_low,
        "ci95_mean_high": ci_high,
    }


def safe_correlation(
    x: np.ndarray, y: np.ndarray, method: str
) -> dict[str, float]:
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return {"coefficient": 0.0, "p_value": 1.0}
    result = pearsonr(x, y) if method == "pearson" else spearmanr(x, y)
    return {"coefficient": float(result.statistic), "p_value": float(result.pvalue)}


def save_figure(
    energy2: np.ndarray,
    energy4: np.ndarray,
    share2: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    scatter = axes[0].scatter(
        energy4,
        energy2,
        c=share2,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=18,
        alpha=0.72,
        linewidths=0,
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Scale-4 energy ratio $E_4$")
    axes[0].set_ylabel(r"Scale-2 energy ratio $E_2$")
    axes[0].set_title("Per-image residual composition")
    axes[0].grid(alpha=0.22)
    colorbar = fig.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar.set_label(r"Relative scale-2 share $q_2$")

    axes[1].hist(share2, bins=35, color="#4c78a8", alpha=0.88)
    axes[1].axvline(
        float(np.median(share2)),
        color="#c51b32",
        linestyle="--",
        linewidth=1.6,
        label=f"Median = {np.median(share2):.3f}",
    )
    axes[1].axvspan(
        float(np.quantile(share2, 0.25)),
        float(np.quantile(share2, 0.75)),
        color="#f2a65a",
        alpha=0.22,
        label="IQR",
    )
    axes[1].set_xlabel(r"$q_2=E_2/(E_2+E_4)$")
    axes[1].set_ylabel("Number of images")
    axes[1].set_title("Relative scale contribution")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.22)

    sorted_share = np.sort(share2)
    percentile = 100.0 * (np.arange(len(sorted_share)) + 0.5) / len(sorted_share)
    axes[2].plot(percentile, sorted_share, color="#3b7a57", linewidth=2.0)
    axes[2].fill_between(percentile, 0.0, sorted_share, color="#7fbf7b", alpha=0.22)
    axes[2].set_xlabel("Image percentile")
    axes[2].set_ylabel(r"Relative scale-2 share $q_2$")
    axes[2].set_title("Image-to-image variation")
    axes[2].set_xlim(0.0, 100.0)
    axes[2].set_ylim(0.0, min(1.0, max(0.25, float(sorted_share.max()) * 1.08)))
    axes[2].grid(alpha=0.22)

    fig.suptitle(
        "Variation of Stage-2 residual composition across images",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.005,
        "$q_2$ and $q_4$ are descriptive shares within the two modeled scales; "
        "they are not orthogonal explained-variance percentages.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.94))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = read_rows(args.per_image_csv)

    energy2 = np.asarray(
        [float(row["energy_ratio_scale2"]) for row in source_rows], dtype=np.float64
    )
    energy4 = np.asarray(
        [float(row["energy_ratio_scale4"]) for row in source_rows], dtype=np.float64
    )
    if np.any(energy2 < 0.0) or np.any(energy4 < 0.0):
        raise ValueError("Component energy ratios must be non-negative")

    denominator = energy2 + energy4
    if np.any(denominator <= 1e-20):
        bad = int(np.sum(denominator <= 1e-20))
        raise ValueError(f"Found {bad} images with zero energy at both modeled scales")

    share2 = energy2 / denominator
    share4 = energy4 / denominator
    log_ratio_2_to_4 = np.log10((energy2 + 1e-20) / (energy4 + 1e-20))

    augmented_rows = []
    for row, q2, q4, log_ratio in zip(
        source_rows, share2, share4, log_ratio_2_to_4
    ):
        augmented_rows.append(
            {
                "sample_id": row["sample_id"],
                "energy_ratio_scale2": row["energy_ratio_scale2"],
                "energy_ratio_scale4": row["energy_ratio_scale4"],
                "relative_share_scale2_q2": float(q2),
                "relative_share_scale4_q4": float(q4),
                "log10_energy_ratio_scale2_to_scale4": float(log_ratio),
            }
        )

    csv_path = args.output_dir / "residual_scale_variation_per_image.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(augmented_rows[0]))
        writer.writeheader()
        writer.writerows(augmented_rows)

    rng = np.random.default_rng(args.seed)
    summary = {
        "metadata": {
            "n": len(source_rows),
            "source_csv": str(args.per_image_csv.resolve()),
            "uses_gate_model": False,
            "inputs_to_source_analysis": "ground truth and frozen Stage-2 predictions",
        },
        "definitions": {
            "q2": "E2 / (E2 + E4)",
            "q4": "E4 / (E2 + E4)",
            "note": (
                "q2 and q4 sum to one by definition and describe relative energy "
                "within the two modeled scales. The resize-based components are "
                "non-orthogonal, so these are not explained-variance percentages."
            ),
        },
        "statistics": {
            "energy_ratio_scale2": describe(energy2, args.bootstrap, rng),
            "energy_ratio_scale4": describe(energy4, args.bootstrap, rng),
            "relative_share_scale2_q2": describe(share2, args.bootstrap, rng),
            "relative_share_scale4_q4": describe(share4, args.bootstrap, rng),
            "log10_energy_ratio_scale2_to_scale4": describe(
                log_ratio_2_to_4, args.bootstrap, rng
            ),
        },
        "correlations": {
            "energy_scale2_vs_scale4_pearson": safe_correlation(
                energy2, energy4, "pearson"
            ),
            "energy_scale2_vs_scale4_spearman": safe_correlation(
                energy2, energy4, "spearman"
            ),
        },
        "predefined_descriptive_fractions": {
            "q2_below_0.10": float(np.mean(share2 < 0.10)),
            "q2_between_0.10_and_0.25": float(
                np.mean((share2 >= 0.10) & (share2 < 0.25))
            ),
            "q2_at_least_0.25": float(np.mean(share2 >= 0.25)),
            "scale2_share_exceeds_scale4_share": float(np.mean(share2 > 0.50)),
        },
        "interpretation_guardrails": [
            "Variation in q2 motivates testing image-conditioned fusion but does not prove it is superior.",
            "Fixed versus learned fusion must still be compared experimentally.",
            "This analysis does not show that scale 2 and scale 4 are globally optimal choices.",
        ],
    }
    json_path = args.output_dir / "residual_scale_variation_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figure_path = args.output_dir / "residual_scale_variation.png"
    save_figure(energy2, energy4, share2, figure_path)

    q2_stats = summary["statistics"]["relative_share_scale2_q2"]
    pearson = summary["correlations"]["energy_scale2_vs_scale4_pearson"]
    spearman = summary["correlations"]["energy_scale2_vs_scale4_spearman"]
    print("\n=== STAGE-2 RESIDUAL SCALE VARIATION ===")
    print(f"Images: {len(source_rows)}")
    print(
        "q2: "
        f"mean={q2_stats['mean']:.6f}, sd={q2_stats['sd']:.6f}, "
        f"median={q2_stats['median']:.6f}, "
        f"IQR=[{q2_stats['q25']:.6f}, {q2_stats['q75']:.6f}], "
        f"5-95%=[{q2_stats['q05']:.6f}, {q2_stats['q95']:.6f}]"
    )
    print(
        f"E2/E4 correlation: Pearson r={pearson['coefficient']:.6f}, "
        f"Spearman rho={spearman['coefficient']:.6f}"
    )
    print(f"Saved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

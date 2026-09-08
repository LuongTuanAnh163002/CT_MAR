#!/usr/bin/env python3
"""
analyze_fmb_features.py
=======================

Diagnostic for the hypothesis:

    Multi-metal cases may produce more heterogeneous directional features,
    while MARMamba's multiplicative FMB fusion can suppress branch-specific
    responses when the three directional branches disagree.

THIS SCRIPT DOES NOT TRAIN OR MODIFY THE MODEL WEIGHTS.

It:
1) loads the best baseline MARMamba checkpoint;
2) instruments every baseline MambaAttention/FMB;
3) captures the three branch outputs immediately before multiplicative fusion;
4) verifies that the instrumented forward is numerically equivalent to the
   original baseline forward on a real AAPM sample;
5) runs AAPM test inference and computes feature diagnostics;
6) groups them by metal count, Single/Multi, size group, and FMB layer;
7) correlates the feature diagnostics with non-metal PSNR.

Core diagnostics
----------------
For branch magnitudes a0=|F0|, a1=|F1|, a2=|F2|:

Arithmetic mean:
    AM = (a0 + a1 + a2) / 3

Geometric mean:
    GM = (a0 * a1 * a2)^(1/3)

GM/AM ratio:
    agreement = GM / (AM + eps)

By AM-GM, agreement is approximately in [0, 1].
- near 1: the three branch magnitudes agree;
- near 0: one/more branches are weak relative to another branch.

Suppression index:
    suppression = 1 - agreement

This is preferable to comparing raw product norms across groups because the
raw product changes scale cubically with feature amplitude.

We also compute:
- pairwise cosine similarity among F0/F1/F2;
- branch dominance = max(a0,a1,a2)/(a0+a1+a2);
- strong-disagreement fraction:
    among the top-q branch-active positions, fraction whose GM/AM ratio is
    below a configurable threshold.

Expected pattern supporting the Residual-Fusion hypothesis:
    metal count ↑
        GM/AM agreement ↓
        cosine similarity ↓
        suppression index ↑
        branch dominance ↑
    and
        suppression index ↑  <->  non-metal PSNR ↓

Do NOT treat any single metric as proof. Look for a consistent pattern across
metal counts and across multiple FMB layers.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import types
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# -------------------------------------------------------------------------
# Project imports
# -------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.mamba import MambaFormer, MambaAttention
from utils.metrics import calculate_psnr

try:
    from scipy.stats import pearsonr, spearmanr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# -------------------------------------------------------------------------
# AAPM indexing/loading
# Kept aligned with the user's test/test_aapm.py protocol.
# -------------------------------------------------------------------------

HU_MIN, HU_MAX = -1000.0, 3000.0
FNAME_ID_RE = re.compile(r"img(\d+)")
DIMS_RE = re.compile(r"(\d+)x(\d+)x(\d+)")


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
        raise ValueError(f"{path}: found {arr.size} values, expected {expected}")
    return arr.reshape(rows, cols) if slices == 1 else arr.reshape(slices, rows, cols)


def hu_to_unit(img):
    img = np.clip(img, HU_MIN, HU_MAX)
    return ((img - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def find_test_samples(test_data_dir: str):
    def _index_one(anatomy_dir):
        prefix = os.path.basename(os.path.normpath(anatomy_dir))
        baseline_dir = os.path.join(anatomy_dir, "Baseline")
        target_dir = os.path.join(anatomy_dir, "Target")
        mask_dir = os.path.join(anatomy_dir, "Mask")

        if not (os.path.isdir(baseline_dir) and os.path.isdir(target_dir)):
            return []

        target_map = {
            FNAME_ID_RE.search(os.path.basename(f)).group(1): f
            for f in glob.glob(os.path.join(target_dir, "*.raw"))
            if "img" in os.path.basename(f)
            and FNAME_ID_RE.search(os.path.basename(f))
        }

        mask_map = {}
        n_materials_map = {}

        if os.path.isdir(mask_dir):
            for f in glob.glob(os.path.join(mask_dir, "*.raw")):
                m = FNAME_ID_RE.search(os.path.basename(f))
                if m:
                    mask_map[m.group(1)] = f

            for f in glob.glob(os.path.join(mask_dir, "*.json")):
                m = re.search(r"metalinfo(\d+)", os.path.basename(f))
                if m:
                    try:
                        with open(f, "r", encoding="utf-8") as jf:
                            n_materials_map[m.group(1)] = json.load(jf).get(
                                "n_materials", None
                            )
                    except Exception:
                        pass

        samples = []
        for bf in glob.glob(os.path.join(baseline_dir, "*.raw")):
            if "img" not in os.path.basename(bf):
                continue

            m = FNAME_ID_RE.search(os.path.basename(bf))
            if not m:
                continue

            img_id = m.group(1)
            if img_id not in target_map:
                continue

            samples.append(
                {
                    "id": f"{prefix}_{img_id}",
                    "baseline": bf,
                    "target": target_map[img_id],
                    "mask": mask_map.get(img_id),
                    "n_materials": n_materials_map.get(img_id),
                }
            )

        return samples

    if os.path.isdir(os.path.join(test_data_dir, "Baseline")):
        return _index_one(test_data_dir)

    all_samples = []
    for entry in sorted(os.listdir(test_data_dir)):
        sub = os.path.join(test_data_dir, entry)
        if os.path.isdir(sub) and os.path.isdir(os.path.join(sub, "Baseline")):
            all_samples.extend(_index_one(sub))

    return all_samples


def compute_quartiles(sizes):
    q25, q50, q75 = np.percentile(np.asarray(sizes), [25, 50, 75])
    return float(q25), float(q50), float(q75)


def size_to_group(size, q25, q50, q75):
    if size >= q75:
        return "Large"
    if size >= q50:
        return "Medium"
    if size >= q25:
        return "Small"
    return "Tiny"


def to_uint8_bgr3(img_float01):
    img_u8 = np.clip(img_float01 * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)


# -------------------------------------------------------------------------
# Checkpoint loading
# -------------------------------------------------------------------------

def load_baseline_model(checkpoint_path: str, device: torch.device):
    """
    Single-device evaluation is intentional:
    - batch size is one;
    - easier and safer feature instrumentation;
    - checkpoints saved via DataParallel are supported by stripping "module.".
    """
    net = MambaFormer(in_channels=1).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a valid state_dict.")

    if any(k.startswith("module.") for k in state.keys()):
        state = {
            k[len("module."):]: v
            for k, v in state.items()
        }

    net.load_state_dict(state, strict=True)
    net.eval()
    return net


# -------------------------------------------------------------------------
# FMB diagnostics
# -------------------------------------------------------------------------

class FMBCollector:
    def __init__(
        self,
        eps=1e-8,
        strong_quantile=0.75,
        disagreement_threshold=0.25,
    ):
        self.eps = float(eps)
        self.strong_quantile = float(strong_quantile)
        self.disagreement_threshold = float(disagreement_threshold)
        self.records = []
        self.enabled = True

    def clear(self):
        self.records = []

    @torch.no_grad()
    def add(self, layer_name, f0, f1, f2):
        if not self.enabled:
            return

        # Expected branch shape: [B, HW, C_branch].
        if f0.shape != f1.shape or f0.shape != f2.shape:
            raise RuntimeError(
                f"Branch shape mismatch at {layer_name}: "
                f"{tuple(f0.shape)}, {tuple(f1.shape)}, {tuple(f2.shape)}"
            )

        eps = self.eps

        a0 = f0.abs()
        a1 = f1.abs()
        a2 = f2.abs()

        arithmetic_mean = (a0 + a1 + a2) / 3.0
        geometric_mean = torch.pow(
            (a0 + eps) * (a1 + eps) * (a2 + eps),
            1.0 / 3.0,
        )

        agreement = geometric_mean / (arithmetic_mean + eps)
        # Numerical noise can put a tiny number above 1.
        agreement = agreement.clamp(0.0, 1.0)

        suppression = 1.0 - agreement

        stacked_abs = torch.stack([a0, a1, a2], dim=0)
        max_abs = stacked_abs.max(dim=0).values
        sum_abs = stacked_abs.sum(dim=0)
        dominance = max_abs / (sum_abs + eps)

        # Strong disagreement is only a secondary, thresholded diagnostic.
        flat_max = max_abs.reshape(-1)
        q = torch.quantile(flat_max, self.strong_quantile)
        strong = max_abs >= q
        low_agreement = agreement < self.disagreement_threshold

        strong_count = int(strong.sum().item())
        if strong_count > 0:
            strong_disagreement_fraction = float(
                (strong & low_agreement).float().sum().item() / strong_count
            )
        else:
            strong_disagreement_fraction = float("nan")

        def cosine(x, y):
            xx = x.reshape(x.shape[0], -1)
            yy = y.reshape(y.shape[0], -1)
            return float(
                F.cosine_similarity(xx, yy, dim=1, eps=eps).mean().item()
            )

        cos01 = cosine(f0, f1)
        cos02 = cosine(f0, f2)
        cos12 = cosine(f1, f2)

        fused_product = f0 * f1 * f2

        rec = {
            "layer": layer_name,
            "mean_abs_f0": float(a0.mean().item()),
            "mean_abs_f1": float(a1.mean().item()),
            "mean_abs_f2": float(a2.mean().item()),
            "mean_abs_product": float(fused_product.abs().mean().item()),
            "gm_am_agreement": float(agreement.mean().item()),
            "suppression_index": float(suppression.mean().item()),
            "branch_dominance": float(dominance.mean().item()),
            "cosine_01": cos01,
            "cosine_02": cos02,
            "cosine_12": cos12,
            "cosine_mean": float((cos01 + cos02 + cos12) / 3.0),
            "strong_disagreement_fraction": strong_disagreement_fraction,
        }
        self.records.append(rec)


def patch_mamba_attention_for_diagnostics(
    model: nn.Module,
    collector: FMBCollector,
):
    """
    Reproduce the baseline MambaAttention forward while recording F0/F1/F2.

    IMPORTANT:
    The three branch transformations below match the branch path used by the
    metal-guided experiment, which explicitly preserved baseline branch
    transformations and changed only the fusion.

    A mandatory forward-equivalence check is performed later. If this patched
    forward does not reproduce the original MARMamba output, the script aborts.
    """
    patched = []

    for module_name, module in model.named_modules():
        if not isinstance(module, MambaAttention):
            continue

        original_forward = module.forward

        def diagnostic_forward(
            self,
            x,
            _module_name=module_name,
            _collector=collector,
        ):
            _, _, H, W = x.shape

            x_proj = self.proj_1(x)
            normal, flip_x1, flip_x2 = torch.chunk(x_proj, 3, dim=1)

            # Exact directional transformations used by baseline branches.
            flip_x1 = flip_x1.flatten(2).transpose(1, 2)
            flip_x1 = torch.flip(flip_x1, dims=[-1])

            flip_x2 = flip_x2.flatten(2).transpose(1, 2)
            flip_x2 = torch.flip(flip_x2, dims=[-2])

            normal = normal.flatten(2).transpose(1, 2)

            normal = self.mamba_norm(normal)
            flip_x1 = self.mamba_flip1(flip_x1)
            flip_x2 = self.mamba_flip2(flip_x2)

            flip_x1 = torch.flip(flip_x1, dims=[-1])
            flip_x2 = torch.flip(flip_x2, dims=[-2])

            _collector.add(_module_name, normal, flip_x1, flip_x2)

            # Baseline multiplicative FMB fusion.
            fused = normal * flip_x1 * flip_x2
            fused = rearrange(
                fused,
                "b (h w) c -> b c h w",
                h=H,
                w=W,
            )
            return self.proj_2(fused)

        module.forward = types.MethodType(diagnostic_forward, module)
        patched.append((module_name, module, original_forward))

    if not patched:
        raise RuntimeError(
            "No MambaAttention modules found. "
            "Check whether model/mamba.py changed."
        )

    return patched


def restore_original_forwards(patched):
    for _, module, original_forward in patched:
        module.forward = original_forward


# -------------------------------------------------------------------------
# Evaluation helpers
# -------------------------------------------------------------------------

def build_input(sample, device):
    baseline_hu = load_raw(sample["baseline"])
    target_hu = load_raw(sample["target"])

    xma = hu_to_unit(baseline_hu)
    xgt = hu_to_unit(target_hu)

    input_t = (
        torch.from_numpy((xma - 0.5) / 0.5)
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    mask = None
    if sample.get("mask") is not None:
        mask = load_raw(sample["mask"], dtype=np.float32) > 0.5

    return input_t, xgt, mask


def compute_psnr_pair(pred, gt, mask):
    # Full-image PSNR
    pred_full_bgr = to_uint8_bgr3(pred)
    gt_full_bgr = to_uint8_bgr3(gt)
    psnr_full = float(
        calculate_psnr(
            pred_full_bgr,
            gt_full_bgr,
            test_y_channel=True,
        )
    )

    # Non-metal PSNR: exactly aligned with user's AAPM evaluator.
    pred_nm = pred.copy()
    gt_nm = gt.copy()
    if mask is not None:
        pred_nm[mask] = 0.0
        gt_nm[mask] = 0.0

    pred_nm_bgr = to_uint8_bgr3(pred_nm)
    gt_nm_bgr = to_uint8_bgr3(gt_nm)
    psnr_nonmetal = float(
        calculate_psnr(
            pred_nm_bgr,
            gt_nm_bgr,
            test_y_channel=True,
        )
    )

    return psnr_full, psnr_nonmetal


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_case_records(layer_rows):
    """
    Average layer diagnostics per case.
    PSNR and metadata are identical across layer rows.
    """
    grouped = defaultdict(list)
    for row in layer_rows:
        grouped[row["id"]].append(row)

    metrics = [
        "mean_abs_f0",
        "mean_abs_f1",
        "mean_abs_f2",
        "mean_abs_product",
        "gm_am_agreement",
        "suppression_index",
        "branch_dominance",
        "cosine_01",
        "cosine_02",
        "cosine_12",
        "cosine_mean",
        "strong_disagreement_fraction",
    ]

    out = []
    for case_id, rows in grouped.items():
        base = rows[0]
        rec = {
            "id": case_id,
            "n_materials": base["n_materials"],
            "is_multi": base["is_multi"],
            "mask_size": base["mask_size"],
            "size_group": base["size_group"],
            "psnr_full": base["psnr_full"],
            "psnr_nonmetal": base["psnr_nonmetal"],
            "num_fmb_layers": len(rows),
        }

        for key in metrics:
            vals = np.asarray([r[key] for r in rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            rec[key] = float(vals.mean()) if vals.size else float("nan")

        out.append(rec)

    return out


def summarize_groups(rows, group_key, metric_keys):
    buckets = defaultdict(list)
    for row in rows:
        key = row[group_key]
        if key is None or key == "":
            continue
        buckets[key].append(row)

    summary = []
    for key in sorted(buckets.keys(), key=lambda x: str(x)):
        group_rows = buckets[key]
        rec = {
            group_key: key,
            "n": len(group_rows),
        }

        for metric in metric_keys:
            vals = [float(r[metric]) for r in group_rows]
            m, s = mean_std(vals)
            rec[f"{metric}_mean"] = m
            rec[f"{metric}_std"] = s

        summary.append(rec)

    return summary


def summarize_by_count_and_layer(layer_rows, metric_keys):
    buckets = defaultdict(list)

    for row in layer_rows:
        n = row["n_materials"]
        if n is None:
            continue
        buckets[(int(n), row["layer"])].append(row)

    result = []
    for (n, layer), rows in sorted(
        buckets.items(),
        key=lambda kv: (kv[0][0], kv[0][1]),
    ):
        rec = {
            "n_materials": n,
            "layer": layer,
            "n": len(rows),
        }

        for metric in metric_keys:
            vals = [float(r[metric]) for r in rows]
            m, s = mean_std(vals)
            rec[f"{metric}_mean"] = m
            rec[f"{metric}_std"] = s

        result.append(rec)

    return result


def compute_correlations(case_rows):
    pairs = [
        ("n_materials", "gm_am_agreement"),
        ("n_materials", "suppression_index"),
        ("n_materials", "cosine_mean"),
        ("n_materials", "branch_dominance"),
        ("n_materials", "strong_disagreement_fraction"),
        ("suppression_index", "psnr_nonmetal"),
        ("gm_am_agreement", "psnr_nonmetal"),
        ("cosine_mean", "psnr_nonmetal"),
        ("branch_dominance", "psnr_nonmetal"),
    ]

    out = []

    for x_key, y_key in pairs:
        xy = []
        for row in case_rows:
            x = row.get(x_key)
            y = row.get(y_key)
            if x is None or y is None:
                continue
            try:
                x = float(x)
                y = float(y)
            except Exception:
                continue
            if np.isfinite(x) and np.isfinite(y):
                xy.append((x, y))

        if len(xy) < 3:
            continue

        x = np.asarray([p[0] for p in xy], dtype=np.float64)
        y = np.asarray([p[1] for p in xy], dtype=np.float64)

        rec = {
            "x": x_key,
            "y": y_key,
            "n": len(x),
        }

        if HAVE_SCIPY:
            p_r = pearsonr(x, y)
            s_r = spearmanr(x, y)
            rec.update(
                {
                    "pearson_r": float(p_r.statistic),
                    "pearson_p": float(p_r.pvalue),
                    "spearman_rho": float(s_r.statistic),
                    "spearman_p": float(s_r.pvalue),
                }
            )
        else:
            rec.update(
                {
                    "pearson_r": float(np.corrcoef(x, y)[0, 1]),
                    "pearson_p": float("nan"),
                    "spearman_rho": float("nan"),
                    "spearman_p": float("nan"),
                }
            )

        out.append(rec)

    return out


def make_plots(output_dir, count_summary, case_rows):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] Could not import matplotlib: {e}")
        return

    if count_summary:
        count_summary = sorted(
            count_summary,
            key=lambda r: int(r["n_materials"]),
        )
        counts = [int(r["n_materials"]) for r in count_summary]

        for metric, ylabel, filename in [
            (
                "gm_am_agreement",
                "GM/AM branch agreement",
                "agreement_vs_metal_count.png",
            ),
            (
                "suppression_index",
                "Suppression index",
                "suppression_vs_metal_count.png",
            ),
            (
                "cosine_mean",
                "Mean pairwise cosine similarity",
                "cosine_vs_metal_count.png",
            ),
        ]:
            means = [r[f"{metric}_mean"] for r in count_summary]
            stds = [r[f"{metric}_std"] for r in count_summary]

            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ax.errorbar(counts, means, yerr=stds, marker="o", capsize=3)
            ax.set_xlabel("Number of metallic objects")
            ax.set_ylabel(ylabel)
            ax.set_xticks(counts)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(
                os.path.join(output_dir, filename),
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(fig)

    # Suppression vs non-metal PSNR scatter.
    xs = []
    ys = []
    for r in case_rows:
        x = float(r["suppression_index"])
        y = float(r["psnr_nonmetal"])
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x)
            ys.append(y)

    if xs:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.scatter(xs, ys, s=12, alpha=0.45)
        ax.set_xlabel("Mean FMB suppression index")
        ax.set_ylabel("Non-metal PSNR (dB)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, "suppression_vs_psnr.png"),
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)


def find_correlation(correlation_rows, x, y):
    for r in correlation_rows:
        if r["x"] == x and r["y"] == y:
            return r
    return None


def write_hypothesis_summary(
    path,
    count_summary,
    single_multi_summary,
    correlations,
    num_layers,
    equivalence_error,
):
    by_count = {
        int(r["n_materials"]): r
        for r in count_summary
    }

    lines = [
        "FMB MULTIPLICATIVE-FUSION DIAGNOSTIC",
        "=" * 72,
        f"Instrumented FMB layers: {num_layers}",
        f"Forward-equivalence max abs error: {equivalence_error:.10e}",
        "",
        "Hypothesis being tested:",
        "  Increasing metal complexity may create more heterogeneous directional",
        "  branch representations, causing multiplicative FMB fusion to suppress",
        "  branch-specific information more strongly.",
        "",
        "A descriptively supportive pattern would be:",
        "  metal count increases -> GM/AM agreement decreases",
        "  metal count increases -> cosine similarity decreases",
        "  metal count increases -> suppression index increases",
        "  metal count increases -> branch dominance increases",
        "  suppression index increases -> non-metal PSNR decreases",
        "",
        "IMPORTANT: these diagnostics establish association, not causality.",
        "",
    ]

    if 1 in by_count and 5 in by_count:
        r1 = by_count[1]
        r5 = by_count[5]

        lines.extend(
            [
                "Count 1 vs Count 5 (case-level averages across all FMB layers):",
                f"  GM/AM agreement:     {r1['gm_am_agreement_mean']:.6f} -> "
                f"{r5['gm_am_agreement_mean']:.6f} "
                f"(delta={r5['gm_am_agreement_mean'] - r1['gm_am_agreement_mean']:+.6f})",
                f"  Suppression index:   {r1['suppression_index_mean']:.6f} -> "
                f"{r5['suppression_index_mean']:.6f} "
                f"(delta={r5['suppression_index_mean'] - r1['suppression_index_mean']:+.6f})",
                f"  Cosine mean:         {r1['cosine_mean_mean']:.6f} -> "
                f"{r5['cosine_mean_mean']:.6f} "
                f"(delta={r5['cosine_mean_mean'] - r1['cosine_mean_mean']:+.6f})",
                f"  Branch dominance:    {r1['branch_dominance_mean']:.6f} -> "
                f"{r5['branch_dominance_mean']:.6f} "
                f"(delta={r5['branch_dominance_mean'] - r1['branch_dominance_mean']:+.6f})",
                f"  Non-metal PSNR:      {r1['psnr_nonmetal_mean']:.4f} -> "
                f"{r5['psnr_nonmetal_mean']:.4f} dB",
                "",
            ]
        )

    lines.append("Correlations:")
    for x, y in [
        ("n_materials", "suppression_index"),
        ("n_materials", "gm_am_agreement"),
        ("n_materials", "cosine_mean"),
        ("suppression_index", "psnr_nonmetal"),
    ]:
        r = find_correlation(correlations, x, y)
        if r is None:
            continue
        lines.append(
            f"  {x} vs {y}: "
            f"Pearson r={r['pearson_r']:.4f}, p={r['pearson_p']:.3e}; "
            f"Spearman rho={r['spearman_rho']:.4f}, p={r['spearman_p']:.3e}"
        )

    lines.extend(
        [
            "",
            "Decision guidance:",
            "  - If the expected directional pattern is consistent across metal",
            "    counts AND multiple FMB layers, Residual Fusion has a stronger",
            "    empirical motivation.",
            "  - If the diagnostics are flat/inconsistent, do NOT spend a full",
            "    320k-step run on Residual Fusion solely based on this hypothesis.",
            "",
            "Inspect summary_by_metal_count_and_layer.csv before making the final",
            "decision; a global average can hide layer-specific behavior.",
        ]
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze baseline MARMamba FMB branch agreement/suppression "
            "on AAPM CT-MAR."
        )
    )

    parser.add_argument(
        "--test_data_dir",
        required=True,
        help="AAPM test root.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Best baseline MARMamba checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        default="./fmb_feature_analysis",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help=(
            "Optional quick test. Example: --max_samples 20. "
            "Omit for all 1000 cases."
        ),
    )
    parser.add_argument(
        "--strong_quantile",
        type=float,
        default=0.75,
        help="Top branch-activation quantile for the secondary disagreement metric.",
    )
    parser.add_argument(
        "--disagreement_threshold",
        type=float,
        default=0.25,
        help="GM/AM threshold for strong-disagreement fraction.",
    )
    parser.add_argument(
        "--equivalence_tolerance",
        type=float,
        default=1e-5,
        help=(
            "Abort if instrumented forward differs from baseline by more than "
            "this max absolute error."
        ),
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Save diagnostic PNG plots.",
    )

    args = parser.parse_args()

    if not (0.0 < args.strong_quantile < 1.0):
        raise ValueError("--strong_quantile must be between 0 and 1.")
    if not (0.0 < args.disagreement_threshold < 1.0):
        raise ValueError("--disagreement_threshold must be between 0 and 1.")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    device = torch.device(args.device)

    # -------------------------------------------------------------
    # 1. Dataset metadata and exact AAPM grouping
    # -------------------------------------------------------------
    samples = find_test_samples(args.test_data_dir)
    if not samples:
        raise RuntimeError(f"No AAPM test samples found in {args.test_data_dir}")

    print(f"Found {len(samples)} AAPM test cases.")

    print("Computing true metal-mask areas...")
    for s in samples:
        if s["mask"] is None:
            s["mask_size"] = 0
        else:
            mask_arr = load_raw(s["mask"], dtype=np.float32)
            s["mask_size"] = int((mask_arr > 0.5).sum())

    sizes = [s["mask_size"] for s in samples]
    q25, q50, q75 = compute_quartiles(sizes)
    print(
        f"Metal-mask quartiles: "
        f"q25={q25:.0f}, q50={q50:.0f}, q75={q75:.0f}"
    )

    for s in samples:
        s["size_group"] = size_to_group(
            s["mask_size"],
            q25,
            q50,
            q75,
        )
        s["is_multi"] = (
            s["n_materials"] is not None
            and int(s["n_materials"]) >= 2
        )

    if args.max_samples is not None:
        samples = samples[: args.max_samples]
        print(f"Quick-test mode: using first {len(samples)} cases.")

    # -------------------------------------------------------------
    # 2. Load untouched baseline model
    # -------------------------------------------------------------
    print(f"Loading baseline checkpoint: {args.checkpoint}")
    net = load_baseline_model(args.checkpoint, device)

    # -------------------------------------------------------------
    # 3. Mandatory forward-equivalence check
    # -------------------------------------------------------------
    print("\nRunning mandatory forward-equivalence check...")

    first_input, _, _ = build_input(samples[0], device)

    with torch.inference_mode():
        reference_output = net(first_input).detach().clone()

    collector = FMBCollector(
        strong_quantile=args.strong_quantile,
        disagreement_threshold=args.disagreement_threshold,
    )
    collector.enabled = False

    patched = patch_mamba_attention_for_diagnostics(net, collector)
    print(f"Instrumented {len(patched)} MambaAttention/FMB modules:")
    for name, _, _ in patched:
        print(f"  - {name}")

    with torch.inference_mode():
        diagnostic_output = net(first_input).detach()

    max_abs_error = float(
        (reference_output - diagnostic_output).abs().max().item()
    )
    mean_abs_error = float(
        (reference_output - diagnostic_output).abs().mean().item()
    )

    print(
        f"Forward equivalence: max_abs_error={max_abs_error:.10e}, "
        f"mean_abs_error={mean_abs_error:.10e}"
    )

    if (
        not np.isfinite(max_abs_error)
        or max_abs_error > args.equivalence_tolerance
    ):
        restore_original_forwards(patched)
        raise RuntimeError(
            "\nInstrumented FMB forward does NOT reproduce baseline output.\n"
            f"max_abs_error={max_abs_error:.10e} > "
            f"tolerance={args.equivalence_tolerance:.10e}\n"
            "Analysis aborted intentionally. Do not trust feature statistics "
            "until model/mamba.py is inspected and the diagnostic forward "
            "is corrected."
        )

    print("PASS: diagnostic forward reproduces baseline within tolerance.")

    collector.enabled = True
    collector.clear()

    # -------------------------------------------------------------
    # 4. Run diagnostic inference
    # -------------------------------------------------------------
    layer_rows = []

    with torch.inference_mode():
        for idx, s in enumerate(samples):
            input_t, xgt, mask = build_input(s, device)

            collector.clear()

            pred_t = net(input_t)
            pred = pred_t.squeeze().detach().cpu().numpy()
            pred = np.clip(pred, 0.0, 1.0)

            psnr_full, psnr_nonmetal = compute_psnr_pair(
                pred,
                xgt,
                mask,
            )

            case_layer_records = list(collector.records)

            if len(case_layer_records) != len(patched):
                raise RuntimeError(
                    f"Case {s['id']}: expected {len(patched)} FMB records, "
                    f"got {len(case_layer_records)}."
                )

            for rec in case_layer_records:
                row = {
                    "id": s["id"],
                    "n_materials": (
                        int(s["n_materials"])
                        if s["n_materials"] is not None
                        else None
                    ),
                    "is_multi": "multi" if s["is_multi"] else "single",
                    "mask_size": int(s["mask_size"]),
                    "size_group": s["size_group"],
                    "psnr_full": psnr_full,
                    "psnr_nonmetal": psnr_nonmetal,
                }
                row.update(rec)
                layer_rows.append(row)

            if (idx + 1) % 50 == 0 or idx == 0:
                print(
                    f"Processed {idx + 1}/{len(samples)} cases | "
                    f"PSNR(non-metal)={psnr_nonmetal:.3f} dB"
                )

    # Restore model methods after diagnostics.
    restore_original_forwards(patched)

    # -------------------------------------------------------------
    # 5. Aggregate and save
    # -------------------------------------------------------------
    case_rows = aggregate_case_records(layer_rows)

    metric_keys = [
        "psnr_nonmetal",
        "gm_am_agreement",
        "suppression_index",
        "cosine_mean",
        "branch_dominance",
        "strong_disagreement_fraction",
    ]

    count_summary = summarize_groups(
        case_rows,
        "n_materials",
        metric_keys,
    )

    single_multi_summary = summarize_groups(
        case_rows,
        "is_multi",
        metric_keys,
    )

    size_summary = summarize_groups(
        case_rows,
        "size_group",
        metric_keys,
    )

    layer_metric_keys = [
        "gm_am_agreement",
        "suppression_index",
        "cosine_mean",
        "branch_dominance",
        "strong_disagreement_fraction",
    ]

    count_layer_summary = summarize_by_count_and_layer(
        layer_rows,
        layer_metric_keys,
    )

    correlations = compute_correlations(case_rows)

    layer_csv = os.path.join(
        args.output_dir,
        "fmb_feature_per_layer.csv",
    )
    case_csv = os.path.join(
        args.output_dir,
        "fmb_feature_per_case.csv",
    )
    count_csv = os.path.join(
        args.output_dir,
        "summary_by_metal_count.csv",
    )
    single_multi_csv = os.path.join(
        args.output_dir,
        "summary_single_multi.csv",
    )
    size_csv = os.path.join(
        args.output_dir,
        "summary_by_size.csv",
    )
    count_layer_csv = os.path.join(
        args.output_dir,
        "summary_by_metal_count_and_layer.csv",
    )
    corr_csv = os.path.join(
        args.output_dir,
        "correlations.csv",
    )
    summary_txt = os.path.join(
        args.output_dir,
        "hypothesis_summary.txt",
    )

    write_csv(layer_csv, layer_rows)
    write_csv(case_csv, case_rows)
    write_csv(count_csv, count_summary)
    write_csv(single_multi_csv, single_multi_summary)
    write_csv(size_csv, size_summary)
    write_csv(count_layer_csv, count_layer_summary)
    write_csv(corr_csv, correlations)

    write_hypothesis_summary(
        summary_txt,
        count_summary,
        single_multi_summary,
        correlations,
        num_layers=len(patched),
        equivalence_error=max_abs_error,
    )

    if args.save_plots:
        make_plots(
            args.output_dir,
            count_summary,
            case_rows,
        )

    # -------------------------------------------------------------
    # 6. Console summary
    # -------------------------------------------------------------
    print("\n" + "=" * 86)
    print("FMB FEATURE DIAGNOSTIC — SUMMARY BY METAL COUNT")
    print("=" * 86)
    print(
        f"{'Count':>5} {'n':>5} "
        f"{'GM/AM':>11} {'Suppress':>11} "
        f"{'Cosine':>11} {'Dominance':>11} {'PSNR-NM':>11}"
    )

    for r in sorted(
        count_summary,
        key=lambda x: int(x["n_materials"]),
    ):
        print(
            f"{int(r['n_materials']):>5d} "
            f"{int(r['n']):>5d} "
            f"{r['gm_am_agreement_mean']:>11.6f} "
            f"{r['suppression_index_mean']:>11.6f} "
            f"{r['cosine_mean_mean']:>11.6f} "
            f"{r['branch_dominance_mean']:>11.6f} "
            f"{r['psnr_nonmetal_mean']:>11.3f}"
        )

    print("\nKey correlations:")
    for r in correlations:
        if (
            (r["x"], r["y"])
            in {
                ("n_materials", "suppression_index"),
                ("n_materials", "gm_am_agreement"),
                ("n_materials", "cosine_mean"),
                ("suppression_index", "psnr_nonmetal"),
            }
        ):
            print(
                f"  {r['x']} vs {r['y']}: "
                f"Pearson r={r['pearson_r']:.4f}, "
                f"Spearman rho={r['spearman_rho']:.4f}"
            )

    print("\nSaved:")
    for path in [
        layer_csv,
        case_csv,
        count_csv,
        single_multi_csv,
        size_csv,
        count_layer_csv,
        corr_csv,
        summary_txt,
    ]:
        print(f"  {path}")

    if args.save_plots:
        print("  diagnostic PNG plots")

    print(
        "\nRead hypothesis_summary.txt AND "
        "summary_by_metal_count_and_layer.csv before deciding whether "
        "Residual Fusion is worth a training run."
    )


if __name__ == "__main__":
    main()

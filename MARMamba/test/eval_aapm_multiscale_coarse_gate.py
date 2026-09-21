"""
Full AAPM evaluator for dynamic-gated Multi-Coarse.

Reports:
- PSNR / SSIM / RMSE / LPIPS
- non-metal and metal-included
- size groups
- metal-count groups
- single / multi
- overall
- gate statistics overall / by size / by metal count / single-multi
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.mamba_multiscale_coarse_gate import (
    GatedMultiScaleCoarseRefinedMambaFormer,
)
from utils.metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_rmse,
)


HU_MIN, HU_MAX = -1000.0, 3000.0
FNAME_ID_RE = re.compile(r"img(\d+)")
DIMS_RE = re.compile(r"(\d+)x(\d+)x(\d+)")
METRICS = ("psnr", "ssim", "rmse", "lpips")
REGIONS = ("non_metal", "metal_included")
SIZE_GROUPS = ("Large", "Medium", "Small", "Tiny")


def parse_dims(filename):
    m = DIMS_RE.search(filename)
    if not m:
        raise ValueError(f"Cannot parse dimensions from: {filename}")
    w, h, d = [int(x) for x in m.groups()]
    return h, w, d


def load_raw(path, dtype=np.float32):
    rows, cols, slices = parse_dims(os.path.basename(path))
    arr = np.fromfile(path, dtype=dtype)
    expected = rows * cols * slices

    if arr.size != expected:
        raise ValueError(
            f"{path}: found {arr.size}, expected {expected}"
        )

    if slices == 1:
        return arr.reshape(rows, cols)

    return arr.reshape(slices, rows, cols)


def hu_to_unit(img):
    img = np.clip(img, HU_MIN, HU_MAX)
    return ((img - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def find_test_samples(test_data_dir):
    def index_one(anatomy_dir):
        anatomy = os.path.basename(os.path.normpath(anatomy_dir))
        bdir = os.path.join(anatomy_dir, "Baseline")
        tdir = os.path.join(anatomy_dir, "Target")
        mdir = os.path.join(anatomy_dir, "Mask")

        if not os.path.isdir(bdir) or not os.path.isdir(tdir):
            return []

        target_map = {}
        for f in glob.glob(os.path.join(tdir, "*.raw")):
            m = FNAME_ID_RE.search(os.path.basename(f))
            if m:
                target_map[m.group(1)] = f

        mask_map = {}
        count_map = {}

        if os.path.isdir(mdir):
            for f in glob.glob(os.path.join(mdir, "*.raw")):
                m = FNAME_ID_RE.search(os.path.basename(f))
                if m:
                    mask_map[m.group(1)] = f

            for f in glob.glob(os.path.join(mdir, "*.json")):
                m = re.search(r"metalinfo(\d+)", os.path.basename(f))
                if not m:
                    continue
                try:
                    with open(f, "r", encoding="utf-8") as jf:
                        count_map[m.group(1)] = json.load(jf).get(
                            "n_materials"
                        )
                except Exception:
                    pass

        samples = []
        for bf in glob.glob(os.path.join(bdir, "*.raw")):
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
                    "n_materials": count_map.get(image_id),
                }
            )

        return samples

    if os.path.isdir(os.path.join(test_data_dir, "Baseline")):
        return index_one(test_data_dir)

    samples = []
    for name in sorted(os.listdir(test_data_dir)):
        sub = os.path.join(test_data_dir, name)
        if (
            os.path.isdir(sub)
            and os.path.isdir(os.path.join(sub, "Baseline"))
        ):
            samples.extend(index_one(sub))

    return samples


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


def to_lpips_tensor(img_bgr3_uint8, device):
    img_rgb = cv2.cvtColor(img_bgr3_uint8, cv2.COLOR_BGR2RGB)
    img_np = img_rgb.astype("float32") / 127.5 - 1.0

    return (
        torch.from_numpy(img_np)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )


def evaluate_image(
    pred,
    gt,
    mask,
    lpips_fn,
    device,
    exclude_metal,
):
    pred_eval = pred.copy()
    gt_eval = gt.copy()

    if exclude_metal and mask is not None:
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
        "lpips": float(
            lpips_fn(
                to_lpips_tensor(pred_bgr, device),
                to_lpips_tensor(gt_bgr, device),
            ).item()
        ),
    }


def load_model(
    checkpoint,
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

    state = torch.load(checkpoint, map_location=device)

    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    if any(k.startswith("module.") for k in state):
        state = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state.items()
        }

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def summarize(entries, region_key, model_key):
    if not entries:
        return None

    out = {"n": len(entries)}

    for metric in METRICS:
        arr = np.asarray(
            [
                e["metrics"][region_key][model_key][metric]
                for e in entries
            ],
            dtype=np.float64,
        )
        out[metric] = float(arr.mean())
        out[f"{metric}_std"] = float(arr.std())

    return out


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


def summarize_gate(entries):
    if not entries:
        return None

    out = {"n": len(entries)}

    for key in ("alpha2", "alpha4"):
        arr = np.asarray(
            [e[key] for e in entries],
            dtype=np.float64,
        )

        out[f"{key}_mean"] = float(arr.mean())
        out[f"{key}_std"] = float(arr.std())
        out[f"{key}_min"] = float(arr.min())
        out[f"{key}_max"] = float(arr.max())

    return out


def print_group(label, entries, region_key):
    if not entries:
        return

    pair = summarize_pair(entries, region_key)
    b = pair["baseline"]
    r = pair["refined"]
    d = pair["delta"]

    print(
        f"{label:>10} | n={r['n']:4d} | "
        f"PSNR {b['psnr']:.4f}->{r['psnr']:.4f} "
        f"({d['psnr']:+.4f}) | "
        f"SSIM {b['ssim']:.6f}->{r['ssim']:.6f} "
        f"({d['ssim']:+.6f}) | "
        f"RMSE {b['rmse']:.4f}->{r['rmse']:.4f} "
        f"({d['rmse']:+.4f}) | "
        f"LPIPS {b['lpips']:.5f}->{r['lpips']:.5f} "
        f"({d['lpips']:+.5f})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output_dir",
        default="./eval_multiscale_coarse_gate",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden_channels", default=32, type=int)
    parser.add_argument("--gate_hidden", default=32, type=int)
    parser.add_argument("--gate_range", default=0.5, type=float)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    samples = find_test_samples(args.test_data_dir)
    if not samples:
        raise RuntimeError("No AAPM test samples found.")

    count_hist = {n: 0 for n in range(1, 6)}
    missing_masks = []

    for sample in samples:
        if sample["n_materials"] is None:
            raise RuntimeError(
                f"Missing n_materials for {sample['id']}"
            )

        n_materials = int(sample["n_materials"])

        if n_materials not in count_hist:
            raise RuntimeError(
                f"Unexpected n_materials={n_materials} "
                f"for {sample['id']}"
            )

        count_hist[n_materials] += 1

        if sample["mask"] is None:
            sample["mask_size"] = 0
            missing_masks.append(sample["id"])
        else:
            mask = (
                load_raw(
                    sample["mask"],
                    dtype=np.float32,
                ) > 0.5
            )
            sample["mask_size"] = int(mask.sum())

    sizes = np.asarray(
        [sample["mask_size"] for sample in samples],
        dtype=np.float64,
    )
    q25, q50, q75 = np.percentile(sizes, [25, 50, 75])

    for sample in samples:
        sample["size_group"] = size_to_group(
            sample["mask_size"],
            q25,
            q50,
            q75,
        )

    print(f"Found {len(samples)} test cases.")
    print(
        f"Mask quartiles: q25={q25:.0f}, "
        f"q50={q50:.0f}, q75={q75:.0f}"
    )
    print(f"Metal-count histogram: {count_hist}")

    if missing_masks:
        print(
            f"WARNING: {len(missing_masks)} samples have no mask."
        )

    model = load_model(
        args.checkpoint,
        device,
        args.hidden_channels,
        args.gate_hidden,
        args.gate_range,
    )

    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()

    entries = []

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
                    ) > 0.5
                )

            input_t = (
                torch.from_numpy((input_img - 0.5) / 0.5)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
                .to(device)
            )

            refined_t, baseline_t, _, stats = model(
                input_t,
                return_refinement=True,
            )

            refined = np.clip(
                refined_t.squeeze().cpu().numpy(),
                0.0,
                1.0,
            )
            baseline = np.clip(
                baseline_t.squeeze().cpu().numpy(),
                0.0,
                1.0,
            )

            metrics = {}

            for exclude_metal, region_key in (
                (True, "non_metal"),
                (False, "metal_included"),
            ):
                metrics[region_key] = {
                    "baseline": evaluate_image(
                        baseline,
                        gt_img,
                        mask,
                        lpips_fn,
                        device,
                        exclude_metal,
                    ),
                    "refined": evaluate_image(
                        refined,
                        gt_img,
                        mask,
                        lpips_fn,
                        device,
                        exclude_metal,
                    ),
                }

            n_materials = int(sample["n_materials"])

            entries.append(
                {
                    "id": sample["id"],
                    "n_materials": n_materials,
                    "size_group": sample["size_group"],
                    "multiplicity": (
                        "multi" if n_materials >= 2 else "single"
                    ),
                    "metrics": metrics,
                    "alpha2": float(
                        stats["alpha2_mean"].item()
                    ),
                    "alpha4": float(
                        stats["alpha4_mean"].item()
                    ),
                    "delta_abs_mean": float(
                        stats["delta_abs_mean"].item()
                    ),
                    "delta_rms": float(
                        stats["delta_rms"].item()
                    ),
                    "detail2_abs_mean": float(
                        stats["detail2_abs_mean"].item()
                    ),
                    "coarse4_abs_mean": float(
                        stats["coarse4_abs_mean"].item()
                    ),
                }
            )

            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"Processed {idx + 1}/{len(samples)}")

    print("\n=== NON-METAL BY SIZE ===")
    for group in SIZE_GROUPS:
        group_entries = [
            e for e in entries if e["size_group"] == group
        ]
        print_group(group, group_entries, "non_metal")

    print("\n=== NON-METAL BY METAL COUNT ===")
    for n in range(1, 6):
        group_entries = [
            e for e in entries if e["n_materials"] == n
        ]
        print_group(f"{n} metal", group_entries, "non_metal")

    print("\n=== NON-METAL SINGLE / MULTI / OVERALL ===")
    for group in ("single", "multi"):
        group_entries = [
            e for e in entries
            if e["multiplicity"] == group
        ]
        print_group(group, group_entries, "non_metal")

    print_group("overall", entries, "non_metal")

    print("\n=== GATE BY SIZE ===")
    for group in SIZE_GROUPS:
        group_entries = [
            e for e in entries if e["size_group"] == group
        ]
        gate = summarize_gate(group_entries)
        print(
            f"{group:>10} | "
            f"a2={gate['alpha2_mean']:.4f}±{gate['alpha2_std']:.4f} | "
            f"a4={gate['alpha4_mean']:.4f}±{gate['alpha4_std']:.4f}"
        )

    print("\n=== GATE BY METAL COUNT ===")
    for n in range(1, 6):
        group_entries = [
            e for e in entries if e["n_materials"] == n
        ]
        gate = summarize_gate(group_entries)
        print(
            f"{n:>10} | "
            f"a2={gate['alpha2_mean']:.4f}±{gate['alpha2_std']:.4f} | "
            f"a4={gate['alpha4_mean']:.4f}±{gate['alpha4_std']:.4f}"
        )

    summary = {
        "metadata": {
            "n_samples": len(entries),
            "metal_count_histogram": {
                str(k): int(v)
                for k, v in count_hist.items()
            },
            "mask_size_quartiles": {
                "q25": float(q25),
                "q50": float(q50),
                "q75": float(q75),
            },
            "regions": {
                "non_metal": (
                    "Prediction and GT are both zeroed inside the "
                    "metal mask before metrics."
                ),
                "metal_included": (
                    "Metrics are computed on the full image without "
                    "masking metal pixels."
                ),
            },
        },
        "by_size": {},
        "by_metal_count": {},
        "by_multiplicity": {},
        "overall": {},
        "gate_stats": {
            "overall": summarize_gate(entries),
            "by_size": {},
            "by_metal_count": {},
            "by_multiplicity": {},
        },
        "refinement_stats": {
            "delta_abs_mean_overall": float(
                np.mean([e["delta_abs_mean"] for e in entries])
            ),
            "delta_rms_overall": float(
                np.mean([e["delta_rms"] for e in entries])
            ),
            "detail2_abs_mean_overall": float(
                np.mean([e["detail2_abs_mean"] for e in entries])
            ),
            "coarse4_abs_mean_overall": float(
                np.mean([e["coarse4_abs_mean"] for e in entries])
            ),
        },
    }

    for group in SIZE_GROUPS:
        group_entries = [
            e for e in entries if e["size_group"] == group
        ]

        summary["by_size"][group] = {
            region: summarize_pair(group_entries, region)
            for region in REGIONS
        }
        summary["gate_stats"]["by_size"][group] = (
            summarize_gate(group_entries)
        )

    for n in range(1, 6):
        group_entries = [
            e for e in entries if e["n_materials"] == n
        ]

        summary["by_metal_count"][str(n)] = {
            region: summarize_pair(group_entries, region)
            for region in REGIONS
        }
        summary["gate_stats"]["by_metal_count"][str(n)] = (
            summarize_gate(group_entries)
        )

    for group in ("single", "multi"):
        group_entries = [
            e for e in entries
            if e["multiplicity"] == group
        ]

        summary["by_multiplicity"][group] = {
            region: summarize_pair(group_entries, region)
            for region in REGIONS
        }
        summary["gate_stats"]["by_multiplicity"][group] = (
            summarize_gate(group_entries)
        )

    summary["overall"] = {
        region: summarize_pair(entries, region)
        for region in REGIONS
    }

    out_path = os.path.join(
        args.output_dir,
        "eval_summary.json",
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

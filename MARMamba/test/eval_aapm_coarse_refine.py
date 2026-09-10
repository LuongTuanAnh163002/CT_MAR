"""
AAPM evaluator for CoarseRefinedMambaFormer.

Protocol:
- same AAPM raw HU -> [0,1] conversion;
- model input normalized to [-1,1];
- evaluates BOTH:
    * non_metal: zero BOTH prediction and GT inside metal mask;
    * metal_included: evaluate the full image, including metal region;
- PSNR / SSIM / RMSE on uint8 BGR;
- LPIPS VGG;
- reports:
    * mask-size groups: Large / Medium / Small / Tiny;
    * exact metal count: 1 / 2 / 3 / 4 / 5;
    * single / multi;
    * overall;
    * baseline prediction and refined prediction;
    * explicit refined - baseline metric deltas.

This keeps the coarse-refinement evaluator aligned with the baseline AAPM evaluator.
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

from model.mamba_coarse_refine import CoarseRefinedMambaFormer
from utils.metrics import calculate_psnr, calculate_ssim, calculate_rmse


HU_MIN, HU_MAX = -1000.0, 3000.0
FNAME_ID_RE = re.compile(r"img(\d+)")
DIMS_RE = re.compile(r"(\d+)x(\d+)x(\d+)")
REGION_KEYS = ("non_metal", "metal_included")
METRICS = ("psnr", "ssim", "rmse", "lpips")
SIZE_GROUPS = ("Large", "Medium", "Small", "Tiny")


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


def find_test_samples(test_data_dir):
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


def compute_quartiles(sizes):
    q25, q50, q75 = np.percentile(
        np.asarray(sizes, dtype=np.float64),
        [25, 50, 75],
    )
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
    img_u8 = np.clip(
        img_float01 * 255.0,
        0,
        255,
    ).astype(np.uint8)

    return cv2.cvtColor(
        img_u8,
        cv2.COLOR_GRAY2BGR,
    )


def to_lpips_tensor(img_bgr3_uint8, device):
    img_rgb = cv2.cvtColor(
        img_bgr3_uint8,
        cv2.COLOR_BGR2RGB,
    )

    img_np = (
        img_rgb.astype("float32")
        / 127.5
        - 1.0
    )

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
):
    model = CoarseRefinedMambaFormer(
        in_channels=1,
        hidden_channels=hidden_channels,
    ).to(device)

    state = torch.load(
        checkpoint,
        map_location=device,
    )

    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    if any(k.startswith("module.") for k in state):
        state = {
            (
                k[len("module."):]
                if k.startswith("module.")
                else k
            ): v
            for k, v in state.items()
        }

    model.load_state_dict(
        state,
        strict=True,
    )
    model.eval()

    return model


def summarize(entries, region_key, model_key):
    if not entries:
        return None

    out = {"n": len(entries)}

    for metric in METRICS:
        values = np.asarray(
            [
                e["metrics"][region_key][model_key][metric]
                for e in entries
            ],
            dtype=np.float64,
        )

        out[metric] = float(values.mean())
        out[f"{metric}_std"] = float(values.std())

    return out


def metric_delta(baseline_summary, refined_summary):
    if baseline_summary is None or refined_summary is None:
        return None

    return {
        metric: float(
            refined_summary[metric]
            - baseline_summary[metric]
        )
        for metric in METRICS
    }


def summarize_pair(entries, region_key):
    if not entries:
        return None

    baseline = summarize(
        entries,
        region_key,
        "baseline",
    )
    refined = summarize(
        entries,
        region_key,
        "refined",
    )

    return {
        "baseline": baseline,
        "refined": refined,
        "delta": metric_delta(
            baseline,
            refined,
        ),
    }


def print_group(label, entries, region_key):
    if not entries:
        print(f"{label:>10} | no samples")
        return

    pair = summarize_pair(entries, region_key)
    b = pair["baseline"]
    r = pair["refined"]
    d = pair["delta"]

    print(
        f"{label:>10} | n={r['n']:4d} | "
        f"PSNR {b['psnr']:.3f} -> {r['psnr']:.3f} "
        f"({d['psnr']:+.3f}) | "
        f"SSIM {b['ssim']:.4f} -> {r['ssim']:.4f} "
        f"({d['ssim']:+.4f}) | "
        f"RMSE {b['rmse']:.3f} -> {r['rmse']:.3f} "
        f"({d['rmse']:+.3f}) | "
        f"LPIPS {b['lpips']:.4f} -> {r['lpips']:.4f} "
        f"({d['lpips']:+.4f})"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_data_dir",
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        default="./eval_coarse_refine",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
    )
    parser.add_argument(
        "--hidden_channels",
        default=16,
        type=int,
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    device = torch.device(
        args.device
    )

    samples = find_test_samples(
        args.test_data_dir
    )

    if not samples:
        raise RuntimeError(
            "No AAPM test samples found."
        )

    print(
        f"Found {len(samples)} test cases."
    )

    # ------------------------------------------------------------------
    # Metadata + mask-size grouping
    # ------------------------------------------------------------------
    count_hist = {
        n: 0
        for n in range(1, 6)
    }

    missing_masks = []

    print(
        "Computing metal-mask area and reading n_materials..."
    )

    for s in samples:
        if s["n_materials"] is None:
            raise RuntimeError(
                f"Missing n_materials for {s['id']}"
            )

        n = int(s["n_materials"])

        if n not in count_hist:
            raise RuntimeError(
                f"Unexpected n_materials={n} for {s['id']}; "
                "expected one of 1..5."
            )

        count_hist[n] += 1

        if s["mask"] is None:
            s["mask_size"] = 0
            missing_masks.append(s["id"])
        else:
            mask = (
                load_raw(
                    s["mask"],
                    dtype=np.float32,
                )
                > 0.5
            )
            s["mask_size"] = int(mask.sum())

    if missing_masks:
        print(
            f"WARNING: {len(missing_masks)} samples have no mask. "
            "Their mask_size is set to 0 and non-metal evaluation "
            "cannot exclude metal pixels for those samples."
        )

    sizes = [
        s["mask_size"]
        for s in samples
    ]

    q25, q50, q75 = compute_quartiles(
        sizes
    )

    print(
        "Mask-area quartiles: "
        f"q25={q25:.0f}, "
        f"q50={q50:.0f}, "
        f"q75={q75:.0f}"
    )

    for s in samples:
        s["size_group"] = size_to_group(
            s["mask_size"],
            q25,
            q50,
            q75,
        )

    print(
        f"Metal-count histogram: {count_hist}"
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = load_model(
        args.checkpoint,
        device,
        args.hidden_channels,
    )

    lpips_fn = lpips.LPIPS(
        net="vgg",
    ).to(device).eval()

    entries = []

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    with torch.inference_mode():
        for idx, sample in enumerate(samples):
            input_img = hu_to_unit(
                load_raw(
                    sample["baseline"]
                )
            )

            gt_img = hu_to_unit(
                load_raw(
                    sample["target"]
                )
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
                    (input_img - 0.5)
                    / 0.5
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
                .to(device)
            )

            refined_t, base_t, delta_t, stats = model(
                input_t,
                return_refinement=True,
            )

            refined = np.clip(
                refined_t.squeeze()
                .cpu()
                .numpy(),
                0.0,
                1.0,
            )

            baseline = np.clip(
                base_t.squeeze()
                .cpu()
                .numpy(),
                0.0,
                1.0,
            )

            region_metrics = {}

            for exclude_metal, region_key in (
                (True, "non_metal"),
                (False, "metal_included"),
            ):
                base_metrics = evaluate_image(
                    baseline,
                    gt_img,
                    mask,
                    lpips_fn,
                    device,
                    exclude_metal=exclude_metal,
                )

                refined_metrics = evaluate_image(
                    refined,
                    gt_img,
                    mask,
                    lpips_fn,
                    device,
                    exclude_metal=exclude_metal,
                )

                region_metrics[region_key] = {
                    "baseline": base_metrics,
                    "refined": refined_metrics,
                }

            n_materials = int(
                sample["n_materials"]
            )

            entries.append(
                {
                    "id": sample["id"],
                    "n_materials": n_materials,
                    "multiplicity": (
                        "multi"
                        if n_materials >= 2
                        else "single"
                    ),
                    "mask_size": int(
                        sample["mask_size"]
                    ),
                    "size_group": sample[
                        "size_group"
                    ],
                    "metrics": region_metrics,
                    "delta_abs_mean": float(
                        stats[
                            "delta_abs_mean"
                        ].item()
                    ),
                    "delta_rms": float(
                        stats[
                            "delta_rms"
                        ].item()
                    ),
                }
            )

            if (
                (idx + 1) % 50 == 0
                or idx == 0
            ):
                print(
                    f"Processed "
                    f"{idx+1}/{len(samples)}"
                )

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------
    for region_key, region_label in (
        (
            "non_metal",
            "NON-METAL (metal pixels excluded)",
        ),
        (
            "metal_included",
            "METAL-INCLUDED (full image)",
        ),
    ):
        print(
            f"\n=== {region_label}: BY SIZE ==="
        )

        for size_group in SIZE_GROUPS:
            group = [
                e
                for e in entries
                if e["size_group"] == size_group
            ]
            print_group(
                size_group,
                group,
                region_key,
            )

        print(
            f"\n=== {region_label}: METAL COUNT 1..5 ==="
        )

        for n in range(1, 6):
            group = [
                e
                for e in entries
                if e["n_materials"] == n
            ]
            print_group(
                f"{n} metal",
                group,
                region_key,
            )

        print(
            f"\n=== {region_label}: SINGLE / MULTI / OVERALL ==="
        )

        single = [
            e
            for e in entries
            if e["multiplicity"] == "single"
        ]
        multi = [
            e
            for e in entries
            if e["multiplicity"] == "multi"
        ]

        print_group(
            "single",
            single,
            region_key,
        )
        print_group(
            "multi",
            multi,
            region_key,
        )
        print_group(
            "overall",
            entries,
            region_key,
        )

    # ------------------------------------------------------------------
    # JSON summary
    # ------------------------------------------------------------------
    summary = {
        "metadata": {
            "n_samples": len(entries),
            "metal_count_histogram": {
                str(k): int(v)
                for k, v in count_hist.items()
            },
            "mask_size_quartiles": {
                "q25": q25,
                "q50": q50,
                "q75": q75,
            },
            "regions": {
                "non_metal": (
                    "Prediction and GT are both zeroed "
                    "inside the metal mask before metrics."
                ),
                "metal_included": (
                    "Metrics are computed on the full image "
                    "without masking metal pixels."
                ),
            },
        },
        "by_size": {},
        "by_metal_count": {},
        "by_multiplicity": {},
        "overall": {},
        "refinement_stats": {},
    }

    for size_group in SIZE_GROUPS:
        group = [
            e
            for e in entries
            if e["size_group"] == size_group
        ]

        summary["by_size"][size_group] = {
            region_key: summarize_pair(
                group,
                region_key,
            )
            for region_key in REGION_KEYS
        }

    for n in range(1, 6):
        group = [
            e
            for e in entries
            if e["n_materials"] == n
        ]

        summary["by_metal_count"][str(n)] = {
            region_key: summarize_pair(
                group,
                region_key,
            )
            for region_key in REGION_KEYS
        }

    single = [
        e
        for e in entries
        if e["multiplicity"] == "single"
    ]
    multi = [
        e
        for e in entries
        if e["multiplicity"] == "multi"
    ]

    summary["by_multiplicity"]["single"] = {
        region_key: summarize_pair(
            single,
            region_key,
        )
        for region_key in REGION_KEYS
    }

    summary["by_multiplicity"]["multi"] = {
        region_key: summarize_pair(
            multi,
            region_key,
        )
        for region_key in REGION_KEYS
    }

    summary["overall"] = {
        region_key: summarize_pair(
            entries,
            region_key,
        )
        for region_key in REGION_KEYS
    }

    summary["refinement_stats"] = {
        "delta_abs_mean_overall": float(
            np.mean(
                [
                    e["delta_abs_mean"]
                    for e in entries
                ]
            )
        ),
        "delta_rms_overall": float(
            np.mean(
                [
                    e["delta_rms"]
                    for e in entries
                ]
            )
        ),
    }

    out_path = os.path.join(
        args.output_dir,
        "eval_summary.json",
    )

    with open(
        out_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nSaved: {out_path}"
    )


if __name__ == "__main__":
    main()

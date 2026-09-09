"""
AAPM evaluation for LowFrequencyMambaFormer.

Protocol is kept aligned with the existing AAPM evaluator:
- true metal-mask quartiles;
- non-metal evaluation zeros BOTH prediction and GT inside mask;
- metal-included evaluation unchanged;
- PSNR/SSIM/RMSE on uint8 BGR;
- LPIPS(VGG);
- Single vs Multi groups.

Additionally reports n_materials = 1..5 because this experiment specifically
targets degradation with metal complexity.
"""

import argparse
import glob
import json
import os
import re
import sys

import cv2
import lpips
import numpy as np
import torch

sys.path.append("../")
sys.path.insert(0, ".")

from model.mamba_lowfreq import LowFrequencyMambaFormer
from utils.metrics import calculate_psnr, calculate_ssim, calculate_rmse

HU_MIN, HU_MAX = -1000.0, 3000.0
FNAME_ID_RE = re.compile(r"img(\d+)")
DIMS_RE = re.compile(r"(\d+)x(\d+)x(\d+)")


def parse_dims(filename):
    m = DIMS_RE.search(filename)
    if not m:
        raise ValueError(f"Không đọc được kích thước từ: {filename}")
    w, h, d = (int(x) for x in m.groups())
    return h, w, d


def load_raw(path, dtype=np.float32):
    rows, cols, slices = parse_dims(os.path.basename(path))
    arr = np.fromfile(path, dtype=dtype)
    expected = rows * cols * slices
    if arr.size != expected:
        raise ValueError(f"{path}: có {arr.size}, cần {expected}")
    return arr.reshape(rows, cols) if slices == 1 else arr.reshape(slices, rows, cols)


def hu_to_unit(img):
    img = np.clip(img, HU_MIN, HU_MAX)
    return ((img - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def find_test_samples(test_data_dir):
    def _index_one(anatomy_dir):
        prefix = os.path.basename(os.path.normpath(anatomy_dir))
        baseline_dir = os.path.join(anatomy_dir, "Baseline")
        target_dir = os.path.join(anatomy_dir, "Target")
        mask_dir = os.path.join(anatomy_dir, "Mask")

        if not (
            os.path.isdir(baseline_dir)
            and os.path.isdir(target_dir)
        ):
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
                                "n_materials"
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

            image_id = m.group(1)
            if image_id not in target_map:
                continue

            samples.append({
                "id": f"{prefix}_{image_id}",
                "baseline": bf,
                "target": target_map[image_id],
                "mask": mask_map.get(image_id),
                "n_materials": n_materials_map.get(image_id),
            })

        return samples

    if os.path.isdir(os.path.join(test_data_dir, "Baseline")):
        return _index_one(test_data_dir)

    samples = []
    for name in sorted(os.listdir(test_data_dir)):
        sub = os.path.join(test_data_dir, name)
        if os.path.isdir(sub) and os.path.isdir(os.path.join(sub, "Baseline")):
            samples.extend(_index_one(sub))

    return samples


def compute_quartiles(sizes):
    return np.percentile(np.asarray(sizes), [25, 50, 75])


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


def load_model(checkpoint_path, device):
    net = LowFrequencyMambaFormer(in_channels=1).to(device)

    state = torch.load(checkpoint_path, map_location=device)

    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    if any(k.startswith("module.") for k in state.keys()):
        state = {
            k[len("module."):]: v
            for k, v in state.items()
        }

    net.load_state_dict(state, strict=True)
    net.eval()
    return net


def summarize(entries):
    if not entries:
        return None

    arr = {
        k: np.asarray([e[k] for e in entries])
        for k in ("psnr", "ssim", "rmse", "lpips")
    }

    return {
        "n": len(entries),
        "psnr": f"{arr['psnr'].mean():.2f} ± {arr['psnr'].std():.2f}",
        "ssim": f"{arr['ssim'].mean():.4f} ± {arr['ssim'].std():.4f}",
        "rmse": f"{arr['rmse'].mean():.2f} ± {arr['rmse'].std():.2f}",
        "lpips": f"{arr['lpips'].mean():.4f} ± {arr['lpips'].std():.4f}",
    }


def run_eval(test_data_dir, checkpoint, output_dir, device="cuda:0"):
    os.makedirs(output_dir, exist_ok=True)

    samples = find_test_samples(test_data_dir)
    if not samples:
        raise RuntimeError(
            f"Không tìm được sample nào trong {test_data_dir}"
        )

    print(f"Tổng số sample test: {len(samples)}")

    for sample in samples:
        if sample["mask"] is None:
            sample["mask_size"] = 0
        else:
            mask = load_raw(
                sample["mask"],
                dtype=np.float32,
            )
            sample["mask_size"] = int((mask > 0.5).sum())

    q25, q50, q75 = compute_quartiles(
        [s["mask_size"] for s in samples]
    )

    print(
        f"Ngưỡng quartile: "
        f"q25={q25:.0f}, q50={q50:.0f}, q75={q75:.0f}"
    )

    for s in samples:
        s["group"] = size_to_group(
            s["mask_size"], q25, q50, q75
        )
        s["is_multi"] = (
            s["n_materials"] is not None
            and int(s["n_materials"]) >= 2
        )

    net = load_model(checkpoint, device)
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()

    by_size = {
        g: {"non_metal": [], "metal_included": []}
        for g in ("Large", "Medium", "Small", "Tiny")
    }

    by_multiplicity = {
        g: {"non_metal": [], "metal_included": []}
        for g in ("single", "multi")
    }

    by_count = {
        i: {"non_metal": [], "metal_included": []}
        for i in range(1, 6)
    }

    with torch.no_grad():
        for idx, s in enumerate(samples):
            input_img = hu_to_unit(load_raw(s["baseline"]))
            gt_img = hu_to_unit(load_raw(s["target"]))

            mask = None
            if s["mask"] is not None:
                mask = (
                    load_raw(
                        s["mask"],
                        dtype=np.float32,
                    )
                    > 0.5
                )

            input_t = (
                torch.from_numpy(
                    (input_img - 0.5) / 0.5
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
                .to(device)
            )

            pred = (
                net(input_t)
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )
            pred = np.clip(pred, 0.0, 1.0)

            for exclude_metal, key in [
                (True, "non_metal"),
                (False, "metal_included"),
            ]:
                pred_eval = pred.copy()
                gt_eval = gt_img.copy()

                if exclude_metal and mask is not None:
                    pred_eval[mask] = 0.0
                    gt_eval[mask] = 0.0

                pred_bgr = to_uint8_bgr3(pred_eval)
                gt_bgr = to_uint8_bgr3(gt_eval)

                entry = {
                    "id": s["id"],
                    "psnr": calculate_psnr(
                        pred_bgr,
                        gt_bgr,
                        test_y_channel=True,
                    ),
                    "ssim": calculate_ssim(
                        pred_bgr,
                        gt_bgr,
                        test_y_channel=True,
                    ),
                    "rmse": calculate_rmse(
                        pred_bgr,
                        gt_bgr,
                    ),
                    "lpips": lpips_fn(
                        to_lpips_tensor(pred_bgr, device),
                        to_lpips_tensor(gt_bgr, device),
                    ).item(),
                }

                by_size[s["group"]][key].append(entry)

                group = "multi" if s["is_multi"] else "single"
                by_multiplicity[group][key].append(entry)

                if s["n_materials"] is not None:
                    n = int(s["n_materials"])
                    if n in by_count:
                        by_count[n][key].append(entry)

            if (idx + 1) % 50 == 0:
                print(f"Processed {idx+1}/{len(samples)}")

    summary = {
        "by_size": {
            group: {
                key: summarize(by_size[group][key])
                for key in ("non_metal", "metal_included")
            }
            for group in ("Large", "Medium", "Small", "Tiny")
        },
        "by_multiplicity": {
            group: {
                key: summarize(by_multiplicity[group][key])
                for key in ("non_metal", "metal_included")
            }
            for group in ("single", "multi")
        },
        "by_metal_count": {
            str(n): {
                key: summarize(by_count[n][key])
                for key in ("non_metal", "metal_included")
            }
            for n in range(1, 6)
        },
    }

    output_json = os.path.join(
        output_dir,
        "eval_summary.json",
    )

    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== NON-METAL: METAL COUNT 1..5 ===")
    for n in range(1, 6):
        r = summary["by_metal_count"][str(n)]["non_metal"]
        if r:
            print(
                f"{n} metal | n={r['n']} | "
                f"PSNR={r['psnr']} | "
                f"SSIM={r['ssim']} | "
                f"RMSE={r['rmse']} | "
                f"LPIPS={r['lpips']}"
            )

    print("\n=== NON-METAL: SINGLE / MULTI ===")
    for group in ("single", "multi"):
        r = summary["by_multiplicity"][group]["non_metal"]
        print(
            f"{group} | n={r['n']} | "
            f"PSNR={r['psnr']} | "
            f"SSIM={r['ssim']} | "
            f"RMSE={r['rmse']} | "
            f"LPIPS={r['lpips']}"
        )

    print(f"\nSaved: {output_json}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output_dir",
        default="./eval_lowfreq",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    run_eval(
        args.test_data_dir,
        args.checkpoint,
        args.output_dir,
        args.device,
    )

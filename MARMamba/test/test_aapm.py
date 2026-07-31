"""
eval_aapm.py
=============
Tương đương test/test_deeplesion.py (script eval gốc của MARMamba dùng để
tạo Table I-IV), viết lại cho dữ liệu AAPM CT-MAR thay vì SynDeepLesion.
"""

import os
import re
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import cv2
import lpips
from PIL import Image

import sys
sys.path.append('../')
sys.path.insert(0, ".")

from model.mamba import MambaFormer
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
        if not (os.path.isdir(baseline_dir) and os.path.isdir(target_dir)):
            return []
        target_map = {
            FNAME_ID_RE.search(os.path.basename(f)).group(1): f
            for f in glob.glob(os.path.join(target_dir, "*.raw"))
            if "img" in os.path.basename(f) and FNAME_ID_RE.search(os.path.basename(f))
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
                        with open(f) as jf:
                            n_materials_map[m.group(1)] = json.load(jf).get("n_materials", None)
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
            samples.append({
                "id": f"{prefix}_{img_id}",
                "baseline": bf,
                "target": target_map[img_id],
                "mask": mask_map.get(img_id),
                "n_materials": n_materials_map.get(img_id),
            })
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
    q25, q50, q75 = np.percentile(np.array(sizes), [25, 50, 75])
    return q25, q50, q75


def size_to_group(size, q25, q50, q75):
    if size >= q75:
        return "Large"
    elif size >= q50:
        return "Medium"
    elif size >= q25:
        return "Small"
    else:
        return "Tiny"


def to_uint8_bgr3(img_float01):
    img_u8 = np.clip(img_float01 * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)


def to_lpips_tensor(img_bgr3_uint8, device):
    img_rgb = cv2.cvtColor(img_bgr3_uint8, cv2.COLOR_BGR2RGB)
    img_np = img_rgb.astype("float32") / 127.5 - 1.0
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    return img_t.to(device)


def load_model(checkpoint_path, device):
    net = MambaFormer(in_channels=1).to(device)
    device_ids = [i for i in range(torch.cuda.device_count())] or [0]
    net = nn.DataParallel(net, device_ids=device_ids)
    state = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(state)
    net.eval()
    return net


def run_eval(test_data_dir, checkpoint, output_dir, device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    samples = find_test_samples(test_data_dir)
    if not samples:
        raise RuntimeError(f"Không tìm được sample nào trong {test_data_dir}")
    print(f"Tổng số sample test: {len(samples)}")

    print("Đang tính diện tích vùng kim loại (mask thật) cho từng sample...")
    for s in samples:
        if s["mask"] is None:
            s["mask_size"] = 0
            continue
        mask = load_raw(s["mask"], dtype=np.float32)
        s["mask_size"] = int((mask > 0.5).sum())

    sizes = [s["mask_size"] for s in samples]
    q25, q50, q75 = compute_quartiles(sizes)
    print(f"Ngưỡng quartile (pixel mask): q25={q25:.0f}, q50={q50:.0f}, q75={q75:.0f}")
    for s in samples:
        s["group"] = size_to_group(s["mask_size"], q25, q50, q75)
        s["is_multi"] = (s["n_materials"] is not None and s["n_materials"] >= 2)

    net = load_model(checkpoint, device)
    lpips_fn = lpips.LPIPS(net="vgg").to(device)

    results = {g: {"non_metal": [], "metal_included": []} for g in ["Large", "Medium", "Small", "Tiny"]}
    multi_results = {"single": {"non_metal": [], "metal_included": []},
                      "multi": {"non_metal": [], "metal_included": []}}

    with torch.no_grad():
        for idx, s in enumerate(samples):
            baseline_hu = load_raw(s["baseline"])
            target_hu = load_raw(s["target"])
            Xma = hu_to_unit(baseline_hu)
            Xgt = hu_to_unit(target_hu)

            mask = None
            if s["mask"] is not None:
                mask = load_raw(s["mask"], dtype=np.float32) > 0.5

            input_t = torch.from_numpy((Xma - 0.5) / 0.5).unsqueeze(0).unsqueeze(0).float().to(device)
            output = net(input_t)
            pred = output.squeeze().cpu().numpy()
            pred = np.clip(pred, 0.0, 1.0)

            for exclude_metal, key in [(True, "non_metal"), (False, "metal_included")]:
                pred_eval = pred.copy()
                gt_eval = Xgt.copy()
                if exclude_metal and mask is not None:
                    pred_eval[mask] = 0.0
                    gt_eval[mask] = 0.0

                res_bgr = to_uint8_bgr3(pred_eval)
                gt_bgr = to_uint8_bgr3(gt_eval)

                cur_psnr = calculate_psnr(res_bgr, gt_bgr, test_y_channel=True)
                cur_ssim = calculate_ssim(res_bgr, gt_bgr, test_y_channel=True)
                cur_rmse = calculate_rmse(res_bgr, gt_bgr)

                res_t = to_lpips_tensor(res_bgr, device)
                gt_t = to_lpips_tensor(gt_bgr, device)
                cur_lpips = lpips_fn(res_t, gt_t).item()

                entry = {"id": s["id"], "psnr": cur_psnr, "ssim": cur_ssim,
                         "rmse": cur_rmse, "lpips": cur_lpips}
                results[s["group"]][key].append(entry)
                multi_results["multi" if s["is_multi"] else "single"][key].append(entry)

            if (idx + 1) % 50 == 0:
                print(f"  đã xử lý {idx+1}/{len(samples)}")

    def summarize(entries):
        if not entries:
            return None
        arr = {k: np.array([e[k] for e in entries]) for k in ["psnr", "ssim", "rmse", "lpips"]}
        return {
            "n": len(entries),
            "psnr": f"{arr['psnr'].mean():.2f} ± {arr['psnr'].std():.2f}",
            "ssim": f"{arr['ssim'].mean():.4f} ± {arr['ssim'].std():.4f}",
            "rmse": f"{arr['rmse'].mean():.2f} ± {arr['rmse'].std():.2f}",
            "lpips": f"{arr['lpips'].mean():.4f} ± {arr['lpips'].std():.4f}",
        }

    summary = {"by_size": {}, "by_multiplicity": {}}
    for group in ["Large", "Medium", "Small", "Tiny"]:
        summary["by_size"][group] = {k: summarize(results[group][k]) for k in ["non_metal", "metal_included"]}
    for group in ["single", "multi"]:
        summary["by_multiplicity"][group] = {k: summarize(multi_results[group][k]) for k in ["non_metal", "metal_included"]}

    with open(os.path.join(output_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== BẢNG CHÍNH: theo kích thước (Large/Medium/Small/Tiny) ===")
    for group in ["Large", "Medium", "Small", "Tiny"]:
        print(f"\n-- {group} --")
        for key, label in [("non_metal", "Non-metallic area"), ("metal_included", "Metal-included region")]:
            r = summary["by_size"][group][key]
            if r is None:
                print(f"  {label}: (không có sample)")
                continue
            print(f"  {label} (n={r['n']}): PSNR={r['psnr']} SSIM={r['ssim']} RMSE={r['rmse']} LPIPS={r['lpips']}")

    print("\n=== BẢNG PHỤ: Single-artifact vs Multi-artifact ===")
    for group in ["single", "multi"]:
        print(f"\n-- {group} --")
        for key, label in [("non_metal", "Non-metallic area"), ("metal_included", "Metal-included region")]:
            r = summary["by_multiplicity"][group][key]
            if r is None:
                print(f"  {label}: (không có sample)")
                continue
            print(f"  {label} (n={r['n']}): PSNR={r['psnr']} SSIM={r['ssim']} RMSE={r['rmse']} LPIPS={r['lpips']}")

    print(f"\nĐã lưu chi tiết: {os.path.join(output_dir, 'eval_summary.json')}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="./eval_results")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_eval(args.test_data_dir, args.checkpoint, args.output_dir, device=args.device)
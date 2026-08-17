"""
analyze_psnr_vs_nmaterials.py
================================
Do PSNR THAT tren checkpoint baseline da co, theo TUNG gia tri n_materials
cu the (1,2,3,4,5+) -- de biet DUNG dang ham can dung cho trong so,
thay vi doan mo (tuyen tinh/mu/log).

CHAY TREN CHECKPOINT DA CO SAN, KHONG CAN TRAIN GI CA.

CACH DUNG:
    python analyze_psnr_vs_nmaterials.py \
        -checkpoint checkpoint/stage2_baseline/297000_ckpt \
        -test_data_dir /kaggle/working/data_CT_MAR_split/test
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

import sys
sys.path.append('.')
from model.mamba import MambaFormer
from utils.metrics import calculate_psnr
from utils.aapm_dataset import hu_to_unit, load_raw, _find_anatomy_folders

FNAME_ID_RE = re.compile(r"img(\d+)")


def load_model(checkpoint_path, device):
    net = MambaFormer(in_channels=1).to(device)
    net = nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())) or [0])
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net.eval()
    return net


def index_samples_with_nmaterials(test_data_dir):
    """Giong _index_one_anatomy trong aapm_dataset.py, nhung THEM doc n_materials tu JSON."""
    samples = []
    for anatomy in sorted(os.listdir(test_data_dir)):
        anatomy_dir = os.path.join(test_data_dir, anatomy)
        baseline_dir = os.path.join(anatomy_dir, "Baseline")
        target_dir = os.path.join(anatomy_dir, "Target")
        mask_dir = os.path.join(anatomy_dir, "Mask")
        if not os.path.isdir(baseline_dir):
            continue

        target_files = {}
        for f in glob.glob(os.path.join(target_dir, "*.raw")):
            m = FNAME_ID_RE.search(os.path.basename(f))
            if m:
                target_files[m.group(1)] = f

        for bf in glob.glob(os.path.join(baseline_dir, "*.raw")):
            m = FNAME_ID_RE.search(os.path.basename(bf))
            if not m or m.group(1) not in target_files:
                continue
            img_id = m.group(1)

            n_materials = None
            json_files = glob.glob(os.path.join(mask_dir, f"*metalinfo{img_id}.json"))
            if json_files:
                with open(json_files[0]) as jf:
                    n_materials = json.load(jf).get("n_materials")

            if n_materials is None:
                continue  # bo qua neu khong co JSON (chua bo sung)

            samples.append({
                "baseline": bf, "target": target_files[img_id], "n_materials": n_materials
            })
    return samples


def to_uint8_bgr3(img_float01):
    img_u8 = np.clip(img_float01 * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-checkpoint", required=True)
    parser.add_argument("-test_data_dir", required=True)
    parser.add_argument("-device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    net = load_model(args.checkpoint, device)
    samples = index_samples_with_nmaterials(args.test_data_dir)
    print(f"Tim thay {len(samples)} anh co du JSON n_materials")

    results_by_n = {}
    with torch.no_grad():
        for s in samples:
            baseline_hu = load_raw(s["baseline"])
            target_hu = load_raw(s["target"])
            Xma = hu_to_unit(baseline_hu)
            Xgt = hu_to_unit(target_hu)
            input_t = torch.from_numpy((Xma - 0.5) / 0.5).unsqueeze(0).unsqueeze(0).float().to(device)
            output = net(input_t)
            pred = np.clip(output.squeeze().cpu().numpy(), 0.0, 1.0)
            psnr = calculate_psnr(to_uint8_bgr3(pred), to_uint8_bgr3(Xgt), test_y_channel=True)

            n = s["n_materials"]
            results_by_n.setdefault(n, []).append(psnr)

    print(f"\n{'n_materials':<15}{'n_anh':<10}{'PSNR trung binh':<20}{'std':<10}")
    print("-" * 55)
    for n in sorted(results_by_n.keys()):
        vals = np.array(results_by_n[n])
        print(f"{n:<15}{len(vals):<10}{vals.mean():<20.3f}{vals.std():<10.3f}")

    print("\n=== Goi y doc ket qua ===")
    print("Neu PSNR giam DEU giua cac buoc (vd: 44->43->42->41...) -> ham TUYEN TINH hop ly")
    print("Neu giam NHANH luc dau roi CHAM dan -> ham LOG hop ly")
    print("Neu giam CHAM luc dau roi NHANH dan -> ham MU/BAC 2 hop ly")

"""
mamba_shedder_pruning.py
===========================
Structured pruning kieu Mamba-Shedder cho MARMamba.
Dung dung cau hinh THAT: depth=[1, 2, 2, 4, 1]
  -> down1_stage(1 block), down2_stage(2), down3_stage(2), down4_stage(4), refine_stage(1)
  -> Tong 10 block MambaBlock

Vi moi block trong 1 Stage nhan/tra ve CUNG SHAPE (thiet ke kieu ResNet),
xoa bot 1 block trong 1 Stage KHONG lam vo shape -- an toan ve kien truc,
khac han ExitHead (phai tu che decode head moi).

CHI CAN FORWARD PASS de do importance -- KHONG can backward/train,
nen chi phi GPU rat re (khac han train tu dau hay ExitHead).
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import cv2

import sys
sys.path.append('.')
from model.mamba import MambaFormer
from utils.metrics import calculate_psnr
from utils.aapm_dataset import hu_to_unit, load_raw, _find_anatomy_folders, _index_one_anatomy


DEPTH_CONFIG = [1, 2, 2, 4, 1]
STAGE_NAMES = ["down1_stage", "down2_stage", "down3_stage", "down4_stage", "refine_stage"]


def load_model(checkpoint_path, device):
    net = MambaFormer(in_channels=1).to(device)
    net = nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())) or [0])
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net.eval()
    return net


def to_uint8_bgr3(img_float01):
    img_u8 = np.clip(img_float01 * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)


def get_val_subset(val_data_dir, n_samples, seed=42):
    anatomy_dirs = _find_anatomy_folders(val_data_dir)
    all_samples = []
    for d in anatomy_dirs:
        all_samples.extend(_index_one_anatomy(d))
    rng = np.random.RandomState(seed)
    if n_samples < len(all_samples):
        idx = rng.choice(len(all_samples), size=n_samples, replace=False)
        return [all_samples[i] for i in idx]
    return all_samples


def eval_psnr_on_subset(net, samples, device):
    psnrs = []
    with torch.no_grad():
        for s in samples:
            baseline_hu = load_raw(s["baseline"])
            target_hu = load_raw(s["target"])
            Xma = hu_to_unit(baseline_hu)
            Xgt = hu_to_unit(target_hu)
            input_t = torch.from_numpy((Xma - 0.5) / 0.5).unsqueeze(0).unsqueeze(0).float().to(device)
            output = net(input_t)
            pred = np.clip(output.squeeze().cpu().numpy(), 0.0, 1.0)
            pred_bgr = to_uint8_bgr3(pred)
            gt_bgr = to_uint8_bgr3(Xgt)
            psnrs.append(calculate_psnr(pred_bgr, gt_bgr, test_y_channel=True))
    return np.mean(psnrs)


def get_all_removable_blocks(net):
    """
    Tra ve list (stage_name, block_index, n_blocks_in_stage) cho MOI block co the thu xoa.
    KHONG bao gom stage chi co 1 block (xoa het se lam Stage rong -> shape van OK vi Sequential
    rong tra ve nguyen input, nhung day la truong hop dac biet, tam thoi LOAI TRU de an toan).
    """
    removable = []
    for stage_name, n_blocks in zip(STAGE_NAMES, DEPTH_CONFIG):
        if n_blocks <= 1:
            continue  # khong xoa Stage chi co dung 1 block -- tranh lam rong hoan toan 1 tang
        for block_idx in range(n_blocks):
            removable.append((stage_name, block_idx, n_blocks))
    return removable


def remove_block_temporarily(net, stage_name, block_idx):
    """
    Tra ve (block_da_xoa, stage_module) de co the KHOI PHUC lai sau khi do xong.
    Xoa THAT SU khoi nn.ModuleList (khong phai mask/zero-out) -- dam bao do dung
    toc do that neu sau nay xoa vinh vien.
    """
    stage_module = getattr(net.module, stage_name)
    removed_block = stage_module.blocks[block_idx]
    # Tao ModuleList moi, KHONG co block bi xoa
    new_blocks = nn.ModuleList([b for i, b in enumerate(stage_module.blocks) if i != block_idx])
    stage_module.blocks = new_blocks
    return removed_block, stage_module


def restore_block(stage_module, removed_block, block_idx):
    """Khoi phuc lai block da xoa, dung vi tri cu."""
    blocks_list = list(stage_module.blocks)
    blocks_list.insert(block_idx, removed_block)
    stage_module.blocks = nn.ModuleList(blocks_list)


def compute_importance_scores(net, val_samples, device, verbose=True):
    """
    Voi MOI block con lai co the xoa, thu xoa TAM THOI, do PSNR, roi KHOI PHUC lai.
    Tra ve: dict {(stage_name, block_idx): importance_score}
    importance_score = baseline_psnr - psnr_khi_thieu_block_do
    (cang NHO = cang "vo hai" khi xoa = cang nen xoa truoc)
    """
    baseline_psnr = eval_psnr_on_subset(net, val_samples, device)
    if verbose:
        print(f"PSNR day du (chua xoa gi): {baseline_psnr:.3f}")

    scores = {}
    for stage_name, block_idx, n_blocks in get_all_removable_blocks(net):
        removed_block, stage_module = remove_block_temporarily(net, stage_name, block_idx)
        psnr_without = eval_psnr_on_subset(net, val_samples, device)
        restore_block(stage_module, removed_block, block_idx)

        score = baseline_psnr - psnr_without
        scores[(stage_name, block_idx)] = score
        if verbose:
            print(f"  Xoa {stage_name}[{block_idx}]: PSNR={psnr_without:.3f} "
                  f"(giam {score:.3f} -- cang nho cang 'vo hai')")

    return scores, baseline_psnr


def iterative_prune(net, val_samples, device, target_n_removed=3, verbose=True):
    """
    Lap lai: tinh importance score cho cac block CON LAI, xoa VINH VIEN block
    co score nho nhat (it anh huong nhat), cho toi khi xoa du target_n_removed block.
    """
    removed_log = []

    for iteration in range(target_n_removed):
        print(f"\n{'='*50}")
        print(f"  Vong lap {iteration + 1}/{target_n_removed}")
        print(f"{'='*50}")

        scores, current_psnr = compute_importance_scores(net, val_samples, device, verbose=verbose)

        if not scores:
            print("Khong con block nao co the xoa them (moi Stage chi con 1 block).")
            break

        best_to_remove = min(scores, key=scores.get)
        stage_name, block_idx = best_to_remove
        score = scores[best_to_remove]

        print(f"\n>>> XOA VINH VIEN: {stage_name}[{block_idx}] (importance score={score:.3f})")

        # Xoa THAT, khong khoi phuc lai lan nay
        removed_block, stage_module = remove_block_temporarily(net, stage_name, block_idx)
        removed_log.append({
            "stage": stage_name, "block_idx": block_idx,
            "score": score, "psnr_after": current_psnr - score
        })

    final_psnr = eval_psnr_on_subset(net, val_samples, device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"\n{'='*50}")
    print(f"KET QUA CUOI: da xoa {len(removed_log)} block")
    print(f"PSNR cuoi cung: {final_psnr:.3f}")
    print(f"So tham so con lai: {n_params:,} ({n_params/1e6:.3f}M)")
    print(f"{'='*50}")

    return removed_log, final_psnr


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-checkpoint", required=True)
    parser.add_argument("-val_data_dir", required=True)
    parser.add_argument("-n_val_samples", type=int, default=50)
    parser.add_argument("-target_n_removed", type=int, default=3)
    parser.add_argument("-output_checkpoint", default="pruned_model.pth")
    parser.add_argument("-output_log", default="pruned_removed_log.json")
    parser.add_argument("-device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    net = load_model(args.checkpoint, device)
    val_samples = get_val_subset(args.val_data_dir, args.n_val_samples)
    print(f"Dung {len(val_samples)} anh validation (co dinh seed=42)")

    removed_log, final_psnr = iterative_prune(net, val_samples, device,
                                                target_n_removed=args.target_n_removed)

    torch.save(net.state_dict(), args.output_checkpoint)
    print(f"\nDa luu model da pruned: {args.output_checkpoint}")

    # THEM MOI: luu lai removed_log ra JSON -- can thiet de finetune_pruned.py
    # sau nay biet CHINH XAC can tai tao lai kien truc nao (khoi nao da bi xoa)
    import json
    with open(args.output_log, "w") as f:
        json.dump({
            "removed_blocks": [{"stage": r["stage"], "block_idx": r["block_idx"]} for r in removed_log],
            "final_psnr": final_psnr,
        }, f, indent=2)
    print(f"Da luu log cac khoi da xoa: {args.output_log}")

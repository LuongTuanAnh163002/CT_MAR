"""
train_exit_heads.py
======================
Train CAC ExitHead, dong bang hoan toan backbone MambaFormer da train xong.
Chi chay SAU KHI da co checkpoint cuoi cung (het Stage 3).

CACH DUNG:
    python train_exit_heads.py \
        --backbone_checkpoint checkpoint/stage3/320000_ckpt \
        --train_data_dir /path/to/data_CT_MAR_split/train \
        --val_data_dir /path/to/data_CT_MAR_split/test \
        --num_steps 5000 \
        --exp_name checkpoint/exit_heads
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.mamba_exithead import MambaFormerWithExit as MambaFormer, ExitHead, compute_confidence_label
from utils.aapm_dataset import AAPMTrainDataset


def load_frozen_backbone(checkpoint_path, device):
    net = MambaFormer(in_channels=1)
    net = net.to(device)
    net = nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())) or [0])
    state = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(state)

    # DONG BANG toan bo backbone -- khong cap nhat trong so nay nua
    for param in net.parameters():
        param.requires_grad = False
    net.eval()
    return net


def build_exit_heads(sample_intermediates, device):
    """Tao ExitHead khop dung channel/downsample_factor cua tung diem thoat, dua tren
    1 lan forward_get_intermediate() mau de biet truoc kich thuoc."""
    heads = nn.ModuleList([
        ExitHead(in_channels=feat.shape[1], downsample_factor=factor)
        for feat, factor in sample_intermediates
    ]).to(device)
    return heads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-backbone_checkpoint", required=True)
    parser.add_argument("-train_data_dir", required=True)
    parser.add_argument("-val_data_dir", required=True)
    parser.add_argument("-crop_size", nargs="+", type=int, default=[256, 256])
    parser.add_argument("-train_batch_size", type=int, default=8)
    parser.add_argument("-num_steps", type=int, default=5000)
    parser.add_argument("-learning_rate", type=float, default=1e-4)
    parser.add_argument("-psnr_gap_threshold", type=float, default=0.5,
                         help="Nguong (dB) de tu tao nhan confidence -- xem exit_head.py")
    parser.add_argument("-conf_loss_weight", type=float, default=0.3)
    parser.add_argument("-save_step", type=int, default=500)
    parser.add_argument("-exp_name", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.exp_name, exist_ok=True)

    # --- Load backbone da train xong, dong bang ---
    print("Dang load backbone da train xong (dong bang hoan toan)...")
    net = load_frozen_backbone(args.backbone_checkpoint, device)

    # --- Data ---
    train_dataset = AAPMTrainDataset(args.crop_size, args.train_data_dir,
                                       random_flip=True, random_rotate=True)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                                shuffle=True, num_workers=4)

    # --- Tao ExitHead, khop dung shape thuc te ---
    sample_x, _ = train_dataset[0]
    sample_x = sample_x.unsqueeze(0).to(device)
    with torch.no_grad():
        sample_intermediates, _, _, _ = net.module.forward_get_intermediate(sample_x)
    exit_heads = build_exit_heads(sample_intermediates, device)
    n_params = sum(p.numel() for p in exit_heads.parameters())
    print(f"Tong tham so ExitHead (se train): {n_params:,} "
          f"(nho hon nhieu so voi backbone ~0.59M da dong bang)")

    optimizer = torch.optim.Adam(exit_heads.parameters(), lr=args.learning_rate)

    total_steps = 0
    exit_heads.train()
    while True:
        for input_image, gt in train_loader:
            input_image = input_image.to(device)
            gt = gt.to(device)

            with torch.no_grad():
                intermediates, x_ori, ori_h, ori_w = net.module.forward_get_intermediate(input_image)
                full_output = net(input_image)  # dung lam "chuan tham chieu" de tao nhan confidence

            optimizer.zero_grad()
            total_loss = 0.0
            log_parts = []

            for i, ((feat, factor), head) in enumerate(zip(intermediates, exit_heads)):
                early_output, confidence = head(feat, x_ori, ori_h, ori_w)

                restore_loss = F.smooth_l1_loss(early_output, gt)

                with torch.no_grad():
                    conf_label = compute_confidence_label(
                        early_output.detach(), full_output.detach(), gt,
                        psnr_gap_threshold=args.psnr_gap_threshold)
                conf_loss = F.binary_cross_entropy(confidence, conf_label)

                total_loss = total_loss + restore_loss + args.conf_loss_weight * conf_loss
                log_parts.append(f"exit{i+1}[restore={restore_loss.item():.4f},conf={conf_loss.item():.4f}]")

            total_loss.backward()
            optimizer.step()
            total_steps += 1

            if total_steps % 10 == 0:
                print(f"Step {total_steps}: total_loss={total_loss.item():.4f}  " + "  ".join(log_parts))

            if total_steps % args.save_step == 0:
                torch.save(exit_heads.state_dict(),
                           os.path.join(args.exp_name, f"{total_steps}_exitheads.pth"))
                print(f"  -> da luu checkpoint ExitHead tai step {total_steps}")

            if total_steps >= args.num_steps:
                print("Hoan thanh train ExitHead!")
                return


if __name__ == "__main__":
    main()

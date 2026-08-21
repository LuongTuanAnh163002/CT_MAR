"""
train_multi_artifact.py
==========================
File train RIENG BIET cho huong cai thien Multi-artifact Count-aware Loss
Weighting -- KHONG dung cham gi toi train_step.py / utils/aapm_dataset.py goc.

Cong thuc trong so (co co so thuc nghiem, KHONG phai doan mo):
    Da do PSNR that theo tung n_materials tren checkpoint baseline:
        PSNR ~ 45.312 - 0.698 * n_materials   (R^2 = 0.994, quan he TUYEN TINH)
    => weight = 1.0 + lambda * (n_materials - 1), lambda=0.125 (weight(n=5)=1.5)

Ho tro CA HAI cach dung:
  1. Train/fine-tune CHI voi multi-artifact weighting (dat -gamma 0)
  2. Ket hop voi contrast-aware loss da co (dat -gamma > 0, vd 0.1)

CACH DUNG (fine-tune tu checkpoint da co, RE hon train tu dau):
    python train_multi_artifact.py \
        -checkpoint checkpoint/stage2_contrast_aware/299000_ckpt \
        -learning_rate 0.00003 \
        -num_steps 20000 \
        -lambda_weight 0.125 \
        -gamma 0.1 \
        -exp_name checkpoint/stage2_multi_artifact \
        -train_data_dir /path/to/data_CT_MAR_split/train \
        -val_data_dir /path/to/data_CT_MAR_split/test
"""

import os
import re
import glob
import json
import time
import argparse
import numpy as np
import torch
torch.set_num_threads(1)  # gioi han so luong CPU threads PyTorch tu sinh ra
                          # -- can thiet vi container gioi han pids.max=256 (rat thap)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')  # tranh loi het /dev/shm khi dung DataLoader multiprocessing
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as tvu
import cv2

import sys
sys.path.append('.')
from model.mamba import MambaFormer
from utils.metrics import calculate_psnr, calculate_ssim, calculate_rmse
from utils.aapm_dataset import AAPMTrainDataset, test_image

import lpips

FNAME_ID_RE = re.compile(r"img(\d+)")


# ============================================================
# 1. DATASET -- ke thua AAPMTrainDataset, THEM n_materials
#    (khong sua file aapm_dataset.py goc)
# ============================================================
class AAPMTrainDatasetWithMaterials(AAPMTrainDataset):
    def __init__(self, crop_size, train_data_dir, random_flip, random_rotate):
        super().__init__(crop_size, train_data_dir, random_flip, random_rotate)
        self._load_n_materials()

    def _load_n_materials(self):
        """Doc them n_materials tu file JSON (*metalinfo<ID>.json) cung thu muc Mask/.
        Neu khong tim thay JSON -> mac dinh n_materials=1 (khong phat sinh loi)."""
        n_found = 0
        for s in self.samples:
            n_materials = 1
            if s.get("mask"):
                m = FNAME_ID_RE.search(os.path.basename(s["mask"]))
                if m:
                    img_id = m.group(1)
                    mask_dir = os.path.dirname(s["mask"])
                    json_files = glob.glob(os.path.join(mask_dir, f"*metalinfo{img_id}.json"))
                    if json_files:
                        with open(json_files[0]) as jf:
                            n_materials = json.load(jf).get("n_materials", 1)
                        n_found += 1
            s["n_materials"] = n_materials
        print(f"Da doc duoc n_materials cho {n_found}/{len(self.samples)} anh "
              f"(con lai mac dinh = 1)")

    def __getitem__(self, idx):
        input_im, gt = super().__getitem__(idx)
        n_materials = self.samples[idx]["n_materials"]
        return input_im, gt, n_materials


# ============================================================
# 2. LOSS -- multi-artifact weighting + contrast-aware (tuy chon)
# ============================================================
def multi_artifact_weight(n_materials, lambda_weight=0.125):
    """
    Cong thuc co co so thuc nghiem: PSNR ~ 45.312 - 0.698*n_materials (R^2=0.994).
    lambda=0.125 -> weight(n=1)=1.0, weight(n=5)=1.5.
    """
    return 1.0 + lambda_weight * (n_materials - 1)


def hub_loss_weighted(img, gt, sample_weight):
    """Giong hub_loss goc trong train_step.py, nhung nhan them trong so THEO TUNG ANH trong batch."""
    c = 0.03
    diff = torch.sqrt(torch.pow(img - gt, 2) + c ** 2)
    loss = diff - c
    weighted_loss = loss * sample_weight.view(-1, 1, 1, 1)
    return weighted_loss.sum() / weighted_loss.numel()


def local_std(gt, window=5):
    """Dung lai tu contrast-aware loss (huong #1) -- de co the KET HOP ca 2 huong."""
    pad = window // 2
    gt_padded = F.pad(gt, (pad, pad, pad, pad), mode='reflect')
    mean = F.avg_pool2d(gt_padded, kernel_size=window, stride=1)
    mean_sq = F.avg_pool2d(gt_padded ** 2, kernel_size=window, stride=1)
    var = (mean_sq - mean ** 2).clamp(min=0)
    return torch.sqrt(var + 1e-8)


def contrast_aware_loss(pred, gt, window=5, eps=0.00125):
    std = local_std(gt, window=window)
    normalized_error = torch.abs(pred - gt) / (std + eps)
    return normalized_error.mean()


# ============================================================
# 3. TRAIN LOOP -- doc lap, khong dung chung voi train_step.py
# ============================================================
def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    tvu.save_image(img, file_directory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-learning_rate', default=2e-4, type=float)
    parser.add_argument('-crop_size', default=[256, 256], nargs='+', type=int)
    parser.add_argument('-train_batch_size', default=8, type=int)
    parser.add_argument('-exp_name', required=True, type=str)
    parser.add_argument('-seed', default=19, type=int)
    parser.add_argument('-num_steps', default=100000, type=int)
    parser.add_argument('-checkpoint', type=str, default=None,
                         help="Checkpoint de fine-tune tiep (khuyen nghi dung checkpoint da co, RE hon train tu dau)")
    parser.add_argument('-save_step', default=1000, type=int)
    parser.add_argument('-train_data_dir', required=True, type=str)
    parser.add_argument('-val_data_dir', required=True, type=str)
    parser.add_argument('-warm_up', default=False, type=bool)
    parser.add_argument('-Tmax', default=10000, type=int)
    parser.add_argument('-lambda_weight', default=0.125, type=float,
                         help="He so trong so multi-artifact (co co so thuc nghiem, xem docstring dau file)")
    parser.add_argument('-gamma', default=0.0, type=float,
                         help="Trong so contrast-aware loss. Dat 0 neu CHI muon test rieng multi-artifact weighting")
    parser.add_argument('-ca_window', default=5, type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.exp_name, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print('--- Hyper-parameters ---')
    print(f'learning_rate={args.learning_rate}, crop_size={args.crop_size}, '
          f'lambda_weight={args.lambda_weight}, gamma={args.gamma}')

    net = MambaFormer(in_channels=1)
    total = sum(p.nelement() for p in net.parameters())
    print(f"So tham so: {total/1e6:.2f}M")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    if args.warm_up:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.Tmax, eta_min=1e-8)

    device_ids = list(range(torch.cuda.device_count())) or [0]
    net = net.to(device)
    net = nn.DataParallel(net, device_ids=device_ids)

    if args.checkpoint is not None:
        net.load_state_dict(torch.load(args.checkpoint))
        print(f'--- da load checkpoint: {args.checkpoint} ---')

    train_dataset = AAPMTrainDatasetWithMaterials(
        args.crop_size, args.train_data_dir, random_flip=True, random_rotate=True)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                                shuffle=True, num_workers=0)

    lpips_loss_fn = lpips.LPIPS(net='vgg', spatial=False).to(device)

    # Gop chung ve "./eva/" -- NHAT QUAN voi noi test_image() da ghi anh output/gt,
    # tach rieng thu muc chi de luu 1 file .txt la khong can thiet
    file_path = "./eva/eva.txt"
    total_steps = 0
    if args.checkpoint:
        try:
            total_steps = int(os.path.basename(args.checkpoint).split("_")[0])
        except ValueError:
            total_steps = 0  # checkpoint khong theo dung dinh dang "<step>_ckpt", bat dau tu 0

    net.train()
    while True:
        for input_image, gt, n_materials_batch in train_loader:
            input_image = input_image.to(device)
            gt = gt.to(device)
            sample_weights = multi_artifact_weight(
                n_materials_batch.float(), lambda_weight=args.lambda_weight).to(device)

            optimizer.zero_grad()
            net.train()
            pred_image = net(input_image)

            huber_term = hub_loss_weighted(pred_image, gt, sample_weights)
            lpips_term = lpips_loss_fn(pred_image, gt).mean()
            loss = 0.8 * huber_term + 0.2 * lpips_term

            if args.gamma > 0:
                ca_term = contrast_aware_loss(pred_image, gt, window=args.ca_window)
                loss = loss + args.gamma * ca_term

            loss.backward()
            optimizer.step()
            if args.warm_up:
                scheduler.step()
            total_steps += 1

            if total_steps % 10 == 0:
                print(f'Steps: {total_steps}, loss: {loss.item():.4f}, '
                      f'weight_range=[{sample_weights.min().item():.2f},{sample_weights.max().item():.2f}]')

            if total_steps % 100 == 0:
                with torch.no_grad():
                    save_image(pred_image, "./train_res_multi_artifact/output.png")
                    save_image(gt, "./train_res_multi_artifact/gt.png")
                    save_image(input_image * 0.5 + 0.5, "./train_res_multi_artifact/input.png")

            if total_steps % args.save_step == 0:
                net.eval()
                with torch.no_grad():
                    total_image, time_avg = test_image(args.val_data_dir, net)
                    print(f"test speed: {time_avg} per image")
                    # SUA LOI: test_image() (trong utils/aapm_dataset.py) GHI CUNG vao
                    # "./eva/output" va "./eva/gt" -- KHONG PHAI duong dan tuy dat
                    # "./eva_multi_artifact/..." -- phai doc DUNG cho khop
                    results_path, gt_path = "./eva/output", "./eva/gt"
                    imgsName = sorted(os.listdir(results_path))
                    gtsName = sorted(os.listdir(gt_path))
                    assert len(imgsName) == len(gtsName)

                    cumulative_psnr, cumulative_ssim, rmse_all = 0, 0, 0
                    for i in range(len(imgsName)):
                        res = cv2.imread(os.path.join(results_path, imgsName[i]), cv2.IMREAD_COLOR)
                        gt_img = cv2.imread(os.path.join(gt_path, gtsName[i]), cv2.IMREAD_COLOR)
                        cumulative_psnr += calculate_psnr(res, gt_img, test_y_channel=True)
                        cumulative_ssim += calculate_ssim(res, gt_img, test_y_channel=True)
                        rmse_all += calculate_rmse(res, gt_img)

                    psnr = cumulative_psnr / len(imgsName)
                    ssim = cumulative_ssim / len(imgsName)
                    rmse_avg = rmse_all / len(imgsName)
                    print(f'Testing: PSNR={psnr:.4f} SSIM={ssim:.4f} RMSE={rmse_avg:.4f}')

                    # SUA LOI: open() KHONG tu tao thu muc cha (khac save_image() co tu tao)
                    # -- phai tao thu muc "./eva_multi_artifact/" TRUOC khi ghi file
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    mode = 'a' if os.path.exists(file_path) else 'w'
                    with open(file_path, mode) as f:
                        f.write(f"steps:{total_steps}, PSNR:{psnr}, SSIM:{ssim}, RMSE:{rmse_avg}\n")

                    torch.save(net.state_dict(), f'./{args.exp_name}/{total_steps}_ckpt')
                torch.cuda.empty_cache()
                net.train()

            if total_steps == args.num_steps:
                print("Finish!")
                return


if __name__ == "__main__":
    main()

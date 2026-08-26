"""Training script for Exp1: Metal-Geometry Guided Dynamic FMB.

Baseline files remain untouched:
  - train_step.py
  - model/mamba.py
  - utils/aapm_dataset.py
"""

import argparse
import os
import random

import cv2
import lpips
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as tvu
from torch.utils.data import DataLoader

from model.mamba_exp1 import MetalGuidedMambaFormer
from utils.aapm_dataset_exp1 import AAPMTrainDatasetExp1, test_image_exp1
from utils.metrics import calculate_psnr, calculate_ssim, calculate_rmse


parser = argparse.ArgumentParser(description='Hyper-parameters for MARMamba Exp1')
parser.add_argument('-learning_rate', default=2e-4, type=float)
parser.add_argument('-crop_size', default=[128, 128], nargs='+', type=int)
parser.add_argument('-train_batch_size', default=18, type=int)
parser.add_argument('-epoch_start', default=0, type=int)
parser.add_argument('-val_batch_size', default=1, type=int)
parser.add_argument('-exp_name', type=str, required=True)
parser.add_argument('-seed', default=19, type=int)
parser.add_argument('-num_epochs', default=200, type=int)
parser.add_argument('-num_steps', default=90000, type=int)
parser.add_argument('-checkpoint', type=str)
parser.add_argument('-save_epoch', default=10, type=int)
parser.add_argument('-save_step', default=1000, type=int)
parser.add_argument('-train_data_dir', type=str, required=True)
parser.add_argument('-val_data_dir', type=str, required=True)
parser.add_argument('-warm_up', action='store_true')
parser.add_argument('-Tmax', default=10000, type=int)
parser.add_argument('-guidance_temperature', default=1.0, type=float)
args = parser.parse_args()

learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
exp_name = args.exp_name
num_steps = args.num_steps
save_step = args.save_step
train_data_dir = args.train_data_dir
val_data_dir = args.val_data_dir
warm_up = args.warm_up
Tmax = args.Tmax

os.makedirs(exp_name, exist_ok=True)
train_res_dir = os.path.join(exp_name, 'train_res')
eva_root = os.path.join(exp_name, 'eva')
os.makedirs(train_res_dir, exist_ok=True)
os.makedirs(eva_root, exist_ok=True)


def save_image(img, file_directory):
    os.makedirs(os.path.dirname(file_directory), exist_ok=True)
    tvu.save_image(img, file_directory)


seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

print('--- Hyper-parameters for Exp1 training ---')
print(
    f'learning_rate: {learning_rate}\n'
    f'crop_size: {crop_size}\n'
    f'train_batch_size: {train_batch_size}\n'
    f'train dataset: {train_data_dir}\n'
    f'validation dataset: {val_data_dir}\n'
    f'guidance_temperature: {args.guidance_temperature}'
)

device_ids = list(range(torch.cuda.device_count()))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = MetalGuidedMambaFormer(
    in_channels=1,
    guidance_temperature=args.guidance_temperature,
)
total = sum(param.nelement() for param in net.parameters())
print('Number of parameter: %.2fM' % (total / 1e6))

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
if warm_up:
    print(f'Using CosineAnnealingLR, T_max = {Tmax}')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Tmax, eta_min=1e-8
    )

net = net.to(device)
if len(device_ids) > 1:
    net = nn.DataParallel(net, device_ids=device_ids)

chk = args.checkpoint
if chk is not None:
    if not os.path.isfile(chk):
        raise FileNotFoundError(f"The file at path '{chk}' does not exist.")
    state = torch.load(chk, map_location=device)
    net.load_state_dict(state)
    print('--- Exp1 weight loaded ---')

train_loader = DataLoader(
    AAPMTrainDatasetExp1(
        crop_size,
        train_data_dir,
        random_flip=True,
        random_rotate=True,
    ),
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=8,
)


def hub_loss(img, gt):
    c = 0.03
    diff = torch.sqrt(torch.pow(img - gt, 2) + c ** 2)
    return (diff - c).sum() / diff.numel()


lpips_loss = lpips.LPIPS(net='vgg', spatial=False).to(device)
file_path = os.path.join(eva_root, 'eva.txt')
total_steps = 0
net.train()

if chk:
    try:
        total_steps = int(os.path.basename(chk).split('_')[0])
    except ValueError:
        print('Could not infer total_steps from checkpoint name; starting counter at 0.')

while True:
    for _, train_data in enumerate(train_loader):
        input_image, gt, metal_mask = train_data
        input_image = input_image.to(device)
        gt = gt.to(device)
        metal_mask = metal_mask.to(device)

        optimizer.zero_grad()
        net.train()

        pred_image = net(input_image, metal_mask)

        # Keep the baseline loss unchanged for a clean Exp1 architecture ablation.
        loss = 0.8 * hub_loss(pred_image, gt) + 0.2 * lpips_loss(pred_image, gt).mean()

        loss.backward()
        optimizer.step()
        if warm_up:
            scheduler.step()

        total_steps += 1

        if total_steps % 10 == 0:
            # Inspect routing behavior without changing optimization.
            with torch.no_grad():
                model_ref = net.module if isinstance(net, nn.DataParallel) else net
                weights = model_ref.metal_guidance(metal_mask)
                mean_w = weights.mean(dim=0).detach().cpu().tolist()
            print(
                f'Steps: {total_steps}, loss: {loss.item():.6f}, '
                f'mean branch weights: [{mean_w[0]:.3f}, {mean_w[1]:.3f}, {mean_w[2]:.3f}]'
            )

        if total_steps % 100 == 0:
            with torch.no_grad():
                if warm_up:
                    print(f'Current Learning Rate: {scheduler.get_last_lr()[0]}')
                save_image(pred_image, os.path.join(train_res_dir, 'output.png'))
                save_image(gt, os.path.join(train_res_dir, 'gt.png'))
                save_image(input_image * 0.5 + 0.5, os.path.join(train_res_dir, 'input.png'))
                save_image(metal_mask, os.path.join(train_res_dir, 'mask.png'))

        if total_steps % save_step == 0:
            net.eval()
            with torch.no_grad():
                total_image, time_avg = test_image_exp1(
                    val_data_dir,
                    net,
                    save_root=eva_root,
                )
                print(f'test speed: {time_avg} per image')

                results_path = os.path.join(eva_root, 'output')
                gt_path = os.path.join(eva_root, 'gt')
                imgs_name = sorted(os.listdir(results_path))
                gts_name = sorted(os.listdir(gt_path))
                assert len(imgs_name) == len(gts_name)

                cumulative_psnr = 0.0
                cumulative_ssim = 0.0
                rmse_all = 0.0

                for i in range(len(imgs_name)):
                    res = cv2.imread(
                        os.path.join(results_path, imgs_name[i]), cv2.IMREAD_COLOR
                    )
                    gt_img = cv2.imread(
                        os.path.join(gt_path, gts_name[i]), cv2.IMREAD_COLOR
                    )
                    cumulative_psnr += calculate_psnr(res, gt_img, test_y_channel=True)
                    cumulative_ssim += calculate_ssim(res, gt_img, test_y_channel=True)
                    rmse_all += calculate_rmse(res, gt_img)

                psnr = cumulative_psnr / len(imgs_name)
                ssim = cumulative_ssim / len(imgs_name)
                rmse_avg = rmse_all / len(imgs_name)

                print(
                    'Testing set, PSNR is %.4f and SSIM is %.4f, RMSE is %.4f'
                    % (psnr, ssim, rmse_avg)
                )

                with open(file_path, 'a') as f:
                    f.write(
                        f'steps:{total_steps}, PSNR:{psnr}, SSIM:{ssim}, RMSE:{rmse_avg}\n'
                    )

                torch.save(
                    net.state_dict(),
                    os.path.join(exp_name, f'{total_steps}_ckpt'),
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if total_steps >= num_steps:
            print('Finish!')
            raise SystemExit(0)

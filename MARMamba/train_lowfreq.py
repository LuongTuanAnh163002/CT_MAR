"""
Pilot training for Low-Frequency Global Branch.

Recommended first experiment:
- initialize from the best BASELINE MARMamba checkpoint;
- freeze baseline weights;
- train only the low-frequency branch for 10k-20k steps;
- keep the original 0.8 Huber + 0.2 LPIPS loss unchanged.

This is a cheap hypothesis test, NOT the final fair full-training comparison.
If the pilot is promising, do a full baseline-vs-new-model comparison under
the same initialization/schedule.
"""

import argparse
import json
import os
import random
import time

import cv2
import lpips
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.mamba_lowfreq import (
    LowFrequencyMambaFormer,
    load_baseline_weights,
)
from utils.aapm_dataset import (
    AAPMTrainDataset,
    _find_anatomy_folders,
    _index_one_anatomy,
    hu_to_unit,
    load_raw,
)
from utils.metrics import calculate_psnr, calculate_ssim, calculate_rmse


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("true", "1", "yes", "y"):
        return True
    if v in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser(description="Low-frequency MARMamba pilot")
parser.add_argument("-learning_rate", default=2e-5, type=float)
parser.add_argument("-crop_size", default=[256, 256], nargs="+", type=int)
parser.add_argument("-train_batch_size", default=8, type=int)
parser.add_argument("-exp_name", type=str, required=True)
parser.add_argument("-seed", default=19, type=int)
parser.add_argument("-num_steps", default=20000, type=int)
parser.add_argument("-save_step", default=1000, type=int)
parser.add_argument("-eval_step", default=1000, type=int)
parser.add_argument("-train_data_dir", type=str, required=True)
parser.add_argument("-val_data_dir", type=str, required=True)
parser.add_argument("-init_baseline", type=str, required=True)
parser.add_argument("-resume", type=str, default=None)
parser.add_argument("-freeze_baseline", default=True, type=str2bool)
parser.add_argument("-warm_up", default=False, type=str2bool)
parser.add_argument("-Tmax", default=1000, type=int)
parser.add_argument("-num_workers", default=0, type=int)
parser.add_argument(
    "-max_val_samples",
    default=None,
    type=int,
    help="Optional fixed-size validation subset for fast pilot evaluation.",
)
args = parser.parse_args()


# -------------------------------------------------------------------------
# Reproducibility / dirs
# -------------------------------------------------------------------------

os.makedirs(args.exp_name, exist_ok=True)
train_res_dir = os.path.join(args.exp_name, "train_res")
os.makedirs(train_res_dir, exist_ok=True)

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = list(range(torch.cuda.device_count()))


# -------------------------------------------------------------------------
# Model initialization
# -------------------------------------------------------------------------

net = LowFrequencyMambaFormer(in_channels=1)
missing = load_baseline_weights(
    net,
    args.init_baseline,
    map_location="cpu",
)

print("Loaded BASELINE checkpoint.")
print(f"New low-frequency keys initialized separately: {len(missing)}")

if args.freeze_baseline:
    for p in net.parameters():
        p.requires_grad = False
    for p in net.lowfreq_branch.parameters():
        p.requires_grad = True

total_params = sum(p.numel() for p in net.parameters())
trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

print(f"Total parameters:     {total_params / 1e6:.4f} M")
print(f"Trainable parameters: {trainable_params / 1e6:.4f} M")
print(f"Freeze baseline:      {args.freeze_baseline}")

net = net.to(device)

if len(device_ids) > 1:
    net = nn.DataParallel(net, device_ids=device_ids)


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


optimizer = torch.optim.Adam(
    [p for p in net.parameters() if p.requires_grad],
    lr=args.learning_rate,
)

scheduler = None
if args.warm_up:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.Tmax,
        eta_min=1e-8,
    )


# -------------------------------------------------------------------------
# Resume full training state if requested
# -------------------------------------------------------------------------

total_steps = 0

if args.resume is not None:
    state = torch.load(args.resume, map_location=device)

    if "model" not in state or "optimizer" not in state or "step" not in state:
        raise RuntimeError(
            "-resume must point to *_train_state.pt created by this script."
        )

    model_state = state["model"]
    if any(k.startswith("module.") for k in model_state):
        model_state = {
            k[len("module."):]: v
            for k, v in model_state.items()
        }

    unwrap_model(net).load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(state["optimizer"])

    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])

    total_steps = int(state["step"])
    print(f"Resumed low-frequency pilot from step {total_steps}.")


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------

train_loader = DataLoader(
    AAPMTrainDataset(
        args.crop_size,
        args.train_data_dir,
        random_flip=True,
        random_rotate=True,
    ),
    batch_size=args.train_batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=torch.cuda.is_available(),
)


# -------------------------------------------------------------------------
# Loss: unchanged baseline objective
# -------------------------------------------------------------------------

def hub_loss(img, gt):
    c = 0.03
    diff = torch.sqrt(torch.pow(img - gt, 2) + c ** 2)
    return (diff - c).mean()


lpips_loss = lpips.LPIPS(net="vgg", spatial=False).to(device)
lpips_loss.eval()
for p in lpips_loss.parameters():
    p.requires_grad = False


# -------------------------------------------------------------------------
# Validation using same 8-bit metric conversion and non-metal masking
# -------------------------------------------------------------------------

def to_uint8_bgr3(img_float01):
    img_u8 = np.clip(img_float01 * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)


@torch.no_grad()
def evaluate_validation(model):
    model.eval()

    samples = []
    for anatomy_dir in _find_anatomy_folders(args.val_data_dir):
        samples.extend(_index_one_anatomy(anatomy_dir))

    samples = sorted(samples, key=lambda s: s["id"])

    if args.max_val_samples is not None:
        samples = samples[:args.max_val_samples]

    all_metrics = []
    single_metrics = []
    multi_metrics = []

    start = time.time()

    for sample in samples:
        input_img = hu_to_unit(load_raw(sample["baseline"]))
        gt_img = hu_to_unit(load_raw(sample["target"]))

        mask = None
        if sample.get("mask") is not None:
            mask = load_raw(sample["mask"], dtype=np.float32) > 0.5

        input_t = (
            torch.from_numpy((input_img - 0.5) / 0.5)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(device)
        )

        pred = model(input_t).squeeze().detach().cpu().numpy()
        pred = np.clip(pred, 0.0, 1.0)

        pred_eval = pred.copy()
        gt_eval = gt_img.copy()

        if mask is not None:
            pred_eval[mask] = 0.0
            gt_eval[mask] = 0.0

        pred_bgr = to_uint8_bgr3(pred_eval)
        gt_bgr = to_uint8_bgr3(gt_eval)

        m = {
            "psnr": calculate_psnr(pred_bgr, gt_bgr, test_y_channel=True),
            "ssim": calculate_ssim(pred_bgr, gt_bgr, test_y_channel=True),
            "rmse": calculate_rmse(pred_bgr, gt_bgr),
        }

        all_metrics.append(m)

        n_materials = sample.get("n_materials")
        if n_materials is not None and int(n_materials) >= 2:
            multi_metrics.append(m)
        else:
            single_metrics.append(m)

    def summarize(items):
        if not items:
            return None
        return {
            key: float(np.mean([m[key] for m in items]))
            for key in ("psnr", "ssim", "rmse")
        } | {"n": len(items)}

    elapsed = time.time() - start

    return {
        "overall": summarize(all_metrics),
        "single": summarize(single_metrics),
        "multi": summarize(multi_metrics),
        "seconds": elapsed,
    }


def save_checkpoint(step):
    base = unwrap_model(net)

    # Raw weights: easy to use with evaluation scripts.
    raw_path = os.path.join(args.exp_name, f"{step}_ckpt")
    torch.save(base.state_dict(), raw_path)

    # Full state: safe resume including Adam + scheduler.
    state_path = os.path.join(args.exp_name, f"{step}_train_state.pt")
    torch.save(
        {
            "model": base.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "step": step,
            "args": vars(args),
        },
        state_path,
    )

    return raw_path, state_path


# -------------------------------------------------------------------------
# Train
# -------------------------------------------------------------------------

print("\n--- Low-Frequency Pilot ---")
print(f"LR:              {args.learning_rate}")
print(f"Crop:            {args.crop_size}")
print(f"Batch:           {args.train_batch_size}")
print(f"Target steps:    {args.num_steps}")
print(f"Eval step:       {args.eval_step}")
print(f"Workers:         {args.num_workers}")
print()

best_multi_psnr = -float("inf")

while total_steps < args.num_steps:
    for train_data in train_loader:
        if total_steps >= args.num_steps:
            break

        # Baseline AAPM loader should return (input, gt).
        if not isinstance(train_data, (tuple, list)) or len(train_data) < 2:
            raise RuntimeError(
                "AAPMTrainDataset must return at least (input_image, gt)."
            )

        input_image, gt = train_data[:2]
        input_image = input_image.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        net.train()

        pred = net(input_image)

        loss_h = hub_loss(pred, gt)
        loss_p = lpips_loss(pred, gt).mean()
        loss = 0.8 * loss_h + 0.2 * loss_p

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_steps += 1

        if total_steps % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"step={total_steps:6d} "
                f"loss={loss.item():.6f} "
                f"huber={loss_h.item():.6f} "
                f"lpips={loss_p.item():.6f} "
                f"lr={lr:.3e}"
            )

        if total_steps % args.eval_step == 0:
            val = evaluate_validation(net)
            print("\nValidation:")
            print(json.dumps(val, indent=2))

            multi_psnr = (
                val["multi"]["psnr"]
                if val["multi"] is not None
                else val["overall"]["psnr"]
            )

            with open(
                os.path.join(args.exp_name, "validation.jsonl"),
                "a",
                encoding="utf-8",
            ) as f:
                f.write(
                    json.dumps(
                        {"step": total_steps, **val},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if multi_psnr > best_multi_psnr:
                best_multi_psnr = multi_psnr
                best_path = os.path.join(args.exp_name, "best_multi_psnr_ckpt")
                torch.save(unwrap_model(net).state_dict(), best_path)
                print(
                    f"New best multi-metal validation PSNR: "
                    f"{best_multi_psnr:.4f} dB"
                )

        if total_steps % args.save_step == 0:
            raw_path, state_path = save_checkpoint(total_steps)
            print(f"Saved weights: {raw_path}")
            print(f"Saved train state: {state_path}")

print("Pilot finished.")

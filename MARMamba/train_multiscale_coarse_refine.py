"""
Train Multi-Scale Coarse Residual Refinement on AAPM CT-MAR.

Main experiment defaults:
- Stage-2 final MARMamba checkpoint as frozen backbone
- crop 416x416
- Adam, lr 1e-4
- hidden channels 32
- direct hierarchical residual supervision
- auxiliary final-image SmoothL1
- validation every 500 steps
- best checkpoint selected by OVERALL non-metal PSNR

The original utils/aapm_dataset.py and utils/metrics.py are reused unchanged.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as tvu
from torch.utils.data import DataLoader

from model.mamba_multiscale_coarse_refine import (
    MultiScaleCoarseRefinedMambaFormer,
    load_baseline_weights,
)
from utils.aapm_dataset import AAPMTrainDataset
from utils.metrics import calculate_psnr, calculate_ssim, calculate_rmse


# -------------------------------------------------------------------------
# Args
# -------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Multi-scale direct-delta coarse refinement"
)
parser.add_argument("-learning_rate", default=1e-4, type=float)
parser.add_argument("-crop_size", default=[416, 416], nargs="+", type=int)
parser.add_argument("-train_batch_size", default=8, type=int)
parser.add_argument("-exp_name", type=str, required=True)
parser.add_argument("-seed", default=19, type=int)
parser.add_argument("-num_steps", default=5000, type=int)
parser.add_argument("-save_step", default=500, type=int)
parser.add_argument("-eval_step", default=500, type=int)
parser.add_argument("-train_data_dir", type=str, required=True)
parser.add_argument("-val_data_dir", type=str, required=True)
parser.add_argument("-init_baseline", type=str, required=True)
parser.add_argument("-resume", type=str, default=None)
parser.add_argument("-num_workers", default=0, type=int)
parser.add_argument("-hidden_channels", default=32, type=int)
parser.add_argument("-delta_beta", default=0.01, type=float)
parser.add_argument("-lambda_scale2", default=1.0, type=float)
parser.add_argument("-lambda_scale4", default=1.0, type=float)
parser.add_argument("-lambda_final", default=1.0, type=float)
parser.add_argument(
    "-max_val_samples",
    default=None,
    type=int,
    help="Optional fast-pilot validation subset; default uses all val samples.",
)
args = parser.parse_args()

if len(args.crop_size) != 2:
    raise ValueError("-crop_size must contain exactly two integers.")
if args.delta_beta <= 0:
    raise ValueError("-delta_beta must be > 0.")

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
print(f"Device: {device}")


# -------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------

net = MultiScaleCoarseRefinedMambaFormer(
    in_channels=1,
    hidden_channels=args.hidden_channels,
)

load_baseline_weights(
    net,
    args.init_baseline,
    map_location="cpu",
)

net = net.to(device)

trainable_params = [p for p in net.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_params, lr=args.learning_rate)

total_params = sum(p.numel() for p in net.parameters())
backbone_params = sum(p.numel() for p in net.backbone.parameters())
refiner_params = total_params - backbone_params
trainable_count = sum(p.numel() for p in trainable_params)

print("\n--- Multi-Scale Coarse Refinement ---")
print(f"Total parameters:      {total_params:,}")
print(f"Backbone parameters:   {backbone_params:,}")
print(f"Refiner parameters:    {refiner_params:,}")
print(f"Trainable parameters:  {trainable_count:,}")
print(f"Parameter overhead:    {100.0 * refiner_params / backbone_params:.2f}%")
print(f"Crop size:             {args.crop_size}")
print(f"Learning rate:         {args.learning_rate}")
print(f"Hidden channels:       {args.hidden_channels}")
print(f"SmoothL1 beta:         {args.delta_beta}")
print(
    "Loss weights:         "
    f"s2={args.lambda_scale2}, "
    f"s4={args.lambda_scale4}, "
    f"final={args.lambda_final}"
)


# -------------------------------------------------------------------------
# Resume
# -------------------------------------------------------------------------

total_steps = 0
best_overall_psnr = -float("inf")

if args.resume is not None:
    state = torch.load(args.resume, map_location=device)
    required = {"model", "optimizer", "step"}
    if not required.issubset(state):
        raise RuntimeError(
            "-resume must point to *_train_state.pt created by this script."
        )

    model_state = state["model"]
    if any(k.startswith("module.") for k in model_state):
        model_state = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in model_state.items()
        }

    net.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(state["optimizer"])
    total_steps = int(state["step"])
    best_overall_psnr = float(
        state.get("best_overall_psnr", -float("inf"))
    )
    print(f"Resumed from step {total_steps}.")


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
# Hierarchical targets + loss
# -------------------------------------------------------------------------

def make_targets(gt, baseline):
    residual = gt - baseline

    target4_low = F.avg_pool2d(
        residual,
        kernel_size=4,
        stride=4,
    )

    target2_low = F.avg_pool2d(
        residual,
        kernel_size=2,
        stride=2,
    )

    target4_to_2 = F.interpolate(
        target4_low,
        size=target2_low.shape[-2:],
        mode="nearest",
    )

    # Scale-2 branch learns only what scale-4 cannot represent.
    target_detail2_low = target2_low - target4_to_2

    return target_detail2_low, target4_low


def compute_loss(outputs, gt):
    baseline = outputs["baseline"]

    with torch.no_grad():
        target_detail2_low, target4_low = make_targets(
            gt,
            baseline,
        )

    loss_s2 = F.smooth_l1_loss(
        outputs["pred_detail2_low"],
        target_detail2_low,
        beta=args.delta_beta,
        reduction="mean",
    )
    loss_s4 = F.smooth_l1_loss(
        outputs["pred_coarse4_low"],
        target4_low,
        beta=args.delta_beta,
        reduction="mean",
    )
    loss_final = F.smooth_l1_loss(
        outputs["refined"],
        gt,
        beta=args.delta_beta,
        reduction="mean",
    )

    total = (
        args.lambda_scale2 * loss_s2
        + args.lambda_scale4 * loss_s4
        + args.lambda_final * loss_final
    )

    return total, loss_s2, loss_s4, loss_final, target_detail2_low, target4_low


def branch_diagnostics(outputs, target_detail2_low, target4_low):
    with torch.no_grad():
        def one(pred, target):
            p = pred.detach().float().reshape(pred.shape[0], -1)
            t = target.detach().float().reshape(target.shape[0], -1)
            pnorm = torch.linalg.vector_norm(p, dim=1)
            tnorm = torch.linalg.vector_norm(t, dim=1)
            cosine = torch.sum(p * t, dim=1) / (pnorm * tnorm + 1e-12)
            return {
                "cos": float(cosine.mean().item()),
                "mag": float((pnorm / (tnorm + 1e-12)).mean().item()),
            }

        return {
            "s2": one(outputs["pred_detail2_low"], target_detail2_low),
            "s4": one(outputs["pred_coarse4_low"], target4_low),
            "delta_abs": float(outputs["delta"].detach().abs().mean().item()),
        }


# -------------------------------------------------------------------------
# Validation helpers
# -------------------------------------------------------------------------

HU_MIN, HU_MAX = -1000.0, 3000.0
FNAME_ID_RE = re.compile(r"img(\d+)")
DIMS_RE = re.compile(r"(\d+)x(\d+)x(\d+)")


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
        raise ValueError(f"{path}: found {arr.size}, expected {expected}")
    return arr.reshape(rows, cols) if slices == 1 else arr.reshape(
        slices, rows, cols
    )


def hu_to_unit(img):
    img = np.clip(img, HU_MIN, HU_MAX)
    return ((img - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def find_eval_samples(root_dir):
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

    if os.path.isdir(os.path.join(root_dir, "Baseline")):
        return index_one(root_dir)

    samples = []
    for name in sorted(os.listdir(root_dir)):
        sub = os.path.join(root_dir, name)
        if (
            os.path.isdir(sub)
            and os.path.isdir(os.path.join(sub, "Baseline"))
        ):
            samples.extend(index_one(sub))
    return samples


def to_uint8_bgr3(img_float01):
    img_u8 = np.clip(img_float01 * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)


def metric_one(pred, gt, mask):
    pred_eval = pred.copy()
    gt_eval = gt.copy()

    if mask is not None:
        pred_eval[mask] = 0.0
        gt_eval[mask] = 0.0

    pred_bgr = to_uint8_bgr3(pred_eval)
    gt_bgr = to_uint8_bgr3(gt_eval)

    return {
        "psnr": float(
            calculate_psnr(pred_bgr, gt_bgr, test_y_channel=True)
        ),
        "ssim": float(
            calculate_ssim(pred_bgr, gt_bgr, test_y_channel=True)
        ),
        "rmse": float(calculate_rmse(pred_bgr, gt_bgr)),
    }


def summarize(records, key):
    if not records:
        return None

    out = {"n": len(records)}
    for metric in ("psnr", "ssim", "rmse"):
        arr = np.asarray(
            [r[key][metric] for r in records],
            dtype=np.float64,
        )
        out[metric] = float(arr.mean())
        out[f"{metric}_std"] = float(arr.std())
    return out


def evaluate_validation(model):
    samples = find_eval_samples(args.val_data_dir)
    if args.max_val_samples is not None:
        samples = samples[: args.max_val_samples]

    if not samples:
        raise RuntimeError(
            f"No AAPM validation samples found in {args.val_data_dir}"
        )

    records = []
    start = time.time()

    model.eval()

    with torch.inference_mode():
        for sample in samples:
            input_img = hu_to_unit(load_raw(sample["baseline"]))
            gt_img = hu_to_unit(load_raw(sample["target"]))

            mask = None
            if sample["mask"] is not None:
                mask = (
                    load_raw(sample["mask"], dtype=np.float32) > 0.5
                )

            input_t = (
                torch.from_numpy((input_img - 0.5) / 0.5)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
                .to(device)
            )

            refined_t, baseline_t, _, _ = model(
                input_t,
                return_refinement=True,
            )

            baseline = np.clip(
                baseline_t.squeeze().cpu().numpy(),
                0.0,
                1.0,
            )
            refined = np.clip(
                refined_t.squeeze().cpu().numpy(),
                0.0,
                1.0,
            )

            n_materials = sample["n_materials"]
            n_materials = (
                int(n_materials)
                if n_materials is not None
                else None
            )

            records.append(
                {
                    "id": sample["id"],
                    "n_materials": n_materials,
                    "group": (
                        "multi"
                        if n_materials is not None and n_materials >= 2
                        else "single"
                    ),
                    "baseline": metric_one(
                        baseline, gt_img, mask
                    ),
                    "refined": metric_one(
                        refined, gt_img, mask
                    ),
                }
            )

    result = {
        "seconds": time.time() - start,
        "overall": {
            "baseline": summarize(records, "baseline"),
            "refined": summarize(records, "refined"),
        },
        "single": {},
        "multi": {},
        "by_metal_count": {},
    }

    single_records = [r for r in records if r["group"] == "single"]
    multi_records = [r for r in records if r["group"] == "multi"]

    for name, group_records in (
        ("single", single_records),
        ("multi", multi_records),
    ):
        result[name] = {
            "baseline": summarize(group_records, "baseline"),
            "refined": summarize(group_records, "refined"),
        }

    for n in range(1, 6):
        group_records = [
            r for r in records if r["n_materials"] == n
        ]
        result["by_metal_count"][str(n)] = {
            "baseline": summarize(group_records, "baseline"),
            "refined": summarize(group_records, "refined"),
        }

    return result


def print_pair(label, pair):
    if pair["baseline"] is None or pair["refined"] is None:
        return
    b, r = pair["baseline"], pair["refined"]
    print(
        f"{label:>8} | n={r['n']:4d} | "
        f"PSNR {b['psnr']:.4f} -> {r['psnr']:.4f} "
        f"({r['psnr'] - b['psnr']:+.4f}) | "
        f"SSIM {b['ssim']:.6f} -> {r['ssim']:.6f} "
        f"({r['ssim'] - b['ssim']:+.6f}) | "
        f"RMSE {b['rmse']:.4f} -> {r['rmse']:.4f} "
        f"({r['rmse'] - b['rmse']:+.4f})"
    )


def print_validation(step, result):
    print("\n" + "=" * 120)
    print(f"MULTISCALE VALIDATION @ STEP {step}")
    print("=" * 120)
    for n in range(1, 6):
        print_pair(
            f"{n} metal",
            result["by_metal_count"][str(n)],
        )
    print("-" * 120)
    for name in ("single", "multi", "overall"):
        print_pair(name, result[name])
    print(f"Validation time: {result['seconds']:.1f}s")


# -------------------------------------------------------------------------
# Checkpoints
# -------------------------------------------------------------------------

def save_checkpoint(step):
    raw_path = os.path.join(args.exp_name, f"{step}_ckpt")
    state_path = os.path.join(
        args.exp_name,
        f"{step}_train_state.pt",
    )

    torch.save(net.state_dict(), raw_path)
    torch.save(
        {
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_overall_psnr": best_overall_psnr,
            "args": vars(args),
        },
        state_path,
    )
    return raw_path, state_path


validation_history = []


# -------------------------------------------------------------------------
# Train
# -------------------------------------------------------------------------

while total_steps < args.num_steps:
    for train_data in train_loader:
        if total_steps >= args.num_steps:
            break

        if not isinstance(train_data, (tuple, list)) or len(train_data) < 2:
            raise RuntimeError(
                "AAPMTrainDataset must return at least (input_image, gt)."
            )

        input_image, gt = train_data[:2]
        input_image = input_image.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        net.train()
        optimizer.zero_grad(set_to_none=True)

        outputs = net(input_image, return_aux=True)

        (
            loss,
            loss_s2,
            loss_s4,
            loss_final,
            target_detail2_low,
            target4_low,
        ) = compute_loss(outputs, gt)

        loss.backward()

        grad_norm_sq = 0.0
        for p in trainable_params:
            if p.grad is not None:
                g = float(p.grad.detach().norm().item())
                grad_norm_sq += g * g
        grad_norm = math.sqrt(grad_norm_sq)

        optimizer.step()
        total_steps += 1

        if total_steps % 10 == 0 or total_steps == 1:
            diag = branch_diagnostics(
                outputs,
                target_detail2_low,
                target4_low,
            )
            print(
                f"step={total_steps:6d} "
                f"loss={loss.item():.7f} "
                f"s2={loss_s2.item():.7f} "
                f"s4={loss_s4.item():.7f} "
                f"final={loss_final.item():.7f} "
                f"s2_cos={diag['s2']['cos']:+.3f} "
                f"s2_mag={diag['s2']['mag']:.3f} "
                f"s4_cos={diag['s4']['cos']:+.3f} "
                f"s4_mag={diag['s4']['mag']:.3f} "
                f"delta_abs={diag['delta_abs']:.6e} "
                f"grad={grad_norm:.6e}"
            )

        if total_steps % 100 == 0:
            with torch.no_grad():
                tvu.save_image(
                    input_image * 0.5 + 0.5,
                    os.path.join(train_res_dir, "input.png"),
                )
                tvu.save_image(
                    outputs["baseline"],
                    os.path.join(train_res_dir, "baseline.png"),
                )
                tvu.save_image(
                    outputs["refined"],
                    os.path.join(train_res_dir, "refined.png"),
                )
                tvu.save_image(
                    gt,
                    os.path.join(train_res_dir, "gt.png"),
                )

                detail_vis = torch.clamp(
                    outputs["pred_detail2"] / 0.2 + 0.5,
                    0.0,
                    1.0,
                )
                coarse_vis = torch.clamp(
                    outputs["pred_coarse4"] / 0.2 + 0.5,
                    0.0,
                    1.0,
                )
                tvu.save_image(
                    detail_vis,
                    os.path.join(train_res_dir, "detail2.png"),
                )
                tvu.save_image(
                    coarse_vis,
                    os.path.join(train_res_dir, "coarse4.png"),
                )

        if total_steps % args.eval_step == 0:
            val_result = evaluate_validation(net)
            print_validation(total_steps, val_result)

            validation_history.append(
                {
                    "step": total_steps,
                    **val_result,
                }
            )

            with open(
                os.path.join(
                    args.exp_name,
                    "checkpoint_result.json",
                ),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    validation_history,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            cur_psnr = val_result["overall"]["refined"]["psnr"]

            if cur_psnr > best_overall_psnr:
                best_overall_psnr = cur_psnr
                torch.save(
                    net.state_dict(),
                    os.path.join(
                        args.exp_name,
                        "best_overall_psnr_ckpt",
                    ),
                )
                with open(
                    os.path.join(
                        args.exp_name,
                        "best_overall_psnr.json",
                    ),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        {
                            "step": total_steps,
                            "overall_psnr": cur_psnr,
                            "result": val_result,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
                print(
                    f"NEW BEST overall PSNR: "
                    f"{best_overall_psnr:.6f} @ {total_steps}"
                )

        if total_steps % args.save_step == 0:
            raw_path, state_path = save_checkpoint(total_steps)
            print(f"Saved: {raw_path}")
            print(f"Saved: {state_path}")

print("Training finished.")

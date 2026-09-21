"""
Train dynamic-gated Multi-Coarse on AAPM CT-MAR.

For a clean ablation:
- initialize from the same Stage-2 final checkpoint as fixed Multi-Coarse
- crop 416x416
- Adam, lr 1e-4
- same hierarchical targets and same SmoothL1 losses
- no severity weighting
- only change: fixed sum -> per-image gated fusion
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
from torch.utils.data import DataLoader

from model.mamba_multiscale_coarse_gate import (
    GatedMultiScaleCoarseRefinedMambaFormer,
    load_baseline_weights,
)
from utils.aapm_dataset import AAPMTrainDataset
from utils.metrics import calculate_psnr, calculate_ssim, calculate_rmse


parser = argparse.ArgumentParser()
parser.add_argument("-learning_rate", default=1e-4, type=float)
parser.add_argument("-crop_size", default=[416, 416], nargs="+", type=int)
parser.add_argument("-train_batch_size", default=8, type=int)
parser.add_argument("-exp_name", required=True, type=str)
parser.add_argument("-seed", default=19, type=int)
parser.add_argument("-num_steps", default=5000, type=int)
parser.add_argument("-save_step", default=500, type=int)
parser.add_argument("-eval_step", default=500, type=int)
parser.add_argument("-train_data_dir", required=True, type=str)
parser.add_argument("-val_data_dir", required=True, type=str)
parser.add_argument("-init_baseline", required=True, type=str)
parser.add_argument("-resume", default=None, type=str)
parser.add_argument("-num_workers", default=0, type=int)

parser.add_argument("-hidden_channels", default=32, type=int)
parser.add_argument("-gate_hidden", default=32, type=int)
parser.add_argument("-gate_range", default=0.5, type=float)

parser.add_argument("-delta_beta", default=0.01, type=float)
parser.add_argument("-lambda_scale2", default=1.0, type=float)
parser.add_argument("-lambda_scale4", default=1.0, type=float)
parser.add_argument("-lambda_final", default=1.0, type=float)
parser.add_argument("-max_val_samples", default=None, type=int)
args = parser.parse_args()

if len(args.crop_size) != 2:
    raise ValueError("-crop_size must contain exactly two integers.")
if args.delta_beta <= 0:
    raise ValueError("-delta_beta must be > 0.")
if args.gate_range <= 0:
    raise ValueError("-gate_range must be > 0.")

os.makedirs(args.exp_name, exist_ok=True)

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

net = GatedMultiScaleCoarseRefinedMambaFormer(
    in_channels=1,
    hidden_channels=args.hidden_channels,
    gate_hidden=args.gate_hidden,
    gate_range=args.gate_range,
)

load_baseline_weights(
    net,
    args.init_baseline,
    map_location="cpu",
)
net = net.to(device)

trainable_params = [p for p in net.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(
    trainable_params,
    lr=args.learning_rate,
)

total_params = sum(p.numel() for p in net.parameters())
backbone_params = sum(p.numel() for p in net.backbone.parameters())
trainable_count = sum(p.numel() for p in trainable_params)

print("\n--- Gated Multi-Coarse ---")
print(f"Total params:       {total_params:,}")
print(f"Backbone params:    {backbone_params:,}")
print(f"Trainable params:   {trainable_count:,}")
print(f"Trainable/backbone: {100.0 * trainable_count / backbone_params:.2f}%")
print(f"Gate range:         [{1-args.gate_range:.2f}, {1+args.gate_range:.2f}]")


# ---------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------

total_steps = 0
best_overall_psnr = -float("inf")

if args.resume:
    state = torch.load(args.resume, map_location=device)
    net.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    total_steps = int(state["step"])
    best_overall_psnr = float(
        state.get("best_overall_psnr", -float("inf"))
    )
    print(f"Resumed from step {total_steps}")


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Same targets/loss as fixed Multi-Coarse
# ---------------------------------------------------------------------

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

    target_detail2_low = target2_low - target4_to_2
    return target_detail2_low, target4_low


def compute_loss(outputs, gt):
    with torch.no_grad():
        target_detail2_low, target4_low = make_targets(
            gt,
            outputs["baseline"],
        )

    loss_s2 = F.smooth_l1_loss(
        outputs["pred_detail2_low"],
        target_detail2_low,
        beta=args.delta_beta,
    )

    loss_s4 = F.smooth_l1_loss(
        outputs["pred_coarse4_low"],
        target4_low,
        beta=args.delta_beta,
    )

    loss_final = F.smooth_l1_loss(
        outputs["refined"],
        gt,
        beta=args.delta_beta,
    )

    loss = (
        args.lambda_scale2 * loss_s2
        + args.lambda_scale4 * loss_s4
        + args.lambda_final * loss_final
    )

    return loss, loss_s2, loss_s4, loss_final


# ---------------------------------------------------------------------
# Validation: same full-image protocol, non-metal metrics for selection
# ---------------------------------------------------------------------

HU_MIN, HU_MAX = -1000.0, 3000.0
FNAME_ID_RE = re.compile(r"img(\d+)")
DIMS_RE = re.compile(r"(\d+)x(\d+)x(\d+)")


def parse_dims(filename):
    m = DIMS_RE.search(filename)
    if not m:
        raise ValueError(f"Cannot parse dimensions from {filename}")
    w, h, d = [int(x) for x in m.groups()]
    return h, w, d


def load_raw(path, dtype=np.float32):
    rows, cols, slices = parse_dims(os.path.basename(path))
    arr = np.fromfile(path, dtype=dtype)
    expected = rows * cols * slices
    if arr.size != expected:
        raise ValueError(
            f"{path}: got {arr.size}, expected {expected}"
        )
    if slices == 1:
        return arr.reshape(rows, cols)
    return arr.reshape(slices, rows, cols)


def hu_to_unit(img):
    img = np.clip(img, HU_MIN, HU_MAX)
    return ((img - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def find_samples(root):
    def index_one(anatomy_dir):
        anatomy = os.path.basename(os.path.normpath(anatomy_dir))
        bdir = os.path.join(anatomy_dir, "Baseline")
        tdir = os.path.join(anatomy_dir, "Target")
        mdir = os.path.join(anatomy_dir, "Mask")

        if not os.path.isdir(bdir) or not os.path.isdir(tdir):
            return []

        target_map = {}
        for f in glob.glob(os.path.join(tdir, "*.raw")):
            m = FNAME_ID_RE.search(os.path.basename(f))
            if m:
                target_map[m.group(1)] = f

        mask_map = {}
        count_map = {}

        if os.path.isdir(mdir):
            for f in glob.glob(os.path.join(mdir, "*.raw")):
                m = FNAME_ID_RE.search(os.path.basename(f))
                if m:
                    mask_map[m.group(1)] = f

            for f in glob.glob(os.path.join(mdir, "*.json")):
                m = re.search(r"metalinfo(\d+)", os.path.basename(f))
                if not m:
                    continue
                try:
                    with open(f, "r", encoding="utf-8") as jf:
                        count_map[m.group(1)] = json.load(jf).get(
                            "n_materials"
                        )
                except Exception:
                    pass

        samples = []
        for bf in glob.glob(os.path.join(bdir, "*.raw")):
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
                    "n_materials": count_map.get(image_id),
                }
            )
        return samples

    if os.path.isdir(os.path.join(root, "Baseline")):
        return index_one(root)

    samples = []
    for name in sorted(os.listdir(root)):
        sub = os.path.join(root, name)
        if (
            os.path.isdir(sub)
            and os.path.isdir(os.path.join(sub, "Baseline"))
        ):
            samples.extend(index_one(sub))
    return samples


def to_bgr_u8(img):
    u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)


def metrics_non_metal(pred, gt, mask):
    p = pred.copy()
    g = gt.copy()

    if mask is not None:
        p[mask] = 0.0
        g[mask] = 0.0

    p = to_bgr_u8(p)
    g = to_bgr_u8(g)

    return {
        "psnr": float(calculate_psnr(p, g, test_y_channel=True)),
        "ssim": float(calculate_ssim(p, g, test_y_channel=True)),
        "rmse": float(calculate_rmse(p, g)),
    }


def mean_std(records, key):
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


def gate_summary(records):
    if not records:
        return None

    out = {"n": len(records)}
    for key in ("alpha2", "alpha4"):
        arr = np.asarray([r[key] for r in records], dtype=np.float64)
        out[f"{key}_mean"] = float(arr.mean())
        out[f"{key}_std"] = float(arr.std())
        out[f"{key}_min"] = float(arr.min())
        out[f"{key}_max"] = float(arr.max())
    return out


def evaluate(model):
    samples = find_samples(args.val_data_dir)
    if args.max_val_samples is not None:
        samples = samples[: args.max_val_samples]

    records = []
    model.eval()

    with torch.inference_mode():
        for sample in samples:
            inp = hu_to_unit(load_raw(sample["baseline"]))
            gt = hu_to_unit(load_raw(sample["target"]))

            mask = None
            if sample["mask"] is not None:
                mask = load_raw(
                    sample["mask"],
                    dtype=np.float32,
                ) > 0.5

            inp_t = (
                torch.from_numpy((inp - 0.5) / 0.5)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
                .to(device)
            )

            refined_t, baseline_t, _, stats = model(
                inp_t,
                return_refinement=True,
            )

            refined = np.clip(
                refined_t.squeeze().cpu().numpy(),
                0.0,
                1.0,
            )
            baseline = np.clip(
                baseline_t.squeeze().cpu().numpy(),
                0.0,
                1.0,
            )

            n_materials = sample["n_materials"]
            if n_materials is not None:
                n_materials = int(n_materials)

            records.append(
                {
                    "id": sample["id"],
                    "n_materials": n_materials,
                    "baseline": metrics_non_metal(
                        baseline, gt, mask
                    ),
                    "refined": metrics_non_metal(
                        refined, gt, mask
                    ),
                    "alpha2": float(stats["alpha2_mean"].item()),
                    "alpha4": float(stats["alpha4_mean"].item()),
                }
            )

    result = {
        "overall": {
            "baseline": mean_std(records, "baseline"),
            "refined": mean_std(records, "refined"),
        },
        "gate_overall": gate_summary(records),
        "by_metal_count": {},
        "gate_by_metal_count": {},
    }

    for n in range(1, 6):
        group = [r for r in records if r["n_materials"] == n]
        result["by_metal_count"][str(n)] = {
            "baseline": mean_std(group, "baseline"),
            "refined": mean_std(group, "refined"),
        }
        result["gate_by_metal_count"][str(n)] = gate_summary(group)

    return result


def print_eval(step, result):
    b = result["overall"]["baseline"]
    r = result["overall"]["refined"]
    gate = result["gate_overall"]

    print("\n" + "=" * 110)
    print(f"VALIDATION @ {step}")
    print(
        f"Overall PSNR: {b['psnr']:.6f} -> {r['psnr']:.6f} "
        f"({r['psnr'] - b['psnr']:+.6f})"
    )
    print(
        f"Overall SSIM: {b['ssim']:.6f} -> {r['ssim']:.6f}"
    )
    print(
        f"Overall RMSE: {b['rmse']:.6f} -> {r['rmse']:.6f}"
    )
    print(
        f"Gate: a2={gate['alpha2_mean']:.4f}±{gate['alpha2_std']:.4f} "
        f"[{gate['alpha2_min']:.4f},{gate['alpha2_max']:.4f}] | "
        f"a4={gate['alpha4_mean']:.4f}±{gate['alpha4_std']:.4f} "
        f"[{gate['alpha4_min']:.4f},{gate['alpha4_max']:.4f}]"
    )

    for n in range(1, 6):
        pair = result["by_metal_count"][str(n)]
        if pair["refined"] is None:
            continue
        print(
            f"{n} metal: "
            f"{pair['baseline']['psnr']:.4f} -> "
            f"{pair['refined']['psnr']:.4f}"
        )


def save_state(step):
    torch.save(
        net.state_dict(),
        os.path.join(args.exp_name, f"{step}_ckpt"),
    )

    torch.save(
        {
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_overall_psnr": best_overall_psnr,
            "args": vars(args),
        },
        os.path.join(args.exp_name, f"{step}_train_state.pt"),
    )


history = []


# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------

while total_steps < args.num_steps:
    for batch in train_loader:
        if total_steps >= args.num_steps:
            break

        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise RuntimeError(
                "AAPMTrainDataset must return at least (input, gt)."
            )

        inp, gt = batch[:2]
        inp = inp.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        net.train()
        optimizer.zero_grad(set_to_none=True)

        outputs = net(inp, return_aux=True)
        loss, loss_s2, loss_s4, loss_final = compute_loss(
            outputs, gt
        )

        loss.backward()

        grad_sq = 0.0
        for p in trainable_params:
            if p.grad is not None:
                g = p.grad.detach().norm().item()
                grad_sq += g * g

        optimizer.step()
        total_steps += 1

        if total_steps == 1 or total_steps % 10 == 0:
            a2 = outputs["alpha2"].detach().flatten()
            a4 = outputs["alpha4"].detach().flatten()

            print(
                f"step={total_steps:6d} "
                f"loss={loss.item():.7f} "
                f"s2={loss_s2.item():.7f} "
                f"s4={loss_s4.item():.7f} "
                f"final={loss_final.item():.7f} "
                f"a2={a2.mean().item():.3f}±{a2.std(unbiased=False).item():.3f} "
                f"a4={a4.mean().item():.3f}±{a4.std(unbiased=False).item():.3f} "
                f"grad={math.sqrt(grad_sq):.6e}"
            )

        if total_steps % args.eval_step == 0:
            t0 = time.time()
            result = evaluate(net)
            print_eval(total_steps, result)
            print(f"Validation time: {time.time()-t0:.1f}s")

            history.append(
                {
                    "step": total_steps,
                    **result,
                }
            )

            with open(
                os.path.join(args.exp_name, "checkpoint_result.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

            cur_psnr = result["overall"]["refined"]["psnr"]

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
                            "result": result,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )

                print(
                    f"NEW BEST: {best_overall_psnr:.6f} "
                    f"@ step {total_steps}"
                )

        if total_steps % args.save_step == 0:
            save_state(total_steps)

print("Training finished.")

"""
Direct-delta supervision pilot for Output-Space Coarse Residual Refinement.

What changes compared with the previous pilot?
------------------------------------------------
ARCHITECTURE: unchanged.

Previous training:
    final = baseline + predicted_delta
    loss(final, GT)

This pilot:
    oracle coarse target =
        UpNearest(AvgPool2x2(GT - baseline))

    loss =
        SmoothL1(predicted_delta, oracle_coarse_target)

So the refinement head is explicitly taught WHAT coarse correction it should
predict instead of receiving only an indirect final-image restoration loss.

Important:
- Baseline MARMamba is frozen.
- No new dataset or preprocessing files are required.
- No mask, metal count, sinogram, or GT is needed at inference.
- GT is used only during training, as usual.
- This is a diagnostic pilot, not yet the final paper training protocol.

Validation reports baseline -> refined on the SAME samples, so there is no
need to remember baseline numbers from an older run.
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
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvu
from torch.utils.data import DataLoader

from model.mamba_coarse_refine import (
    CoarseRefinedMambaFormer,
    load_baseline_weights,
)
from utils.aapm_dataset import AAPMTrainDataset
from utils.metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_rmse,
)


# =========================================================================
# Arguments
# =========================================================================

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("true", "1", "yes", "y"):
        return True
    if v in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser(
    description="Direct-delta coarse-refinement pilot"
)

parser.add_argument("-learning_rate", default=1e-4, type=float)
parser.add_argument("-crop_size", default=[256, 256], nargs="+", type=int)
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
parser.add_argument("-hidden_channels", default=16, type=int)

# SmoothL1 transition point.
# Residual values are in normalized image space [roughly -1, 1].
# 0.01 keeps the loss quadratic very close to zero but L1-like for larger
# correction errors. It is configurable so this choice is explicit.
parser.add_argument("-delta_beta", default=0.01, type=float)

parser.add_argument(
    "-max_val_samples",
    default=None,
    type=int,
    help="Optional subset for a very fast pilot; default evaluates all val cases.",
)

args = parser.parse_args()

if len(args.crop_size) != 2:
    raise ValueError("-crop_size must contain exactly two integers.")

if args.delta_beta <= 0:
    raise ValueError("-delta_beta must be > 0.")


# =========================================================================
# Reproducibility / device
# =========================================================================

os.makedirs(args.exp_name, exist_ok=True)
train_res_dir = os.path.join(args.exp_name, "train_res")
os.makedirs(train_res_dir, exist_ok=True)

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

# Keep this pilot single-device. It avoids DataParallel tuple/dict gathering
# surprises and matches the single-GPU experimental setting.
print(f"Device: {device}")


# =========================================================================
# Model
# =========================================================================

net = CoarseRefinedMambaFormer(
    in_channels=1,
    hidden_channels=args.hidden_channels,
)

load_baseline_weights(
    net,
    args.init_baseline,
    map_location="cpu",
)

# This experiment explicitly freezes the baseline.
for p in net.backbone.parameters():
    p.requires_grad = False

for p in net.refinement.parameters():
    p.requires_grad = True

net = net.to(device)

total_params = sum(p.numel() for p in net.parameters())
backbone_params = sum(p.numel() for p in net.backbone.parameters())
refiner_params = sum(p.numel() for p in net.refinement.parameters())
trainable_params = sum(
    p.numel() for p in net.parameters() if p.requires_grad
)

print("\n--- Direct-Delta Coarse Refinement Pilot ---")
print(f"Total parameters:      {total_params:,}")
print(f"Baseline parameters:   {backbone_params:,}")
print(f"Refinement parameters: {refiner_params:,}")
print(
    f"Parameter overhead:    "
    f"{100.0 * refiner_params / backbone_params:.2f}%"
)
print(f"Trainable parameters:  {trainable_params:,}")
print(f"Learning rate:         {args.learning_rate}")
print(f"SmoothL1 beta:         {args.delta_beta}")
print(f"Target steps:          {args.num_steps}")
print(f"Validate every:        {args.eval_step}")
print(f"Validation dir:        {args.val_data_dir}")


optimizer = torch.optim.Adam(
    net.refinement.parameters(),
    lr=args.learning_rate,
)


# =========================================================================
# Resume
# =========================================================================

total_steps = 0
best_multi_psnr = -float("inf")

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
    best_multi_psnr = float(
        state.get("best_multi_psnr", -float("inf"))
    )

    print(f"Resumed from step {total_steps}.")


# =========================================================================
# Train dataset
# =========================================================================

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


# =========================================================================
# Direct-delta target
# =========================================================================

def make_coarse_delta_target(
    gt: torch.Tensor,
    baseline_pred: torch.Tensor,
) -> torch.Tensor:
    """
    Construct exactly the scale-2 coarse correction used by the oracle test:

        residual = GT - baseline
        coarse   = AvgPool2d(residual, 2, 2)
        target   = nearest-upsample(coarse)

    The model's refinement head also outputs a nearest-upsampled scale-2
    correction, so supervising at full resolution is mathematically
    equivalent to supervising each coarse cell before nearest upsampling.
    """
    if gt.shape != baseline_pred.shape:
        raise ValueError(
            f"GT/baseline shape mismatch: {gt.shape} vs {baseline_pred.shape}"
        )

    residual = gt - baseline_pred

    coarse = F.avg_pool2d(
        residual,
        kernel_size=2,
        stride=2,
    )

    target = F.interpolate(
        coarse,
        size=baseline_pred.shape[-2:],
        mode="nearest",
    )

    return target


def delta_diagnostics(
    pred_delta: torch.Tensor,
    target_delta: torch.Tensor,
):
    """Small diagnostics to tell WHY the pilot succeeds/fails."""
    with torch.no_grad():
        p = pred_delta.detach().float().reshape(pred_delta.shape[0], -1)
        t = target_delta.detach().float().reshape(target_delta.shape[0], -1)

        p_norm = torch.linalg.vector_norm(p, dim=1)
        t_norm = torch.linalg.vector_norm(t, dim=1)

        cos = torch.sum(p * t, dim=1) / (
            p_norm * t_norm + 1e-12
        )

        mag_ratio = p_norm / (t_norm + 1e-12)

        mae = torch.mean(
            torch.abs(pred_delta.detach() - target_delta.detach())
        )

        return {
            "cosine": float(cos.mean().item()),
            "mag_ratio": float(mag_ratio.mean().item()),
            "mae": float(mae.item()),
            "pred_abs": float(pred_delta.detach().abs().mean().item()),
            "target_abs": float(target_delta.detach().abs().mean().item()),
        }


# =========================================================================
# AAPM validation indexing
# =========================================================================

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
        raise ValueError(
            f"{path}: found {arr.size}, expected {expected}"
        )

    if slices == 1:
        return arr.reshape(rows, cols)

    return arr.reshape(slices, rows, cols)


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
                m = re.search(
                    r"metalinfo(\d+)",
                    os.path.basename(f),
                )
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


val_samples = find_eval_samples(args.val_data_dir)

if not val_samples:
    raise RuntimeError(
        f"No validation samples found in {args.val_data_dir}"
    )

val_samples = sorted(val_samples, key=lambda x: x["id"])

if args.max_val_samples is not None:
    val_samples = val_samples[:args.max_val_samples]

missing_n = [
    s["id"] for s in val_samples if s["n_materials"] is None
]

if missing_n:
    raise RuntimeError(
        "Validation metadata is missing n_materials for "
        f"{len(missing_n)} cases. First examples: {missing_n[:5]}"
    )

count_hist = {i: 0 for i in range(1, 6)}

for s in val_samples:
    n = int(s["n_materials"])
    if n in count_hist:
        count_hist[n] += 1

print(f"Validation cases:       {len(val_samples)}")
print(f"Validation metal histogram: {count_hist}")


# =========================================================================
# Validation metrics
# =========================================================================

def to_uint8_bgr3(img_float01):
    img_u8 = np.clip(
        img_float01 * 255.0,
        0,
        255,
    ).astype(np.uint8)

    return cv2.cvtColor(
        img_u8,
        cv2.COLOR_GRAY2BGR,
    )


def metric_one(pred, gt, mask):
    pred_eval = pred.copy()
    gt_eval = gt.copy()

    # Same non-metal protocol used in previous AAPM evaluation.
    if mask is not None:
        pred_eval[mask] = 0.0
        gt_eval[mask] = 0.0

    pred_bgr = to_uint8_bgr3(pred_eval)
    gt_bgr = to_uint8_bgr3(gt_eval)

    return {
        "psnr": float(
            calculate_psnr(
                pred_bgr,
                gt_bgr,
                test_y_channel=True,
            )
        ),
        "ssim": float(
            calculate_ssim(
                pred_bgr,
                gt_bgr,
                test_y_channel=True,
            )
        ),
        "rmse": float(
            calculate_rmse(
                pred_bgr,
                gt_bgr,
            )
        ),
    }


def summarize_metric_records(records, key):
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


def summarize_diag_records(records):
    if not records:
        return None

    out = {}

    for metric in (
        "cosine",
        "mag_ratio",
        "mae",
        "pred_abs",
        "target_abs",
    ):
        arr = np.asarray(
            [r["diag"][metric] for r in records],
            dtype=np.float64,
        )
        out[metric] = float(arr.mean())

    return out


@torch.no_grad()
def evaluate_validation(model):
    model.eval()

    records = []
    start = time.time()

    for sample in val_samples:
        input_img = hu_to_unit(load_raw(sample["baseline"]))
        gt_img = hu_to_unit(load_raw(sample["target"]))

        mask = None
        if sample["mask"] is not None:
            mask = (
                load_raw(sample["mask"], dtype=np.float32)
                > 0.5
            )

        input_t = (
            torch.from_numpy((input_img - 0.5) / 0.5)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(device)
        )

        gt_t = (
            torch.from_numpy(gt_img)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(device)
        )

        refined_t, baseline_t, pred_delta_t, _ = model(
            input_t,
            return_refinement=True,
        )

        target_delta_t = make_coarse_delta_target(
            gt_t,
            baseline_t,
        )

        diag = delta_diagnostics(
            pred_delta_t,
            target_delta_t,
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

        n_materials = int(sample["n_materials"])

        records.append(
            {
                "id": sample["id"],
                "n_materials": n_materials,
                "group": "multi" if n_materials >= 2 else "single",
                "baseline": metric_one(
                    baseline,
                    gt_img,
                    mask,
                ),
                "refined": metric_one(
                    refined,
                    gt_img,
                    mask,
                ),
                "diag": diag,
            }
        )

    groups = {
        "overall": records,
        "single": [
            r for r in records if r["group"] == "single"
        ],
        "multi": [
            r for r in records if r["group"] == "multi"
        ],
    }

    result = {
        "by_metal_count": {},
        "seconds": time.time() - start,
    }

    for group_name, group_records in groups.items():
        result[group_name] = {
            "baseline": summarize_metric_records(
                group_records,
                "baseline",
            ),
            "refined": summarize_metric_records(
                group_records,
                "refined",
            ),
            "delta_diag": summarize_diag_records(
                group_records
            ),
        }

    for n in range(1, 6):
        group_records = [
            r for r in records if r["n_materials"] == n
        ]

        result["by_metal_count"][str(n)] = {
            "baseline": summarize_metric_records(
                group_records,
                "baseline",
            ),
            "refined": summarize_metric_records(
                group_records,
                "refined",
            ),
            "delta_diag": summarize_diag_records(
                group_records
            ),
        }

    return result


def print_one_group(label, group):
    if (
        group is None
        or group["baseline"] is None
        or group["refined"] is None
    ):
        return

    b = group["baseline"]
    r = group["refined"]
    d = group["delta_diag"]

    print(
        f"{label:>8} | n={r['n']:3d} | "
        f"PSNR {b['psnr']:.3f} -> {r['psnr']:.3f} "
        f"({r['psnr'] - b['psnr']:+.3f}) | "
        f"SSIM {b['ssim']:.4f} -> {r['ssim']:.4f} "
        f"({r['ssim'] - b['ssim']:+.4f}) | "
        f"RMSE {b['rmse']:.3f} -> {r['rmse']:.3f} "
        f"({r['rmse'] - b['rmse']:+.3f}) | "
        f"cos={d['cosine']:+.3f} | "
        f"mag={d['mag_ratio']:.3f}"
    )


def print_validation(step, result):
    print("\n" + "=" * 130)
    print(f"DIRECT-DELTA VALIDATION @ STEP {step}")
    print("=" * 130)

    for n in range(1, 6):
        print_one_group(
            f"{n} metal",
            result["by_metal_count"][str(n)],
        )

    print("-" * 130)

    for group_name in ("single", "multi", "overall"):
        print_one_group(
            group_name,
            result[group_name],
        )

    print(
        f"Validation time: {result['seconds']:.1f}s"
    )


# =========================================================================
# Checkpoint save
# =========================================================================

def save_checkpoint(step):
    raw_path = os.path.join(
        args.exp_name,
        f"{step}_ckpt",
    )

    state_path = os.path.join(
        args.exp_name,
        f"{step}_train_state.pt",
    )

    torch.save(
        net.state_dict(),
        raw_path,
    )

    torch.save(
        {
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_multi_psnr": best_multi_psnr,
            "args": vars(args),
        },
        state_path,
    )

    return raw_path, state_path


# =========================================================================
# Train
# =========================================================================

while total_steps < args.num_steps:
    for train_data in train_loader:
        if total_steps >= args.num_steps:
            break

        if (
            not isinstance(train_data, (tuple, list))
            or len(train_data) < 2
        ):
            raise RuntimeError(
                "AAPMTrainDataset must return at least (input_image, gt)."
            )

        input_image, gt = train_data[:2]

        input_image = input_image.to(
            device,
            non_blocking=True,
        )
        gt = gt.to(
            device,
            non_blocking=True,
        )

        net.train()
        # Baseline is frozen and should remain deterministic.
        net.backbone.eval()

        optimizer.zero_grad(set_to_none=True)

        refined, baseline_pred, pred_delta, _ = net(
            input_image,
            return_refinement=True,
        )

        # GT is used ONLY to construct the training target.
        # Detach target so no gradient can flow through baseline/GT target path.
        with torch.no_grad():
            target_delta = make_coarse_delta_target(
                gt,
                baseline_pred,
            )

        loss = F.smooth_l1_loss(
            pred_delta,
            target_delta,
            beta=args.delta_beta,
            reduction="mean",
        )

        loss.backward()

        grad_norm_sq = 0.0
        for p in net.refinement.parameters():
            if p.grad is not None:
                g = float(p.grad.detach().norm().item())
                grad_norm_sq += g * g

        grad_norm = math.sqrt(grad_norm_sq)

        optimizer.step()

        total_steps += 1

        if total_steps % 10 == 0 or total_steps == 1:
            diag = delta_diagnostics(
                pred_delta,
                target_delta,
            )

            print(
                f"step={total_steps:6d} "
                f"delta_loss={loss.item():.7f} "
                f"cos={diag['cosine']:+.4f} "
                f"mag={diag['mag_ratio']:.4f} "
                f"pred_abs={diag['pred_abs']:.6e} "
                f"target_abs={diag['target_abs']:.6e} "
                f"mae={diag['mae']:.6e} "
                f"grad={grad_norm:.6e}"
            )

        if total_steps % 100 == 0:
            with torch.no_grad():
                tvu.save_image(
                    input_image * 0.5 + 0.5,
                    os.path.join(train_res_dir, "input.png"),
                )
                tvu.save_image(
                    baseline_pred,
                    os.path.join(train_res_dir, "baseline.png"),
                )
                tvu.save_image(
                    refined,
                    os.path.join(train_res_dir, "refined.png"),
                )
                tvu.save_image(
                    gt,
                    os.path.join(train_res_dir, "gt.png"),
                )

                # Signed maps: roughly [-0.1, 0.1] -> [0, 1].
                pred_vis = torch.clamp(
                    pred_delta / 0.2 + 0.5,
                    0.0,
                    1.0,
                )
                target_vis = torch.clamp(
                    target_delta / 0.2 + 0.5,
                    0.0,
                    1.0,
                )

                tvu.save_image(
                    pred_vis,
                    os.path.join(train_res_dir, "pred_delta.png"),
                )
                tvu.save_image(
                    target_vis,
                    os.path.join(train_res_dir, "target_delta.png"),
                )

        # -----------------------------------------------------------------
        # Validation
        # -----------------------------------------------------------------
        if total_steps % args.eval_step == 0:
            val_result = evaluate_validation(net)
            print_validation(total_steps, val_result)

            validation_path = os.path.join(
                args.exp_name,
                "validation.jsonl",
            )

            with open(
                validation_path,
                "a",
                encoding="utf-8",
            ) as f:
                f.write(
                    json.dumps(
                        {
                            "step": total_steps,
                            **val_result,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            current_multi_psnr = (
                val_result["multi"]["refined"]["psnr"]
            )

            if current_multi_psnr > best_multi_psnr:
                best_multi_psnr = current_multi_psnr

                best_path = os.path.join(
                    args.exp_name,
                    "best_multi_psnr_ckpt",
                )

                torch.save(
                    net.state_dict(),
                    best_path,
                )

                print(
                    f"New best validation Multi PSNR: "
                    f"{best_multi_psnr:.4f} dB"
                )

            # Restore train mode after validation, keeping backbone frozen/eval.
            net.train()
            net.backbone.eval()

        # -----------------------------------------------------------------
        # Periodic checkpoint
        # -----------------------------------------------------------------
        if total_steps % args.save_step == 0:
            raw_path, state_path = save_checkpoint(total_steps)
            print(f"Saved weights:     {raw_path}")
            print(f"Saved train state: {state_path}")


print("Direct-delta pilot finished.")

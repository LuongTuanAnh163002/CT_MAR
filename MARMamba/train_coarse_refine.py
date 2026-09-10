"""
Pilot training for Output-Space Coarse Residual Refinement.

Training:
- load best baseline MARMamba checkpoint;
- freeze the baseline;
- train ONLY the output-space refinement head;
- keep the original restoration loss:
      0.8 * Pseudo-Huber + 0.2 * LPIPS

Validation:
- evaluate on val_data_dir every eval_step;
- use the same non-metal protocol:
      prediction[metal_mask] = 0
      GT[metal_mask] = 0
- report:
      overall
      single / multi
      metal count 1..5
- save best_multi_psnr_ckpt according to validation Multi-metal PSNR.

The test set is NOT used here. Use test/eval_aapm_coarse_refine.py only for
the final evaluation.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import time

import cv2
import lpips
import numpy as np
import torch
import torch.nn as nn
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


# -------------------------------------------------------------------------
# Args
# -------------------------------------------------------------------------

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("true", "1", "yes", "y"):
        return True
    if v in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(
        "Boolean value expected."
    )


parser = argparse.ArgumentParser(
    description="Output-space coarse-refinement pilot"
)

parser.add_argument(
    "-learning_rate",
    default=1e-4,
    type=float,
)
parser.add_argument(
    "-crop_size",
    default=[256, 256],
    nargs="+",
    type=int,
)
parser.add_argument(
    "-train_batch_size",
    default=8,
    type=int,
)
parser.add_argument(
    "-exp_name",
    type=str,
    required=True,
)
parser.add_argument(
    "-seed",
    default=19,
    type=int,
)
parser.add_argument(
    "-num_steps",
    default=5000,
    type=int,
)
parser.add_argument(
    "-save_step",
    default=500,
    type=int,
)
parser.add_argument(
    "-eval_step",
    default=500,
    type=int,
)
parser.add_argument(
    "-train_data_dir",
    type=str,
    required=True,
)
parser.add_argument(
    "-val_data_dir",
    type=str,
    required=True,
)
parser.add_argument(
    "-init_baseline",
    type=str,
    required=True,
)
parser.add_argument(
    "-resume",
    type=str,
    default=None,
)
parser.add_argument(
    "-freeze_backbone",
    default=True,
    type=str2bool,
)
parser.add_argument(
    "-num_workers",
    default=0,
    type=int,
)
parser.add_argument(
    "-hidden_channels",
    default=16,
    type=int,
)
parser.add_argument(
    "-max_val_samples",
    default=None,
    type=int,
    help=(
        "Optional fast-pilot subset. "
        "Leave empty to evaluate the full validation set."
    ),
)

args = parser.parse_args()


# -------------------------------------------------------------------------
# Reproducibility / dirs
# -------------------------------------------------------------------------

os.makedirs(
    args.exp_name,
    exist_ok=True,
)

train_res_dir = os.path.join(
    args.exp_name,
    "train_res",
)
os.makedirs(
    train_res_dir,
    exist_ok=True,
)

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

device = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

device_ids = list(
    range(torch.cuda.device_count())
)


# -------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------

net = CoarseRefinedMambaFormer(
    in_channels=1,
    hidden_channels=args.hidden_channels,
)

load_baseline_weights(
    net,
    args.init_baseline,
    map_location="cpu",
)

if args.freeze_backbone:
    for p in net.backbone.parameters():
        p.requires_grad = False

for p in net.refinement.parameters():
    p.requires_grad = True


total_params = sum(
    p.numel()
    for p in net.parameters()
)

backbone_params = sum(
    p.numel()
    for p in net.backbone.parameters()
)

refiner_params = sum(
    p.numel()
    for p in net.refinement.parameters()
)

trainable_params = sum(
    p.numel()
    for p in net.parameters()
    if p.requires_grad
)


print("--- Output-Space Coarse Refinement Pilot ---")
print(f"Total parameters:      {total_params:,}")
print(f"Baseline parameters:   {backbone_params:,}")
print(f"Refinement parameters: {refiner_params:,}")
print(
    f"Parameter overhead:    "
    f"{100.0 * refiner_params / backbone_params:.2f}%"
)
print(f"Trainable parameters:  {trainable_params:,}")
print(f"Freeze backbone:       {args.freeze_backbone}")
print(f"Learning rate:         {args.learning_rate}")
print(f"Target steps:          {args.num_steps}")
print(f"Save every:            {args.save_step}")
print(f"Validate every:        {args.eval_step}")
print(f"Validation dir:        {args.val_data_dir}")

net = net.to(device)

if len(device_ids) > 1:
    net = nn.DataParallel(
        net,
        device_ids=device_ids,
    )


def unwrap(model):
    return (
        model.module
        if isinstance(model, nn.DataParallel)
        else model
    )


optimizer = torch.optim.Adam(
    [
        p
        for p in net.parameters()
        if p.requires_grad
    ],
    lr=args.learning_rate,
)


# -------------------------------------------------------------------------
# Resume
# -------------------------------------------------------------------------

total_steps = 0
best_multi_psnr = -float("inf")

if args.resume is not None:
    state = torch.load(
        args.resume,
        map_location=device,
    )

    required = {
        "model",
        "optimizer",
        "step",
    }

    if not required.issubset(state):
        raise RuntimeError(
            "-resume must point to *_train_state.pt "
            "created by this script."
        )

    model_state = state["model"]

    if any(
        k.startswith("module.")
        for k in model_state
    ):
        model_state = {
            (
                k[len("module."):]
                if k.startswith("module.")
                else k
            ): v
            for k, v in model_state.items()
        }

    unwrap(net).load_state_dict(
        model_state,
        strict=True,
    )

    optimizer.load_state_dict(
        state["optimizer"]
    )

    total_steps = int(
        state["step"]
    )

    best_multi_psnr = float(
        state.get(
            "best_multi_psnr",
            -float("inf"),
        )
    )

    print(
        f"Resumed from step {total_steps}."
    )


# -------------------------------------------------------------------------
# Train dataset
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
# Original baseline loss
# -------------------------------------------------------------------------

def hub_loss(img, gt):
    c = 0.03
    diff = torch.sqrt(
        torch.pow(
            img - gt,
            2,
        )
        + c ** 2
    )
    return (
        diff - c
    ).mean()


lpips_loss = lpips.LPIPS(
    net="vgg",
    spatial=False,
).to(device)

lpips_loss.eval()

for p in lpips_loss.parameters():
    p.requires_grad = False


# -------------------------------------------------------------------------
# AAPM validation indexing
# -------------------------------------------------------------------------

HU_MIN, HU_MAX = -1000.0, 3000.0
FNAME_ID_RE = re.compile(r"img(\d+)")
DIMS_RE = re.compile(r"(\d+)x(\d+)x(\d+)")


def parse_dims(filename):
    m = DIMS_RE.search(filename)

    if not m:
        raise ValueError(
            f"Cannot parse dimensions from: {filename}"
        )

    w, h, d = (
        int(x)
        for x in m.groups()
    )

    return h, w, d


def load_raw(
    path,
    dtype=np.float32,
):
    rows, cols, slices = parse_dims(
        os.path.basename(path)
    )

    arr = np.fromfile(
        path,
        dtype=dtype,
    )

    expected = (
        rows * cols * slices
    )

    if arr.size != expected:
        raise ValueError(
            f"{path}: found {arr.size}, "
            f"expected {expected}"
        )

    if slices == 1:
        return arr.reshape(
            rows,
            cols,
        )

    return arr.reshape(
        slices,
        rows,
        cols,
    )


def hu_to_unit(img):
    img = np.clip(
        img,
        HU_MIN,
        HU_MAX,
    )

    return (
        (img - HU_MIN)
        / (HU_MAX - HU_MIN)
    ).astype(np.float32)


def find_eval_samples(root_dir):
    def index_one(anatomy_dir):
        anatomy = os.path.basename(
            os.path.normpath(anatomy_dir)
        )

        baseline_dir = os.path.join(
            anatomy_dir,
            "Baseline",
        )

        target_dir = os.path.join(
            anatomy_dir,
            "Target",
        )

        mask_dir = os.path.join(
            anatomy_dir,
            "Mask",
        )

        if not (
            os.path.isdir(baseline_dir)
            and os.path.isdir(target_dir)
        ):
            return []

        target_map = {}

        for f in glob.glob(
            os.path.join(
                target_dir,
                "*.raw",
            )
        ):
            m = FNAME_ID_RE.search(
                os.path.basename(f)
            )

            if m:
                target_map[
                    m.group(1)
                ] = f

        mask_map = {}
        n_materials_map = {}

        if os.path.isdir(mask_dir):
            for f in glob.glob(
                os.path.join(
                    mask_dir,
                    "*.raw",
                )
            ):
                m = FNAME_ID_RE.search(
                    os.path.basename(f)
                )

                if m:
                    mask_map[
                        m.group(1)
                    ] = f

            for f in glob.glob(
                os.path.join(
                    mask_dir,
                    "*.json",
                )
            ):
                m = re.search(
                    r"metalinfo(\d+)",
                    os.path.basename(f),
                )

                if not m:
                    continue

                try:
                    with open(
                        f,
                        "r",
                        encoding="utf-8",
                    ) as jf:
                        n_materials_map[
                            m.group(1)
                        ] = json.load(
                            jf
                        ).get(
                            "n_materials"
                        )
                except Exception:
                    pass

        samples = []

        for bf in glob.glob(
            os.path.join(
                baseline_dir,
                "*.raw",
            )
        ):
            m = FNAME_ID_RE.search(
                os.path.basename(bf)
            )

            if not m:
                continue

            image_id = m.group(1)

            if image_id not in target_map:
                continue

            samples.append(
                {
                    "id": (
                        f"{anatomy}_{image_id}"
                    ),
                    "baseline": bf,
                    "target": target_map[
                        image_id
                    ],
                    "mask": mask_map.get(
                        image_id
                    ),
                    "n_materials": (
                        n_materials_map.get(
                            image_id
                        )
                    ),
                }
            )

        return samples

    if os.path.isdir(
        os.path.join(
            root_dir,
            "Baseline",
        )
    ):
        return index_one(
            root_dir
        )

    samples = []

    for name in sorted(
        os.listdir(
            root_dir
        )
    ):
        sub = os.path.join(
            root_dir,
            name,
        )

        if (
            os.path.isdir(sub)
            and os.path.isdir(
                os.path.join(
                    sub,
                    "Baseline",
                )
            )
        ):
            samples.extend(
                index_one(sub)
            )

    return samples


val_samples = find_eval_samples(
    args.val_data_dir
)

if not val_samples:
    raise RuntimeError(
        f"No validation samples found in "
        f"{args.val_data_dir}"
    )

val_samples = sorted(
    val_samples,
    key=lambda x: x["id"],
)

if args.max_val_samples is not None:
    val_samples = val_samples[
        :args.max_val_samples
    ]

print(
    f"Validation cases:       {len(val_samples)}"
)

# Explicitly verify n_materials rather than silently labeling unknown cases.
missing_n = [
    s["id"]
    for s in val_samples
    if s["n_materials"] is None
]

if missing_n:
    raise RuntimeError(
        "Validation metadata is missing n_materials "
        f"for {len(missing_n)} cases. "
        f"First examples: {missing_n[:5]}"
    )

count_hist = {
    i: 0
    for i in range(1, 6)
}

for s in val_samples:
    n = int(
        s["n_materials"]
    )

    if n in count_hist:
        count_hist[n] += 1

print(
    f"Validation metal histogram: {count_hist}"
)


# -------------------------------------------------------------------------
# Validation metrics
# -------------------------------------------------------------------------

def to_uint8_bgr3(
    img_float01,
):
    img_u8 = np.clip(
        img_float01 * 255.0,
        0,
        255,
    ).astype(np.uint8)

    return cv2.cvtColor(
        img_u8,
        cv2.COLOR_GRAY2BGR,
    )


def metric_one(
    pred,
    gt,
    mask,
):
    pred_eval = pred.copy()
    gt_eval = gt.copy()

    # Same non-metal protocol as the test evaluator.
    if mask is not None:
        pred_eval[mask] = 0.0
        gt_eval[mask] = 0.0

    pred_bgr = to_uint8_bgr3(
        pred_eval
    )

    gt_bgr = to_uint8_bgr3(
        gt_eval
    )

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


def summarize_metrics(items):
    if not items:
        return None

    result = {
        "n": len(items),
    }

    for key in (
        "psnr",
        "ssim",
        "rmse",
    ):
        arr = np.asarray(
            [
                x[key]
                for x in items
            ],
            dtype=np.float64,
        )

        result[key] = float(
            arr.mean()
        )

        result[
            f"{key}_std"
        ] = float(
            arr.std()
        )

    return result


@torch.no_grad()
def evaluate_validation(model):
    was_training = model.training
    model.eval()

    by_count = {
        i: []
        for i in range(1, 6)
    }

    single = []
    multi = []
    overall = []

    start = time.time()

    for sample in val_samples:
        input_img = hu_to_unit(
            load_raw(
                sample["baseline"]
            )
        )

        gt_img = hu_to_unit(
            load_raw(
                sample["target"]
            )
        )

        mask = None

        if sample["mask"] is not None:
            mask = (
                load_raw(
                    sample["mask"],
                    dtype=np.float32,
                )
                > 0.5
            )

        input_t = (
            torch.from_numpy(
                (input_img - 0.5)
                / 0.5
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(device)
        )

        pred = (
            model(input_t)
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        pred = np.clip(
            pred,
            0.0,
            1.0,
        )

        metrics = metric_one(
            pred,
            gt_img,
            mask,
        )

        n_materials = int(
            sample["n_materials"]
        )

        overall.append(
            metrics
        )

        if n_materials in by_count:
            by_count[
                n_materials
            ].append(
                metrics
            )

        if n_materials >= 2:
            multi.append(
                metrics
            )
        else:
            single.append(
                metrics
            )

    result = {
        "overall": summarize_metrics(
            overall
        ),
        "single": summarize_metrics(
            single
        ),
        "multi": summarize_metrics(
            multi
        ),
        "by_metal_count": {
            str(i): summarize_metrics(
                by_count[i]
            )
            for i in range(1, 6)
        },
        "seconds": (
            time.time()
            - start
        ),
    }

    if was_training:
        model.train()

        if args.freeze_backbone:
            unwrap(
                model
            ).backbone.eval()

    return result


def print_validation(
    step,
    result,
):
    print(
        "\n"
        + "=" * 95
    )
    print(
        f"VALIDATION @ STEP {step}"
    )
    print(
        "=" * 95
    )

    for n in range(1, 6):
        r = result[
            "by_metal_count"
        ][str(n)]

        if r is None:
            continue

        print(
            f"{n} metal | "
            f"n={r['n']:3d} | "
            f"PSNR={r['psnr']:.3f} | "
            f"SSIM={r['ssim']:.4f} | "
            f"RMSE={r['rmse']:.3f}"
        )

    print(
        "-" * 95
    )

    for group in (
        "single",
        "multi",
        "overall",
    ):
        r = result[group]

        if r is None:
            continue

        print(
            f"{group:>7} | "
            f"n={r['n']:3d} | "
            f"PSNR={r['psnr']:.3f} | "
            f"SSIM={r['ssim']:.4f} | "
            f"RMSE={r['rmse']:.3f}"
        )

    print(
        f"Validation time: "
        f"{result['seconds']:.1f}s"
    )


# -------------------------------------------------------------------------
# Save
# -------------------------------------------------------------------------

def save_checkpoint(step):
    model = unwrap(net)

    raw_path = os.path.join(
        args.exp_name,
        f"{step}_ckpt",
    )

    state_path = os.path.join(
        args.exp_name,
        f"{step}_train_state.pt",
    )

    torch.save(
        model.state_dict(),
        raw_path,
    )

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_multi_psnr": best_multi_psnr,
            "args": vars(args),
        },
        state_path,
    )

    return raw_path, state_path


# -------------------------------------------------------------------------
# Train
# -------------------------------------------------------------------------

while total_steps < args.num_steps:
    for train_data in train_loader:
        if total_steps >= args.num_steps:
            break

        if (
            not isinstance(
                train_data,
                (tuple, list),
            )
            or len(train_data) < 2
        ):
            raise RuntimeError(
                "AAPMTrainDataset must return "
                "(input_image, gt)."
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

        optimizer.zero_grad(
            set_to_none=True,
        )

        net.train()

        # Frozen baseline should remain deterministic.
        if args.freeze_backbone:
            unwrap(
                net
            ).backbone.eval()

        pred, baseline_pred, delta, stats = net(
            input_image,
            return_refinement=True,
        )

        loss_h = hub_loss(
            pred,
            gt,
        )

        loss_p = lpips_loss(
            pred,
            gt,
        ).mean()

        loss = (
            0.8 * loss_h
            + 0.2 * loss_p
        )

        loss.backward()

        grad_norm = 0.0

        for p in unwrap(
            net
        ).refinement.parameters():
            if p.grad is not None:
                grad_norm += float(
                    p.grad.detach()
                    .norm()
                    .item() ** 2
                )

        grad_norm = (
            grad_norm ** 0.5
        )

        optimizer.step()

        total_steps += 1

        if (
            total_steps % 10 == 0
            or total_steps == 1
        ):
            print(
                f"step={total_steps:6d} "
                f"loss={loss.item():.6f} "
                f"huber={loss_h.item():.6f} "
                f"lpips={loss_p.item():.6f} "
                f"delta_abs={stats['delta_abs_mean'].item():.6e} "
                f"delta_rms={stats['delta_rms'].item():.6e} "
                f"refine_grad={grad_norm:.6e}"
            )

        if total_steps % 100 == 0:
            with torch.no_grad():
                tvu.save_image(
                    input_image * 0.5 + 0.5,
                    os.path.join(
                        train_res_dir,
                        "input.png",
                    ),
                )

                tvu.save_image(
                    baseline_pred,
                    os.path.join(
                        train_res_dir,
                        "baseline.png",
                    ),
                )

                tvu.save_image(
                    pred,
                    os.path.join(
                        train_res_dir,
                        "refined.png",
                    ),
                )

                tvu.save_image(
                    gt,
                    os.path.join(
                        train_res_dir,
                        "gt.png",
                    ),
                )

                delta_vis = torch.clamp(
                    delta / 0.2 + 0.5,
                    0.0,
                    1.0,
                )

                tvu.save_image(
                    delta_vis,
                    os.path.join(
                        train_res_dir,
                        "delta.png",
                    ),
                )

        # -------------------------------------------------------------
        # Validation
        # -------------------------------------------------------------
        if (
            total_steps
            % args.eval_step
            == 0
        ):
            val_result = evaluate_validation(
                net
            )

            print_validation(
                total_steps,
                val_result,
            )

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
                val_result[
                    "multi"
                ]["psnr"]
            )

            if (
                current_multi_psnr
                > best_multi_psnr
            ):
                best_multi_psnr = (
                    current_multi_psnr
                )

                best_path = os.path.join(
                    args.exp_name,
                    "best_multi_psnr_ckpt",
                )

                torch.save(
                    unwrap(
                        net
                    ).state_dict(),
                    best_path,
                )

                print(
                    f"New best validation "
                    f"Multi PSNR: "
                    f"{best_multi_psnr:.4f} dB"
                )

        # -------------------------------------------------------------
        # Periodic save
        # -------------------------------------------------------------
        if (
            total_steps
            % args.save_step
            == 0
        ):
            raw_path, state_path = save_checkpoint(
                total_steps
            )

            print(
                f"Saved weights:     "
                f"{raw_path}"
            )

            print(
                f"Saved train state: "
                f"{state_path}"
            )


print(
    "Pilot finished."
)

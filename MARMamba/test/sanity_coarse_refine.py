"""
Sanity check for Output-Space Coarse Residual Refinement.

Required:
1) new model == baseline at initialization;
2) final refinement projection receives gradient at step 0;
3) after one optimizer step, earlier refinement layers receive gradient;
4) print parameter overhead.

Run from repository root:
    python test/sanity_coarse_refine.py --checkpoint <baseline_ckpt>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.mamba import MambaFormer
from model.mamba_coarse_refine import (
    CoarseRefinedMambaFormer,
    load_baseline_weights,
)


def extract_baseline_state(
    checkpoint,
    device,
):
    obj = torch.load(
        checkpoint,
        map_location=device,
    )

    if (
        isinstance(obj, dict)
        and "model" in obj
    ):
        obj = obj["model"]

    if any(
        k.startswith("module.")
        for k in obj
    ):
        obj = {
            (
                k[len("module."):]
                if k.startswith("module.")
                else k
            ): v
            for k, v in obj.items()
        }

    return obj


parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint",
    required=True,
)
parser.add_argument(
    "--device",
    default="cuda:0",
)
parser.add_argument(
    "--size",
    default=256,
    type=int,
)
parser.add_argument(
    "--tol",
    default=1e-6,
    type=float,
)
parser.add_argument(
    "--hidden_channels",
    default=16,
    type=int,
)
args = parser.parse_args()

device = torch.device(
    args.device
)

baseline = MambaFormer(
    in_channels=1,
).to(device).eval()

baseline.load_state_dict(
    extract_baseline_state(
        args.checkpoint,
        device,
    ),
    strict=True,
)

model = CoarseRefinedMambaFormer(
    in_channels=1,
    hidden_channels=args.hidden_channels,
).to(device).eval()

load_baseline_weights(
    model,
    args.checkpoint,
    map_location=device,
)

x = torch.randn(
    1,
    1,
    args.size,
    args.size,
    device=device,
)

with torch.no_grad():
    y_base = baseline(x)
    y_new = model(x)

max_err = float(
    (y_base - y_new).abs().max().item()
)
mean_err = float(
    (y_base - y_new).abs().mean().item()
)

print(
    f"Forward max abs error : {max_err:.10e}"
)
print(
    f"Forward mean abs error: {mean_err:.10e}"
)

if (
    not np.isfinite(max_err)
    or max_err > args.tol
):
    raise RuntimeError(
        "FAIL: refined model is not "
        "baseline-equivalent at initialization."
    )

print(
    "PASS 1: baseline-equivalent forward."
)


# ------------------------------------------------------------------
# Gradient step 0: out_proj must learn.
# ------------------------------------------------------------------

model.train()
model.backbone.eval()

for p in model.backbone.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(
    model.refinement.parameters(),
    lr=1e-4,
)

target = (
    y_base.detach()
    + 0.01 * torch.randn_like(y_base)
)

optimizer.zero_grad(
    set_to_none=True,
)

y = model(x)
loss = (y - target).square().mean()
loss.backward()

out_grad = (
    model.refinement.out_proj.weight.grad
)

if out_grad is None:
    raise RuntimeError(
        "FAIL: out_proj has no gradient."
    )

if not torch.isfinite(out_grad).all():
    raise RuntimeError(
        "FAIL: out_proj gradient has NaN/Inf."
    )

out_grad_mean = float(
    out_grad.abs().mean().item()
)

if out_grad_mean == 0.0:
    raise RuntimeError(
        "FAIL: out_proj gradient is zero."
    )

print(
    f"out_proj grad mean abs: "
    f"{out_grad_mean:.10e}"
)
print(
    "PASS 2: output projection can learn."
)

optimizer.step()


# ------------------------------------------------------------------
# After out_proj moves away from zero, earlier blocks must receive grad.
# ------------------------------------------------------------------

optimizer.zero_grad(
    set_to_none=True,
)

y = model(x)
loss = (y - target).square().mean()
loss.backward()

stem_grad = (
    model.refinement.stem[0].weight.grad
)

if stem_grad is None:
    raise RuntimeError(
        "FAIL: stem has no gradient after "
        "out_proj became non-zero."
    )

if not torch.isfinite(stem_grad).all():
    raise RuntimeError(
        "FAIL: stem gradient has NaN/Inf."
    )

stem_grad_mean = float(
    stem_grad.abs().mean().item()
)

if stem_grad_mean == 0.0:
    raise RuntimeError(
        "FAIL: stem gradient is still zero."
    )

print(
    f"stem grad mean abs: "
    f"{stem_grad_mean:.10e}"
)
print(
    "PASS 3: full refinement head can learn."
)


# ------------------------------------------------------------------
# Params
# ------------------------------------------------------------------

base_params = sum(
    p.numel()
    for p in baseline.parameters()
)
new_params = sum(
    p.numel()
    for p in model.parameters()
)
extra = new_params - base_params

print(
    f"Baseline params : {base_params:,}"
)
print(
    f"New model params: {new_params:,}"
)
print(
    f"Extra params    : {extra:,} "
    f"({100.0 * extra / base_params:.2f}%)"
)

print(
    "\nALL SANITY CHECKS PASSED."
)

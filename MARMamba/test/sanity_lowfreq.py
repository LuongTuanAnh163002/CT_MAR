"""
Sanity check BEFORE training Low-Frequency Global Branch.

Required PASS conditions:
1) new model loaded from baseline checkpoint produces exactly/nearly the same
   output as baseline because lowfreq out_proj is zero-initialized;
2) lowfreq out_proj receives non-zero finite gradient;
3) parameter counts are printed.

Run from repository root.
"""

import argparse
import numpy as np
import torch

from model.mamba import MambaFormer
from model.mamba_lowfreq import (
    LowFrequencyMambaFormer,
    load_baseline_weights,
)


def extract_state(path, device):
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model" in obj:
        obj = obj["model"]
    if any(k.startswith("module.") for k in obj.keys()):
        obj = {
            k[len("module."):]: v
            for k, v in obj.items()
        }
    return obj


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--size", type=int, default=256)
parser.add_argument("--tol", type=float, default=1e-6)
args = parser.parse_args()

device = torch.device(args.device)

baseline = MambaFormer(in_channels=1).to(device).eval()
baseline.load_state_dict(
    extract_state(args.checkpoint, device),
    strict=True,
)

new_model = LowFrequencyMambaFormer(in_channels=1).to(device).eval()
missing = load_baseline_weights(
    new_model,
    args.checkpoint,
    map_location=device,
)

print(f"Allowed missing lowfreq keys: {len(missing)}")

x = torch.randn(
    1, 1, args.size, args.size,
    device=device,
)

with torch.no_grad():
    y_base = baseline(x)
    y_new = new_model(x)

max_err = (y_base - y_new).abs().max().item()
mean_err = (y_base - y_new).abs().mean().item()

print(f"Forward max abs error : {max_err:.10e}")
print(f"Forward mean abs error: {mean_err:.10e}")

if not np.isfinite(max_err) or max_err > args.tol:
    raise RuntimeError(
        f"FAIL: new model is not baseline-equivalent at initialization. "
        f"max_err={max_err:.3e}, tol={args.tol:.3e}"
    )

print("PASS 1: baseline-equivalent forward.")

# Gradient check
new_model.train()
new_model.zero_grad(set_to_none=True)

y = new_model(x)
loss = y.square().mean()
loss.backward()

grad = new_model.lowfreq_branch.out_proj.weight.grad

if grad is None:
    raise RuntimeError("FAIL: lowfreq out_proj has no gradient.")
if not torch.isfinite(grad).all():
    raise RuntimeError("FAIL: lowfreq out_proj gradient contains NaN/Inf.")
if grad.abs().mean().item() == 0.0:
    raise RuntimeError("FAIL: lowfreq out_proj gradient is exactly zero.")

print(
    "Lowfreq out_proj grad mean abs: "
    f"{grad.abs().mean().item():.10e}"
)
print("PASS 2: low-frequency branch can learn.")

base_params = sum(p.numel() for p in baseline.parameters())
new_params = sum(p.numel() for p in new_model.parameters())
extra = new_params - base_params

print(f"Baseline params : {base_params:,}")
print(f"New model params: {new_params:,}")
print(f"Extra params    : {extra:,} ({100*extra/base_params:.2f}%)")
print("\nALL SANITY CHECKS PASSED.")

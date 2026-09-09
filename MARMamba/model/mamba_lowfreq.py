"""
Low-Frequency Global Branch for MARMamba.

This file does NOT modify model/mamba.py.

Design
------
Baseline embedded feature F:
    F -> original MARMamba path
    F -> fixed Haar-LL -> lightweight MambaBlock -> bilinear upsample
      -> zero-initialized 1x1 projection -> residual correction

Then:
    F_enhanced = F + correction

The output projection of the new branch is initialized to zero, therefore
after loading a baseline MARMamba checkpoint the network starts EXACTLY from
the baseline function (up to floating-point determinism).

No metal mask, sinogram, or extra inference input is required.
The original restoration loss is unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mamba import MambaFormer, MambaBlock


class FixedHaarLL(nn.Module):
    """Per-channel one-level Haar LL transform with stride 2.

    For each 2x2 block:
        LL = (a + b + c + d) / 2

    This matches the orthonormal Haar-LL scaling used in the residual
    diagnostic, differing from avg_pool2d only by a constant factor.
    """

    def __init__(self, channels: int):
        super().__init__()
        kernel = torch.tensor(
            [[0.5, 0.5],
             [0.5, 0.5]],
            dtype=torch.float32,
        ).view(1, 1, 2, 2)
        self.register_buffer(
            "weight",
            kernel.repeat(channels, 1, 1, 1),
            persistent=False,
        )
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"Expected [B,{self.channels},H,W], got {tuple(x.shape)}"
            )
        return F.conv2d(
            x,
            self.weight.to(dtype=x.dtype),
            stride=2,
            padding=0,
            groups=self.channels,
        )


class LowFrequencyGlobalBranch(nn.Module):
    """Extract and process coarse/low-frequency feature information."""

    def __init__(
        self,
        dim: int,
        bias: bool = True,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()

        self.haar_ll = FixedHaarLL(dim)

        # A single MARMamba block at half spatial resolution:
        # lightweight, but still able to model long-range context.
        self.global_block = MambaBlock(
            dim=dim,
            bias=bias,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.out_proj = nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Critical for safe baseline initialization:
        # correction == 0 at step 0.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]

        low = self.haar_ll(x)              # [B,C,H/2,W/2]
        low = self.global_block(low)        # process coarse/global context
        low = F.interpolate(
            low,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        correction = self.out_proj(low)     # zero at initialization
        return correction


class LowFrequencyMambaFormer(MambaFormer):
    """MARMamba + one low-frequency global residual branch after embedder."""

    def __init__(
        self,
        input_size=256,
        in_channels=3,
        depth=(1, 2, 2, 4, 1),
        dim=12,
        bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
    ):
        super().__init__(
            input_size=input_size,
            in_channels=in_channels,
            depth=list(depth),
            dim=dim,
            bias=bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
        )

        self.lowfreq_branch = LowFrequencyGlobalBranch(
            dim=dim,
            bias=bias,
        )

    def forward(self, x, return_lowfreq=False):
        x_ori = x.clone()
        _, _, ori_h, ori_w = x.shape

        x = self.pad_to_multiple_of_eight(x)
        x = self.embedder(x)

        # New architecture-level intervention.
        lowfreq_correction = self.lowfreq_branch(x)
        x = x + lowfreq_correction

        # Baseline MARMamba path remains unchanged.
        x = self.down1_stage(x)
        x_to_s1 = x.clone()

        x = self.down1(x)
        x = self.down2_stage(x)
        x_to_s2 = x.clone()

        x = self.down2(x)
        x = self.down3_stage(x)
        x_to_s3 = x.clone()

        x = self.down3(x)
        x = self.down4_stage(x)

        x = self.up1(x)
        x = torch.cat((x, x_to_s3), dim=1)
        x = self.fusion2(x)
        x = self.up3_stage(x)

        x = self.up2(x)
        x = torch.cat((x, x_to_s2), dim=1)
        x = self.fusion3(x)
        x = self.up2_stage(x)

        x = self.up3(x)
        x = torch.cat((x, x_to_s1), dim=1)
        x = self.fusion4(x)
        x = self.up1_stage(x)

        if self.refine_stage is not None:
            x = self.refine_stage(x)

        x = self.unembedder(x)
        x = x[:, :, :ori_h, :ori_w] + x_ori

        if return_lowfreq:
            stats = {
                "correction_abs_mean": lowfreq_correction.detach().abs().mean(),
                "correction_rms": torch.sqrt(
                    torch.mean(lowfreq_correction.detach() ** 2)
                ),
            }
            return x, stats

        return x


def _extract_state_dict(obj):
    """Accept raw state_dict or a training-state dict containing 'model'."""
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        obj = obj["model"]
    if not isinstance(obj, dict):
        raise TypeError("Checkpoint does not contain a valid state_dict.")
    return obj


def _strip_module_prefix(state):
    if any(k.startswith("module.") for k in state.keys()):
        return {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state.items()
        }
    return state


def load_baseline_weights(
    model: LowFrequencyMambaFormer,
    checkpoint_path: str,
    map_location="cpu",
):
    """Load a BASELINE MARMamba checkpoint into the augmented model.

    Only lowfreq_branch.* keys are allowed to be missing.
    Any other mismatch aborts, preventing accidental loading of an
    incompatible experiment checkpoint.
    """
    obj = torch.load(checkpoint_path, map_location=map_location)
    state = _strip_module_prefix(_extract_state_dict(obj))

    incompatible = model.load_state_dict(state, strict=False)

    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    illegal_missing = [
        k for k in missing
        if not k.startswith("lowfreq_branch.")
    ]

    if unexpected or illegal_missing:
        raise RuntimeError(
            "Baseline checkpoint is not compatible with LowFrequencyMambaFormer.\n"
            f"Unexpected keys: {unexpected}\n"
            f"Illegal missing keys: {illegal_missing}\n"
            f"Allowed missing keys: {[k for k in missing if k.startswith('lowfreq_branch.')]}"
        )

    return missing


def load_lowfreq_weights(
    model: LowFrequencyMambaFormer,
    checkpoint_path: str,
    map_location="cpu",
):
    """Strictly load an already-trained LowFrequencyMambaFormer checkpoint."""
    obj = torch.load(checkpoint_path, map_location=map_location)
    state = _strip_module_prefix(_extract_state_dict(obj))
    model.load_state_dict(state, strict=True)

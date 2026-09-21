"""
Output-Space Coarse Residual Refinement for MARMamba.

This file DOES NOT modify model/mamba.py.

Inference:
    corrupted CT x
         |
         v
    frozen/trainable MARMamba backbone
         |
         v
    baseline prediction y
         |
         +-----------------------------+
         |                             |
         |  [x_unit, y, x_unit - y]    |
         |             |               |
         |        AvgPool x2            |
         |             |               |
         |   lightweight dilated CNN    |
         |             |               |
         |      coarse delta (H/2,W/2) |
         |             |               |
         |       nearest upsample x2    |
         |             |               |
         +-----------> delta -----------+
                       |
                       v
                 final = y + delta

Why scale=2?
------------
The oracle diagnostic showed the largest headroom at scale 2:
- single-metal: about +3.19 dB oracle PSNR
- multi-metal:  about +4.07 dB oracle PSNR
- 5-metal:      about +4.53 dB oracle PSNR

The refinement head predicts ONLY a coarse correction. It does not use GT,
metal masks, sinograms, or extra inference inputs.

The final output convolution is zero-initialized, therefore after loading a
baseline checkpoint:
    refined_model(x) == baseline_model(x)
at initialization (up to floating-point determinism).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mamba import MambaFormer


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=True,
        )
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=True,
        )

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))


class CoarseResidualHead(nn.Module):
    """
    Predict a one-channel residual correction at half resolution.

    Input channels:
      0: corrupted CT converted from [-1,1] -> [0,1]
      1: baseline MARMamba prediction
      2: corrupted CT - baseline prediction

    Dilations [1,2,4,8] provide a relatively broad receptive field at the
    half-resolution grid without adding much parameter cost.
    """

    def __init__(
        self,
        hidden_channels: int = 16,
        dilations=(1, 2, 4, 8),
        scale: int = 2,
    ):
        super().__init__()

        if scale != 2:
            raise ValueError(
                "This pilot is intentionally fixed to scale=2 because that "
                "is the scale supported most strongly by the oracle test."
            )

        self.scale = scale

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            *[
                DilatedResidualBlock(
                    hidden_channels,
                    dilation=d,
                )
                for d in dilations
            ]
        )

        self.out_proj = nn.Conv2d(
            hidden_channels,
            1,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        # Critical: model starts exactly as the baseline.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        normalized_input: torch.Tensor,
        baseline_pred: torch.Tensor,
    ):
        if normalized_input.ndim != 4 or baseline_pred.ndim != 4:
            raise ValueError("Expected BCHW tensors.")

        if normalized_input.shape != baseline_pred.shape:
            raise ValueError(
                f"Input/prediction shape mismatch: "
                f"{tuple(normalized_input.shape)} vs "
                f"{tuple(baseline_pred.shape)}"
            )

        # Training/eval input to baseline is normalized by:
        # Normalize(mean=0.5, std=0.5), hence convert it back to [0,1].
        input_unit = normalized_input * 0.5 + 0.5

        features = torch.cat(
            [
                input_unit,
                baseline_pred,
                input_unit - baseline_pred,
            ],
            dim=1,
        )

        coarse = F.avg_pool2d(
            features,
            kernel_size=self.scale,
            stride=self.scale,
        )

        coarse = self.stem(coarse)
        coarse = self.blocks(coarse)
        delta_coarse = self.out_proj(coarse)

        delta = F.interpolate(
            delta_coarse,
            size=baseline_pred.shape[-2:],
            mode="nearest",
        )

        return delta


class CoarseRefinedMambaFormer(nn.Module):
    """Baseline MARMamba followed by an output-space coarse residual head."""

    def __init__(
        self,
        input_size=256,
        in_channels=1,
        depth=(1, 2, 2, 4, 1),
        dim=12,
        bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        hidden_channels=16,
    ):
        super().__init__()

        self.backbone = MambaFormer(
            input_size=input_size,
            in_channels=in_channels,
            depth=list(depth),
            dim=dim,
            bias=bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
        )

        self.refinement = CoarseResidualHead(
            hidden_channels=hidden_channels,
            scale=2,
        )

    def forward(self, x, return_refinement=False):
        baseline_pred = self.backbone(x)
        delta = self.refinement(x, baseline_pred)
        final = baseline_pred + delta

        if return_refinement:
            stats = {
                "delta_abs_mean": delta.detach().abs().mean(),
                "delta_rms": torch.sqrt(
                    torch.mean(delta.detach() ** 2)
                ),
            }
            return final, baseline_pred, delta, stats

        return final


def _extract_state_dict(obj):
    if (
        isinstance(obj, dict)
        and "model" in obj
        and isinstance(obj["model"], dict)
    ):
        obj = obj["model"]

    if not isinstance(obj, dict):
        raise TypeError(
            "Checkpoint does not contain a valid state_dict."
        )

    return obj


def _strip_module_prefix(state):
    if any(k.startswith("module.") for k in state.keys()):
        return {
            (
                k[len("module."):]
                if k.startswith("module.")
                else k
            ): v
            for k, v in state.items()
        }
    return state


def load_baseline_weights(
    model: CoarseRefinedMambaFormer,
    checkpoint_path: str,
    map_location="cpu",
):
    """
    Load a BASELINE MARMamba checkpoint into model.backbone.

    This is intentionally strict: a wrong baseline checkpoint should fail
    immediately rather than silently initialize part of the model.
    """
    obj = torch.load(
        checkpoint_path,
        map_location=map_location,
    )
    state = _strip_module_prefix(
        _extract_state_dict(obj)
    )

    model.backbone.load_state_dict(
        state,
        strict=True,
    )


def load_refined_weights(
    model: CoarseRefinedMambaFormer,
    checkpoint_path: str,
    map_location="cpu",
):
    """Strict load of an already-trained CoarseRefinedMambaFormer."""
    obj = torch.load(
        checkpoint_path,
        map_location=map_location,
    )
    state = _strip_module_prefix(
        _extract_state_dict(obj)
    )

    model.load_state_dict(
        state,
        strict=True,
    )

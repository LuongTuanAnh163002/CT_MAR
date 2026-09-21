"""
Dynamic-gated multi-scale coarse residual refinement for MARMamba.

Frozen backbone:
    x -> MARMamba -> baseline b

Refinement:
    z = concat[x_unit, b, x_unit - b]
    scale-2 branch -> detail residual
    scale-4 branch -> coarse residual

Adaptive fusion:
    delta = alpha2(x) * delta2 + alpha4(x) * delta4
    refined = b + delta

The gate is initialized with alpha2 = alpha4 = 1.0, so training starts
from the exact fixed-fusion Multi-Coarse behavior.
"""

from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mamba import MambaFormer


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, 3,
            padding=dilation, dilation=dilation
        )
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            channels, channels, 3,
            padding=dilation, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act(self.conv1(x)))


class CoarseBranch(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 32,
        dilations=(1, 2, 4, 8),
    ):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            *[
                DilatedResidualBlock(hidden_channels, d)
                for d in dilations
            ]
        )

        self.out_proj = nn.Conv2d(
            hidden_channels, 1, 3, padding=1
        )

        # Start from zero residual correction.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, return_feature: bool = False):
        feat = self.in_proj(x)
        feat = self.blocks(feat)
        residual = self.out_proj(feat)

        if return_feature:
            return residual, feat
        return residual


class ScaleGate(nn.Module):
    """
    Per-image scale weights:
        alpha = 1 + gate_range * tanh(raw)

    Final linear layer is zero-initialized, so initially:
        alpha2 = alpha4 = 1
    """

    def __init__(
        self,
        hidden_channels: int = 32,
        gate_hidden: int = 32,
        gate_range: float = 0.5,
    ):
        super().__init__()
        self.gate_range = float(gate_range)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 2),
        )

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feat2: torch.Tensor, feat4: torch.Tensor):
        g2 = F.adaptive_avg_pool2d(feat2, 1).flatten(1)
        g4 = F.adaptive_avg_pool2d(feat4, 1).flatten(1)

        raw = self.mlp(torch.cat([g2, g4], dim=1))
        alpha = 1.0 + self.gate_range * torch.tanh(raw)

        alpha2 = alpha[:, 0].view(-1, 1, 1, 1)
        alpha4 = alpha[:, 1].view(-1, 1, 1, 1)
        return alpha2, alpha4


class GatedMultiScaleCoarseRefinedMambaFormer(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        dilations=(1, 2, 4, 8),
        gate_hidden: int = 32,
        gate_range: float = 0.5,
    ):
        super().__init__()

        self.backbone = MambaFormer(in_channels=in_channels)

        self.refinement_scale2 = CoarseBranch(
            in_channels=3,
            hidden_channels=hidden_channels,
            dilations=dilations,
        )
        self.refinement_scale4 = CoarseBranch(
            in_channels=3,
            hidden_channels=hidden_channels,
            dilations=dilations,
        )

        self.scale_gate = ScaleGate(
            hidden_channels=hidden_channels,
            gate_hidden=gate_hidden,
            gate_range=gate_range,
        )

        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(
        self,
        x: torch.Tensor,
        return_refinement: bool = False,
        return_aux: bool = False,
    ):
        with torch.no_grad():
            baseline = self.backbone(x)

        # MARMamba input is [-1,1]; head uses image-like [0,1].
        x_unit = x * 0.5 + 0.5
        z = torch.cat(
            [x_unit, baseline, x_unit - baseline],
            dim=1,
        )

        h, w = baseline.shape[-2:]

        z2 = F.avg_pool2d(z, 2, 2)
        z4 = F.avg_pool2d(z, 4, 4)

        pred_detail2_low, feat2 = self.refinement_scale2(
            z2, return_feature=True
        )
        pred_coarse4_low, feat4 = self.refinement_scale4(
            z4, return_feature=True
        )

        pred_detail2 = F.interpolate(
            pred_detail2_low,
            size=(h, w),
            mode="nearest",
        )
        pred_coarse4 = F.interpolate(
            pred_coarse4_low,
            size=(h, w),
            mode="nearest",
        )

        alpha2, alpha4 = self.scale_gate(feat2, feat4)

        delta = (
            alpha2 * pred_detail2
            + alpha4 * pred_coarse4
        )
        refined = baseline + delta

        stats: Dict[str, Any] = {
            "delta_abs_mean": delta.detach().abs().mean(),
            "delta_rms": torch.sqrt(torch.mean(delta.detach().float() ** 2)),
            "detail2_abs_mean": pred_detail2.detach().abs().mean(),
            "coarse4_abs_mean": pred_coarse4.detach().abs().mean(),
            "alpha2_mean": alpha2.detach().mean(),
            "alpha4_mean": alpha4.detach().mean(),
            "alpha2_std": alpha2.detach().std(unbiased=False),
            "alpha4_std": alpha4.detach().std(unbiased=False),
        }

        if return_aux:
            return {
                "refined": refined,
                "baseline": baseline,
                "delta": delta,
                "pred_detail2_low": pred_detail2_low,
                "pred_coarse4_low": pred_coarse4_low,
                "pred_detail2": pred_detail2,
                "pred_coarse4": pred_coarse4,
                "alpha2": alpha2,
                "alpha4": alpha4,
                "stats": stats,
            }

        if return_refinement:
            return refined, baseline, delta, stats

        return refined


def _extract_state_dict(state):
    if isinstance(state, dict):
        for key in ("model", "state_dict", "net"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a state_dict.")

    if any(k.startswith("module.") for k in state):
        state = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state.items()
        }

    return state


def load_baseline_weights(
    model: GatedMultiScaleCoarseRefinedMambaFormer,
    checkpoint_path: str,
    map_location="cpu",
) -> None:
    state = torch.load(checkpoint_path, map_location=map_location)
    state = _extract_state_dict(state)

    if any(k.startswith("backbone.") for k in state):
        state = {
            k[len("backbone."):]: v
            for k, v in state.items()
            if k.startswith("backbone.")
        }

    missing, unexpected = model.backbone.load_state_dict(
        state, strict=False
    )

    if missing or unexpected:
        raise RuntimeError(
            "Baseline checkpoint is incompatible with MambaFormer.\n"
            f"Missing ({len(missing)}): {missing[:10]}\n"
            f"Unexpected ({len(unexpected)}): {unexpected[:10]}"
        )

    model.freeze_backbone()

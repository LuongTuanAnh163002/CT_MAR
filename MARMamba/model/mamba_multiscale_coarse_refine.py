"""
Multi-scale coarse residual refinement for MARMamba.

Design
------
Frozen MARMamba backbone:
    x_norm [-1,1] -> baseline b [0,1]

Refinement input:
    z = concat[x_unit, b, x_unit - b]

Two hierarchical residual branches:
    scale-4 branch learns the very-coarse residual.
    scale-2 branch learns the residual detail that remains between scale 2
    and scale 4.

Final:
    delta = Up2(detail2) + Up4(coarse4)
    refined = baseline + delta

No metal mask / metal count / GT is required at inference.
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
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act(self.conv1(x)))


class CoarseBranch(nn.Module):
    """Predict a one-channel residual at the branch's pooled resolution."""

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
                DilatedResidualBlock(hidden_channels, dilation=d)
                for d in dilations
            ]
        )

        self.out_proj = nn.Conv2d(
            hidden_channels,
            1,
            kernel_size=3,
            padding=1,
        )

        # Start exactly from the frozen-backbone prediction.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.blocks(self.in_proj(x)))


class MultiScaleCoarseRefinedMambaFormer(nn.Module):
    """
    MARMamba + dual-scale hierarchical coarse residual refinement.

    The backbone is intentionally frozen.  Input to forward() follows the
    existing AAPM/MARMamba convention: corrupted image normalized to [-1, 1].
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        dilations=(1, 2, 4, 8),
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

        self.freeze_backbone()

    @property
    def refinement_parameters(self):
        return list(self.refinement_scale2.parameters()) + list(
            self.refinement_scale4.parameters()
        )

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the frozen backbone deterministic.
        self.backbone.eval()
        return self

    def forward(
        self,
        x: torch.Tensor,
        return_refinement: bool = False,
        return_aux: bool = False,
    ):
        # Backbone receives the normal MARMamba [-1,1] input.
        with torch.no_grad():
            baseline = self.backbone(x)

        # Head works in the same [0,1]-like image space as baseline/GT.
        x_unit = x * 0.5 + 0.5

        z = torch.cat(
            [x_unit, baseline, x_unit - baseline],
            dim=1,
        )

        h, w = baseline.shape[-2:]

        z2 = F.avg_pool2d(z, kernel_size=2, stride=2)
        z4 = F.avg_pool2d(z, kernel_size=4, stride=4)

        pred_detail2_low = self.refinement_scale2(z2)
        pred_coarse4_low = self.refinement_scale4(z4)

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

        delta = pred_detail2 + pred_coarse4
        refined = baseline + delta

        stats: Dict[str, Any] = {
            "delta_abs_mean": delta.detach().abs().mean(),
            "delta_rms": torch.sqrt(
                torch.mean(delta.detach().float() ** 2)
            ),
            "detail2_abs_mean": pred_detail2.detach().abs().mean(),
            "coarse4_abs_mean": pred_coarse4.detach().abs().mean(),
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
                "stats": stats,
            }

        # Keep the same convenient shape used by the previous evaluator.
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
        raise TypeError("Checkpoint does not contain a state_dict-like mapping.")

    # Strip DataParallel prefix.
    if any(k.startswith("module.") for k in state):
        state = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state.items()
        }

    return state


def load_baseline_weights(
    model: MultiScaleCoarseRefinedMambaFormer,
    checkpoint_path: str,
    map_location="cpu",
) -> None:
    """
    Load a plain Stage-2/Stage-3 MARMamba checkpoint into model.backbone.

    Handles the common original checkpoint form:
        module.xxx -> xxx
    """
    state = torch.load(checkpoint_path, map_location=map_location)
    state = _extract_state_dict(state)

    # If a wrapped model was accidentally supplied, retain only backbone keys.
    if any(k.startswith("backbone.") for k in state):
        state = {
            k[len("backbone."):]: v
            for k, v in state.items()
            if k.startswith("backbone.")
        }

    missing, unexpected = model.backbone.load_state_dict(
        state,
        strict=False,
    )

    if missing or unexpected:
        raise RuntimeError(
            "Baseline checkpoint is not compatible with MambaFormer.\n"
            f"Missing keys ({len(missing)}): {missing[:10]}\n"
            f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}"
        )

    model.freeze_backbone()

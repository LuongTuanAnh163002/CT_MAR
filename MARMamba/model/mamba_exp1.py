"""
Exp1 model: Metal-Geometry Guided Dynamic FMB.

This file intentionally does NOT modify model/mamba.py.
The baseline MARMamba layers are reused wherever possible.  The only model-side
change is the FMB fusion: three Mamba branch outputs are dynamically weighted by
three per-image weights predicted from the AAPM metal mask.
"""

import torch
import torch.nn as nn
from einops import rearrange

from model.mamba import MambaFormer, MambaAttention, MambaBlock


class MetalGeometryGuidance(nn.Module):
    """Predict three branch weights from a binary metal mask.

    Adaptive 4x4 pooling intentionally keeps coarse spatial geometry (position,
    extent, and rough shape) instead of collapsing the mask immediately to one
    global scalar.
    """

    def __init__(self, hidden_dim=32, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.head = nn.Linear(hidden_dim * 4 * 4, 3)

        # Start from an unbiased 1/3, 1/3, 1/3 routing distribution.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, metal_mask):
        if metal_mask is None:
            raise ValueError("Exp1 requires metal_mask, got None")
        if metal_mask.ndim != 4 or metal_mask.shape[1] != 1:
            raise ValueError(
                f"metal_mask must have shape [B, 1, H, W], got {tuple(metal_mask.shape)}"
            )

        logits = self.head(self.encoder(metal_mask.float()))
        return torch.softmax(logits / self.temperature, dim=-1)


class MetalGuidedMambaAttention(MambaAttention):
    """Baseline FMB branches + mask-conditioned dynamic fusion."""

    def forward(self, x, guidance_weights):
        _, _, H, W = x.shape

        x = self.proj_1(x)
        normal, flip_x1, flip_x2 = torch.chunk(x, 3, dim=1)

        # Keep the exact branch transformations/scanning behavior of baseline.
        flip_x1 = flip_x1.flatten(2).transpose(1, 2)
        flip_x1 = torch.flip(flip_x1, dims=[-1])

        flip_x2 = flip_x2.flatten(2).transpose(1, 2)
        flip_x2 = torch.flip(flip_x2, dims=[-2])

        normal = normal.flatten(2).transpose(1, 2)

        normal = self.mamba_norm(normal)
        flip_x1 = self.mamba_flip1(flip_x1)
        flip_x2 = self.mamba_flip2(flip_x2)

        flip_x1 = torch.flip(flip_x1, dims=[-1])
        flip_x2 = torch.flip(flip_x2, dims=[-2])

        if guidance_weights.ndim != 2 or guidance_weights.shape[-1] != 3:
            raise ValueError(
                "guidance_weights must have shape [B, 3], "
                f"got {tuple(guidance_weights.shape)}"
            )

        w0 = guidance_weights[:, 0].view(-1, 1, 1)
        w1 = guidance_weights[:, 1].view(-1, 1, 1)
        w2 = guidance_weights[:, 2].view(-1, 1, 1)

        # Exp1 change: fixed multiplicative fusion -> metal-conditioned fusion.
        fused = w0 * normal + w1 * flip_x1 + w2 * flip_x2

        fused = rearrange(fused, 'b (h w) c -> b c h w', h=H, w=W)
        return self.proj_2(fused)


class MetalGuidedMambaBlock(MambaBlock):
    def __init__(self, dim, bias=True, d_state=16, d_conv=4, expand=2, LayerNorm_type='WithBias'):
        super().__init__(
            dim=dim,
            bias=bias,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            LayerNorm_type=LayerNorm_type,
        )
        # Replace only the attention/FMB component.  Norm, AMFN/MLP and layer
        # scales remain inherited from the baseline implementation.
        self.attn = MetalGuidedMambaAttention(
            dim=dim,
            bias=False,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x, guidance_weights):
        x = x + self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(
            self.norm1(x), guidance_weights
        )
        x = x + self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))
        return x


class MetalGuidedStage(nn.Module):
    def __init__(self, dim, depth, bias=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            MetalGuidedMambaBlock(dim=dim, bias=bias)
            for _ in range(depth)
        ])

    def forward(self, x, guidance_weights):
        for block in self.blocks:
            x = block(x, guidance_weights)
        return x


class MetalGuidedMambaFormer(MambaFormer):
    """MARMamba Exp1 network.

    Forward signature:
        pred = model(artifact_image, metal_mask)
    """

    def __init__(
        self,
        input_size=256,
        in_channels=3,
        depth=(1, 2, 2, 4, 1),
        dim=12,
        bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        guidance_hidden_dim=32,
        guidance_temperature=1.0,
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

        # Replace baseline stages with Exp1 stages; all down/up/fusion convolutions
        # from MambaFormer stay untouched.
        self.down1_stage = MetalGuidedStage(dim=dim, depth=depth[0])
        self.down2_stage = MetalGuidedStage(dim=dim * 2, depth=depth[1])
        self.down3_stage = MetalGuidedStage(dim=dim * 4, depth=depth[2])
        self.down4_stage = MetalGuidedStage(dim=dim * 8, depth=depth[3])
        self.up1_stage = MetalGuidedStage(dim=dim, depth=depth[0])
        self.up2_stage = MetalGuidedStage(dim=dim * 2, depth=depth[1])
        self.up3_stage = MetalGuidedStage(dim=dim * 4, depth=depth[2])
        self.refine_stage = MetalGuidedStage(dim=dim, depth=depth[4]) if depth[4] != 0 else None

        self.metal_guidance = MetalGeometryGuidance(
            hidden_dim=guidance_hidden_dim,
            temperature=guidance_temperature,
        )

    def forward(self, x, metal_mask, return_guidance=False):
        x_ori = x.clone()
        _, _, ori_h, ori_w = x.shape

        # One routing vector per image.  The same case-level geometry prior is
        # supplied to every FMB scale; the Mamba features themselves remain
        # scale-specific.
        guidance_weights = self.metal_guidance(metal_mask)

        x = self.pad_to_multiple_of_eight(x)
        x = self.embedder(x)

        x = self.down1_stage(x, guidance_weights)
        x_to_s1 = x.clone()

        x = self.down1(x)
        x = self.down2_stage(x, guidance_weights)
        x_to_s2 = x.clone()

        x = self.down2(x)
        x = self.down3_stage(x, guidance_weights)
        x_to_s3 = x.clone()

        x = self.down3(x)
        x = self.down4_stage(x, guidance_weights)

        x = self.up1(x)
        x = torch.cat((x, x_to_s3), dim=1)
        x = self.fusion2(x)
        x = self.up3_stage(x, guidance_weights)

        x = self.up2(x)
        x = torch.cat((x, x_to_s2), dim=1)
        x = self.fusion3(x)
        x = self.up2_stage(x, guidance_weights)

        x = self.up3(x)
        x = torch.cat((x, x_to_s1), dim=1)
        x = self.fusion4(x)
        x = self.up1_stage(x, guidance_weights)

        if self.refine_stage is not None:
            x = self.refine_stage(x, guidance_weights)

        x = self.unembedder(x)
        x = x[:, :, :ori_h, :ori_w] + x_ori

        if return_guidance:
            return x, guidance_weights
        return x

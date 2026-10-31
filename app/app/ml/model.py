"""Spinal-Cord MRI Segmentation & Classification Model Suite.

Includes 3-Way Architectures:
  1. UNetWithClassifier (Baseline Standard U-Net)
  2. AttentionUNetWithClassifier (Attention U-Net with Additive Attention Gates)
  3. NeuroCordResAttnNet (Proposed Model: Multi-Scale Residual-SE U-Net with ASPP & Cross-Attention)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Standard Conv Block
# ---------------------------------------------------------------------------
def _conv_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
# 2. Standard U-Net with Classification Head (Baseline)
# ---------------------------------------------------------------------------
class UNetWithClassifier(nn.Module):
    """Standard U-Net backbone with bottleneck classification head."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4, base: int = 32):
        super().__init__()
        self.name = "Standard U-Net"
        # Encoder
        self.enc1 = _conv_block(in_channels, base)
        self.enc2 = _conv_block(base, base * 2)
        self.enc3 = _conv_block(base * 2, base * 4)
        self.enc4 = _conv_block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = _conv_block(base * 8, base * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = _conv_block(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _conv_block(base * 2, base)

        # Segmentation head
        self.seg_head = nn.Conv2d(base, 1, 1)

        # Classification head from bottleneck features
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        seg_logits = self.seg_head(d1)
        cls_logits = self.cls_head(b)
        return seg_logits, cls_logits


# ---------------------------------------------------------------------------
# 3. Attention Gate (Oktay et al., 2018)
# ---------------------------------------------------------------------------
class AttentionGate(nn.Module):
    """Additive Attention Gate for suppressing irrelevant background activations."""

    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int),
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # g: gating signal from coarser layer, x: skip connection from encoder
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        return x * alpha, alpha


# ---------------------------------------------------------------------------
# 4. Attention U-Net with Classification Head
# ---------------------------------------------------------------------------
class AttentionUNetWithClassifier(nn.Module):
    """Attention U-Net with Additive Attention Gates at skip connections."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4, base: int = 32):
        super().__init__()
        self.name = "Attention U-Net"
        # Encoder
        self.enc1 = _conv_block(in_channels, base)
        self.enc2 = _conv_block(base, base * 2)
        self.enc3 = _conv_block(base * 2, base * 4)
        self.enc4 = _conv_block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = _conv_block(base * 8, base * 16)

        # Attention Gates
        self.ag4 = AttentionGate(f_g=base * 8, f_l=base * 8, f_int=base * 4)
        self.ag3 = AttentionGate(f_g=base * 4, f_l=base * 4, f_int=base * 2)
        self.ag2 = AttentionGate(f_g=base * 2, f_l=base * 2, f_int=base)
        self.ag1 = AttentionGate(f_g=base, f_l=base, f_int=base // 2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = _conv_block(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _conv_block(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _conv_block(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _conv_block(base * 2, base)

        # Segmentation head
        self.seg_head = nn.Conv2d(base, 1, 1)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        u4 = self.up4(b)
        e4_att, alpha4 = self.ag4(g=u4, x=e4)
        d4 = self.dec4(torch.cat([u4, e4_att], dim=1))

        u3 = self.up3(d4)
        e3_att, alpha3 = self.ag3(g=u3, x=e3)
        d3 = self.dec3(torch.cat([u3, e3_att], dim=1))

        u2 = self.up2(d3)
        e2_att, alpha2 = self.ag2(g=u2, x=e2)
        d2 = self.dec2(torch.cat([u2, e2_att], dim=1))

        u1 = self.up1(d2)
        e1_att, alpha1 = self.ag1(g=u1, x=e1)
        d1 = self.dec1(torch.cat([u1, e1_att], dim=1))

        seg_logits = self.seg_head(d1)
        cls_logits = self.cls_head(b)

        if return_attention:
            attn_maps = {"ag4": alpha4, "ag3": alpha3, "ag2": alpha2, "ag1": alpha1}
            return seg_logits, cls_logits, attn_maps
        return seg_logits, cls_logits


# ---------------------------------------------------------------------------
# 5. Proposed Model Building Blocks: Squeeze-and-Excitation & Residual Conv
# ---------------------------------------------------------------------------
class SqueezeExcitation(nn.Module):
    """Channel attention mechanism to re-calibrate feature maps."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class ResidualSEBlock(nn.Module):
    """Residual convolutional block enhanced with Squeeze-and-Excitation channel gating."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.se = SqueezeExcitation(out_c)
        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.conv(x)
        out = self.se(out)
        return self.relu(out + res)


# ---------------------------------------------------------------------------
# 6. Atrous Spatial Pyramid Pooling (ASPP) Context Bottleneck
# ---------------------------------------------------------------------------
class ASPPBottleneck(nn.Module):
    """Multi-scale dilated receptive field bottleneck to capture both macroscopic cord deformation and micro-lesions."""

    def __init__(self, in_c: int, out_c: int, rates: Tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        self.branches = nn.ModuleList()
        # 1x1 conv branch
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_c, out_c // 4, 1, bias=False),
                nn.BatchNorm2d(out_c // 4),
                nn.ReLU(inplace=True),
            )
        )
        # Dilated 3x3 branches
        for r in rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_c, out_c // 4, 3, padding=r, dilation=r, bias=False),
                    nn.BatchNorm2d(out_c // 4),
                    nn.ReLU(inplace=True),
                )
            )
        # Global pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, out_c // 4, 1, bias=True),
            nn.ReLU(inplace=True),
        )
        total_in = (len(rates) + 2) * (out_c // 4)
        self.project = nn.Sequential(
            nn.Conv2d(total_in, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        feats = [b(x) for b in self.branches]
        gp = F.interpolate(self.global_pool(x), size=(h, w), mode="bilinear", align_corners=True)
        feats.append(gp)
        concat = torch.cat(feats, dim=1)
        return self.project(concat)


# ---------------------------------------------------------------------------
# 7. Proposed Model: NeuroCord-ResAttnNet
# ---------------------------------------------------------------------------
class NeuroCordResAttnNet(nn.Module):
    """Proposed Model: Dual-Task Multi-Scale Res-UNet with Residual-SE, ASPP Context Bottleneck, and Cross-Attention."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4, base: int = 32):
        super().__init__()
        self.name = "NeuroCord-ResAttnNet (Proposed)"
        # Residual-SE Encoder
        self.enc1 = ResidualSEBlock(in_channels, base)
        self.enc2 = ResidualSEBlock(base, base * 2)
        self.enc3 = ResidualSEBlock(base * 2, base * 4)
        self.enc4 = ResidualSEBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        # Multi-Scale ASPP Bottleneck
        self.bottleneck = ASPPBottleneck(base * 8, base * 16, rates=(2, 4, 6))

        # Cross-Attention Skip Gates
        self.ag4 = AttentionGate(f_g=base * 8, f_l=base * 8, f_int=base * 4)
        self.ag3 = AttentionGate(f_g=base * 4, f_l=base * 4, f_int=base * 2)
        self.ag2 = AttentionGate(f_g=base * 2, f_l=base * 2, f_int=base)
        self.ag1 = AttentionGate(f_g=base, f_l=base, f_int=base // 2)

        # Residual Decoder
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = ResidualSEBlock(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ResidualSEBlock(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ResidualSEBlock(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ResidualSEBlock(base * 2, base)

        # Segmentation Head with Deep Supervision feature
        self.seg_head = nn.Sequential(
            nn.Conv2d(base, base // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base // 2, 1, 1),
        )

        # Multi-Scale Joint Classification Head
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 16, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        u4 = self.up4(b)
        e4_att, alpha4 = self.ag4(g=u4, x=e4)
        d4 = self.dec4(torch.cat([u4, e4_att], dim=1))

        u3 = self.up3(d4)
        e3_att, alpha3 = self.ag3(g=u3, x=e3)
        d3 = self.dec3(torch.cat([u3, e3_att], dim=1))

        u2 = self.up2(d3)
        e2_att, alpha2 = self.ag2(g=u2, x=e2)
        d2 = self.dec2(torch.cat([u2, e2_att], dim=1))

        u1 = self.up1(d2)
        e1_att, alpha1 = self.ag1(g=u1, x=e1)
        d1 = self.dec1(torch.cat([u1, e1_att], dim=1))

        seg_logits = self.seg_head(d1)
        cls_logits = self.cls_head(b)

        if return_attention:
            attn_maps = {"ag4": alpha4, "ag3": alpha3, "ag2": alpha2, "ag1": alpha1}
            return seg_logits, cls_logits, attn_maps
        return seg_logits, cls_logits

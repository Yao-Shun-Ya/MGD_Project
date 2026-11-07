"""MGD-PriorNet：解剖先验门控双解码器网络（Phase 3 核心创新模型）。

三大模块（论文 / 专利表述保持与需求文档一致）：
- FCAS：频域-对比度自适应增强前端。拉普拉斯高频残差 + 通道注意力，与 CLAHE 互补；
- ASAB：各向异性条带感知卷积块。并联 1×k（水平上下文）与 k×1（纵向腺体连续性）卷积；
- APG-DD：解剖先验门控双解码器。眼睑分支生成 A_eyelid，以软门控
  F_gland' = F_gland ⊙ (1 + A_eyelid) 引导腺体分支，抑制眼睑外伪影。

输出约定：`torch.cat([gland_logits, eyelid_logits], dim=1)`，即 2 通道 logits。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """通道注意力（squeeze-excitation）。"""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class FCASStem(nn.Module):
    """Frequency-Contrast Adaptive Stem。

    高频分支用固定的 3×3 拉普拉斯核提取条索纹理，再由可学习 1×1/3×3 卷积投影；
    低频分支是常规卷积。两者按通道注意力融合，端到端学习红外低信噪比增强。
    """

    def __init__(self, in_channels: int = 1, channels: int = 24):
        super().__init__()
        kernel = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=torch.float32)
        self.register_buffer("laplace_kernel", kernel.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))
        self.high_proj = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.low_proj = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            SEBlock(channels),
        )
        self.groups = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        high = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), self.laplace_kernel, groups=self.groups)
        high = self.high_proj(high)
        low = self.low_proj(x)
        return self.fuse(torch.cat([low, high], dim=1))


class ASAB(nn.Module):
    """Anisotropic Strip-Aware Block：水平 1×k + 垂直 k×1 + 3×3 并联。"""

    def __init__(self, in_channels: int, out_channels: int, k: int = 7):
        super().__init__()
        pad_h = (0, k // 2)
        pad_v = (k // 2, 0)
        self.horizontal = nn.Conv2d(in_channels, out_channels, kernel_size=(1, k), padding=pad_h)
        self.vertical = nn.Conv2d(in_channels, out_channels, kernel_size=(k, 1), padding=pad_v)
        self.local = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.horizontal(x)
        v = self.vertical(x)
        l = self.local(x)
        return self.fuse(torch.cat([h, v, l], dim=1))


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int = 7, downsample: bool = True):
        super().__init__()
        self.block = nn.Sequential(ASAB(in_channels, out_channels, k), ASAB(out_channels, out_channels, k))
        self.downsample = downsample
        self.pool = nn.MaxPool2d(2) if downsample else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.block(x))


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, k: int = 7):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = nn.Sequential(
            ASAB(out_channels + skip_channels, out_channels, k),
            ASAB(out_channels, out_channels, k),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([skip, x], dim=1))


class MGDPriorNet(nn.Module):
    """MGD-PriorNet：双任务（gland / eyelid）分割网络。"""

    def __init__(self, in_channels: int = 1, out_channels: int = 2, stem_channels: int = 24, k: int = 7):
        super().__init__()
        if int(out_channels) != 2:
            raise ValueError("MGD-PriorNet 必须使用 out_channels=2（gland + eyelid）")
        self.out_channels = 2
        self.stem = FCASStem(in_channels, stem_channels)
        self.enc1 = EncoderStage(stem_channels, stem_channels, k)
        self.enc2 = EncoderStage(stem_channels, stem_channels * 2, k)
        self.enc3 = EncoderStage(stem_channels * 2, stem_channels * 4, k)
        self.enc4 = EncoderStage(stem_channels * 4, stem_channels * 8, k)

        self.gland_dec4 = DecoderStage(stem_channels * 8, stem_channels * 4, stem_channels * 4, k)
        self.gland_dec3 = DecoderStage(stem_channels * 4, stem_channels * 2, stem_channels * 2, k)
        self.gland_dec2 = DecoderStage(stem_channels * 2, stem_channels, stem_channels, k)
        self.gland_dec1 = DecoderStage(stem_channels, stem_channels, stem_channels, k)

        self.eyelid_dec4 = DecoderStage(stem_channels * 8, stem_channels * 4, stem_channels * 2, k)
        self.eyelid_dec3 = DecoderStage(stem_channels * 2, stem_channels * 2, stem_channels, k)
        self.eyelid_dec2 = DecoderStage(stem_channels, stem_channels, stem_channels, k)
        self.eyelid_dec1 = DecoderStage(stem_channels, stem_channels, stem_channels, k)

        self.gland_head = nn.Conv2d(stem_channels, 1, kernel_size=1)
        self.eyelid_head = nn.Conv2d(stem_channels, 1, kernel_size=1)
        # 多尺度眼睑门控头：通道数分别为 dec4=2c / dec3=c / dec2=c
        self.gate4 = nn.Conv2d(stem_channels * 2, 1, kernel_size=1)
        self.gate3 = nn.Conv2d(stem_channels, 1, kernel_size=1)
        self.gate2 = nn.Conv2d(stem_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        s = self.stem(x)
        e1 = self.enc1(s)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        e_dec4 = self.eyelid_dec4(e4, e3)
        e_dec3 = self.eyelid_dec3(e_dec4, e2)
        e_dec2 = self.eyelid_dec2(e_dec3, e1)
        e_dec1 = self.eyelid_dec1(e_dec2, s)
        eyelid_logits = self.eyelid_head(e_dec1)

        # APG-DD：用眼睑分支的多尺度概率作为软门控（1 + A_eyelid），只增强有效区域
        gate4 = torch.sigmoid(self.gate4(F.interpolate(e_dec4, size=e3.shape[-2:], mode="bilinear", align_corners=False)))
        g_dec4 = self.gland_dec4(e4, e3) * (1.0 + gate4)
        gate3 = torch.sigmoid(self.gate3(F.interpolate(e_dec3, size=e2.shape[-2:], mode="bilinear", align_corners=False)))
        g_dec3 = self.gland_dec3(g_dec4, e2) * (1.0 + gate3)
        gate2 = torch.sigmoid(self.gate2(F.interpolate(e_dec2, size=e1.shape[-2:], mode="bilinear", align_corners=False)))
        g_dec2 = self.gland_dec2(g_dec3, e1) * (1.0 + gate2)
        gate1 = torch.sigmoid(eyelid_logits)
        g_dec1 = self.gland_dec1(g_dec2, s) * (1.0 + gate1)
        gland_logits = self.gland_head(g_dec1)

        if gland_logits.shape[-2:] != size:
            gland_logits = F.interpolate(gland_logits, size=size, mode="bilinear", align_corners=False)
            eyelid_logits = F.interpolate(eyelid_logits, size=size, mode="bilinear", align_corners=False)
        return torch.cat([gland_logits, eyelid_logits], dim=1)

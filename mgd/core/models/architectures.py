"""模型架构的唯一实现（S1 抽核目标）。

三种架构此前在 4 份副本中重复定义，这里保留唯一一份：
- UNet：沿用 MONAI UNet 的超参（64..1024, 2 res units, PRELU, INSTANCE, dropout=0.1）
- AttentionUNet：带注意力门控的 UNet
- WRCANet：宽残差通道注意力增强 + 轻量分割头

所有架构输出均为 **logits**（1 通道），sigmoid 由调用方决定。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionBlock(nn.Module):
    """注意力门控：对 skip 连接做空间加权。"""

    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(f_int),
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(f_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        psi = self.relu(self.w_g(g) + self.w_x(x))
        return x * self.psi(psi)


class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, bilinear: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = _AttentionUp(1024, 512 // factor, bilinear)
        self.up2 = _AttentionUp(512, 256 // factor, bilinear)
        self.up3 = _AttentionUp(256, 128 // factor, bilinear)
        self.up4 = _AttentionUp(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class _AttentionUp(Up):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__(in_channels, out_channels, bilinear)
        self.attention = AttentionBlock(
            f_g=in_channels // 2, f_l=out_channels, f_int=max(1, out_channels // 2)
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x2 = self.attention(g=x1, x=x2)
        return self.conv(torch.cat([x2, x1], dim=1))


class RCAB(nn.Module):
    """Residual Channel Attention Block。"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.ca(out) * out
        return out + x


class ResidualGroup(nn.Module):
    def __init__(self, channels: int, num_rcab: int = 10):
        super().__init__()
        self.rcabs = nn.Sequential(*[RCAB(channels) for _ in range(num_rcab)])
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def _run(inp: torch.Tensor) -> torch.Tensor:
            return self.conv(self.rcabs(inp))

        # 全分辨率 80 个 RCAB 的激活值会轻易吃满 8 GB 显存；检查组在训练时
        # 用梯度检查点重算，数学等价，推理不受影响。
        if self.training and x.requires_grad:
            from torch.utils.checkpoint import checkpoint

            out = checkpoint(_run, x, use_reentrant=False)
        else:
            out = _run(x)
        return out + x


class WRCANet(nn.Module):
    """图像增强主干。"""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, num_groups: int = 8,
                 num_rcab: int = 10, channels: int = 64):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.rgs = nn.Sequential(*[ResidualGroup(channels, num_rcab) for _ in range(num_groups)])
        self.conv_out = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_out(self.rgs(self.conv_in(x))) + x


class WRCANetForSegmentation(nn.Module):
    """增强分支 + 三级 UNet 分割头。out_channels=2 时为 gland + eyelid 双任务。"""

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.out_channels = int(out_channels)
        self.enhancer = WRCANet(in_channels=in_channels, out_channels=in_channels)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        # 必须使用转置卷积（bilinear=False）：Up 的 conv 按 cat 后通道数构造，
        # 若走 Upsample 分支，cat 通道为 in_channels + out_channels，与 conv 不符。
        self.up1 = Up(512, 256, bilinear=False)
        self.up2 = Up(256, 128, bilinear=False)
        self.up3 = Up(128, 64, bilinear=False)
        self.outc = OutConv(64, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enhancer(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)


def build_unet(in_channels: int = 1, out_channels: int = 1) -> nn.Module:
    from monai.networks.nets import UNet

    model = UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(64, 128, 256, 512, 1024),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        act="PRELU",
        norm="INSTANCE",
        dropout=0.1,
    )
    model.out_channels = int(out_channels)
    return model

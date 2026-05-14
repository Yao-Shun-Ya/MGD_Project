"""
WRCANet 模型 - Wide Residual Channel Attention Network
用于图像增强任务
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RCAB(nn.Module):
    """
    Residual Channel Attention Block
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.ca(out) * out
        out += residual
        return out


class RG(nn.Module):
    """
    Residual Group
    """
    def __init__(self, channels, num_rcab=10):
        super().__init__()
        self.rcabs = nn.Sequential(*[RCAB(channels) for _ in range(num_rcab)])
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        residual = x
        out = self.rcabs(x)
        out = self.conv(out)
        out += residual
        return out


class WRCANet(nn.Module):
    """
    Wide Residual Channel Attention Network
    主要用于医学影像增强任务
    """
    def __init__(self, in_channels=1, out_channels=1, num_groups=8, num_rcab=10, channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Shallow Feature Extraction
        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        
        # Residual Groups
        self.rgs = nn.Sequential(*[RG(channels, num_rcab) for _ in range(num_groups)])
        
        # Global Residual Learning
        self.conv_out = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        residual = x
        out = self.conv_in(x)
        out = self.rgs(out)
        out = self.conv_out(out)
        out += residual
        return out


class WRCANetForSegmentation(nn.Module):
    """
    WRCANet 用于分割任务的变体
    将增强后的图像输入到分割头
    """
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        # 图像增强分支
        self.enhancer = WRCANet(in_channels=in_channels, out_channels=in_channels)
        
        # 分割分支（简化的 UNet 结构）
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        
        self.outc = OutConv(64, out_channels)
    
    def forward(self, x):
        # 先增强图像
        enhanced = self.enhancer(x)
        # 再进行分割
        x1 = self.inc(enhanced)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        
        return logits


class DoubleConv(nn.Module):
    """
    双重卷积模块
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    下采样模块
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    上采样模块
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    输出卷积模块
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)
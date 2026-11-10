"""MGD-Former：SegFormer-B2(MiT-B2) 编码器 + 高分辨率融合解码器。

架构决议见 docs/engineering/ARCHITECTURE_DECISION.md。**实现现状（如实记录）**：
- 编码器 4 阶段 stride 4/8/16/32，通道 64/128/320/512；
- 解码器：四级特征经 1×1 proj 统一到 decoder_dim 后，在 **stride 32 分辨率**做
  自顶向下融合（c3/c2/c1 依次下采样到 c4 尺度相加），再经 head + 32× 双线性
  上采样回到原图。注意：这与 ADR 中「在 1/4 与 1/8 分辨率融合、stride=4 输出」
  的描述**不一致**——实际为 stride-32 融合 + 单次上采样；已发表指标
  （2026-11 患者级基准）均在该实现上测得。真正的 FPN 式 stride-4 融合属于
  v2.4.0 架构演进项，变更后必须重训并在 5 折上重新对比；
- 深监督：3 个 1×1 辅助头（当前实现共享 stride-32 融合特征，插值到 1/4、1/8、
  1/16——多尺度多样性有限，见 KNOWN_LIMITATIONS）；
- 双任务头：腺体 + 眼睑（与旧模型 out_channels=2 契约一致）。

只依赖纯 PyTorch，不引入自定义 CUDA 算子，保证 Windows / 离线可安装。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MiTConfig:
    dims: tuple[int, ...] = (64, 128, 320, 512)
    depths: tuple[int, ...] = (3, 4, 6, 3)
    heads: tuple[int, ...] = (1, 2, 5, 8)
    sr_ratios: tuple[int, ...] = (8, 4, 2, 1)
    mlp_ratio: int = 4
    decoder_dim: int = 128


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, kernel: int, stride: int, padding: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel, stride, padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        height, width = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, height, width


class EfficientSelfAttention(nn.Module):
    """带空间缩减（SR）的自注意力；SR=1 时退化为普通多头注意力。"""

    def __init__(self, dim: int, heads: int, sr_ratio: int = 1):
        super().__init__()
        self.heads = int(heads)
        self.scale = (dim // heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.sr_ratio = int(sr_ratio) if sr_ratio > 1 else 1
        if self.sr_ratio > 1:
            self.reduce = nn.Conv2d(dim, dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, tokens, dim = x.shape
        q = self.q(x).reshape(batch, tokens, self.heads, dim // self.heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            grid = x.transpose(1, 2).reshape(batch, dim, height, width)
            grid = self.reduce(grid).flatten(2).transpose(1, 2)
            grid = self.norm(grid)
            kv = self.kv(grid)
        else:
            kv = self.kv(x)
        kv = kv.reshape(batch, -1, 2, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4)
        key, value = kv[0], kv[1]
        attn = (q @ key.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ value).transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj(out)


class MixFFN(nn.Module):
    def __init__(self, dim: int, ratio: int = 4):
        super().__init__()
        hidden = dim * ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.dw = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = self.fc1(x)
        batch, tokens, hidden = x.shape
        grid = x.transpose(1, 2).reshape(batch, hidden, height, width)
        grid = self.act(self.dw(grid))
        x = grid.flatten(2).transpose(1, 2)
        return self.fc2(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, sr_ratio: int, drop_path: float = 0.0, ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim, ratio)
        self.drop = DropPath(drop_path)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), height, width))
        x = x + self.drop(self.ffn(self.norm2(x), height, width))
        return x


class MiTEncoder(nn.Module):
    """MiT-B2 结构（不含 ImageNet/ADE 分类头）。"""

    def __init__(self, in_channels: int = 1, config: MiTConfig | None = None, drop_path: float = 0.1):
        super().__init__()
        self.config = config or MiTConfig()
        dims = self.config.dims
        self.patch1 = OverlapPatchEmbed(in_channels, dims[0], 7, 4, 3)
        self.patch2 = OverlapPatchEmbed(dims[0], dims[1], 3, 2, 1)
        self.patch3 = OverlapPatchEmbed(dims[1], dims[2], 3, 2, 1)
        self.patch4 = OverlapPatchEmbed(dims[2], dims[3], 3, 2, 1)
        total = sum(self.config.depths)
        self.blocks = nn.ModuleList()
        for stage in range(4):
            layer_blocks = []
            for index in range(self.config.depths[stage]):
                # 线性递增的 drop path（SegFormer 官方实现同款）
                rate = drop_path * index / max(1, total - 1)
                layer_blocks.append(
                    TransformerBlock(
                        dims[stage], self.config.heads[stage], self.config.sr_ratios[stage], rate,
                        self.config.mlp_ratio,
                    )
                )
            self.blocks.append(nn.ModuleList(layer_blocks))
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for dim in dims])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        x, height, width = self.patch1(x)
        for index, block in enumerate(self.blocks[0]):
            x = block(x, height, width)
        x = self.norms[0](x).transpose(1, 2).reshape(x.shape[0], -1, height, width)
        features.append(x)

        x, height, width = self.patch2(features[-1])
        for block in self.blocks[1]:
            x = block(x, height, width)
        x = self.norms[1](x).transpose(1, 2).reshape(x.shape[0], -1, height, width)
        features.append(x)

        x, height, width = self.patch3(features[-1])
        for block in self.blocks[2]:
            x = block(x, height, width)
        x = self.norms[2](x).transpose(1, 2).reshape(x.shape[0], -1, height, width)
        features.append(x)

        x, height, width = self.patch4(features[-1])
        for block in self.blocks[3]:
            x = block(x, height, width)
        x = self.norms[3](x).transpose(1, 2).reshape(x.shape[0], -1, height, width)
        features.append(x)
        return features


class FusionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class MGDFormer(nn.Module):
    """默认主干。输出 dict：{"logits": (N,C,H,W), "aux": [(N,1,H/4,W/4), ...]}"""

    ARCH = "MGD-Former"

    def __init__(self, in_channels: int = 1, out_channels: int = 2, config: MiTConfig | None = None,
                 deep_supervision: bool = True, drop_path: float = 0.1):
        super().__init__()
        self.config = config or MiTConfig()
        self.encoder = MiTEncoder(in_channels, self.config, drop_path=drop_path)
        decoder = self.config.decoder_dim
        self.deep_supervision = bool(deep_supervision)
        for index, dim in enumerate(self.config.dims):
            setattr(self, f"proj{index + 1}", nn.Conv2d(dim, decoder, 1))
        self.fuse16 = FusionBlock(decoder)
        self.fuse8 = FusionBlock(decoder)
        self.fuse4 = FusionBlock(decoder)
        self.head = nn.Sequential(
            nn.Conv2d(decoder, decoder, 3, 1, 1),
            nn.BatchNorm2d(decoder),
            nn.GELU(),
            nn.Conv2d(decoder, int(out_channels), 1),
        )
        self.aux_heads = nn.ModuleList(
            [nn.Conv2d(decoder, 1, 1) for _ in range(3)]
        ) if self.deep_supervision else nn.ModuleList()
        self.out_channels = int(out_channels)

    def forward(self, x: torch.Tensor) -> dict:
        height, width = x.shape[-2], x.shape[-1]
        features = self.encoder(x)
        c1, c2, c3, c4 = (
            getattr(self, f"proj{index + 1}")(feature) for index, feature in enumerate(features)
        )
        # 自顶向下逐级融合
        fused = self.fuse16(c4 + F.interpolate(c3, size=c4.shape[-2:], mode="bilinear", align_corners=False))
        fused = self.fuse8(fused + F.interpolate(c2, size=fused.shape[-2:], mode="bilinear", align_corners=False))
        fused = self.fuse4(fused + F.interpolate(c1, size=fused.shape[-2:], mode="bilinear", align_corners=False))
        logits = self.head(fused)
        logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)

        aux: list[torch.Tensor] = []
        if self.deep_supervision:
            aux_sources = (fused, self.fuse8_out(c1, c2, c3, c4), self.fuse16_out(c3, c4))
            positions = (4, 8, 16)
            for head, source, stride in zip(self.aux_heads, aux_sources, positions):
                value = head(source)
                aux.append(
                    F.interpolate(value, size=(max(1, height // stride), max(1, width // stride)),
                                  mode="bilinear", align_corners=False)
                )
        return {"logits": logits, "aux": aux}

    def fuse8_out(self, c1, c2, c3, c4):  # 1/8 辅助头输入
        up = self.fuse16(c4 + F.interpolate(c3, size=c4.shape[-2:], mode="bilinear", align_corners=False))
        return self.fuse8(up + F.interpolate(c2, size=up.shape[-2:], mode="bilinear", align_corners=False))

    def fuse16_out(self, c3, c4):  # 1/16 辅助头输入
        return self.fuse16(c4 + F.interpolate(c3, size=c4.shape[-2:], mode="bilinear", align_corners=False))


def build_mgd_former(in_channels: int = 1, out_channels: int = 2, deep_supervision: bool = True,
                     drop_path: float = 0.1) -> MGDFormer:
    return MGDFormer(in_channels, out_channels, deep_supervision=deep_supervision, drop_path=drop_path)


#: 预训练来源 → HuggingFace 仓库（仅取 encoder 权重）
PRETRAINED_SOURCES = {
    "ade20k": "nvidia/segformer-b2-finetuned-ade-512-512",
    "imagenet": "nvidia/mit-b2",
}


def load_pretrained_encoder(model: MGDFormer, source: str = "auto") -> dict:
    """尝试加载预训练编码器；离线/无缓存时**明确降级**并返回原因，不静默假装成功。"""
    from .pretrained import try_load_encoder

    return try_load_encoder(model, PRETRAINED_SOURCES, source)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

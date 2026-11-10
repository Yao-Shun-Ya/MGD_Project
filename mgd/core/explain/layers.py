"""逐层归因：LayerCAM 与 Score-CAM（与 visualize.TrueGradCAM 并列、同风格）。

相比 Grad-CAM（全局平均池化权重），LayerCAM 保留逐通道空间结构，
对细长低对比结构（睑板腺正是这种形态）定位更准；Score-CAM 无梯度、
通过掩膜前向打分加权，作为梯度法的交叉验证手段。

实现契约（WP-3 验收口径）：
- 输出契约：dict / tuple / Tensor 三种模型输出统一经 unwrap_outputs 解包取 logits；
- 默认全分辨率：内部不做任何输入降采样，CAM 上采样回输入空间尺寸后返回
  （out_size 仅用于显式改变输出尺寸）；
- hook 注册后必须在 finally 中 remove()，异常路径同样清理；
- 计算时 model.eval() + torch.enable_grad（包在 torch.no_grad 外层内执行）；
- 输入使用 requires_grad_() 的副本，不污染调用方张量；
- 选层：显式传入 > 自动探测（取「反向传播后梯度非全零」的最后一个 Conv2d）；
  探测失败 raise RuntimeError（消息含「未找到有梯度响应的卷积层」+原因），
  绝不返回全零图糊弄。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8

#: 自动探测时挂钩的候选卷积层上限：主输出头通常在末尾几个卷积内，
#: 同时避免高分辨率输入缓存全部中间激活造成内存膨胀（与 TrueGradCAM 一致）。
_MAX_PROBE_CONVS = 12


def _channel_probability(logits: torch.Tensor, channel: int) -> torch.Tensor:
    """logits (N,C,H,W) → 目标通道概率 (N,H,W)：多通道 softmax、单通道 sigmoid。"""
    if logits.ndim != 4:
        raise ValueError(f"期望 4D logits (N,C,H,W)，实际形状 {tuple(logits.shape)}")
    num_classes = int(logits.shape[1])
    ch = int(channel)
    if ch < 0 or ch >= num_classes:
        raise ValueError(f"目标通道 {channel} 越界：logits 共 {num_classes} 个通道")
    if num_classes > 1:
        return torch.softmax(logits, dim=1)[:, ch]
    return torch.sigmoid(logits[:, 0])


def _channel_score(logits: torch.Tensor, channel: int) -> torch.Tensor:
    """目标通道概率的空间均值 → 反向传播用的标量分数。"""
    return _channel_probability(logits, channel).mean()


def _conv_candidates(model: nn.Module) -> list[nn.Module]:
    """候选卷积层：末尾 12 个 Conv2d。一个都没有 → 探测失败的诚实报错。"""
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise RuntimeError("未找到有梯度响应的卷积层：模型中不存在任何 Conv2d 层，无法做逐层归因")
    return convs[-_MAX_PROBE_CONVS:]


class _LayerCapture:
    """在候选层上挂 forward/backward hook，捕获激活 A 与梯度 dY/dA。"""

    def __init__(self, candidates: list[nn.Module]) -> None:
        self.candidates = list(candidates)
        self.activations: dict[nn.Module, torch.Tensor] = {}
        self.gradients: dict[nn.Module, torch.Tensor] = {}
        self._handles: list = []

    def attach(self) -> None:
        """注册全部 hook；中途失败时回滚已注册的 handle（不留半注册窗口）。"""
        self._handles = []
        try:
            for module in self.candidates:
                self._handles.append(module.register_forward_hook(self._forward_hook))
                self._handles.append(
                    module.register_full_backward_hook(self._backward_hook)
                )
        except BaseException:
            self.detach()
            raise

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _forward_hook(self, module, inputs, output):
        self.activations[module] = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        if grad_output and grad_output[0] is not None:
            self.gradients[module] = grad_output[0].detach()

    def select_by_gradient(self) -> nn.Module:
        """取「反向传播后梯度非全零」的最后一个候选层；找不到则抛 RuntimeError。"""
        for module in reversed(self.candidates):
            grad = self.gradients.get(module)
            if grad is not None and bool(grad.any()):
                return module
        if not self.candidates:
            reason = "候选层列表为空"
        elif not any(module in self.gradients for module in self.candidates):
            reason = "反向传播未到达任何候选卷积层（检查模型输出与目标通道是否连通）"
        else:
            reason = "反向传播已到达候选卷积层，但所有梯度均为全零（该层对目标通道无响应）"
        raise RuntimeError(f"未找到有梯度响应的卷积层：{reason}")


def _run_grad_pass(model: nn.Module, tensor: torch.Tensor, channel: int, capture: _LayerCapture) -> None:
    """eval + enable_grad（包在 no_grad 外层内）执行一次前向/反向，触发 hook 捕获。

    hook 在本函数 finally 中移除（异常路径同样清理）；attach 本身自带回滚
    （中途注册失败不留半注册窗口）；输入用 requires_grad_() 的副本，不污染
    调用方张量；backward 前保存调用方参数梯度、finally 中原样恢复——
    不污染调用方参数梯度（原来有 grad 的还原原张量，原来没有的置 None）。
    """
    x = tensor.detach().clone().requires_grad_(True)
    was_training = model.training
    saved_grads = [(p, p.grad) for p in model.parameters() if p.requires_grad]
    model.eval()
    try:
        capture.attach()
        with torch.no_grad():
            with torch.enable_grad():
                from ..models.registry import unwrap_outputs

                logits = unwrap_outputs(model(x))[0]
                score = _channel_score(logits, channel)
                model.zero_grad(set_to_none=True)
                score.backward()
    finally:
        capture.detach()
        for param, grad in saved_grads:
            param.grad = grad
        if was_training:
            model.train()


def _probe_target_layer(model: nn.Module, tensor: torch.Tensor, channel: int) -> nn.Module:
    """自动选层：一次带梯度的前向/反向，取「梯度非全零」的最后一个 Conv2d。"""
    capture = _LayerCapture(_conv_candidates(model))
    _run_grad_pass(model, tensor, channel, capture)
    return capture.select_by_gradient()


class LayerCAM:
    """LayerCAM：逐通道 relu(梯度) 空间加权，不做全局平均池化。

    相较 Grad-CAM 对细长低对比结构定位更准。

    实现说明（与 visualize.TrueGradCAM 同风格）：
    - 正向 hook 抓激活 A、反向 hook 抓梯度 dY/dA；
    - cam = ReLU(Σ_c A_c ⊙ ReLU(dY/dA)_c)，归一化到 [0,1]；
    - 分数：logits (N,C,H,W) 的目标通道经 softmax（C>1）或 sigmoid（C=1）后取空间均值；
    - 默认全分辨率：不做输入降采样，CAM 双线性上采样回输入空间尺寸（out_size 可覆盖）；
    - hook 在 finally 中移除（异常路径同样清理）；model.eval() + torch.enable_grad
      包在 torch.no_grad 外层内执行；输入用 requires_grad_() 的副本；
    - 选层：显式 target_layers 优先；否则自动探测「反向传播后梯度非全零」的最后一个
      Conv2d（候选取末尾 12 个卷积）；探测失败 raise RuntimeError，绝不返回全零图糊弄。
    """

    def __init__(self, model: nn.Module, target_layers: list | None = None) -> None:
        self.model = model
        self.target_layers = list(target_layers) if target_layers else None
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

    def generate(
        self,
        tensor: torch.Tensor,
        channel: int = 0,
        out_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """返回 [0,1] 的 LayerCAM 图（float32）；tensor 形状 (1,C,H,W)。

        形状 = out_size（若给）否则输入空间形状 (H,W)。
        """
        if tensor.ndim != 4 or tensor.shape[0] != 1:
            raise ValueError("LayerCAM 仅支持 4D 输入且 batch=1")
        candidates = self.target_layers if self.target_layers is not None else _conv_candidates(self.model)
        capture = _LayerCapture(candidates)
        # 前向/反向 + hook 清理都在 _run_grad_pass 内完成（异常路径同样清理）
        _run_grad_pass(self.model, tensor, channel, capture)
        layer = capture.select_by_gradient()
        activations = capture.activations[layer]
        gradients = capture.gradients[layer]
        self.activations = activations
        self.gradients = gradients
        with torch.no_grad():
            cam = torch.relu((activations * torch.relu(gradients)).sum(dim=1, keepdim=True))
            cam_max = float(cam.max())
            # 有限性守卫：NaN 会与任何比较为 False，靠 <= 0.0 拦不住全 NaN 的 CAM，
            # 必须显式检查 isfinite，绝不静默返回全 NaN 图。
            if not np.isfinite(cam_max) or cam_max <= 0.0:
                raise RuntimeError(
                    "LayerCAM 失败：CAM 含 NaN/Inf 或全零——常见原因是模型权重含 NaN/训练发散；"
                    "全零时也可能是所选层「激活 × ReLU(梯度)」无任何正响应"
                    "（可尝试显式传入 target_layers 或检查目标通道）"
                )
            cam = (cam - cam.min()) / ((cam.max() - cam.min()) + _EPS)
            target = (
                tuple(int(v) for v in out_size)
                if out_size is not None
                else (int(tensor.shape[-2]), int(tensor.shape[-1]))
            )
            cam = F.interpolate(cam, size=target, mode="bilinear", align_corners=False)
            return cam[0, 0].detach().cpu().numpy().astype(np.float32)


class ScoreCAM:
    """Score-CAM：激活通道上采样为掩膜、掩膜前向打分加权。

    max_channels 控制候选通道上限（开销控制）：按激活绝对值的空间均值取 top-k。

    实现说明：
    - target_layer 为空时走与 LayerCAM 相同的自动探测（梯度非全零的最后一个 Conv2d，
      探测失败抛 RuntimeError，绝不返回全零图糊弄）；
    - 掩膜前向批量化：所有候选掩膜拼成一个 batch，一次 forward 完成打分，控制耗时；
    - 每个掩膜 min-max 归一化到 [0,1]；通道权重对打分做 min-max 归一化
      （所有候选打分无差异时退化为均匀权重）；
    - cam = ReLU(Σ_k w_k · mask_k) 归一化到 [0,1]，双线性上采样回输入空间尺寸；
    - 默认全分辨率返回；hook 在 finally 中移除；输入用副本，不污染调用方张量。
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None, max_channels: int = 32) -> None:
        self.model = model
        self.target_layer = target_layer
        self.max_channels = max(1, int(max_channels))

    def generate(
        self,
        tensor: torch.Tensor,
        channel: int = 0,
        out_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """返回 [0,1] 的 Score-CAM 图（float32）；tensor 形状 (1,C,H,W)。"""
        if tensor.ndim != 4 or tensor.shape[0] != 1:
            raise ValueError("Score-CAM 仅支持 4D 输入且 batch=1")
        layer = self.target_layer if self.target_layer is not None else _probe_target_layer(self.model, tensor, channel)

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                from ..models.registry import unwrap_outputs

                x = tensor.detach().clone()
                # 第一步：只挂 forward hook 抓一次激活，抓完立即移除，不污染后续批量前向
                capture = _LayerCapture([layer])
                try:
                    capture.attach()
                    unwrap_outputs(self.model(x))[0]
                finally:
                    capture.detach()
                if layer not in capture.activations:
                    raise RuntimeError("Score-CAM 未捕获到目标层激活（forward hook 未触发）")
                activation = capture.activations[layer].float()  # (1,Ca,h,w)

                num_channels = int(activation.shape[1])
                k = min(self.max_channels, num_channels)
                magnitude = activation.detach().abs().mean(dim=(2, 3)).flatten()
                top = torch.topk(magnitude, k).indices.tolist()

                height, width = int(x.shape[-2]), int(x.shape[-1])
                masks = []
                for idx in top:
                    single = activation[0, int(idx)]
                    up = F.interpolate(
                        single[None, None], size=(height, width), mode="bilinear", align_corners=False
                    )[0, 0]
                    span = float(up.max() - up.min())
                    masks.append((up - up.min()) / (span + _EPS) if span > 0 else torch.zeros_like(up))
                mask_stack = torch.stack(masks, dim=0)  # (k,H,W)，每个掩膜已归一化 [0,1]
                # 掩膜在 float32 域计算；半精度模型（fp16）下必须转回输入 dtype，
                # 否则 float32 掩膜会把 masked_batch 提升成 fp32，与 fp16 权重不匹配报错。
                mask_stack = mask_stack.to(dtype=x.dtype)

                # 第二步：批量化——所有候选掩膜拼一个 batch，一次 forward 完成打分
                masked_batch = x.repeat(k, 1, 1, 1) * mask_stack.unsqueeze(1)
                logits = unwrap_outputs(self.model(masked_batch))[0]
                scores = _channel_probability(logits, channel).mean(dim=(1, 2))  # (k,)
                s_min, s_max = float(scores.min()), float(scores.max())
                if s_max - s_min > 1e-12:
                    weights = (scores - s_min) / (s_max - s_min)
                else:
                    weights = torch.full_like(scores, 1.0 / k)

                cam = torch.relu((weights[:, None, None] * mask_stack).sum(dim=0))
                cam_max = float(cam.max())
                # 有限性守卫：与 LayerCAM 同型——NaN 与任何比较为 False，
                # 必须显式检查 isfinite，绝不静默返回全 NaN 图。
                if not np.isfinite(cam_max) or cam_max <= 0.0:
                    raise RuntimeError(
                        "Score-CAM 失败：CAM 含 NaN/Inf 或全零——常见原因是模型权重含 NaN/训练发散；"
                        "全零时也可能是候选通道激活均为常量（掩膜退化），无法归因"
                    )
                cam = (cam - cam.min()) / ((cam.max() - cam.min()) + _EPS)
                target = (
                    tuple(int(v) for v in out_size)
                    if out_size is not None
                    else (height, width)
                )
                cam = F.interpolate(cam[None, None], size=target, mode="bilinear", align_corners=False)[0, 0]
                return cam.detach().cpu().numpy().astype(np.float32)
        finally:
            if was_training:
                self.model.train()

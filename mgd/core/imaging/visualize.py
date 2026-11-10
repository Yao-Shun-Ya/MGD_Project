"""可视化。全部输出 RGB uint8，供 UI 直接展示。

Phase 6：新增真正的 Grad-CAM（forward/backward hook）与像素级不确定性热力图。
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn


def _ensure_gray_u8(gray: np.ndarray) -> np.ndarray:
    if gray.dtype != np.uint8:
        arr = gray.astype(np.float32)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)
    return gray


def probability_colormap(prob_map: np.ndarray) -> np.ndarray:
    """概率图 → JET 伪彩（注意：这是概率热力图，不是 Grad-CAM）。"""
    prob = np.clip(np.asarray(prob_map, dtype=np.float32), 0.0, 1.0)
    colored = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def generate_heatmap(prob_map: np.ndarray, gray_img: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    gray_u8 = _ensure_gray_u8(gray_img)
    base = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB)
    heat = probability_colormap(prob_map)
    return cv2.addWeighted(base, 1.0 - alpha, heat, alpha, 0)


def binary_overlay(gray_img: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha: float = 0.3) -> np.ndarray:
    gray_u8 = _ensure_gray_u8(gray_img)
    base = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB)
    layer = np.zeros_like(base, dtype=np.uint8)
    layer[np.asarray(mask) > 0] = color
    return cv2.addWeighted(base, 1.0 - alpha, layer, alpha, 0)


def generate_contour_visualization(
    prob_map: np.ndarray,
    gray_img: np.ndarray,
    min_area: int = 80,
    threshold: float = 0.5,
) -> np.ndarray:
    base = cv2.cvtColor(_ensure_gray_u8(gray_img), cv2.COLOR_GRAY2RGB)
    # 轮廓必须与实际使用的二值化阈值一致（默认推理阈值 0.7；0.5 等值线会圈住被排除的区域）
    binary = (np.asarray(prob_map) >= float(threshold)).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = [c for c in contours if cv2.contourArea(c) >= max(1, int(min_area))]
    cv2.drawContours(base, kept, -1, (0, 255, 120), 2)
    return base


def generate_skeleton_visualization(mask: np.ndarray, gray_img: np.ndarray) -> np.ndarray:
    base = cv2.cvtColor(_ensure_gray_u8(gray_img), cv2.COLOR_GRAY2RGB)
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    skeleton = _skeletonize(binary)
    base[skeleton > 0] = (255, 80, 80)
    return base


def _skeletonize(binary: np.ndarray) -> np.ndarray:
    """形态学骨架；不依赖 opencv-contrib 的 ximgproc。

    终止条件必须是「腐蚀不再改变图像」，而不能只看非零像素是否归零：
    孤立像素在默认边界（腐蚀视为 +inf）下永远腐蚀不掉，只看归零会死循环。
    """
    img = binary.copy()
    skeleton = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(img) > 0:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skeleton = cv2.bitwise_or(skeleton, temp)
        if cv2.countNonZero(cv2.subtract(img, eroded)) == 0:
            break
        img = eroded
    return skeleton


def generate_instance_visualization(mask: np.ndarray, gray_img: np.ndarray, min_area: int = 80) -> np.ndarray:
    """不同腺体实例用不同颜色标注。"""
    base = cv2.cvtColor(_ensure_gray_u8(gray_img), cv2.COLOR_GRAY2RGB)
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    palette = [
        (66, 133, 244), (52, 168, 83), (251, 188, 5), (234, 67, 53),
        (155, 114, 203), (0, 172, 193), (255, 138, 128), (140, 195, 74),
    ]
    overlay = base.copy()
    kept = 0
    for index in range(1, num_labels):
        if stats[index, cv2.CC_STAT_AREA] < max(1, int(min_area)):
            continue
        overlay[labels == index] = palette[kept % len(palette)]
        kept += 1
    if kept == 0:
        return base
    return cv2.addWeighted(base, 0.45, overlay, 0.55, 0)

# --------------------------------------------------------------------------- #
# Phase 6: True Grad-CAM + Uncertainty
# --------------------------------------------------------------------------- #

def generate_probability_map(prob_map: np.ndarray, gray_img: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """旧 generate_heatmap 的语义别名（概率伪彩，不是 Grad-CAM）。"""
    return generate_heatmap(prob_map, gray_img, alpha=alpha)


def _conv_layers(model: nn.Module) -> list[nn.Conv2d]:
    layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    if not layers:
        raise ValueError("模型中未找到 Conv2d 层，无法挂载 Grad-CAM")
    return layers


def _last_conv_layer(model: nn.Module) -> nn.Module:
    return _conv_layers(model)[-1]


class TrueGradCAM:
    """基于 forward/backward hook 的 Grad-CAM。

    MGD-Former 的最后一个 Conv2d 可能属于未被主 logits 使用的辅助头，
    因此默认在全部卷积层上挂钩，反向结束后取「最后一个真正收到梯度」的层。
    这样既保留 UNet 的原有行为，也避免 Transformer 解码器选错层导致无梯度。
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model
        self.target_layer = target_layer
        # 只保留末尾 12 个卷积：主输出头通常在此范围内，同时避免在
        # 高分辨率输入上缓存全部中间激活造成显存/内存膨胀。
        self._candidates: list[nn.Module] = (
            [target_layer] if target_layer is not None else _conv_layers(model)[-12:]
        )
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._activations: dict[nn.Module, torch.Tensor] = {}
        self._gradients: dict[nn.Module, torch.Tensor] = {}
        self._handles: list = []

    def _forward_hook(self, module, inputs, output):
        self._activations[module] = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        if grad_output and grad_output[0] is not None:
            self._gradients[module] = grad_output[0].detach()

    def _select_layer(self) -> nn.Module:
        for module in reversed(self._candidates):
            if module in self._gradients:
                return module
        raise RuntimeError("Grad-CAM 未捕获到任何卷积层梯度")

    def generate(self, tensor: torch.Tensor, channel: int = 0, out_size: tuple[int, int] | None = None) -> np.ndarray:
        """返回 [0,1] 的 CAM；tensor 形状 (1,C,H,W)。"""
        if tensor.shape[0] != 1:
            raise ValueError("Grad-CAM 仅支持 batch=1")
        self._activations = {}
        self._gradients = {}
        self._handles = [
            module.register_forward_hook(self._forward_hook)
            for module in self._candidates
        ] + [
            module.register_full_backward_hook(self._backward_hook)
            for module in self._candidates
        ]
        try:
            was_training = self.model.training
            self.model.eval()
            with torch.enable_grad():
                from ..models.registry import unwrap_outputs

                logits = unwrap_outputs(self.model(tensor))[0]
                if logits.shape[1] > 1:
                    logits = logits[:, int(channel): int(channel) + 1]
                score = torch.sigmoid(logits).sum()
                self.model.zero_grad(set_to_none=True)
                score.backward()
            selected = self._select_layer()
            self.target_layer = selected
            self.activations = self._activations[selected]
            self.gradients = self._gradients[selected]
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            target = out_size or (tensor.shape[-2], tensor.shape[-1])
            cam = torch.nn.functional.interpolate(cam, size=target, mode="bilinear", align_corners=False)
            if was_training:
                self.model.train()
            return cam.squeeze().detach().cpu().numpy().astype(np.float32)
        finally:
            for handle in self._handles:
                handle.remove()
            self._handles = []


def gradcam_overlay(cam: np.ndarray, gray_img: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    base = cv2.cvtColor(_ensure_gray_u8(gray_img), cv2.COLOR_GRAY2RGB)
    heat = probability_colormap(cam)
    return cv2.addWeighted(base, 1.0 - alpha, heat, alpha, 0)


def pixel_entropy(prob_map: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """像素级香农熵 H(P)，以 2 为底，取值 [0,1]。"""
    p = np.clip(np.asarray(prob_map, dtype=np.float32), eps, 1.0 - eps)
    return (-(p * np.log2(p)) - ((1.0 - p) * np.log2(1.0 - p))).astype(np.float32)


def tta_variance(probability_stack: np.ndarray) -> np.ndarray:
    """多次 TTA 预测的像素方差；stack 形状 (N,H,W)。"""
    stack = np.asarray(probability_stack, dtype=np.float32)
    if stack.ndim == 2:
        stack = stack[None]
    return np.clip(stack.var(axis=0), 0.0, 1.0)


def uncertainty_map(prob_map: np.ndarray, tta_stack: np.ndarray | None = None) -> np.ndarray:
    """认知不确定性 = 熵 + TTA 方差（可选），用于伪彩展示。"""
    entropy = pixel_entropy(prob_map)
    if tta_stack is not None:
        variance = tta_variance(tta_stack)
        entropy = np.clip(entropy + variance, 0.0, 1.0)
    return entropy


def uncertainty_overlay(uncertainty: np.ndarray, gray_img: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    base = cv2.cvtColor(_ensure_gray_u8(gray_img), cv2.COLOR_GRAY2RGB)
    heat = cv2.applyColorMap(np.clip(uncertainty * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base, 1.0 - alpha, heat, alpha, 0)


def confidence_score(
    prob_map: np.ndarray,
    ambiguous_low: float = 0.4,
    ambiguous_high: float = 0.6,
    review_ratio: float = 0.10,
) -> dict:
    """0–100 置信度评分。

    confidence = 100 × (1 − 模糊像素占比)，模糊像素定义为 p ∈ [0.4, 0.6]；
    模糊占比 > review_ratio（默认 10%）时建议人工复核。
    """
    prob = np.clip(np.asarray(prob_map, dtype=np.float32), 0.0, 1.0)
    ambiguous = (prob >= ambiguous_low) & (prob <= ambiguous_high)
    ratio = float(ambiguous.mean())
    score = float(max(0.0, min(100.0, 100.0 * (1.0 - ratio))))
    return {
        "score": score,
        "ambiguous_ratio": ratio,
        "mean_entropy": float(pixel_entropy(prob).mean()),
        "need_review": bool(ratio > float(review_ratio)),
        "ambiguous_low": float(ambiguous_low),
        "ambiguous_high": float(ambiguous_high),
    }

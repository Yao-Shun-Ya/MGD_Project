"""损失与优化器。

修复点：pos_weight 不再在 CPU 上预创建（否则 CUDA 上会报设备不一致）。
Phase 4 新增：Tversky / FocalTversky / 可微分 clDice / 解剖联合损失。
"""
from __future__ import annotations

import re

import torch
import torch.nn.functional as F
from monai.losses import DiceLoss, FocalLoss

from ..runtime.errors import ConfigError

LOSS_OPTIONS: tuple[str, ...] = (
    "MGD 复合损失",
    "Dice+CE 组合损失",
    "Dice Loss",
    "BCE (pos_weight)",
    "Focal Loss",
    "Tversky Loss",
    "FocalTversky Loss",
    "SoftclDice + Dice",
    "AnatomicalJointLoss (解剖联合)",
)

_EPS = 1e-6


def _key(name: str) -> str:
    return re.sub(r"[^0-9a-z一-龥]", "", str(name).lower())


class WeightedBCE(torch.nn.Module):
    """BCEWithLogits + pos_weight，pos_weight 随预测张量设备创建。"""

    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = float(pos_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = torch.as_tensor(self.pos_weight, dtype=pred.dtype, device=pred.device)
        return F.binary_cross_entropy_with_logits(pred, target, pos_weight=weight)


class CombinedLoss(torch.nn.Module):
    def __init__(self, first: torch.nn.Module, second: torch.nn.Module, weight1: float = 0.5, weight2: float = 0.5):
        super().__init__()
        self.first = first
        self.second = second
        self.weight1 = float(weight1)
        self.weight2 = float(weight2)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.weight1 * self.first(pred, target) + self.weight2 * self.second(pred, target)


class TverskyLoss(torch.nn.Module):
    """Tversky：alpha=0.3, beta=0.7 —— 对漏检（FN）惩罚更重，提高召回。"""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(pred_logits)
        target = (target > 0.5).float()
        dims = tuple(range(1, prob.ndim))
        tp = (prob * target).sum(dims)
        fp = (prob * (1.0 - target)).sum(dims)
        fn = ((1.0 - prob) * target).sum(dims)
        score = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - score.mean()


class FocalTverskyLoss(TverskyLoss):
    """FocalTversky：gamma=1.33，聚焦困难样本。"""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.33, smooth: float = 1.0):
        super().__init__(alpha=alpha, beta=beta, smooth=smooth)
        self.gamma = float(gamma)

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.pow(super().forward(pred_logits, target), self.gamma)


def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iterations: int = 5) -> torch.Tensor:
    """可微分骨架提取（Iterative Soft-Skeletonization，clDice 论文实现）。"""
    img1 = _soft_open(img)
    skeleton = F.relu(img - img1)
    for _ in range(int(iterations)):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


class SoftclDiceLoss(torch.nn.Module):
    """软骨架中心线 Dice 损失：惩罚腺体断裂。"""

    def __init__(self, iterations: int = 5, smooth: float = 1.0):
        super().__init__()
        self.iterations = int(iterations)
        self.smooth = float(smooth)

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(pred_logits)
        target = (target > 0.5).float()
        skel_pred = soft_skeletonize(prob, self.iterations)
        skel_gt = soft_skeletonize(target, self.iterations)
        dims = tuple(range(1, pred_logits.ndim))
        tprec = (skel_pred * target).sum(dims) + self.smooth
        tprec = tprec / (skel_pred.sum(dims) + self.smooth)
        tsens = (skel_gt * prob).sum(dims) + self.smooth
        tsens = tsens / (skel_gt.sum(dims) + self.smooth)
        cldice = 2.0 * tprec * tsens / (tprec + tsens + _EPS)
        return 1.0 - cldice.mean()


class DualTaskLoss(torch.nn.Module):
    """双任务监督：gland 通道 + λ1·eyelid 通道。"""

    dual_task = True

    def __init__(self, base: torch.nn.Module, eyelid_weight: float = 0.3):
        super().__init__()
        self.base = base
        self.eyelid_weight = float(eyelid_weight)

    def forward(self, logits: torch.Tensor, gland: torch.Tensor, eyelid: torch.Tensor | None = None) -> torch.Tensor:
        loss = self.base(logits[:, 0:1], gland)
        if eyelid is not None:
            loss = loss + self.eyelid_weight * self.base(logits[:, 1:2], eyelid)
        return loss


class AnatomicalJointLoss(torch.nn.Module):
    """L_total = L_gland + λ1·L_eyelid + λ2·L_clDice + λ3·L_out-of-ROI。"""

    dual_task = True

    def __init__(
        self,
        gland_loss: torch.nn.Module | None = None,
        eyelid_loss: torch.nn.Module | None = None,
        lambda_eyelid: float = 0.3,
        lambda_cldice: float = 0.5,
        lambda_out: float = 0.2,
        cldice_iterations: int = 5,
    ):
        super().__init__()
        self.gland_loss = gland_loss or CombinedLoss(DiceLoss(sigmoid=True), WeightedBCE(1.0))
        self.eyelid_loss = eyelid_loss or self.gland_loss
        self.lambda_eyelid = float(lambda_eyelid)
        self.lambda_cldice = float(lambda_cldice)
        self.lambda_out = float(lambda_out)
        self.cldice = SoftclDiceLoss(iterations=cldice_iterations)

    def forward(self, logits: torch.Tensor, gland: torch.Tensor, eyelid: torch.Tensor | None = None) -> torch.Tensor:
        if logits.shape[1] < 2:
            raise ConfigError("AnatomicalJointLoss 需要双通道输出（gland + eyelid）")
        gland_prob = torch.sigmoid(logits[:, 0:1])
        loss = self.gland_loss(logits[:, 0:1], gland)
        if eyelid is not None:
            loss = loss + self.lambda_eyelid * self.eyelid_loss(logits[:, 1:2], eyelid)
            # 解剖越界惩罚：腺体概率 × (1 - 眼睑 ROI)
            out_roi = (gland_prob * (1.0 - eyelid)).mean()
            loss = loss + self.lambda_out * out_roi
        loss = loss + self.lambda_cldice * self.cldice(logits[:, 0:1], gland)
        return loss


class MGDCompositeLoss(torch.nn.Module):
    """决议默认损失：0.4 Dice + 0.4 CE + clDice + λ1·眼睑 + 可选边界项。

    clDice 权重由引擎按 warmup 轮数动态改写（`cldice_weight`），
    训练早期为 0，避免骨架噪声主导梯度。
    """

    dual_task = True

    def __init__(
        self,
        pos_weight: float = 1.0,
        eyelid_weight: float = 0.25,
        cldice_weight: float = 0.2,
        boundary_weight: float = 0.0,
        iterations: int = 5,
    ):
        super().__init__()
        self.base = CombinedLoss(DiceLoss(sigmoid=True), WeightedBCE(pos_weight), 0.4, 0.4)
        self.cldice = SoftclDiceLoss(iterations=iterations)
        self.eyelid_weight = float(eyelid_weight)
        self.cldice_weight = float(cldice_weight)
        self.boundary_weight = float(boundary_weight)

    def _boundary_target(self, target: torch.Tensor) -> torch.Tensor:
        eroded = -F.max_pool2d(-target, 3, 1, 1)
        return F.relu(target - eroded)

    def forward(self, logits: torch.Tensor, gland: torch.Tensor, eyelid: torch.Tensor | None = None) -> torch.Tensor:
        loss = self.base(logits[:, 0:1], gland)
        if self.cldice_weight > 0.0:
            loss = loss + self.cldice_weight * self.cldice(logits[:, 0:1], gland)
        if self.boundary_weight > 0.0:
            boundary = self._boundary_target((gland > 0.5).float())
            weight = 1.0 + 3.0 * boundary
            loss = loss + self.boundary_weight * F.binary_cross_entropy_with_logits(
                logits[:, 0:1], gland, weight=weight
            )
        if eyelid is not None and self.eyelid_weight > 0.0:
            loss = loss + self.eyelid_weight * self.base(logits[:, 1:2], eyelid)
        return loss


def deep_supervision_loss(
    aux: list[torch.Tensor],
    gland: torch.Tensor,
    base: torch.nn.Module | None = None,
    weights: tuple[float, ...] = (1.0, 0.5, 0.25),
) -> torch.Tensor | None:
    """辅助头损失：把标注下采样到各尺度，按 1.0/0.5/0.25 加权平均。"""
    if not aux:
        return None
    criterion = base or CombinedLoss(DiceLoss(sigmoid=True), WeightedBCE(1.0))
    total = None
    for tensor, weight in zip(aux, weights):
        target = F.interpolate(gland, size=tensor.shape[-2:], mode="nearest")
        value = criterion(tensor, target)
        total = value * weight if total is None else total + value * weight
    return total / float(sum(weights[: len(aux)]))


def _base_loss(loss_name: str, pos_weight: float) -> torch.nn.Module:
    key = _key(loss_name)
    if "mgd复合" in key or ("复合" in key and "损失" in key) or "composite" in key:
        return MGDCompositeLoss(pos_weight=pos_weight)
    if "anatomical" in key or "解剖联合" in key:
        return AnatomicalJointLoss()
    if "focaltversky" in key:
        return FocalTverskyLoss()
    if "tversky" in key:
        return TverskyLoss()
    if "cldice" in key or "软骨架" in key:
        return CombinedLoss(SoftclDiceLoss(), DiceLoss(sigmoid=True), weight1=0.5, weight2=0.5)
    if "dice" in key and ("组合" in key or "bce" in key):
        return CombinedLoss(DiceLoss(sigmoid=True), WeightedBCE(pos_weight))
    if "dice" in key:
        return DiceLoss(sigmoid=True)
    if key in ("crossentropy", "ce", "bce", "bceloss", "bcewithlogits") or key.startswith("bce"):
        return WeightedBCE(pos_weight)
    if "focal" in key:
        return FocalLoss(alpha=0.25, gamma=2.0)
    raise ConfigError(f"未知损失函数：{loss_name}")


def get_loss_function(
    loss_name: str,
    pos_weight: float = 1.0,
    *,
    dual_task: bool = False,
    eyelid_weight: float = 0.3,
    cldice_weight: float = 0.2,
    boundary_weight: float = 0.0,
) -> torch.nn.Module:
    """返回损失模块。

    dual_task=True 且损失本身不处理双通道时，用 DualTaskLoss 包装为 gland + eyelid 监督。
    双任务损失模块的 forward 签名为 (logits, gland, eyelid)。
    """
    base = _base_loss(loss_name, pos_weight)
    if isinstance(base, MGDCompositeLoss):
        base.eyelid_weight = float(eyelid_weight)
        base.cldice_weight = float(cldice_weight)
        base.boundary_weight = float(boundary_weight)
        return base
    if getattr(base, "dual_task", False):
        if not dual_task:
            raise ConfigError(f"损失 {loss_name} 需要双通道模型（dual_task=True）")
        return base
    return DualTaskLoss(base, eyelid_weight) if dual_task else base


def get_optimizer(optimizer_name: str, model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    key = _key(optimizer_name)
    if key == "adam":
        return torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    if key in ("adamw", "adamw默认"):
        return torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    if key in ("sgd", "sgd默认"):
        return torch.optim.SGD(model.parameters(), lr=float(lr), weight_decay=float(weight_decay), momentum=0.9)
    raise ConfigError(f"未知优化器：{optimizer_name}")


def get_scheduler(
    scheduler_name: str,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    total_batches: int,
):
    """返回 (scheduler, is_batch_level)。"""
    key = _key(scheduler_name)
    total_steps = max(1, int(epochs) * max(1, int(total_batches)))

    if key.startswith("cosineannealingwarmrestarts"):
        return (
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-6),
            True,
        )
    if key.startswith("reducelronplateau") or key in ("plateau", "默认"):
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10),
            False,
        )
    if key.startswith("cosineannealing"):
        eta_min = float(optimizer.param_groups[0]["lr"]) * 0.01
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min), True
    if key.startswith("stepdecay") or key == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(epochs * 0.3)), gamma=0.5), False
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0), False

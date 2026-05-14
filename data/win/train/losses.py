"""
损失函数模块
"""
import torch
from monai.losses import DiceLoss, FocalLoss


class CombinedLoss(torch.nn.Module):
    """
    组合损失函数：将两个损失函数加权组合
    """
    
    def __init__(self, loss_fn1, loss_fn2, weight1=0.5, weight2=0.5):
        super().__init__()
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.weight1 = weight1
        self.weight2 = weight2
    
    def forward(self, pred, target):
        return self.weight1 * self.loss_fn1(pred, target) + self.weight2 * self.loss_fn2(pred, target)


def get_loss_function(loss_name, pos_weight=1.0):
    """
    获取损失函数
    
    Args:
        loss_name: 损失函数名称
        pos_weight: 正负样本权重
    
    Returns:
        损失函数实例
    """
    if loss_name == "Dice Loss":
        return DiceLoss(sigmoid=True)
    elif loss_name == "Cross Entropy":
        return torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    elif loss_name == "Focal Loss":
        return FocalLoss(alpha=0.25, gamma=2.0)
    elif loss_name == "Dice+CE 组合损失":
        loss_fn_main = DiceLoss(sigmoid=True)
        loss_fn_ce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        return CombinedLoss(loss_fn_main, loss_fn_ce)
    else:
        raise ValueError(f"未知损失函数: {loss_name}")


def get_optimizer(optimizer_name, model, lr, weight_decay):
    """
    获取优化器
    
    Args:
        optimizer_name: 优化器名称
        model: 模型
        lr: 学习率
        weight_decay: 权重衰减
    
    Returns:
        优化器实例
    """
    lr_float = float(lr)
    wd_float = float(weight_decay)
    
    if optimizer_name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr_float, weight_decay=wd_float)
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr_float, weight_decay=wd_float)
    elif optimizer_name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr_float, weight_decay=wd_float, momentum=0.9)
    else:
        raise ValueError(f"未知优化器: {optimizer_name}")


def get_scheduler(scheduler_name, optimizer, epochs, total_batches):
    """
    获取学习率调度器
    
    Args:
        scheduler_name: 调度器名称
        optimizer: 优化器
        epochs: 总轮数
        total_batches: 每轮的批次数量
    
    Returns:
        调度器实例和是否为批次级调度
    """
    total_steps = int(epochs) * total_batches
    
    if scheduler_name == "CosineAnnealingWarmRestarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-6), True
    elif scheduler_name == "ReduceLROnPlateau (默认)":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True), False
    elif scheduler_name == "Cosine Annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=float(optimizer.param_groups[0]['lr']) * 0.01), True
    elif scheduler_name == "Step Decay":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(epochs * 0.3)), gamma=0.5), False
    else:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0), False

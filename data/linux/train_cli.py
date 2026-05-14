#!/usr/bin/env python3
"""
睑板腺分割训练脚本 - Linux版本
集成多种模型支持和学术图表输出功能
"""
import torch
import numpy as np
import os
import glob
import time
import argparse
from monai.losses import DiceLoss, FocalLoss
from typing import List, Optional, Dict, Any

# 导入我们的新模块
import data_loader
from models import get_clinical_model, get_available_models
from utils.helpers import cleanup_checkpoints
from utils.plotting import create_training_plot


class CombinedLoss(torch.nn.Module):
    """
    组合损失函数
    """
    def __init__(self, loss_fn1, loss_fn2, weight1=0.5, weight2=0.5):
        super().__init__()
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.weight1 = weight1
        self.weight2 = weight2

    def forward(self, pred, target):
        return self.weight1 * self.loss_fn1(pred, target) + self.weight2 * self.loss_fn2(pred, target)


def get_loss_function(loss_name: str, pos_weight: float = 2.0):
    """
    获取损失函数
    
    Args:
        loss_name: 损失函数名称
        pos_weight: 正负样本权重
    
    Returns:
        损失函数实例
    """
    if loss_name == "Dice":
        return DiceLoss(sigmoid=True)
    elif loss_name == "BCE":
        return torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    elif loss_name == "Focal":
        return FocalLoss(alpha=0.25, gamma=2.0)
    elif loss_name == "Dice+BCE":
        loss_fn_main = DiceLoss(sigmoid=True)
        loss_fn_ce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        return CombinedLoss(loss_fn_main, loss_fn_ce)
    else:
        raise ValueError(f"未知损失函数: {loss_name}")


def get_optimizer(optimizer_name: str, model, lr: float, weight_decay: float = 1e-4):
    """
    获取优化器
    
    Args:
        optimizer_name: 优化器名称
        model: 模型实例
        lr: 学习率
        weight_decay: 权重衰减
    
    Returns:
        优化器实例
    """
    if optimizer_name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"未知优化器: {optimizer_name}")


def get_scheduler(scheduler_name: str, optimizer, epochs: int):
    """
    获取学习率调度器
    
    Args:
        scheduler_name: 调度器名称
        optimizer: 优化器实例
        epochs: 总训练轮数
    
    Returns:
        学习率调度器实例
    """
    if scheduler_name == "CosineAnnealingWarmRestarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    elif scheduler_name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True)
    elif scheduler_name == "CosineAnnealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif scheduler_name == "StepDecay":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(epochs * 0.2)), gamma=0.5)
    else:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)


def main():
    parser = argparse.ArgumentParser(description="\"腺\"而易见 - 睑板腺训练工具 (V100 32GB 优化版)")
    parser.add_argument('--epochs', '-e', type=int, default=100, help='训练轮数 (默认: 100)')
    parser.add_argument('--batch-size', '-b', type=int, default=8, help='批次大小 (默认: 8)')
    parser.add_argument('--save-freq', type=int, default=10, help='保存频率 (默认: 每10轮)')
    parser.add_argument('--max-keep', type=int, default=5, help='保留的checkpoint数量 (默认: 5)')
    parser.add_argument('--mem-limit', type=float, default=0.95, help='显存限制 (0-1, 默认: 0.95)')
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp16', help='精度 (默认: fp16)')
    parser.add_argument('--tile-mode', action='store_true', help='启用分块训练 (V100不需要)')
    parser.add_argument('--tile-height', type=int, default=512, help='分块高度 (默认: 512)')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率 (默认: 0.001)')
    parser.add_argument('--optimizer', choices=['Adam', 'AdamW', 'SGD'], default='AdamW', help='优化器 (默认: AdamW)')
    parser.add_argument('--scheduler', choices=['CosineAnnealingWarmRestarts', 'ReduceLROnPlateau', 'CosineAnnealing', 'StepDecay', 'Fixed'], default='CosineAnnealing', help='学习率调度器 (默认: CosineAnnealing)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减 (默认: 0.0001)')
    parser.add_argument('--loss', choices=['Dice', 'BCE', 'Focal', 'Dice+BCE'], default='Dice+BCE', help='损失函数 (默认: Dice+BCE)')
    parser.add_argument('--pos-weight', type=float, default=2.0, help='正负样本权重 (默认: 2.0)')
    parser.add_argument('--augmentation', action='store_true', default=True, help='启用数据增强 (默认: 开启)')
    parser.add_argument('--no-augmentation', action='store_true', help='禁用数据增强')
    parser.add_argument('--resume', action='store_true', help='从checkpoint继续训练')
    parser.add_argument('--num-workers', type=int, default=8, help='数据加载线程数 (默认: 8)')
    
    # 新增参数 - 模型选择和绘图
    parser.add_argument('--model', choices=get_available_models(), default='UNet', help=f'模型架构 (默认: UNet, 可选: {", ".join(get_available_models())})')
    parser.add_argument('--plot', action='store_true', default=True, help='生成训练图表 (默认: 开启)')
    parser.add_argument('--no-plot', action='store_true', help='禁用训练图表')
    parser.add_argument('--plot-path', type=str, default='training_results', help='图表保存路径 (默认: training_results)')
    
    args = parser.parse_args()
    
    # 处理数据增强
    if args.no_augmentation:
        args.augmentation = False
    
    # 处理绘图选项
    if args.no_plot:
        args.plot = False
    
    print("=" * 70)
    print("🚀 \"腺\"而易见 - 睑板腺训练引擎 (V100 32GB 优化版)")
    print("=" * 70)
    print("配置参数:")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  模型架构: {args.model}")
    print(f"  学习率: {args.lr}")
    print(f"  优化器: {args.optimizer}")
    print(f"  损失函数: {args.loss}")
    print(f"  数据增强: {'开启' if args.augmentation else '关闭'}")
    print(f"  精度: {args.precision}")
    print(f"  分块训练: {'开启' if args.tile_mode else '关闭'}")
    print(f"  数据加载线程: {args.num_workers}")
    print(f"  生成训练图表: {'是' if args.plot else '否'}")
    print("-" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ 设备: {gpu_name}")
        if args.mem_limit < 1.0:
            torch.cuda.set_per_process_memory_fraction(args.mem_limit, 0)
            print(f"   显存限制: {args.mem_limit * 100}%")
        torch.cuda.empty_cache()
    else:
        print("⚠️  设备: CPU (训练速度较慢)")
    
    # 检查数据集
    try:
        dataset_size = len(data_loader.data_dicts)
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return
    
    if dataset_size <= 2:
        print(f"❌ 错误: 数据集仅有 {dataset_size} 张图片")
        print("   请检查 data_loader.py 中的 data_dir 路径是否正确")
        return
    
    print(f"📂 数据集大小: {dataset_size} 张图片")
    
    # 构建训练数据加载器
    train_ds, train_loader = data_loader.build_training_loader(
        batch_size=args.batch_size,
        tile_mode=args.tile_mode,
        tile_height=args.tile_height,
        use_augmentation=args.augmentation,
        num_workers=args.num_workers,
    )
    
    total_batches = len(train_loader)
    
    # 使用新的模型加载函数
    model = get_clinical_model(torch.device("cpu"), model_name=args.model)
    start_epoch_offset = 0
    
    if args.resume:
        best_model_path = "meibomian_model_best.pth"
        if os.path.exists(best_model_path):
            state_dict = torch.load(best_model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            existing_epochs = [int(f.split("_")[-1].split(".")[0]) for f in glob.glob("mgd_epoch_*.pth")]
            if existing_epochs:
                start_epoch_offset = max(existing_epochs)
            print(f"🔌 断点续训: 从第 {start_epoch_offset} 轮继续")
        else:
            print("⚠️  未找到历史模型，从零开始训练")
    
    try:
        model = model.to(device)
    except torch.OutOfMemoryError:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print("❌ 模型加载到GPU时显存不足")
        print("   建议: 减小 batch-size 或启用 --tile-mode")
        return
    
    # 获取损失函数、优化器、调度器
    loss_fn = get_loss_function(args.loss, args.pos_weight)
    optimizer = get_optimizer(args.optimizer, model, args.lr, args.weight_decay)
    scheduler = get_scheduler(args.scheduler, optimizer, args.epochs)
    
    # 设置混合精度训练
    use_amp = args.precision == 'fp16' and device.type == "cuda"
    if device.type == "cuda":
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    else:
        scaler = None
    
    best_loss = float('inf')
    total_epochs = args.epochs
    
    # 用于记录训练历史（用于绘图）
    loss_history: List[float] = []
    dice_history: List[float] = []
    lr_history: List[float] = []
    
    print("")
    print("🚀 开始训练...")
    print("-" * 70)
    
    for epoch_idx in range(total_epochs):
        current_epoch = start_epoch_offset + epoch_idx + 1
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        start_time = time.time()
        
        for batch_idx, batch_data in enumerate(train_loader, 1):
            try:
                inputs = batch_data["image"].to(device)
                labels = batch_data["label"].to(device)
                
                optimizer.zero_grad()
                
                if use_amp and device.type == "cuda":
                    with torch.amp.autocast('cuda', enabled=True):
                        outputs = model(inputs)
                        loss = loss_fn(outputs, labels)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                if args.scheduler in ["CosineAnnealingWarmRestarts", "CosineAnnealing"]:
                    scheduler.step()
                
                epoch_loss += loss.item()
                
                # 计算Dice系数
                with torch.no_grad():
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    labels_bin = (labels > 0.5).float()
                    intersection = (preds * labels_bin).sum(dim=(1, 2, 3))
                    denom = preds.sum(dim=(1, 2, 3)) + labels_bin.sum(dim=(1, 2, 3))
                    batch_dice = ((2.0 * intersection + 1e-6) / (denom + 1e-6)).mean().item()
                    epoch_dice += batch_dice
                
            except torch.OutOfMemoryError:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print("❌ 训练过程中显存不足")
                print("   建议: 减小 batch-size 或启用 --tile-mode")
                return
            
            if batch_idx % max(1, total_batches // 10) == 0 or batch_idx == total_batches:
                print(f"  Epoch {current_epoch:3d}/{start_epoch_offset + total_epochs}  "
                      f"Batch {batch_idx:3d}/{total_batches}  "
                      f"Loss: {loss.item():.4f}  "
                      f"Dice: {batch_dice:.4f}")
        
        epoch_loss /= total_batches
        epoch_dice /= total_batches
        epoch_time = time.time() - start_time
        
        # 记录历史数据
        loss_history.append(epoch_loss)
        dice_history.append(epoch_dice)
        lr_history.append(optimizer.param_groups[0]['lr'])
        
        if args.scheduler == "ReduceLROnPlateau":
            scheduler.step(epoch_loss)
        elif args.scheduler == "StepDecay":
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "meibomian_model_best.pth")
            print(f"")
            print(f"🏆 第 {current_epoch} 轮  平均Loss: {epoch_loss:.4f}  "
                  f"平均Dice: {epoch_dice:.4f}  "
                  f"LR: {current_lr:.6f}  耗时: {epoch_time:.1f}s  [BEST!]")
        else:
            print(f"")
            print(f"✅ 第 {current_epoch} 轮  平均Loss: {epoch_loss:.4f}  "
                  f"平均Dice: {epoch_dice:.4f}  "
                  f"LR: {current_lr:.6f}  耗时: {epoch_time:.1f}s")
        
        if (epoch_idx + 1) % args.save_freq == 0 or (epoch_idx + 1) == total_epochs:
            save_name = f"mgd_epoch_{current_epoch}.pth"
            torch.save(model.state_dict(), save_name)
            print(f"💾 Checkpoint已保存: {save_name}")
            cleanup_checkpoints(max_keep=args.max_keep)
        
        print("-" * 70)
    
    print("")
    print("🎉 训练完成!")
    print(f"📈 最佳Loss: {best_loss:.4f}")
    print("💾 模型已保存为: meibomian_model_best.pth")
    
    # 生成训练图表
    if args.plot:
        print("")
        print("📊 生成训练图表...")
        os.makedirs(args.plot_path, exist_ok=True)
        
        try:
            # 生成主训练图
            plot_name = f"{args.plot_path}/{args.model}_training_plot.png"
            create_training_plot(
                loss_history=loss_history,
                dice_history=dice_history,
                lr_history=lr_history,
                save_path=plot_name,
                title=f"{args.model} Training Process (Epochs: {args.epochs})"
            )
            print(f"✅ 训练图表已保存: {plot_name}")
            
            # 同时保存训练数据为CSV，方便后续分析
            csv_path = f"{args.plot_path}/{args.model}_training_history.csv"
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write("Epoch,Loss,Dice,LearningRate\n")
                for i, (loss, dice, lr) in enumerate(zip(loss_history, dice_history, lr_history)):
                    f.write(f"{i+1},{loss:.6f},{dice:.6f},{lr:.8f}\n")
            print(f"✅ 训练历史已保存: {csv_path}")
            
        except Exception as e:
            print(f"⚠️ 生成图表时出错: {e}")
            print("   训练已完成，图表生成失败不影响模型使用")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

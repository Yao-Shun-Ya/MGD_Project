#!/usr/bin/env python3
import torch
import numpy as np
import os
import glob
import time
import argparse
from monai.networks.nets import UNet
from monai.losses import DiceLoss, FocalLoss
import data_loader


class CombinedLoss(torch.nn.Module):
    def __init__(self, loss_fn1, loss_fn2, weight1=0.5, weight2=0.5):
        super().__init__()
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.weight1 = weight1
        self.weight2 = weight2

    def forward(self, pred, target):
        return self.weight1 * self.loss_fn1(pred, target) + self.weight2 * self.loss_fn2(pred, target)


def cleanup_checkpoints(max_keep=5):
    ckpt_files = glob.glob("mgd_epoch_*.pth")
    if len(ckpt_files) <= max_keep:
        return
    sorted_ckpts = sorted(
        ckpt_files,
        key=lambda f: int(f.split("_")[-1].split(".")[0])
    )
    for stale_ckpt in sorted_ckpts[:-max_keep]:
        try:
            os.remove(stale_ckpt)
        except OSError:
            pass


def get_clinical_model(device):
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(64, 128, 256, 512, 1024),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        act="PRELU",
        norm="INSTANCE",
        dropout=0.1
    ).to(device)
    return model


def main():
    parser = argparse.ArgumentParser(description='"腺"而易见 - 睑板腺训练工具 (V100 32GB 优化版)')
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

    args = parser.parse_args()

    # 处理数据增强
    if args.no_augmentation:
        args.augmentation = False

    print("=" * 70)
    print('🚀 "腺"而易见 - 睑板腺训练引擎 (V100 32GB 优化版)')
    print("=" * 70)
    print("配置参数:")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.lr}")
    print(f"  优化器: {args.optimizer}")
    print(f"  损失函数: {args.loss}")
    print(f"  数据增强: {'开启' if args.augmentation else '关闭'}")
    print(f"  精度: {args.precision}")
    print(f"  分块训练: {'开启' if args.tile_mode else '关闭'}")
    print(f"  数据加载线程: {args.num_workers}")
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

    try:
        dataset_size = len(data_loader.check_ds)
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return

    if dataset_size <= 2:
        print(f"❌ 错误: 数据集仅有 {dataset_size} 张图片")
        print("   请检查 data_loader.py 中的 data_dir 路径是否正确")
        return

    print(f"📂 数据集大小: {dataset_size} 张图片")

    data_loader.check_ds, data_loader.check_loader = data_loader.build_training_loader(
        batch_size=args.batch_size,
        tile_mode=args.tile_mode,
        tile_height=args.tile_height,
        use_augmentation=args.augmentation,
        num_workers=args.num_workers,
    )

    model = get_clinical_model(torch.device("cpu"))
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

    if args.loss == "Dice":
        loss_function = DiceLoss(sigmoid=True)
    elif args.loss == "BCE":
        loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight))
    elif args.loss == "Focal":
        loss_function = FocalLoss(alpha=0.25, gamma=2.0)
    elif args.loss == "Dice+BCE":
        loss_fn_main = DiceLoss(sigmoid=True)
        loss_fn_ce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight))
        loss_function = CombinedLoss(loss_fn_main, loss_fn_ce)

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)

    total_batches = len(data_loader.check_loader)
    if args.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    elif args.scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True)
    elif args.scheduler == "CosineAnnealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == "StepDecay":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(args.epochs * 0.2)), gamma=0.5)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    use_amp = args.precision == 'fp16' and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_loss = float('inf')
    total_epochs = args.epochs

    print("")
    print("🚀 开始训练...")
    print("-" * 70)

    for epoch_idx in range(total_epochs):
        current_epoch = start_epoch_offset + epoch_idx + 1
        model.train()
        epoch_loss = 0
        start_time = time.time()

        for batch_idx, batch_data in enumerate(data_loader.check_loader, 1):
            try:
                inputs = batch_data["image"].to(device)
                labels = batch_data["label"].to(device)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if args.scheduler in ["CosineAnnealingWarmRestarts", "CosineAnnealing"]:
                    scheduler.step()

                epoch_loss += loss.item()

                with torch.no_grad():
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    labels_bin = (labels > 0.5).float()
                    intersection = (preds * labels_bin).sum(dim=(1, 2, 3))
                    denom = preds.sum(dim=(1, 2, 3)) + labels_bin.sum(dim=(1, 2, 3))
                    batch_dice = ((2.0 * intersection + 1e-6) / (denom + 1e-6)).mean().item()

            except torch.OutOfMemoryError:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print("❌ 训练过程中显存溢出")
                print("   建议: 减小 batch-size 或启用 --tile-mode")
                return

            if batch_idx % max(1, total_batches // 10) == 0 or batch_idx == total_batches:
                print(f"  Epoch {current_epoch:3d}/{start_epoch_offset + total_epochs}  "
                      f"Batch {batch_idx:3d}/{total_batches}  "
                      f"Loss: {loss.item():.4f}  "
                      f"Dice: {batch_dice:.4f}")

        epoch_loss /= total_batches
        epoch_time = time.time() - start_time

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
                  f"LR: {current_lr:.6f}  耗时: {epoch_time:.1f}s  [BEST!]")
        else:
            print(f"")
            print(f"✅ 第 {current_epoch} 轮  平均Loss: {epoch_loss:.4f}  "
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
    print("=" * 70)


if __name__ == "__main__":
    main()

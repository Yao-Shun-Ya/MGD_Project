"""
训练引擎模块
"""
import os
import time
import torch
import numpy as np
from models import get_clinical_model
from data import get_data_manager
from utils import cleanup_checkpoints
from config import MODEL_BEST_PATH, get_model_best_path, get_model_checkpoint_path
from .losses import get_loss_function, get_optimizer, get_scheduler

# 安全导入 wandb
wandb = None
wandb_available = False
try:
    import wandb  # type: ignore
    wandb_available = True
except ImportError:
    pass


def professional_training(
    resume,
    epochs,
    batch_size,
    save_freq,
    max_keep,
    mem_limit,
    precision,
    tile_mode,
    tile_height,
    lr,
    optimizer_name,
    scheduler_name,
    weight_decay,
    loss_fn_name,
    pos_weight,
    use_augmentation,
    model_name="UNet",
    wandb_enable=False,
    log_callback=None
):
    """
    专业训练函数（生成器模式，用于 UI 实时更新）
    
    Args:
        resume: 是否从断点续训
        epochs: 训练轮数
        batch_size: 批次大小
        save_freq: 保存频率
        max_keep: 保留的最大检查点数量
        mem_limit: 显存限制
        precision: 精度
        tile_mode: 是否分块
        tile_height: 分块高度
        lr: 学习率
        optimizer_name: 优化器名称
        scheduler_name: 调度器名称
        weight_decay: 权重衰减
        loss_fn_name: 损失函数名称
        pos_weight: 正负样本权重
        use_augmentation: 是否使用数据增强
        log_callback: 日志回调函数
    
    Yields:
        日志字符串
    """
    def log(msg):
        if log_callback:
            log_callback(msg)
        yield msg
    
    yield from log("🚀 [临床 Pro 满血版] 训练引擎初始化...\n")
    
    # 尝试初始化W&B
    run = None
    if wandb_enable:
        try:
            if wandb_available:
                run = wandb.init(
                    project="MGD-Segmentation",
                    name=f"{model_name}_{time.strftime('%Y%m%d_%H%M%S')}",
                    anonymous="allow",
                    config={
                        "model": model_name,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": lr,
                        "optimizer": optimizer_name,
                        "scheduler": scheduler_name,
                        "weight_decay": weight_decay,
                        "loss_function": loss_fn_name,
                    }
                )
                yield from log("✅ Weights & Biases 已连接\n")
        except Exception as e:
            yield from log(f"⚠️ W&B 初始化失败: {e}\n")
    yield from log(f"📋 当前超参数配置：\n")
    yield from log(f"   • 学习率 (LR): {lr}\n")
    yield from log(f"   • 优化器: {optimizer_name}\n")
    yield from log(f"   • 调度器: {scheduler_name}\n")
    yield from log(f"   • 权重衰减: {weight_decay}\n")
    yield from log(f"   • 损失函数: {loss_fn_name}\n")
    yield from log(f"   • 正负样本权重: {pos_weight}\n")
    yield from log(f"   • 数据增强: {'启用' if use_augmentation else '禁用'}\n")
    yield from log("-" * 40 + "\n")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        if float(mem_limit) < 0.99:
            torch.cuda.set_per_process_memory_fraction(float(mem_limit), 0)
            yield from log(f"✅ 设备就绪: {gpu_name} (显存限额: {mem_limit*100}%)\n")
        else:
            yield from log(f"✅ 设备就绪: {gpu_name} (显存限额: 100%，未强制限额)\n")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        yield from log("⚠️ 降级运行: 当前使用 CPU 训练。\n")
    
    dm = get_data_manager()
    
    try:
        dataset_size = dm.get_dataset_size()
    except Exception as e:
        yield from log(f"❌ 数据集加载崩溃: {e}\n")
        return
    
    if dataset_size <= 2:
        yield from log(f"❌ 致命错误：读取到的训练数据仅有 {dataset_size} 张！\n")
        yield from log("⚠️ 训练已拦截！极少量数据会导致模型瞬间遗忘特征（变傻），这是您可视化变差的元凶。\n")
        return
    
    yield from log(f"📂 数据挂载成功：检测到 {dataset_size} 张影像，准备切分批次...\n")
    
    _, train_loader = dm.build_training_loader(
        batch_size=int(batch_size),
        tile_mode=bool(tile_mode),
        tile_height=int(tile_height),
        use_augmentation=bool(use_augmentation),
        num_workers=0,
    )
    
    if tile_mode:
        from data import TARGET_HEIGHT, TARGET_WIDTH
        tile_w = int(max(128, min(int(tile_height) * 2, TARGET_WIDTH)))
        yield from log(f"🧩 分块训练已启用: Tile={int(tile_height)}x{tile_w}\n")
    else:
        yield from log("🖼️ 全图训练模式已启用。\n")
    
    model = get_clinical_model(torch.device("cpu"), model_name=model_name)
    yield from log(f"✅ 模型架构: {model_name}\n")
    
    current_model_best_path = get_model_best_path(model_name)
    yield from log(f"📂 模型保存路径: {current_model_best_path}\n")
    
    start_epoch_offset = 0
    
    if resume:
        if os.path.exists(current_model_best_path):
            state_dict = torch.load(current_model_best_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model_dir = os.path.dirname(current_model_best_path)
            existing_epochs = []
            if os.path.exists(model_dir):
                for f in os.listdir(model_dir):
                    if f.startswith(f"{model_name.lower()}_epoch_") and f.endswith('.pth'):
                        try:
                            epoch = int(f.split('_')[-1].split('.')[0])
                            existing_epochs.append(epoch)
                        except:
                            pass
            if existing_epochs:
                start_epoch_offset = max(existing_epochs)
            yield from log(f"🔌 [断点续训] 成功接通前序记忆！当前进度：第 {start_epoch_offset} 轮...\n")
        else:
            yield from log(f"⚠️ 未找到历史最优模型，将从零开始训练。\n")
    
    try:
        model = model.to(device)
    except torch.OutOfMemoryError:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        yield from log("❌ 模型装载到 GPU 时显存不足。请尝试：显存限额=1.0、训练 Tile 高度降到 192/128、关闭断点续训后重试。\n")
        return
    
    loss_function = get_loss_function(loss_fn_name, pos_weight)
    yield from log(f"✅ 损失函数: {loss_fn_name}\n")
    
    optimizer = get_optimizer(optimizer_name, model, lr, weight_decay)
    yield from log(f"✅ 优化器: {optimizer_name} (lr={lr}, weight_decay={weight_decay})\n")
    
    total_batches = len(train_loader)
    
    # 创建学习率调度器
    scheduler, is_batch_scheduler = get_scheduler(scheduler_name, optimizer, epochs, total_batches)
    yield from log(f"✅ 学习率调度器: {scheduler_name}\n")
    
    use_amp = precision.startswith("FP16") and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    yield from log("-" * 40 + "\n")
    
    best_loss = float('inf')
    
    epoch_dice_list = []
    total_epochs_to_run = int(epochs)
    for epoch_idx in range(total_epochs_to_run):
        current_global_epoch = start_epoch_offset + epoch_idx + 1
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        for batch_idx, batch_data in enumerate(train_loader, 1):
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
                
                if is_batch_scheduler:
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
                yield from log("❌ 训练过程中显存溢出(OOM)。建议：1) 训练 Tile 高度降到 192 或 128；2) Batch Size=1；3) 显存限额调到 1.0；4) 必要时先关闭断点续训。\n")
                return
            
            if batch_idx % max(1, total_batches // 5) == 0 or batch_idx == total_batches:
                current_log = (
                    f"   [Epoch {current_global_epoch}/{start_epoch_offset + total_epochs_to_run}] "
                    f"Batch {batch_idx}/{total_batches} - Loss: {loss.item():.4f} - Dice: {batch_dice:.4f}\n"
                )
                yield from log(current_log)
        
        epoch_loss /= total_batches
        
        # 记录epoch dice（平均最后一个batch的dice）
        epoch_dice_list.append(batch_dice)
        epoch_dice = np.mean(epoch_dice_list[-5:]) if len(epoch_dice_list) > 0 else 0.0
        
        # 记录到W&B
        if run is not None:
            run.log({
                "epoch_loss": epoch_loss,
                "epoch_dice": epoch_dice,
                "learning_rate": current_lr,
                "epoch": current_global_epoch,
            })
        
        if not is_batch_scheduler and scheduler_name == "ReduceLROnPlateau (默认)":
            scheduler.step(epoch_loss)
        elif not is_batch_scheduler:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), current_model_best_path)
            yield from log(f"   🏆 探测到更优模型，已按完整精度(FP32)更新 best 权重！\n")
        
        epoch_time = time.time() - start_time
        yield from log(f"✅ 第 {current_global_epoch} 轮完成 | 平均 Loss: {epoch_loss:.4f} | 学习率: {current_lr:.6f} | 耗时: {epoch_time:.1f}s\n")
        
        # 每5轮上传训练样本可视化
        if run is not None and (current_global_epoch % 5 == 0 or current_global_epoch == 1):
            try:
                dm = get_data_manager()
                vis_ds, _ = dm.build_training_loader(batch_size=1, tile_mode=False, tile_height=256, use_augmentation=False, num_workers=0)
                if len(vis_ds) > 0:
                    sample = vis_ds[0]
                    image = sample["image"].to(device)
                    label = sample["label"].to(device)
                    model.eval()
                    with torch.no_grad():
                        output = model(image)
                        pred = (torch.sigmoid(output) > 0.5).float()
                    img_np = image[0, 0].cpu().numpy()
                    gt_np = label[0, 0].cpu().numpy()
                    pred_np = pred[0, 0].cpu().numpy()
                    # 归一化到0-255
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8) * 255
                    img_np = img_np.astype(np.uint8)
                    gt_np = (gt_np * 255).astype(np.uint8)
                    pred_np = (pred_np * 255).astype(np.uint8)
                    # 创建对比图
                    import cv2
                    combined = np.hstack([img_np, gt_np, pred_np])
                    # 转换为彩色
                    combined_color = cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)
                    run.log({"train_sample": wandb.Image(combined_color, caption=f"Epoch {current_global_epoch}")})
                    yield from log(f"   📊 W&B: 已上传样本可视化\n")
            except Exception as e:
                yield from log(f"   ⚠️ W&B 可视化失败: {e}\n")
        
        if (epoch_idx + 1) % int(save_freq) == 0 or (epoch_idx + 1) == total_epochs_to_run:
            checkpoint_path = get_model_checkpoint_path(model_name, current_global_epoch)
            torch.save(model.state_dict(), checkpoint_path)
            yield from log(f"   💾 保存检查点: {checkpoint_path}\n")
            cleanup_checkpoints(max_keep=int(max_keep), model_name=model_name)
    
    # 结束W&B
    if run is not None:
        run.finish()
        yield from log("✅ Weights & Biases run 已结束\n")
    
    yield from log("-" * 40 + "\n🎉 训练全流程结束！")


def simple_train(max_epochs=20, save_interval=2):
    """
    简单训练函数（用于独立运行 train.py）
    
    Args:
        max_epochs: 最大轮数
        save_interval: 保存间隔
    """
    print("=" * 60)
    print("🚀 '腺'而易见 — 睑板腺 MGD 深度学习模型训练引擎 (满血精度版)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"✅ 算力引擎就绪: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        print("⚠️ 未检测到 GPU，训练速度将受到严重限制。")
    print("=" * 60)
    
    model = get_clinical_model(device)
    
    from monai.losses import TverskyLoss
    loss_function = TverskyLoss(sigmoid=True, alpha=0.3, beta=0.7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    
    dm = get_data_manager()
    print(f"\n📂 训练数据集: {dm.get_dataset_size()} 例专业影像")
    print(f"🔥 训练设置: Epochs={max_epochs}, Batch Size={dm.check_loader.batch_size}")
    print(f"🌟 已激活: 全尺度(64-1024) + Res-Skip + PixelShuffle + FP32 保存")
    print("-" * 40)
    
    best_loss = float('inf')
    
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        step = 0
        start_time = time.time()
        
        for batch_data in dm.check_loader:
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            if step % max(1, len(dm.check_loader) // 5) == 0 or step == len(dm.check_loader):
                print(f"   [轮次 {epoch+1}] 进度: {step}/{len(dm.check_loader)} - Tversky 损失: {loss.item():.4f}")
        
        avg_loss = epoch_loss / step
        epoch_time = time.time() - start_time
        print(f"✅ 第 {epoch+1} 轮完成 | 平均损失: {avg_loss:.4f} | 耗时: {epoch_time:.1f}s")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_BEST_PATH)
            print(f"   🏆 探测到更优模型！已保存完整精度权重: {MODEL_BEST_PATH}")
        
        if (epoch + 1) % save_interval == 0:
            checkpoint_name = f"mgd_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_name)
            print(f"   💾 定期备份已存: {checkpoint_name}")
    
    print("-" * 40)
    print(f"🎉 训练全流程顺利结束！")
    print(f"📈 最优模型损失值: {best_loss:.4f}")
    print(f"💾 请在目录中使用 '{MODEL_BEST_PATH}' 进行临床部署。")
    print("=" * 60)

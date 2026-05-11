import gradio as gr
import torch
import numpy as np
import cv2
import os
import glob
import time
import json
from PIL import Image
from monai.networks.nets import UNet
from skimage import exposure
from skimage.morphology import skeletonize
from monai.losses import TverskyLoss, DiceLoss, FocalLoss, MaskedDiceLoss
from monai.inferers import sliding_window_inference
import data_loader
import train

PRESET_FILE = "infer_presets.json"
BUILTIN_PRESETS = {
    "平衡-默认": {
        "tile_mode": True, "tile_size": 192, "tile_overlap": 0.25,
        "threshold": 0.70, "min_area": 120,
        "smooth_on": True, "smooth_sigma": 1.2, "roi_only": True,
        "tta_mode": "关闭",
    },
    "保守-低误检": {
        "tile_mode": True, "tile_size": 256, "tile_overlap": 0.35,
        "threshold": 0.80, "min_area": 180,
        "smooth_on": True, "smooth_sigma": 1.0, "roi_only": True,
        "tta_mode": "关闭",
    },
    "敏感-高召回": {
        "tile_mode": True, "tile_size": 192, "tile_overlap": 0.30,
        "threshold": 0.60, "min_area": 80,
        "smooth_on": True, "smooth_sigma": 1.3, "roi_only": True,
        "tta_mode": "关闭",
    },
}

class CombinedLoss(torch.nn.Module):
    def __init__(self, loss_fn1, loss_fn2, weight1=0.5, weight2=0.5):
        super().__init__()
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.weight1 = weight1
        self.weight2 = weight2

    def forward(self, pred, target):
        return self.weight1 * self.loss_fn1(pred, target) + self.weight2 * self.loss_fn2(pred, target)

def apply_medical_visualizations(overlay_img, prob_map, binary_mask, min_area=50):
    result_img = overlay_img.copy()

    mask_80 = (prob_map > 0.8).astype(np.uint8) * 255
    mask_50 = (prob_map > 0.5).astype(np.uint8) * 255
    mask_20 = (prob_map > 0.2).astype(np.uint8) * 255

    contours_80, _ = cv2.findContours(mask_80, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_img, contours_80, -1, (255, 0, 0), 2)
    contours_50, _ = cv2.findContours(mask_50, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_img, contours_50, -1, (255, 255, 0), 1)
    contours_20, _ = cv2.findContours(mask_20, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_img, contours_20, -1, (0, 0, 255), 1)

    bool_mask = binary_mask > 0
    skeleton = skeletonize(bool_mask)
    result_img[skeleton] = [255, 0, 0]

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
    instance_count = 1
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 255, 255), 1)
            cv2.putText(result_img, f"#{instance_count}", (cx - 12, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            instance_count += 1

    return result_img

def generate_contour_visualization(prob_map, original_img, min_area=50):
    if len(original_img.shape) == 2:
        result_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    else:
        result_img = original_img.copy()

    mask_80 = (prob_map > 0.8).astype(np.uint8) * 255
    mask_50 = (prob_map > 0.5).astype(np.uint8) * 255
    mask_20 = (prob_map > 0.2).astype(np.uint8) * 255

    contours_80, _ = cv2.findContours(mask_80, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_img, contours_80, -1, (255, 0, 0), 2)
    contours_50, _ = cv2.findContours(mask_50, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_img, contours_50, -1, (255, 255, 0), 1)
    contours_20, _ = cv2.findContours(mask_20, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_img, contours_20, -1, (0, 0, 255), 1)

    return result_img

def generate_skeleton_visualization(binary_mask, original_img):
    if len(original_img.shape) == 2:
        result_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    else:
        result_img = original_img.copy()

    bool_mask = binary_mask > 0
    skeleton = skeletonize(bool_mask)
    result_img[skeleton] = [255, 0, 0]

    return result_img

def generate_instance_visualization(binary_mask, original_img, min_area=50):
    if len(original_img.shape) == 2:
        result_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    else:
        result_img = original_img.copy()

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
    instance_count = 1
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 255, 255), 2)
            cv2.putText(result_img, f"#{instance_count}", (cx - 12, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            instance_count += 1

    return result_img

def cleanup_checkpoints(max_keep=3):
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

def preprocess_for_model(input_image):
    if isinstance(input_image, str):
        from PIL import Image
        img = Image.open(input_image)
        try:
            exif = img.getexif()
            if exif:
                orientation = exif.get(0x0112, 1)
                if orientation == 2:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 3:
                    img = img.rotate(180)
                elif orientation == 4:
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)
                elif orientation == 5:
                    img = img.rotate(-90).transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 6:
                    img = img.rotate(-90)
                elif orientation == 7:
                    img = img.rotate(90).transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 8:
                    img = img.rotate(90)
        except Exception:
            pass
        input_image = np.array(img)

    if input_image.ndim == 2:
        img_gray = input_image
    elif input_image.shape[2] == 3:
        img_gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = cv2.cvtColor(input_image, cv2.COLOR_RGBA2GRAY)

    img_resized = cv2.resize(
        img_gray,
        (data_loader.TARGET_WIDTH, data_loader.TARGET_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )
    img_clahe = exposure.equalize_adapthist(img_resized, clip_limit=0.02).astype(np.float32)
    return img_resized, img_clahe

def professional_training(
    resume, epochs, batch_size, save_freq, max_keep, mem_limit, precision, tile_mode, tile_height,
    lr, optimizer_name, scheduler_name, weight_decay, loss_fn_name, pos_weight, use_augmentation
):
    log_output = "🚀 [临床 Pro 满血版] 训练引擎初始化...\n"
    log_output += f"📋 当前超参数配置：\n"
    log_output += f"   • 学习率 (LR): {lr}\n"
    log_output += f"   • 优化器: {optimizer_name}\n"
    log_output += f"   • 调度器: {scheduler_name}\n"
    log_output += f"   • 权重衰减: {weight_decay}\n"
    log_output += f"   • 损失函数: {loss_fn_name}\n"
    log_output += f"   • 正负样本权重: {pos_weight}\n"
    log_output += f"   • 数据增强: {'启用' if use_augmentation else '禁用'}\n"
    log_output += "-" * 40 + "\n"
    yield log_output

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        log_output += f"✅ 设备就绪: Apple Silicon MPS (GPU 加速)\n"
    else:
        device = torch.device("cpu")
        log_output += "⚠️ 降级运行: 当前使用 CPU 训练。\n"
    yield log_output

    try:
        dataset_size = len(data_loader.check_ds)
    except Exception as e:
        yield log_output + f"❌ 数据集加载崩溃: {e}\n"
        return

    if dataset_size <= 2:
        log_output += f"❌ 致命错误：读取到的训练数据仅有 {dataset_size} 张！\n"
        log_output += "⚠️ 训练已拦截！极少量数据会导致模型瞬间遗忘特征（变傻），这是您可视化变差的元凶。\n"
        log_output += "👉 修复动作：请打开 data_loader.py，检查 data_dir 路径是否正确指向了您的完整数据集文件夹！\n"
        yield log_output
        return

    log_output += f"📂 数据挂载成功：检测到 {dataset_size} 张影像，准备切分批次...\n"
    yield log_output

    data_loader.check_ds, data_loader.check_loader = data_loader.build_training_loader(
        batch_size=int(batch_size),
        tile_mode=bool(tile_mode),
        tile_height=int(tile_height),
        use_augmentation=bool(use_augmentation),
        num_workers=0,
    )
    if tile_mode:
        tile_w = int(max(128, min(int(tile_height) * 2, data_loader.TARGET_WIDTH)))
        log_output += f"🧩 分块训练已启用: Tile={int(tile_height)}x{tile_w}\n"
    else:
        log_output += "🖼️ 全图训练模式已启用。\n"
    yield log_output

    model = get_clinical_model(torch.device("cpu"))
    start_epoch_offset = 0

    if resume:
        best_model_path = "meibomian_model_best.pth"
        if os.path.exists(best_model_path):
            state_dict = torch.load(best_model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            existing_epochs = [int(f.split('_')[-1].split('.')[0]) for f in glob.glob("mgd_epoch_*.pth")]
            if existing_epochs:
                start_epoch_offset = max(existing_epochs)
            log_output += f"🔌 [断点续训] 成功接通前序记忆！当前进度：第 {start_epoch_offset} 轮...\n"
        else:
            log_output += f"⚠️ 未找到历史最优模型，将从零开始训练。\n"
    try:
        model = model.to(device)
    except torch.OutOfMemoryError:
        if device.type == "mps":
            torch.mps.empty_cache()
        log_output += "❌ 模型装载到 GPU 时内存不足。请尝试：训练 Tile 高度降到 192/128、关闭断点续训后重试。\n"
        yield log_output
        return

    if loss_fn_name == "Dice Loss":
        loss_function = DiceLoss(sigmoid=True)
        log_output += f"✅ 损失函数: Dice Loss\n"
    elif loss_fn_name == "Cross Entropy":
        loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        log_output += f"✅ 损失函数: Cross Entropy (pos_weight={pos_weight})\n"
    elif loss_fn_name == "Focal Loss":
        loss_function = FocalLoss(alpha=0.25, gamma=2.0)
        log_output += f"✅ 损失函数: Focal Loss\n"
    elif loss_fn_name == "Dice+CE 组合损失":
        loss_fn_main = DiceLoss(sigmoid=True)
        loss_fn_ce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        loss_function = CombinedLoss(loss_fn_main, loss_fn_ce)
        log_output += f"✅ 损失函数: Dice+CE 组合损失 (pos_weight={pos_weight})\n"

    lr_float = float(lr)
    wd_float = float(weight_decay)
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_float, weight_decay=wd_float)
        log_output += f"✅ 优化器: Adam (lr={lr_float}, weight_decay={wd_float})\n"
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_float, weight_decay=wd_float)
        log_output += f"✅ 优化器: AdamW (lr={lr_float}, weight_decay={wd_float})\n"
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_float, weight_decay=wd_float, momentum=0.9)
        log_output += f"✅ 优化器: SGD (lr={lr_float}, weight_decay={wd_float}, momentum=0.9)\n"

    total_steps = int(epochs) * total_batches
    if scheduler_name == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-6)
        log_output += f"✅ 学习率调度器: CosineAnnealingWarmRestarts (T_0=50, T_mult=1, eta_min=1e-6)\n"
    elif scheduler_name == "ReduceLROnPlateau (默认)":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        log_output += f"✅ 学习率调度器: ReduceLROnPlateau (默认)\n"
    elif scheduler_name == "Cosine Annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr_float * 0.01)
        log_output += f"✅ 学习率调度器: Cosine Annealing (T_max={total_steps})\n"
    elif scheduler_name == "Step Decay":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(epochs * 0.3)), gamma=0.5)
        log_output += f"✅ 学习率调度器: Step Decay (step_size={max(1, int(epochs * 0.3))}, gamma=0.5)\n"
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        log_output += f"✅ 学习率调度器: Fixed LR\n"

    use_amp = precision.startswith("FP16") and device.type == "mps"
    scaler = torch.amp.GradScaler('mps', enabled=use_amp)

    log_output += "-" * 40 + "\n"
    yield log_output

    total_batches = len(data_loader.check_loader)
    best_loss = float('inf')

    total_epochs_to_run = int(epochs)
    for i in range(total_epochs_to_run):
        current_global_epoch = start_epoch_offset + i + 1
        model.train()
        epoch_loss = 0
        start_time = time.time()

        for step, batch_data in enumerate(data_loader.check_loader, 1):
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
                if scheduler_name in ["CosineAnnealingWarmRestarts", "Cosine Annealing"]:
                    scheduler.step()
                elif scheduler_name == "Step Decay" or scheduler_name == "Fixed LR":
                    pass
                epoch_loss += loss.item()
                with torch.no_grad():
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    labels_bin = (labels > 0.5).float()
                    intersection = (preds * labels_bin).sum(dim=(1, 2, 3))
                    denom = preds.sum(dim=(1, 2, 3)) + labels_bin.sum(dim=(1, 2, 3))
                    batch_dice = ((2.0 * intersection + 1e-6) / (denom + 1e-6)).mean().item()
            except torch.OutOfMemoryError:
                if device.type == "mps":
                    torch.mps.empty_cache()
                log_output += (
                    "❌ 训练过程中内存溢出(OOM)。建议："
                    "1) 训练 Tile 高度降到 192 或 128；"
                    "2) Batch Size=1；"
                    "3) 必要时先关闭断点续训。\n"
                )
                yield log_output
                return

            if step % max(1, total_batches // 5) == 0 or step == total_batches:
                current_log = (
                    f"   [Epoch {current_global_epoch}/{start_epoch_offset + total_epochs_to_run}] "
                    f"Batch {step}/{total_batches} - Loss: {loss.item():.4f} - Dice: {batch_dice:.4f}\n"
                )
                yield log_output + current_log

        epoch_loss /= total_batches

        if scheduler_name == "ReduceLROnPlateau (默认)":
            scheduler.step(epoch_loss)
        elif scheduler_name == "Step Decay":
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "meibomian_model_best.pth")
            log_output += f"   🏆 探测到更优模型，已按完整精度(FP32)更新 best 权重！\n"

        epoch_time = time.time() - start_time
        log_output += f"✅ 第 {current_global_epoch} 轮完成 | 平均 Loss: {epoch_loss:.4f} | 学习率: {current_lr:.6f} | 耗时: {epoch_time:.1f}s\n"
        yield log_output

        if (i + 1) % int(save_freq) == 0 or (i + 1) == total_epochs_to_run:
            save_name = f"mgd_epoch_{current_global_epoch}.pth"
            torch.save(model.state_dict(), save_name)
            yield log_output
            cleanup_checkpoints(max_keep=int(max_keep))

    log_output += "-" * 40 + "\n🎉 训练全流程结束！"
    yield log_output

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

def calculate_clinical_metrics(mask):
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = mask.shape[0] * mask.shape[1]
    gland_area = np.sum(mask > 0)
    area_ratio = (gland_area / total_area) * 100 if total_area > 0 else 0

    lengths, widths, centroids = [], [], []
    for cnt in contours:
        if cv2.contourArea(cnt) < 50: continue
        rect = cv2.minAreaRect(cnt)
        w, h = rect[1]
        lengths.append(max(w, h))
        widths.append(min(w, h))
        centroids.append(rect[0])

    avg_spacing = 0
    if len(centroids) > 1:
        centroids = np.array(centroids)
        centroids = centroids[centroids[:, 0].argsort()]
        spacings = np.sqrt(np.sum(np.diff(centroids, axis=0)**2, axis=1))
        avg_spacing = np.mean(spacings)

    return [
        ["腺体总面积占比", f"{area_ratio:.1f}%"],
        ["检出腺体数量", f"{len(lengths)} 个"],
        ["腺体平均长度", f"{np.mean(lengths) if lengths else 0:.1f} px"],
        ["腺体平均宽度", f"{np.mean(widths) if widths else 0:.1f} px"],
        ["平均分布间距", f"{avg_spacing:.1f} px"]
    ]

def generate_heatmap(prob_map, original_img):
    heatmap_gray = np.uint8(255 * prob_map)
    heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
    if len(original_img.shape) == 2:
        original_img_color = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    else:
        original_img_color = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    heatmap_color = cv2.resize(heatmap_color, (original_img_color.shape[1], original_img_color.shape[0]))
    overlay = cv2.addWeighted(original_img_color, 0.5, heatmap_color, 0.5, 0)
    return overlay

def postprocess_binary_mask(prob_map, threshold=0.7, min_area=120, open_kernel=3):
    th = float(max(0.05, min(threshold, 0.98)))
    binary = (prob_map >= th).astype(np.uint8)

    cover = float(binary.mean())
    if cover > 0.25:
        for t in (0.75, 0.8, 0.85, 0.9):
            cand = (prob_map >= t).astype(np.uint8)
            if float(cand.mean()) <= 0.25:
                binary = cand
                th = t
                break

    k = int(max(1, min(open_kernel, 9)))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary, dtype=np.uint8)
    area_th = int(max(1, min_area))
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= area_th:
            cleaned[labels == i] = 1
    return cleaned.astype(np.float32), th

def smooth_probability_map(prob_map, sigma=1.2):
    s = float(max(0.0, min(sigma, 3.0)))
    if s <= 1e-6:
        return prob_map
    smoothed = cv2.GaussianBlur(prob_map.astype(np.float32), (0, 0), sigmaX=s, sigmaY=s)
    return np.clip(smoothed, 0.0, 1.0)

def estimate_eyelid_roi_mask(gray_img):
    if gray_img.dtype != np.uint8:
        img_u8 = np.clip(gray_img, 0, 255).astype(np.uint8)
    else:
        img_u8 = gray_img
    blur = cv2.GaussianBlur(img_u8, (5, 5), 0)
    _, roi = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    roi = roi.astype(np.uint8)

    kernel = np.ones((9, 9), np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = int(np.argmax(areas)) + 1
        roi = (labels == largest).astype(np.uint8)

    roi = cv2.dilate(roi, np.ones((7, 7), np.uint8), iterations=1)

    if float(roi.mean()) < 0.05:
        roi = np.ones_like(roi, dtype=np.uint8)
    return roi.astype(np.float32)

def render_prob_histogram(prob_map):
    hist_img = np.full((220, 360, 3), 245, dtype=np.uint8)
    if prob_map is None:
        return hist_img
    vals = np.clip(prob_map.astype(np.float32).ravel(), 0.0, 1.0)
    if vals.size == 0:
        return hist_img
    hist, _ = np.histogram(vals, bins=20, range=(0.0, 1.0))
    max_h = max(int(hist.max()), 1)
    x0, y0 = 28, 190
    w, h = 300, 150
    cv2.rectangle(hist_img, (x0, y0 - h), (x0 + w, y0), (60, 60, 60), 1)
    bar_w = w // len(hist)
    for i, c in enumerate(hist):
        bh = int((c / max_h) * (h - 4))
        x1 = x0 + i * bar_w + 1
        x2 = x0 + (i + 1) * bar_w - 1
        cv2.rectangle(hist_img, (x1, y0 - bh), (x2, y0), (80, 120, 230), -1)
    cv2.putText(hist_img, "Raw Probability Histogram", (28, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(hist_img, "0.0", (22, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.putText(hist_img, "1.0", (320, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1, cv2.LINE_AA)
    mean_v = float(vals.mean())
    cv2.putText(hist_img, f"mean={mean_v:.3f}", (220, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    return hist_img

def professional_inference(
    input_image,
    tile_infer,
    tile_height,
    tile_overlap,
    seg_threshold,
    min_area,
    smooth_fusion,
    smooth_sigma,
    roi_only,
    tta_mode,
):
    if input_image is None: return None, None, None, {"提示": "请上传红外影像"}
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    img_resized, img_clahe = preprocess_for_model(input_image)
    input_tensor = torch.from_numpy(img_clahe).unsqueeze(0).unsqueeze(0).float().to(device)

    model = get_clinical_model(torch.device("cpu"))
    model_path = "meibomian_model_best.pth"
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        try:
            model = model.to(device)
        except torch.OutOfMemoryError:
            if device.type == "mps":
                torch.mps.empty_cache()
            return input_image, None, None, {"错误": "推理加载模型时内存不足，请调小 Tile 高度后重试"}
    else:
        return input_image, None, None, {"错误": "未找到模型文件，请在训练 Tab 启动训练"}

    model.eval()
    with torch.no_grad():
        if tta_mode == "关闭":
            if tile_infer:
                roi_h = int(max(64, min(int(tile_height), data_loader.TARGET_HEIGHT)))
                roi_w = int(max(128, min(roi_h * 2, data_loader.TARGET_WIDTH)))
                overlap = float(max(0.05, min(float(tile_overlap), 0.95)))
                output = sliding_window_inference(
                    inputs=input_tensor,
                    roi_size=(roi_h, roi_w),
                    sw_batch_size=1,
                    predictor=model,
                    overlap=overlap,
                )
            else:
                output = model(input_tensor)
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
        else:
            probs = []
            if tile_infer:
                roi_h = int(max(64, min(int(tile_height), data_loader.TARGET_HEIGHT)))
                roi_w = int(max(128, min(roi_h * 2, data_loader.TARGET_WIDTH)))
                overlap = float(max(0.05, min(float(tile_overlap), 0.95)))
                output = sliding_window_inference(
                    inputs=input_tensor,
                    roi_size=(roi_h, roi_w),
                    sw_batch_size=1,
                    predictor=model,
                    overlap=overlap,
                )
            else:
                output = model(input_tensor)
            probs.append(torch.sigmoid(output).squeeze())

            input_hflip = torch.flip(input_tensor, dims=[3])
            if tile_infer:
                output_hflip = sliding_window_inference(
                    inputs=input_hflip,
                    roi_size=(roi_h, roi_w),
                    sw_batch_size=1,
                    predictor=model,
                    overlap=overlap,
                )
            else:
                output_hflip = model(input_hflip)
            prob_hflip = torch.flip(torch.sigmoid(output_hflip).squeeze(), dims=[2])
            probs.append(prob_hflip)

            if tta_mode == "4倍增强 (全面翻转)":
                input_vflip = torch.flip(input_tensor, dims=[2])
                if tile_infer:
                    output_vflip = sliding_window_inference(
                        inputs=input_vflip,
                        roi_size=(roi_h, roi_w),
                        sw_batch_size=1,
                        predictor=model,
                        overlap=overlap,
                    )
                else:
                    output_vflip = model(input_vflip)
                prob_vflip = torch.flip(torch.sigmoid(output_vflip).squeeze(), dims=[1])
                probs.append(prob_vflip)

                input_dflip = torch.flip(input_tensor, dims=[2, 3])
                if tile_infer:
                    output_dflip = sliding_window_inference(
                        inputs=input_dflip,
                        roi_size=(roi_h, roi_w),
                        sw_batch_size=1,
                        predictor=model,
                        overlap=overlap,
                    )
                else:
                    output_dflip = model(input_dflip)
                prob_dflip = torch.flip(torch.sigmoid(output_dflip).squeeze(), dims=[1, 2])
                probs.append(prob_dflip)

            prob_mean = torch.mean(torch.stack(probs), dim=0)
            prob_map = prob_mean.cpu().numpy()
    if smooth_fusion:
        prob_map = smooth_probability_map(prob_map, sigma=float(smooth_sigma))

    binary_mask, used_th = postprocess_binary_mask(
        prob_map,
        threshold=float(seg_threshold),
        min_area=int(min_area),
        open_kernel=3,
    )
    if roi_only:
        roi_mask = estimate_eyelid_roi_mask(img_resized)
        binary_mask = (binary_mask * roi_mask).astype(np.float32)
        prob_map = (prob_map * roi_mask).astype(np.float32)

    green_mask = np.zeros((img_resized.shape[0], img_resized.shape[1], 3), dtype=np.uint8)
    green_mask[binary_mask > 0] = [0, 255, 0]
    seg_overlay = cv2.addWeighted(cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB), 0.7, green_mask, 0.3, 0)
    heatmap_overlay = generate_heatmap(prob_map, img_resized)

    img_u8 = img_resized.astype(np.uint8)
    contour_img = generate_contour_visualization(prob_map, img_u8, min_area=int(min_area))
    skeleton_img = generate_skeleton_visualization(binary_mask, img_u8)
    instance_img = generate_instance_visualization(binary_mask, img_u8, min_area=int(min_area))

    metrics = calculate_clinical_metrics(binary_mask)
    metrics.append(["后处理阈值", f"{used_th:.2f}"])
    metrics.append(["边界平滑融合", "开启" if smooth_fusion else "关闭"])
    metrics.append(["ROI 约束", "仅眼睑区域" if roi_only else "全图"])
    return input_image, seg_overlay, heatmap_overlay, metrics, contour_img, skeleton_img, instance_img

def visualize_train_sample(tile_infer, tile_height, tile_overlap, seg_threshold, min_area, smooth_fusion, smooth_sigma, roi_only):
    try:
        vis_ds, _ = data_loader.build_training_loader(
            batch_size=1, tile_mode=False, tile_height=256, use_augmentation=False, num_workers=0
        )
        dataset_size = len(vis_ds)
    except Exception as e:
        return None, None, None, None, None, f"❌ 数据集不可用: {e}"
    if dataset_size == 0:
        return None, None, None, None, None, "❌ 训练集为空，请先检查 data_loader.py 中的数据路径。"

    idx = int(np.random.randint(0, dataset_size))
    sample = vis_ds[idx]

    image = sample["image"]
    label = sample["label"]
    if isinstance(image, torch.Tensor):
        image_np = image.squeeze().cpu().numpy().astype(np.float32)
    else:
        image_np = np.asarray(image).squeeze().astype(np.float32)
    if isinstance(label, torch.Tensor):
        label_np = label.squeeze().cpu().numpy().astype(np.float32)
    else:
        label_np = np.asarray(label).squeeze().astype(np.float32)

    image_u8 = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    label_bin = (label_np > 0.5).astype(np.uint8)
    label_vis = (label_bin * 255).astype(np.uint8)
    label_color = np.dstack([np.zeros_like(label_vis), label_vis, np.zeros_like(label_vis)])

    pred_vis = np.zeros_like(label_vis, dtype=np.uint8)
    pred_color = np.dstack([pred_vis, np.zeros_like(pred_vis), np.zeros_like(pred_vis)])
    prob_map_raw = None
    status = f"📌 抽样索引: {idx} / {dataset_size - 1}（仅显示 GT，未加载模型）"
    model_path = "meibomian_model_best.pth"
    if os.path.exists(model_path):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model = get_clinical_model(torch.device("cpu"))
        model.load_state_dict(state_dict)
        try:
            model = model.to(device)
        except torch.OutOfMemoryError:
            if device.type == "mps":
                torch.mps.empty_cache()
            gt_overlay = cv2.addWeighted(
                cv2.cvtColor(image_u8, cv2.COLOR_GRAY2RGB), 0.75,
                label_color, 0.25, 0
            )
            hist_img = render_prob_histogram(prob_map_raw)
            return image_u8, label_color, pred_color, np.hstack([gt_overlay, gt_overlay]), hist_img, "❌ 可视化加载模型时内存不足，请先关闭其他任务或切到 CPU。", None, None, None
        model.eval()
        with torch.no_grad():
            input_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).float().to(device)
            if tile_infer:
                roi_h = int(max(64, min(int(tile_height), data_loader.TARGET_HEIGHT)))
                roi_w = int(max(128, min(roi_h * 2, data_loader.TARGET_WIDTH)))
                overlap = float(max(0.05, min(float(tile_overlap), 0.95)))
                logits = sliding_window_inference(
                    inputs=input_tensor,
                    roi_size=(roi_h, roi_w),
                    sw_batch_size=1,
                    predictor=model,
                    overlap=overlap,
                )
            else:
                logits = model(input_tensor)
            prob_map_raw = torch.sigmoid(logits).squeeze().cpu().numpy()
        prob_map = smooth_probability_map(prob_map_raw, sigma=float(smooth_sigma) if smooth_fusion else 0.0)
        pred_mask, _ = postprocess_binary_mask(
            prob_map,
            threshold=float(seg_threshold),
            min_area=int(min_area),
            open_kernel=3,
        )
        if roi_only:
            roi_mask = estimate_eyelid_roi_mask(image_u8)
            pred_mask = (pred_mask * roi_mask).astype(np.float32)
        pred_bin = (pred_mask > 0.5).astype(np.uint8)
        pred_vis = (pred_bin * 255).astype(np.uint8)
        pred_color = np.dstack([pred_vis, np.zeros_like(pred_vis), np.zeros_like(pred_vis)])
        inter = (pred_bin * label_bin).sum()
        denom = pred_bin.sum() + label_bin.sum()
        dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
        status = f"📌 抽样索引: {idx} / {dataset_size - 1} | Dice: {dice:.4f}"

    gt_overlay = cv2.addWeighted(
        cv2.cvtColor(image_u8, cv2.COLOR_GRAY2RGB), 0.75,
        label_color, 0.25, 0
    )
    pred_overlay = cv2.addWeighted(
        cv2.cvtColor(image_u8, cv2.COLOR_GRAY2RGB), 0.75,
        pred_color, 0.25, 0
    )
    combined_overlay = np.hstack([gt_overlay, pred_overlay])

    contour_img = None
    skeleton_img = None
    instance_img = None

    if prob_map_raw is not None and 'pred_bin' in locals():
        pred_overlay = apply_medical_visualizations(pred_overlay, prob_map_raw, pred_bin, min_area=int(min_area))
        combined_overlay = np.hstack([gt_overlay, pred_overlay])

        contour_img = generate_contour_visualization(prob_map_raw, image_u8, min_area=int(min_area))
        skeleton_img = generate_skeleton_visualization(pred_bin, image_u8)
        instance_img = generate_instance_visualization(pred_bin, image_u8, min_area=int(min_area))
    hist_img = render_prob_histogram(prob_map_raw)
    return image_u8, label_color, pred_color, combined_overlay, hist_img, status, contour_img, skeleton_img, instance_img

def evaluate_n_train_samples(num_samples, tile_infer, tile_height, tile_overlap, seg_threshold, min_area, smooth_fusion, smooth_sigma, roi_only):
    try:
        vis_ds, _ = data_loader.build_training_loader(
            batch_size=1, tile_mode=False, tile_height=256, use_augmentation=False, num_workers=0
        )
        dataset_size = len(vis_ds)
    except Exception as e:
        return None, None, None, None, None, f"❌ 数据集不可用: {e}"
    if dataset_size == 0:
        return None, None, None, None, None, "❌ 训练集为空，请先检查 data_loader.py 中的数据路径。"

    n = int(max(1, num_samples))
    model_path = "meibomian_model_best.pth"
    if not os.path.exists(model_path):
        return None, None, None, None, None, "❌ 未找到 meibomian_model_best.pth，请先训练模型。"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model = get_clinical_model(torch.device("cpu"))
    model.load_state_dict(state_dict)
    try:
        model = model.to(device)
    except torch.OutOfMemoryError:
        if device.type == "mps":
            torch.mps.empty_cache()
        return None, None, None, None, None, "❌ 批量评估加载模型时内存不足，请先降低内存占用后重试。"
    model.eval()

    dices = []
    best_item = None
    worst_item = None
    best_dice = -1.0
    worst_dice = 2.0
    sampled_indices = []

    with torch.no_grad():
        for _ in range(n):
            idx = int(np.random.randint(0, dataset_size))
            sampled_indices.append(idx)
            sample = vis_ds[idx]
            image = sample["image"]
            label = sample["label"]
            image_np = image.squeeze().cpu().numpy().astype(np.float32) if isinstance(image, torch.Tensor) else np.asarray(image).squeeze().astype(np.float32)
            label_np = label.squeeze().cpu().numpy().astype(np.float32) if isinstance(label, torch.Tensor) else np.asarray(label).squeeze().astype(np.float32)

            label_bin = (label_np > 0.5).astype(np.uint8)
            input_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).float().to(device)
            if tile_infer:
                roi_h = int(max(64, min(int(tile_height), data_loader.TARGET_HEIGHT)))
                roi_w = int(max(128, min(roi_h * 2, data_loader.TARGET_WIDTH)))
                overlap = float(max(0.05, min(float(tile_overlap), 0.95)))
                logits = sliding_window_inference(
                    inputs=input_tensor,
                    roi_size=(roi_h, roi_w),
                    sw_batch_size=1,
                    predictor=model,
                    overlap=overlap,
                )
            else:
                logits = model(input_tensor)
            prob_map_raw = torch.sigmoid(logits).squeeze().cpu().numpy()
            prob_map = smooth_probability_map(prob_map_raw, sigma=float(smooth_sigma) if smooth_fusion else 0.0)
            pred_mask, _ = postprocess_binary_mask(
                prob_map,
                threshold=float(seg_threshold),
                min_area=int(min_area),
                open_kernel=3,
            )
            if roi_only:
                roi_mask = estimate_eyelid_roi_mask(np.clip(image_np * 255.0, 0, 255).astype(np.uint8))
                pred_mask = (pred_mask * roi_mask).astype(np.float32)
            pred_bin = (pred_mask > 0.5).astype(np.uint8)
            inter = (pred_bin * label_bin).sum()
            denom = pred_bin.sum() + label_bin.sum()
            dice = float((2.0 * inter + 1e-6) / (denom + 1e-6))
            dices.append(dice)

            image_u8 = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
            pred_vis = (pred_bin * 255).astype(np.uint8)
            label_vis = (label_bin * 255).astype(np.uint8)
            pred_color = np.dstack([pred_vis, np.zeros_like(pred_vis), np.zeros_like(pred_vis)])
            label_color = np.dstack([np.zeros_like(label_vis), label_vis, np.zeros_like(label_vis)])
            overlay = cv2.addWeighted(
                cv2.cvtColor(image_u8, cv2.COLOR_GRAY2RGB), 0.75,
                np.dstack([pred_vis, label_vis, np.zeros_like(pred_vis)]), 0.25, 0
            )

            item = (idx, image_u8, label_color, pred_color, overlay, dice, prob_map_raw)
            if dice > best_dice:
                best_dice = dice
                best_item = item
            if dice < worst_dice:
                worst_dice = dice
                worst_item = item

    avg_dice = float(np.mean(dices)) if dices else 0.0
    std_dice = float(np.std(dices)) if dices else 0.0
    status = (
        f"✅ 连续抽样 {n} 张 | 平均 Dice: {avg_dice:.4f} | 标准差: {std_dice:.4f} | "
        f"最佳: {best_dice:.4f}(idx={best_item[0]}) | 最差: {worst_dice:.4f}(idx={worst_item[0]})"
    )

    best_overlay = best_item[4]
    worst_overlay = worst_item[4]

    best_pred_mask, _ = postprocess_binary_mask(
        best_item[6],
        threshold=float(seg_threshold),
        min_area=int(min_area),
        open_kernel=3,
    )
    best_pred_bin = (best_pred_mask > 0.5).astype(np.uint8)
    best_overlay = apply_medical_visualizations(best_overlay, best_item[6], best_pred_bin, min_area=int(min_area))

    worst_pred_mask, _ = postprocess_binary_mask(
        worst_item[6],
        threshold=float(seg_threshold),
        min_area=int(min_area),
        open_kernel=3,
    )
    worst_pred_bin = (worst_pred_mask > 0.5).astype(np.uint8)
    worst_overlay = apply_medical_visualizations(worst_overlay, worst_item[6], worst_pred_bin, min_area=int(min_area))

    compare_overlay = np.hstack([best_overlay, worst_overlay])

    best_img_u8 = best_item[1]
    best_prob_map = best_item[6]

    best_pred_mask, _ = postprocess_binary_mask(
        best_prob_map,
        threshold=float(seg_threshold),
        min_area=int(min_area),
        open_kernel=3,
    )
    best_pred_bin = (best_pred_mask > 0.5).astype(np.uint8)

    contour_img = generate_contour_visualization(best_prob_map, best_img_u8, min_area=int(min_area))
    skeleton_img = generate_skeleton_visualization(best_pred_bin, best_img_u8)
    instance_img = generate_instance_visualization(best_pred_bin, best_img_u8, min_area=int(min_area))

    hist_img = render_prob_histogram(best_item[6])
    return best_item[1], best_item[2], best_item[3], compare_overlay, hist_img, status, contour_img, skeleton_img, instance_img

def sync_infer_controls(tile_mode, tile_size, tile_overlap, threshold, min_area, smooth_on, smooth_sigma, roi_only, tta_mode="关闭"):
    tile_mode = bool(tile_mode)
    tile_size = int(max(128, min(int(tile_size), 512)))
    tile_overlap = float(max(0.1, min(float(tile_overlap), 0.7)))
    threshold = float(max(0.4, min(float(threshold), 0.95)))
    min_area = int(max(20, min(int(min_area), 500)))
    smooth_on = bool(smooth_on)
    smooth_sigma = float(max(0.0, min(float(smooth_sigma), 3.0)))
    roi_only = bool(roi_only)
    return (tile_mode, tile_size, tile_overlap, threshold, min_area, smooth_on, smooth_sigma, roi_only, tta_mode,
            tile_mode, tile_size, tile_overlap, threshold, min_area, smooth_on, smooth_sigma, roi_only, tta_mode)

def load_all_presets():
    presets = dict(BUILTIN_PRESETS)
    if os.path.exists(PRESET_FILE):
        try:
            with open(PRESET_FILE, "r", encoding="utf-8") as f:
                custom = json.load(f)
            if isinstance(custom, dict):
                for k, v in custom.items():
                    if isinstance(v, dict):
                        presets[k] = v
        except Exception:
            pass
    return presets

def save_custom_preset_file(all_presets):
    custom = {k: v for k, v in all_presets.items() if k not in BUILTIN_PRESETS}
    with open(PRESET_FILE, "w", encoding="utf-8") as f:
        json.dump(custom, f, ensure_ascii=False, indent=2)

def apply_named_preset(preset_name):
    presets = load_all_presets()
    p = presets.get(preset_name, BUILTIN_PRESETS["平衡-默认"])
    synced = sync_infer_controls(
        p.get("tile_mode", True),
        p.get("tile_size", 192),
        p.get("tile_overlap", 0.25),
        p.get("threshold", 0.70),
        p.get("min_area", 120),
        p.get("smooth_on", True),
        p.get("smooth_sigma", 1.2),
        p.get("roi_only", True),
        p.get("tta_mode", "关闭"),
    )
    names = list(load_all_presets().keys())
    status = f"✅ 已应用预设：{preset_name}"

    return (
        *synced,
        gr.update(choices=names, value=preset_name),
        gr.update(choices=names, value=preset_name),
        status
    )

def save_named_preset(preset_name, tile_mode, tile_size, tile_overlap, threshold, min_area, smooth_on, smooth_sigma, roi_only, tta_mode):
    name = (preset_name or "").strip()
    if not name:
        names = list(load_all_presets().keys())
        return gr.update(choices=names), gr.update(choices=names), "❌ 预设名不能为空"
    if name in BUILTIN_PRESETS:
        names = list(load_all_presets().keys())
        return gr.update(choices=names), gr.update(choices=names), "❌ 内置预设不可覆盖，请换个名字保存"
    presets = load_all_presets()
    synced = sync_infer_controls(tile_mode, tile_size, tile_overlap, threshold, min_area, smooth_on, smooth_sigma, roi_only, tta_mode)
    presets[name] = {
        "tile_mode": synced[0], "tile_size": synced[1], "tile_overlap": synced[2],
        "threshold": synced[3], "min_area": synced[4],
        "smooth_on": synced[5], "smooth_sigma": synced[6], "roi_only": synced[7],
        "tta_mode": synced[8],
    }
    save_custom_preset_file(presets)
    names = list(load_all_presets().keys())
    return gr.update(choices=names, value=name), gr.update(choices=names, value=name), f"💾 已保存预设：{name}"

with gr.Blocks(title="“腺”而易见 - 满血专业版") as app:

    gr.Markdown("# 👁️ “腺”而易见 —— MGD 智能辅助诊疗系统 V2.0")

    with gr.Tabs():
        with gr.TabItem("🩺 临床诊断终端"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_img = gr.Image(label="载入红外影像", type="numpy")
                    infer_preset_diag = gr.Dropdown(
                        choices=list(load_all_presets().keys()),
                        value="平衡-默认",
                        label="推理参数预设（诊断页）"
                    )
                    infer_tile_mode = gr.Checkbox(label="🧩 分块推理 (滑窗)", value=True)
                    infer_tile_size = gr.Slider(128, 512, value=192, step=32, label="推理 Tile 高度")
                    infer_tile_overlap = gr.Slider(0.1, 0.7, value=0.25, step=0.05, label="滑窗重叠比例")
                    infer_threshold = gr.Slider(0.4, 0.95, value=0.7, step=0.05, label="分割阈值（抑制溢出）")
                    infer_min_area = gr.Slider(20, 500, value=120, step=10, label="最小连通域面积")
                    infer_smooth_fusion = gr.Checkbox(label="🧪 边界平滑融合", value=True)
                    infer_smooth_sigma = gr.Slider(0.0, 3.0, value=1.2, step=0.1, label="平滑强度 (sigma)")
                    infer_roi_only = gr.Checkbox(label="🎯 仅眼睑 ROI 内保留", value=True)
                    infer_tta = gr.Dropdown(
                        choices=["关闭", "2倍增强 (水平翻转)", "4倍增强 (全面翻转)"],
                        value="关闭",
                        label="测试时增强 (TTA)"
                    )
                    btn_infer = gr.Button("🚀 运行定量分析", variant="primary", elem_classes="primary-btn")
                    gr.Markdown("### 📊 临床定量报告")
                    out_metrics = gr.Dataframe(
                        headers=["指标名称", "检测数值"],
                        label="AI 自动量化指标",
                        interactive=False
                    )

                with gr.Column(scale=2):
                    with gr.Row():
                        out_original = gr.Image(label="原始影像", height=640, width=1280)
                        out_seg = gr.Image(label="AI 分割结果", height=640, width=1280)
                    with gr.Row():
                        out_contour = gr.Image(label="📊 多级概率等高线", height=640, width=1280)
                        out_skeleton = gr.Image(label="🦴 腺体骨架线", height=640, width=1280)
                    with gr.Row():
                        out_instance = gr.Image(label="🔢 实例轮廓与编号", height=640, width=1280)
                    out_heatmap = gr.Image(label="🔥 AI 决策热力图 (可解释性)", height=640, width=1280)
                    infer_preset_status = gr.Textbox(label="预设状态", lines=1, interactive=False)

            btn_infer.click(
                professional_inference,
                [
                    input_img,
                    infer_tile_mode,
                    infer_tile_size,
                    infer_tile_overlap,
                    infer_threshold,
                    infer_min_area,
                    infer_smooth_fusion,
                    infer_smooth_sigma,
                    infer_roi_only,
                    infer_tta,
                ],
                [out_original, out_seg, out_heatmap, out_metrics, out_contour, out_skeleton, out_instance]
            )

        with gr.TabItem("⚙️ 模型训练与微调"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🛠️ 训练超参数")
                    train_mem = gr.Slider(0.1, 1.0, value=1.0, step=0.1, label="内存限额")
                    train_prec = gr.Radio(["FP16 (混合精度)", "FP32 (标准)"], label="计算精度", value="FP16 (混合精度)")

                    gr.Markdown("#### 📊 优化器与学习率")
                    train_lr = gr.Number(value=0.001, label="初始学习率 (LR)", info="范围: 0.0001 ~ 0.1")
                    train_optimizer = gr.Radio(["Adam", "AdamW", "SGD"], value="AdamW", label="优化器选择")
                    train_scheduler = gr.Dropdown(
                        choices=["ReduceLROnPlateau (默认)", "CosineAnnealingWarmRestarts", "Cosine Annealing", "Step Decay", "Fixed LR"],
                        value="ReduceLROnPlateau (默认)",
                        label="学习率调度器"
                    )
                    train_weight_decay = gr.Number(value=0.00001, label="权重衰减 (Weight Decay)", info="范围: 0 ~ 0.1")

                    gr.Markdown("#### ⚖️ 损失函数配置")
                    train_loss_fn = gr.Radio(["Dice Loss", "Cross Entropy", "Focal Loss", "Dice+CE 组合损失"], value="Dice Loss", label="主损失函数")
                    train_pos_weight = gr.Number(value=1.0, label="正负样本平衡权重", info="范围: 0.1 ~ 10.0")

                    gr.Markdown("#### 🌟 数据增强配置")
                    train_use_augmentation = gr.Checkbox(label="启用数据增强 (Data Augmentation)", value=False, info="随机翻转图像，让模型真正理解解剖结构")

                    train_resume = gr.Checkbox(label="🔌 断点续训", value=True)
                    train_epochs = gr.Number(value=10, label="训练轮数 (Epochs)")
                    train_batch = gr.Slider(1, 8, value=1, step=1, label="批次大小 (Batch Size)")
                    train_tile_mode = gr.Checkbox(label="🧩 分块训练 (随机裁块)", value=True)
                    train_tile_size = gr.Slider(128, 512, value=192, step=32, label="训练 Tile 高度")

                    gr.Markdown("### 📂 模型管家")
                    train_save_freq = gr.Number(value=2, label="保存频率")
                    train_keep_max = gr.Number(value=3, label="最大保留数")

                    btn_train = gr.Button("🚀 启动自动化训练", variant="stop")
                    gr.Markdown("### 🔁 推理参数（与诊断页同步）")
                    infer_preset_train = gr.Dropdown(
                        choices=list(load_all_presets().keys()),
                        value="平衡-默认",
                        label="推理参数预设（训练页）"
                    )
                    infer_preset_name = gr.Textbox(label="保存当前参数为预设", placeholder="输入新预设名，例如：我的稳健方案")
                    btn_save_preset = gr.Button("💾 保存当前参数为预设", variant="secondary")
                    train_infer_tile_mode = gr.Checkbox(label="🧩 分块推理 (滑窗)", value=True)
                    train_infer_tile_size = gr.Slider(128, 512, value=192, step=32, label="推理 Tile 高度")
                    train_infer_tile_overlap = gr.Slider(0.1, 0.7, value=0.25, step=0.05, label="滑窗重叠比例")
                    train_infer_threshold = gr.Slider(0.4, 0.95, value=0.7, step=0.05, label="分割阈值（抑制溢出）")
                    train_infer_min_area = gr.Slider(20, 500, value=120, step=10, label="最小连通域面积")
                    train_infer_smooth_fusion = gr.Checkbox(label="🧪 边界平滑融合", value=True)
                    train_infer_smooth_sigma = gr.Slider(0.0, 3.0, value=1.2, step=0.1, label="平滑强度 (sigma)")
                    train_infer_roi_only = gr.Checkbox(label="🎯 仅眼睑 ROI 内保留", value=True)
                    train_infer_tta = gr.Dropdown(
                        choices=["关闭", "2倍增强 (水平翻转)", "4倍增强 (全面翻转)"],
                        value="关闭",
                        label="测试时增强 (TTA)"
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### 📝 实时训练日志")
                    train_log = gr.Textbox(label="Terminal 输出", lines=20, interactive=False)
                    gr.Markdown("### 🔎 训练集抽样可视化验证")
                    btn_sample_vis = gr.Button("🎯 抽样验证 GT/预测", variant="secondary")
                    sample_n = gr.Slider(1, 100, value=10, step=1, label="连续抽样数量 N")
                    btn_batch_eval = gr.Button("📈 连续抽样 N 张并统计平均 Dice", variant="secondary")
                    sample_status = gr.Textbox(label="抽样状态", lines=2, interactive=False)
                    with gr.Row():
                        sample_img = gr.Image(label="训练图像", height=640, width=1280)
                        sample_gt = gr.Image(label="标准标注 (GT)", height=640, width=1280)
                        sample_pred = gr.Image(label="模型预测", height=640, width=1280)
                    sample_overlay = gr.Image(label="叠加对比（左:GT 右:预测）", height=640, width=2560)
                    with gr.Row():
                        sample_contour = gr.Image(label="📊 多级概率等高线", height=640, width=1280)
                        sample_skeleton = gr.Image(label="🦴 腺体骨架线", height=640, width=1280)
                    with gr.Row():
                        sample_instance = gr.Image(label="🔢 实例轮廓与编号", height=640, width=1280)
                    sample_hist = gr.Image(label="原始概率图直方图", height=640, width=1280)

            for control in [
                infer_tile_mode, infer_tile_size, infer_tile_overlap, infer_threshold,
                infer_min_area, infer_smooth_fusion, infer_smooth_sigma, infer_roi_only, infer_tta
            ]:
                control.change(
                    fn=sync_infer_controls,
                    inputs=[
                        infer_tile_mode, infer_tile_size, infer_tile_overlap, infer_threshold,
                        infer_min_area, infer_smooth_fusion, infer_smooth_sigma, infer_roi_only, infer_tta
                    ],
                    outputs=[
                        infer_tile_mode, infer_tile_size, infer_tile_overlap, infer_threshold,
                        infer_min_area, infer_smooth_fusion, infer_smooth_sigma, infer_roi_only, infer_tta,
                        train_infer_tile_mode, train_infer_tile_size, train_infer_tile_overlap, train_infer_threshold,
                        train_infer_min_area, train_infer_smooth_fusion, train_infer_smooth_sigma, train_infer_roi_only, train_infer_tta
                    ],
                )

            for control in [
                train_infer_tile_mode, train_infer_tile_size, train_infer_tile_overlap, train_infer_threshold,
                train_infer_min_area, train_infer_smooth_fusion, train_infer_smooth_sigma, train_infer_roi_only, train_infer_tta
            ]:
                control.change(
                    fn=sync_infer_controls,
                    inputs=[
                        train_infer_tile_mode, train_infer_tile_size, train_infer_tile_overlap, train_infer_threshold,
                        train_infer_min_area, train_infer_smooth_fusion, train_infer_smooth_sigma, train_infer_roi_only, train_infer_tta
                    ],
                    outputs=[
                        train_infer_tile_mode, train_infer_tile_size, train_infer_tile_overlap, train_infer_threshold,
                        train_infer_min_area, train_infer_smooth_fusion, train_infer_smooth_sigma, train_infer_roi_only, train_infer_tta,
                        infer_tile_mode, infer_tile_size, infer_tile_overlap, infer_threshold,
                        infer_min_area, infer_smooth_fusion, infer_smooth_sigma, infer_roi_only, infer_tta
                    ],
                )

            infer_preset_diag.change(
                fn=apply_named_preset,
                inputs=[infer_preset_diag],
                outputs=[
                    infer_tile_mode, infer_tile_size, infer_tile_overlap, infer_threshold,
                    infer_min_area, infer_smooth_fusion, infer_smooth_sigma, infer_roi_only, infer_tta,
                    train_infer_tile_mode, train_infer_tile_size, train_infer_tile_overlap, train_infer_threshold,
                    train_infer_min_area, train_infer_smooth_fusion, train_infer_smooth_sigma, train_infer_roi_only, train_infer_tta,
                    infer_preset_diag, infer_preset_train, infer_preset_status
                ],
            )
            infer_preset_train.change(
                fn=apply_named_preset,
                inputs=[infer_preset_train],
                outputs=[
                    infer_tile_mode, infer_tile_size, infer_tile_overlap, infer_threshold,
                    infer_min_area, infer_smooth_fusion, infer_smooth_sigma, infer_roi_only, infer_tta,
                    train_infer_tile_mode, train_infer_tile_size, train_infer_tile_overlap, train_infer_threshold,
                    train_infer_min_area, train_infer_smooth_fusion, train_infer_smooth_sigma, train_infer_roi_only, train_infer_tta,
                    infer_preset_diag, infer_preset_train, infer_preset_status
                ],
            )
            btn_save_preset.click(
                fn=save_named_preset,
                inputs=[
                    infer_preset_name,
                    train_infer_tile_mode, train_infer_tile_size, train_infer_tile_overlap, train_infer_threshold,
                    train_infer_min_area, train_infer_smooth_fusion, train_infer_smooth_sigma, train_infer_roi_only, train_infer_tta
                ],
                outputs=[infer_preset_diag, infer_preset_train, infer_preset_status]
            )

            btn_train.click(
                fn=professional_training,
                inputs=[train_resume, train_epochs, train_batch, train_save_freq, train_keep_max, train_mem, train_prec, train_tile_mode, train_tile_size, train_lr, train_optimizer, train_scheduler, train_weight_decay, train_loss_fn, train_pos_weight, train_use_augmentation],
                outputs=[train_log]
            )
            btn_sample_vis.click(
                fn=visualize_train_sample,
                inputs=[
                    train_infer_tile_mode, train_infer_tile_size, train_infer_tile_overlap, train_infer_threshold,
                    train_infer_min_area, train_infer_smooth_fusion, train_infer_smooth_sigma, train_infer_roi_only
                ],
                outputs=[sample_img, sample_gt, sample_pred, sample_overlay, sample_hist, sample_status, sample_contour, sample_skeleton, sample_instance]
            )
            btn_batch_eval.click(
                fn=evaluate_n_train_samples,
                inputs=[
                    sample_n,
                    train_infer_tile_mode, train_infer_tile_size, train_infer_tile_overlap, train_infer_threshold,
                    train_infer_min_area, train_infer_smooth_fusion, train_infer_smooth_sigma, train_infer_roi_only
                ],
                outputs=[sample_img, sample_gt, sample_pred, sample_overlay, sample_hist, sample_status, sample_contour, sample_skeleton, sample_instance]
            )

if __name__ == "__main__":
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    app.launch(server_name="0.0.0.0", server_port=server_port, css="style.css")
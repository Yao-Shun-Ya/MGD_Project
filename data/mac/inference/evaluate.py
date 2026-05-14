"""
评估模块
用于训练集样本可视化和评估
"""
import os
import numpy as np
import torch
import cv2
import pandas as pd
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.transforms import AsDiscrete
from data import get_data_manager
from models import get_clinical_model
from utils import (
    postprocess_binary_mask,
    smooth_probability_map,
    estimate_eyelid_roi_mask,
    apply_medical_visualizations,
    generate_contour_visualization,
    generate_skeleton_visualization,
    generate_instance_visualization,
    render_prob_histogram,
)
from config import MODEL_BEST_PATH, get_model_best_path


def visualize_train_sample(
    model_name,
    tile_infer,
    tile_height,
    tile_overlap,
    seg_threshold,
    min_area,
    smooth_fusion,
    smooth_sigma,
    roi_only,
):
    """
    随机抽样训练集，展示 GT 与模型预测对比
    
    Returns:
        (image_u8, label_color, pred_color, combined_overlay, hist_img, status, contour_img, skeleton_img, instance_img)
    """
    try:
        dm = get_data_manager()
        vis_ds, _ = dm.build_training_loader(
            batch_size=1, tile_mode=False, tile_height=256, use_augmentation=False, num_workers=0
        )
        dataset_size = len(vis_ds)
    except Exception as e:
        return None, None, None, None, None, f"❌ 数据集不可用: {e}", None, None, None
    
    if dataset_size == 0:
        return None, None, None, None, None, "❌ 训练集为空，请先检查数据路径。", None, None, None
    
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
    contour_img = None
    skeleton_img = None
    instance_img = None
    
    model_path = get_model_best_path(model_name)
    
    if os.path.exists(model_path):
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model = get_clinical_model(torch.device("cpu"), model_name=model_name)
        model.load_state_dict(state_dict)
        try:
            model = model.to(device)
        except torch.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                error_msg = "❌ 可视化加载模型时显存不足，请先关闭其他任务或切到 CPU。"
            elif device.type == "mps":
                torch.mps.empty_cache()
                error_msg = "❌ 可视化加载模型时 Apple Silicon GPU 内存不足，请先关闭其他任务或切到 CPU。"
            else:
                error_msg = "❌ 可视化加载模型时内存不足，请先关闭其他任务。"
            gt_overlay = cv2.addWeighted(
                cv2.cvtColor(image_u8, cv2.COLOR_GRAY2RGB), 0.75,
                label_color, 0.25, 0
            )
            hist_img = render_prob_histogram(prob_map_raw)
            return image_u8, label_color, pred_color, np.hstack([gt_overlay, gt_overlay]), hist_img, error_msg, None, None, None
        
        model.eval()
        with torch.no_grad():
            input_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).float().to(device)
            if tile_infer:
                from data import TARGET_HEIGHT, TARGET_WIDTH
                from monai.inferers import sliding_window_inference
                roi_h = int(max(64, min(int(tile_height), TARGET_HEIGHT)))
                roi_w = int(max(128, min(roi_h * 2, TARGET_WIDTH)))
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
        
        contour_img = generate_contour_visualization(prob_map_raw, image_u8, min_area=int(min_area))
        skeleton_img = generate_skeleton_visualization(pred_bin, image_u8)
        instance_img = generate_instance_visualization(pred_bin, image_u8, min_area=int(min_area))
    
    gt_overlay = cv2.addWeighted(
        cv2.cvtColor(image_u8, cv2.COLOR_GRAY2RGB), 0.75,
        label_color, 0.25, 0
    )
    pred_overlay = cv2.addWeighted(
        cv2.cvtColor(image_u8, cv2.COLOR_GRAY2RGB), 0.75,
        pred_color, 0.25, 0
    )
    
    combined_overlay = np.hstack([gt_overlay, pred_overlay])
    
    if prob_map_raw is not None and 'pred_bin' in locals():
        pred_overlay = apply_medical_visualizations(pred_overlay, prob_map_raw, pred_bin, min_area=int(min_area))
        combined_overlay = np.hstack([gt_overlay, pred_overlay])
    
    hist_img = render_prob_histogram(prob_map_raw)
    return image_u8, label_color, pred_color, combined_overlay, hist_img, status, contour_img, skeleton_img, instance_img


def evaluate_n_train_samples(
    num_samples,
    model_name,
    tile_infer,
    tile_height,
    tile_overlap,
    seg_threshold,
    min_area,
    smooth_fusion,
    smooth_sigma,
    roi_only,
):
    """
    连续抽样 N 张，统计平均 Dice，并展示最佳/最差样本
    
    Returns:
        (best_image, best_label, best_pred, compare_overlay, hist_img, status, contour_img, skeleton_img, instance_img)
    """
    try:
        dm = get_data_manager()
        vis_ds, _ = dm.build_training_loader(
            batch_size=1, tile_mode=False, tile_height=256, use_augmentation=False, num_workers=0
        )
        dataset_size = len(vis_ds)
    except Exception as e:
        return None, None, None, None, None, f"❌ 数据集不可用: {e}", None, None, None
    
    if dataset_size == 0:
        return None, None, None, None, None, "❌ 训练集为空，请先检查数据路径。", None, None, None
    
    n = int(max(1, num_samples))
    
    model_path = get_model_best_path(model_name)
    
    if not os.path.exists(model_path):
        return None, None, None, None, None, f"❌ 未找到 {model_name} 模型文件，请先训练该模型。", None, None, None
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model = get_clinical_model(torch.device("cpu"), model_name=model_name)
    model.load_state_dict(state_dict)
    try:
        model = model.to(device)
    except torch.OutOfMemoryError:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            error_msg = "❌ 批量评估加载模型时显存不足，请先降低显存占用后重试。"
        elif device.type == "mps":
            torch.mps.empty_cache()
            error_msg = "❌ 批量评估加载模型时 Apple Silicon GPU 内存不足，请先降低内存占用后重试。"
        else:
            error_msg = "❌ 批量评估加载模型时内存不足，请先降低内存占用后重试。"
        return None, None, None, None, None, error_msg, None, None, None
    
    model.eval()
    
    # 初始化 MONAI 指标（逐样本计算，单通道输出）
    post_label = AsDiscrete(to_onehot=2)
    post_pred = AsDiscrete(argmax=False, threshold=0.5, to_onehot=2)
    
    dices = []
    ious = []
    hd95s = []
    assds = []
    
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
                from data import TARGET_HEIGHT, TARGET_WIDTH
                from monai.inferers import sliding_window_inference
                roi_h = int(max(64, min(int(tile_height), TARGET_HEIGHT)))
                roi_w = int(max(128, min(roi_h * 2, TARGET_WIDTH)))
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
                image_u8 = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
                roi_mask = estimate_eyelid_roi_mask(image_u8)
                pred_mask = (pred_mask * roi_mask).astype(np.float32)
            
            pred_bin = (pred_mask > 0.5).astype(np.uint8)
            
            # 逐样本计算 MONAI 指标
            pred_tensor = torch.from_numpy(pred_bin).unsqueeze(0).unsqueeze(0).float()
            label_tensor = torch.from_numpy(label_bin).unsqueeze(0).unsqueeze(0).float()
            
            # 转换为 one-hot
            pred_onehot = post_pred(pred_tensor)
            label_onehot = post_label(label_tensor)
            
            # 逐样本计算四个指标
            try:
                # Dice
                dice_metric = DiceMetric(include_background=False, reduction="mean")
                dice_metric(y_pred=pred_onehot, y=label_onehot)
                current_dice = float(dice_metric.aggregate().item())
            except:
                current_dice = 0.0
            
            try:
                # IoU
                iou_metric = MeanIoU(include_background=False, reduction="mean")
                iou_metric(y_pred=pred_onehot, y=label_onehot)
                current_iou = float(iou_metric.aggregate().item())
            except:
                current_iou = 0.0
            
            try:
                # HD95
                hd_metric = HausdorffDistanceMetric(include_background=False, percentile=95)
                hd_metric(y_pred=pred_onehot, y=label_onehot)
                current_hd95 = float(hd_metric.aggregate().item())
            except:
                current_hd95 = 0.0
            
            try:
                # ASSD
                assd_metric = SurfaceDistanceMetric(include_background=False)
                assd_metric(y_pred=pred_onehot, y=label_onehot)
                current_assd = float(assd_metric.aggregate().item())
            except:
                current_assd = 0.0
            
            # 保存当前指标
            dices.append(current_dice)
            ious.append(current_iou)
            hd95s.append(current_hd95)
            assds.append(current_assd)
            
            inter = (pred_bin * label_bin).sum()
            denom = pred_bin.sum() + label_bin.sum()
            dice = float((2.0 * inter + 1e-6) / (denom + 1e-6))
            
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
        
        # 计算平均指标和标准差
        avg_dice = float(np.mean(dices)) if dices else 0.0
        std_dice = float(np.std(dices)) if dices else 0.0
        
        avg_iou = float(np.mean(ious)) if ious else 0.0
        std_iou = float(np.std(ious)) if ious else 0.0
        
        avg_hd95 = float(np.mean(hd95s)) if hd95s else 0.0
        std_hd95 = float(np.std(hd95s)) if hd95s else 0.0
        
        avg_assd = float(np.mean(assds)) if assds else 0.0
        std_assd = float(np.std(assds)) if assds else 0.0
        
        # 创建 DataFrame
        metrics_df = pd.DataFrame({
            'Index': sampled_indices,
            'Dice': dices,
            'IoU': ious,
            'HD95': hd95s,
            'ASSD': assds,
        })
        
        # 计算统计量（均值±标准差格式）
        stats_df = pd.DataFrame({
            'Metric': ['Dice', 'IoU', 'HD95', 'ASSD'],
            'Mean': [
                f"{avg_dice:.4f} ± {std_dice:.4f}",
                f"{avg_iou:.4f} ± {std_iou:.4f}",
                f"{avg_hd95:.4f} ± {std_hd95:.4f}",
                f"{avg_assd:.4f} ± {std_assd:.4f}",
            ],
        })
        
        # 导出 CSV
        csv_path = f"{model_name}_metrics.csv"
        metrics_df.to_csv(csv_path, index=False)
        
        # 导出 LaTeX 表格
        latex_path = f"{model_name}_metrics.tex"
        latex_table = stats_df.to_latex(
            index=False,
            caption=f"Segmentation Metrics for {model_name}",
            label=f"tab:{model_name}_metrics",
            escape=False,
        )
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        status = (
            f"✅ 连续抽样 {n} 张 | Dice: {avg_dice:.4f} ± {std_dice:.4f} | "
            f"IoU: {avg_iou:.4f} ± {std_iou:.4f} | HD95: {avg_hd95:.4f} ± {std_hd95:.4f} | ASSD: {avg_assd:.4f} ± {std_assd:.4f} | "
            f"最佳: {best_dice:.4f}(idx={best_item[0]}) | 最差: {worst_dice:.4f}(idx={worst_item[0]}) | "
            f"CSV: {csv_path} | LaTeX: {latex_path}"
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

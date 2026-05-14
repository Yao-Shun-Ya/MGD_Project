"""
推理引擎模块
"""
import os
import torch
import numpy as np
import cv2
from monai.inferers import sliding_window_inference
from models import get_clinical_model
from utils import (
    preprocess_for_model,
    postprocess_binary_mask,
    smooth_probability_map,
    estimate_eyelid_roi_mask,
    generate_heatmap,
    generate_contour_visualization,
    generate_skeleton_visualization,
    generate_instance_visualization,
    calculate_clinical_metrics,
)
from data import TARGET_HEIGHT, TARGET_WIDTH
from config import MODEL_BEST_PATH, get_model_best_path


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
    model_name="UNet",
):
    """
    专业推理函数
    
    Args:
        input_image: 输入图像
        tile_infer: 是否分块推理
        tile_height: 分块高度
        tile_overlap: 重叠比例
        seg_threshold: 分割阈值
        min_area: 最小面积
        smooth_fusion: 是否平滑融合
        smooth_sigma: 平滑系数
        roi_only: 是否仅 ROI
        tta_mode: TTA 模式
    
    Returns:
        (original_img, seg_overlay, heatmap_overlay, metrics, contour_img, skeleton_img, instance_img)
    """
    if input_image is None:
        return None, None, None, {"提示": "请上传红外影像"}, None, None, None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_resized, img_clahe = preprocess_for_model(input_image)
    input_tensor = torch.from_numpy(img_clahe).unsqueeze(0).unsqueeze(0).float().to(device)
    
    model_path = get_model_best_path(model_name)
    
    model = get_clinical_model(torch.device("cpu"), model_name=model_name)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        try:
            model = model.to(device)
        except torch.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return input_image, None, None, {"错误": "推理加载模型时显存不足，请调小 Tile 高度后重试"}, None, None, None
    else:
        return input_image, None, None, {"错误": f"未找到 {model_name} 模型文件，请先在训练 Tab 训练该模型"}, None, None, None
    
    model.eval()
    with torch.no_grad():
        if tta_mode == "关闭":
            if tile_infer:
                roi_h = int(max(64, min(int(tile_height), TARGET_HEIGHT)))
                roi_w = int(max(128, min(roi_h * 2, TARGET_WIDTH)))
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
            
            def _infer_once(tensor):
                if tile_infer:
                    roi_h = int(max(64, min(int(tile_height), TARGET_HEIGHT)))
                    roi_w = int(max(128, min(roi_h * 2, TARGET_WIDTH)))
                    overlap = float(max(0.05, min(float(tile_overlap), 0.95)))
                    return sliding_window_inference(
                        inputs=tensor,
                        roi_size=(roi_h, roi_w),
                        sw_batch_size=1,
                        predictor=model,
                        overlap=overlap,
                    )
                else:
                    return model(tensor)
            
            # 原始
            output = _infer_once(input_tensor)
            probs.append(torch.sigmoid(output).squeeze())
            
            # 水平翻转
            input_hflip = torch.flip(input_tensor, dims=[3])
            output_hflip = _infer_once(input_hflip)
            prob_hflip = torch.flip(torch.sigmoid(output_hflip).squeeze(), dims=[1])
            probs.append(prob_hflip)
            
            if tta_mode == "4倍增强 (全面翻转)":
                # 垂直翻转
                input_vflip = torch.flip(input_tensor, dims=[2])
                output_vflip = _infer_once(input_vflip)
                prob_vflip = torch.flip(torch.sigmoid(output_vflip).squeeze(), dims=[0])
                probs.append(prob_vflip)
                
                # 对角翻转
                input_dflip = torch.flip(input_tensor, dims=[2, 3])
                output_dflip = _infer_once(input_dflip)
                prob_dflip = torch.flip(torch.sigmoid(output_dflip).squeeze(), dims=[0, 1])
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

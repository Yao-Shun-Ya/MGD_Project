"""
可视化工具模块
"""
import numpy as np
import cv2
from skimage.morphology import skeletonize


def apply_medical_visualizations(overlay_img, prob_map, binary_mask, min_area=50):
    """
    应用医疗图像后处理可视化
    
    Args:
        overlay_img: 叠加图像
        prob_map: 概率图
        binary_mask: 二值化掩膜
        min_area: 最小面积
    
    Returns:
        处理后的叠加图像
    """
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
    """
    生成多级概率等高线可视化图
    """
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
    """
    生成腺体骨架线可视化图
    """
    if len(original_img.shape) == 2:
        result_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    else:
        result_img = original_img.copy()
    
    bool_mask = binary_mask > 0
    skeleton = skeletonize(bool_mask)
    result_img[skeleton] = [255, 0, 0]
    
    return result_img


def generate_instance_visualization(binary_mask, original_img, min_area=50):
    """
    生成实例轮廓与编号可视化图
    """
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


def generate_heatmap(prob_map, original_img):
    """
    生成 AI 决策热力图 (可解释性分析)
    """
    heatmap_gray = np.uint8(255 * prob_map)
    heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
    if len(original_img.shape) == 2:
        original_img_color = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    else:
        original_img_color = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    heatmap_color = cv2.resize(heatmap_color, (original_img_color.shape[1], original_img_color.shape[0]))
    overlay = cv2.addWeighted(original_img_color, 0.5, heatmap_color, 0.5, 0)
    return overlay


def render_prob_histogram(prob_map):
    """
    将原始概率图渲染为小型直方图面板
    """
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

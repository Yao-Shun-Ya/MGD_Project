"""
后处理模块
"""
import numpy as np
import cv2


def postprocess_binary_mask(prob_map, threshold=0.7, min_area=120, open_kernel=3):
    """
    抑制假阳性：阈值 + 开运算 + 小连通域过滤
    
    Args:
        prob_map: 概率图
        threshold: 阈值
        min_area: 最小面积
        open_kernel: 开运算核大小
    
    Returns:
        (binary_mask, used_threshold): 处理后的二值掩膜和使用的阈值
    """
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
    """
    边界平滑融合：对概率图做轻量高斯平滑，缓解块边界伪影
    
    Args:
        prob_map: 概率图
        sigma: 高斯平滑系数
    
    Returns:
        平滑后的概率图
    """
    s = float(max(0.0, min(sigma, 3.0)))
    if s <= 1e-6:
        return prob_map
    smoothed = cv2.GaussianBlur(prob_map.astype(np.float32), (0, 0), sigmaX=s, sigmaY=s)
    return np.clip(smoothed, 0.0, 1.0)


def estimate_eyelid_roi_mask(gray_img):
    """
    从红外灰度图估计眼睑有效 ROI，抑制非相关区域误检
    
    Args:
        gray_img: 灰度图像
    
    Returns:
        ROI 掩膜
    """
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

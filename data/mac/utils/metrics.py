"""
临床指标计算模块
"""
import numpy as np
import cv2


def calculate_clinical_metrics(mask):
    """
    基于 OpenCV 的睑板腺多参数定量分析
    
    Args:
        mask: 分割掩膜
    
    Returns:
        metrics: 指标列表 [[名称, 值], ...]
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_area = mask.shape[0] * mask.shape[1]
    gland_area = np.sum(mask > 0)
    area_ratio = (gland_area / total_area) * 100 if total_area > 0 else 0
    
    lengths, widths, centroids = [], [], []
    for cnt in contours:
        if cv2.contourArea(cnt) < 50:
            continue
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

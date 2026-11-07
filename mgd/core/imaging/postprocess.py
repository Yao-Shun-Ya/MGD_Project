"""后处理：阈值 + 开运算 + 小连通域过滤 + 可选 ROI。

修复点：不再**静默**抬升阈值。自动升级必须显式开启，并把 requested / applied 一起返回，
方便 UI 与报告如实展示。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PostprocessResult:
    mask: np.ndarray
    requested_threshold: float
    applied_threshold: float
    escalated: bool
    escalation_reason: str | None = None


def postprocess_binary_mask(
    prob_map: np.ndarray,
    threshold: float = 0.7,
    min_area: int = 120,
    open_kernel: int = 3,
    *,
    auto_escalate: bool = False,
    max_coverage: float = 0.25,
    escalation_schedule: tuple[float, ...] = (0.75, 0.8, 0.85, 0.9),
) -> PostprocessResult:
    """概率图 → 二值掩膜。"""
    requested = float(max(0.05, min(float(threshold), 0.98)))
    applied = requested
    escalated = False
    reason: str | None = None

    binary = (prob_map >= requested).astype(np.uint8)
    if auto_escalate:
        coverage = float(binary.mean())
        if coverage > max_coverage:
            for candidate in escalation_schedule:
                if candidate <= requested:
                    continue
                cand = (prob_map >= candidate).astype(np.uint8)
                if float(cand.mean()) <= max_coverage:
                    binary = cand
                    applied = float(candidate)
                    escalated = True
                    reason = f"覆盖率 {coverage:.2%} 超过 {max_coverage:.0%}，阈值由 {requested:.2f} 抬升至 {applied:.2f}"
                    break

    kernel_size = int(max(1, min(int(open_kernel), 9)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary, dtype=np.uint8)
    area_threshold = int(max(1, min_area))
    for index in range(1, num_labels):
        if stats[index, cv2.CC_STAT_AREA] >= area_threshold:
            cleaned[labels == index] = 1

    return PostprocessResult(
        mask=cleaned.astype(np.float32),
        requested_threshold=requested,
        applied_threshold=applied,
        escalated=escalated,
        escalation_reason=reason,
    )


def smooth_probability_map(prob_map: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    sigma = float(max(0.0, min(float(sigma), 3.0)))
    if sigma <= 1e-6:
        return prob_map.astype(np.float32)
    blurred = cv2.GaussianBlur(prob_map.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(blurred, 0.0, 1.0)


def estimate_eyelid_roi_mask(gray_img: np.ndarray) -> np.ndarray:
    """Otsu 眼睑 ROI 估计（保留旧行为，仅作为启发式兜底）。"""
    img_u8 = gray_img if gray_img.dtype == np.uint8 else np.clip(gray_img, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(img_u8, (5, 5), 0)
    _, roi = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    roi = roi.astype(np.uint8)

    kernel = np.ones((9, 9), np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    if num_labels > 1:
        largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        roi = (labels == largest).astype(np.uint8)

    roi = cv2.dilate(roi, np.ones((7, 7), np.uint8), iterations=1)
    if float(roi.mean()) < 0.05:
        roi = np.ones_like(roi, dtype=np.uint8)
    return roi.astype(np.float32)


def apply_roi(mask: np.ndarray, prob_map: np.ndarray, roi_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (mask * roi_mask).astype(np.float32), (prob_map * roi_mask).astype(np.float32)


def eyelid_roi_from_probability(prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """双任务模型 Channel 1 输出 → 眼睑 ROI（闭运算 + 填洞 + 轻度膨胀）。"""
    roi = (np.asarray(prob_map, dtype=np.float32) >= float(threshold)).astype(np.uint8)
    kernel = np.ones((9, 9), np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(roi)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    filled = cv2.dilate(filled, np.ones((7, 7), np.uint8), iterations=1)
    if float(filled.mean()) < 0.02:
        return np.ones_like(roi, dtype=np.float32)
    return filled.astype(np.float32)

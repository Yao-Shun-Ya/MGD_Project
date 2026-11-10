"""遮挡归因（occlusion）：与模型完全解耦的扰动重要度。

predict_fn(image) -> 概率图 由调用方注入（可以是 ModelService.probability，
也可以是测试中的假函数），因此本模块不依赖 torch，可在纯 numpy 环境运行。

重要度定义：被遮挡窗口内「原概率均值 − 遮挡后概率均值」的窗口均值；
重叠区按窗口计数加权平均。窗口数用 ceil 计算，超出 max_windows 时
stride 自动翻倍直到合规（翻倍不再减少窗口数则按饱和值继续），实际
scale = 实际 stride / 请求 stride 写进返回值并在 note 中记录。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..runtime.logging import get_logger

logger = get_logger("explain")

#: map 归一化除零保护
_EPS = 1e-12


@dataclass(frozen=True)
class OcclusionConfig:
    """滑窗遮挡配置。

    - window: 窗口边长（像素），>= 1；
    - stride: 步长（像素），>= 1；
    - fill: 填充方式，仅 "mean"（全图均值填充）或 "zero"（0 填充），其他值抛 ValueError；
    - max_windows: 窗口数上限，超出时 stride 自动翻倍直到合规（见 occlusion_map 的 note/scale）。
    """

    window: int = 32
    stride: int = 16
    fill: str = "mean"  # "mean"|"zero"，其他值 ValueError
    max_windows: int = 4096

    def __post_init__(self) -> None:
        # 浮点入参规范化：int(window) 截断后写回，保证「校验值 == 生效值」，
        # 否则 window=8.9 校验通过但实际用 8.9 参与运算（frozen dataclass 需
        # object.__setattr__ 绕过不可变约束）。
        object.__setattr__(self, "window", int(self.window))
        object.__setattr__(self, "stride", int(self.stride))
        object.__setattr__(self, "max_windows", int(self.max_windows))
        if self.window < 1:
            raise ValueError(f"window 必须 >= 1，实际 {self.window}")
        if self.stride < 1:
            raise ValueError(f"stride 必须 >= 1，实际 {self.stride}")
        if self.max_windows < 1:
            raise ValueError(f"max_windows 必须 >= 1，实际 {self.max_windows}")
        if str(self.fill) not in ("mean", "zero"):
            raise ValueError(f"fill 仅支持 'mean' | 'zero'，实际 {self.fill!r}")


def aligned_probability(predict_fn, image: np.ndarray) -> np.ndarray:
    """调用 predict_fn 并把概率图对齐到影像空间尺寸。

    predict_fn 返回 2D 概率图；若空间尺寸与输入影像不一致（如模型下采样输出），
    双线性缩放到影像尺寸后再统计（显式对齐，不是静默丢弃信息）。
    """
    prob = np.asarray(predict_fn(image), dtype=np.float32)
    if prob.ndim != 2:
        raise ValueError(f"predict_fn 必须返回 2D 概率图 (H,W)，实际形状 {prob.shape}")
    if prob.shape != image.shape:
        import cv2

        height, width = image.shape
        prob = cv2.resize(prob, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return prob


def _axis_starts(size: int, window: int, stride: int) -> list[int]:
    """单轴窗口起点：数量 = ceil((size − window) / stride) + 1（size > window 时）。

    末尾起点对齐图像边界，保证覆盖到最后一角；size <= window 时整轴一个窗口。
    """
    if size <= window:
        return [0]
    count = int(math.ceil((size - window) / float(stride))) + 1
    return sorted({min(i * stride, size - window) for i in range(count)})


def occlusion_map(predict_fn, image: np.ndarray, config: OcclusionConfig) -> dict:
    """滑窗遮挡重要度图。

    实现：
    - 重要度 = 被遮挡窗口内「原概率均值 − 遮挡后概率均值」的窗口均值
      （predict_fn 在遮挡影像上重新打分）；
    - 重叠区按窗口计数加权平均（sum/count）；无窗口覆盖的像素记 0；
    - 窗口数用 ceil 计算，超出 max_windows 时 stride 自动翻倍直到合规；
      翻倍不再减少窗口数（图像过小/窗口过大）时按饱和 stride 继续，
      两种情况都把实际情况写进 note；
    - map 归一化：importance / max(importance) 后 clip 到 [0,1]
      （负重要度 = 遮挡反而增益，记 0）；全部重要度非正时返回全零图，
      语义为「没有任何区域被遮挡后掉分」，并在 note 中如实说明。

    返回 {"map": float32 [0,1], "windows": int, "scale": float, "note": str}；
    scale = 实际 stride / 请求 stride（>1 表示发生过 stride 翻倍）。
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"occlusion_map 期望 2D 灰度图 (H,W)，实际形状 {img.shape}")
    height, width = img.shape
    window = int(config.window)
    requested_stride = int(config.stride)
    fill_value = float(img.mean()) if config.fill == "mean" else 0.0

    def total_windows(step: int) -> int:
        return len(_axis_starts(height, window, step)) * len(_axis_starts(width, window, step))

    notes: list[str] = []
    effective = requested_stride
    total = total_windows(effective)
    scale = 1.0
    if total > int(config.max_windows):
        requested_total = total
        while total > int(config.max_windows):
            candidate = effective * 2
            candidate_total = total_windows(candidate)
            if candidate_total >= total:
                break  # 翻倍不再减少窗口数 → 已饱和
            effective, total = candidate, candidate_total
        scale = effective / float(requested_stride)
        if total <= int(config.max_windows):
            notes.append(
                f"窗口数 {requested_total} 超出 max_windows={config.max_windows}："
                f"stride 自动翻倍 {requested_stride} -> {effective}（scale={scale:g}），实际窗口数 {total}"
            )
        else:
            notes.append(
                f"窗口数 {requested_total} 超出 max_windows={config.max_windows}："
                f"stride 翻倍至 {effective} 已达下限（仍 {total} 个窗口），按当前 stride 继续"
            )
            logger.warning(notes[-1])

    row_starts = _axis_starts(height, window, effective)
    col_starts = _axis_starts(width, window, effective)
    total = len(row_starts) * len(col_starts)

    base_prob = aligned_probability(predict_fn, img)

    importance_sum = np.zeros((height, width), dtype=np.float64)
    coverage = np.zeros((height, width), dtype=np.float64)
    occluded = img.copy()
    for i in row_starts:
        row_end = min(i + window, height)
        for j in col_starts:
            col_end = min(j + window, width)
            occluded[i:row_end, j:col_end] = fill_value
            prob = aligned_probability(predict_fn, occluded)
            importance_sum[i:row_end, j:col_end] += base_prob[i:row_end, j:col_end] - prob[i:row_end, j:col_end]
            coverage[i:row_end, j:col_end] += 1.0
            occluded[i:row_end, j:col_end] = img[i:row_end, j:col_end]  # 还原窗口

    covered = coverage > 0
    importance = np.zeros((height, width), dtype=np.float64)
    importance[covered] = importance_sum[covered] / coverage[covered]

    max_importance = float(importance.max()) if importance.size else 0.0
    if max_importance > _EPS:
        prob_map = np.clip(importance / max_importance, 0.0, 1.0)
    else:
        prob_map = np.zeros((height, width), dtype=np.float64)
        notes.append("所有窗口的重要度均非正（遮挡未降低概率），map 归零")

    return {
        "map": prob_map.astype(np.float32),
        "windows": int(total),
        "scale": float(scale),
        "note": "；".join(notes),
    }

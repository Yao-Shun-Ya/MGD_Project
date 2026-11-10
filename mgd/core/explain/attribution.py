"""实例级遮挡归因：把遮挡重要度落到连通实例（腺体）粒度。

每个实例整体遮挡，fill 取「该实例区域以外像素的均值」（即用周围背景
外推填充；实例占满全图、无外部像素时回退 0）。截断行为：实例数超过
max_instances 时按面积取前 N，截断说明通过 logger.warning 在返回表之外
给出，不污染 DataFrame 列。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..runtime.logging import get_logger
from .occlusion import aligned_probability

logger = get_logger("explain")

#: 归因表列（与 spec 一致；「面积」仅用于截断排序，不进返回表）
INSTANCE_COLUMNS = ("实例编号", "原始概率均值", "遮挡后概率下降均值", "归因排名")


def _instance_fill(image: np.ndarray, mask: np.ndarray) -> float:
    """实例区域外的均值（外推背景）；实例占满全图时无外部像素，回退 0。"""
    outside = image[~mask]
    return float(outside.mean()) if outside.size else 0.0


def instance_attribution(
    predict_fn,
    image: np.ndarray,
    instances: np.ndarray,
    max_instances: int = 60,
) -> pd.DataFrame:
    """实例级遮挡归因表。

    - image: 2D 灰度图 (H,W)；instances: 同形状 int 标签图，0=背景，1..N 为实例编号；
    - 原始概率均值 = predict_fn(原影像) 在实例区域的均值；
      遮挡后概率下降均值 = 原始均值 − predict_fn(遮挡影像) 在实例区域的均值；
      遮挡方式：整实例区域替换为「该实例区域以外像素的均值」（外推背景），
      实例占满全图无外部像素时回退 0；
    - 归因排名按下降均值降序 1..M（平局按实例编号升序）；
    - 实例数超过 max_instances 时按面积取前 N 并通过 logger.warning 在返回外说明；
    - 零面积编号自动跳过；无任何实例时返回带列名的空表。
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"instance_attribution 期望 2D 灰度图 (H,W)，实际形状 {img.shape}")
    labels = np.asarray(instances)
    if labels.shape != img.shape:
        raise ValueError(f"instances 形状 {labels.shape} 与 image 形状 {img.shape} 不一致")
    max_instances = int(max_instances)
    if max_instances < 1:
        raise ValueError(f"max_instances 必须 >= 1，实际 {max_instances}")

    base_prob = aligned_probability(predict_fn, img)

    records: list[dict] = []
    for label_id in sorted({int(v) for v in np.unique(labels) if int(v) != 0}):
        mask = labels == label_id
        area = int(mask.sum())
        if area == 0:
            continue
        fill = _instance_fill(img, mask)
        occluded = img.copy()
        occluded[mask] = np.float32(fill)
        occluded_prob = aligned_probability(predict_fn, occluded)
        base_mean = float(base_prob[mask].mean())
        records.append(
            {
                "实例编号": label_id,
                "面积": area,
                "原始概率均值": base_mean,
                "遮挡后概率下降均值": base_mean - float(occluded_prob[mask].mean()),
            }
        )

    if len(records) > max_instances:
        total = len(records)
        records.sort(key=lambda r: (-r["面积"], r["实例编号"]))
        records = records[:max_instances]
        logger.warning(
            "实例数 %d 超过 max_instances=%d，已按面积截断保留前 %d 个实例（编号：%s）",
            total, max_instances, max_instances,
            ",".join(str(r["实例编号"]) for r in sorted(records, key=lambda r: r["实例编号"])),
        )

    if not records:
        return pd.DataFrame(columns=list(INSTANCE_COLUMNS))

    records.sort(key=lambda r: (-r["遮挡后概率下降均值"], r["实例编号"]))
    for rank, record in enumerate(records, start=1):
        record["归因排名"] = rank
    records.sort(key=lambda r: r["实例编号"])  # 行序按实例编号升序呈现
    frame = pd.DataFrame(records)
    return frame[list(INSTANCE_COLUMNS)]

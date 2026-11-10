"""拓扑与实例切分指标（WP-1）。

核心价值：两条因细颈/单像素桥相连而黏成一个连通域的平行腺体，
``split_instances(method="cc")`` 只能给出 1 个实例，而
``split_instances(method="watershed")`` 借「骨架连通域作种子 + 距离变换作高程」
的 watershed 切分能给出 2 个实例。

骨架统一复用 ``mgd.core.imaging.clinical._skeleton``（带「腐蚀不再改变即终止」
的防死循环口径），本模块不重复实现骨架。
空输入一律返回 NaN / 全 0 标签图并在返回前记 debug 日志，绝不抛异常、绝不用 0 冒充。
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.segmentation import watershed

from ..imaging.clinical import _skeleton
from ..runtime.logging import get_logger

logger = get_logger("quantification.topology")

#: 3×3 全 1 核（8 邻域统计与 1px 膨胀共用）
_KERNEL_3X3_FLOAT = np.ones((3, 3), dtype=np.float32)
_KERNEL_3X3_UINT8 = np.ones((3, 3), dtype=np.uint8)


def _as_binary2d(mask: object, shape: tuple[int, int] | None = None) -> np.ndarray | None:
    """包内共享助手：把任意输入规整为二维 0/1 uint8 掩膜。

    None / 非二维等非法输入返回 None 并记 debug 日志（绝不抛异常）；
    ``shape`` 给定时尺寸不符则最近邻缩放到该尺寸（与 clinical._binary 口径一致）。
    注意：空掩膜（全 0）是合法返回值，空与否由调用方再判。
    """
    if mask is None:
        logger.debug("_as_binary2d：输入为 None")
        return None
    arr = np.asarray(mask)
    if arr.ndim != 2:
        logger.debug("_as_binary2d：掩膜必须二维，收到 shape=%s", arr.shape)
        return None
    out = (arr > 0).astype(np.uint8)
    if shape is not None and out.shape != tuple(shape):
        out = cv2.resize(out, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_NEAREST)
    return out


def _neighbor_counts(binary: np.ndarray) -> np.ndarray:
    """3×3 邻域（不含自身）内的前景邻居数图，与 binary 同形状。

    用 3×3 全 1 卷积求和再减去中心像素值；图像边界按 0（背景）处理。
    """
    img = (binary > 0).astype(np.float32)
    summed = cv2.filter2D(img, -1, _KERNEL_3X3_FLOAT, borderType=cv2.BORDER_CONSTANT)
    return (summed - img).astype(np.int16)


def _endpoint_branch_counts(skeleton: np.ndarray) -> tuple[int, int]:
    """对给定骨架统计（端点数, 分叉点数）；骨架为空返回 (0, 0)。"""
    skel = (skeleton > 0).astype(np.uint8)
    if int(skel.sum()) == 0:
        return 0, 0
    counts = _neighbor_counts(skel)
    endpoints = int(((skel > 0) & (counts == 1)).sum())
    branches = int(((skel > 0) & (counts >= 3)).sum())
    return endpoints, branches


def endpoints_and_branches(skeleton: np.ndarray | None) -> tuple[float, float]:
    """骨架端点数与分叉点数。

    定义：对骨架每个前景点统计 3×3 邻域（不含自身）内的骨架邻居数 n；
    n == 1 记端点（endpoint），n >= 3 记分叉（branch）。

    返回 (端点数, 分叉点数)。空骨架 / 非法输入返回 ``(nan, nan)``
    （先记 debug 日志再返回），不抛异常。
    """
    skel = _as_binary2d(skeleton)
    if skel is None or int(skel.sum()) == 0:
        logger.debug("endpoints_and_branches：骨架为空或非法，返回 (NaN, NaN)")
        return float("nan"), float("nan")
    endpoints, branches = _endpoint_branch_counts(skel)
    return endpoints, branches


def break_count(mask: np.ndarray | None) -> float:
    """腺体断裂数：Σ_i max(0, E_i − 2)。

    E_i 为第 i 个连通实例（8 连通）骨架的端点数；正常单条腺体骨架恰好
    2 个端点，每多一对端点记一次断裂（如 H 形连通骨架 4 端点 → 2 次断裂）。

    实现（性能优化，结果与旧「整图逐组件扫描」口径一致）：
    ``cv2.connectedComponentsWithStats`` 取每个连通域 bbox，在
    「bbox 外扩 1px 并截断到图像边界」的窗口内 ``labels[窗口] == label``
    逐组件计算。外扩 1px 是等价性关键：clinical._skeleton 的腐蚀/膨胀把
    数组界外当前景处理，紧 bbox 会让组件边缘骨架算错（腐蚀不再从外向内
    侵蚀）；外扩 1px 后窗口内组件像素的邻域与整图完全一致（bbox 本就是
    组件的紧边界，窗外 1px 必为背景）。面积 < 8px 的组件按噪声 / 退化
    片段短路跳过（贡献 0，不对海量噪声点逐个做骨架化）。

    空掩膜 / 非法输入返回 NaN（先记 debug 日志），不抛异常。
    """
    m = _as_binary2d(mask)
    if m is None or int(m.sum()) == 0:
        logger.debug("break_count：掩膜为空或非法，返回 NaN")
        return float("nan")
    height, width = m.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    total = 0
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) < 8:
            continue  # 噪声 / 退化组件：不贡献断裂，短路免骨架化
        y0 = int(stats[label, cv2.CC_STAT_TOP])
        x0 = int(stats[label, cv2.CC_STAT_LEFT])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        ys, ye = max(0, y0 - 1), min(height, y0 + h + 1)
        xs, xe = max(0, x0 - 1), min(width, x0 + w + 1)
        component = (labels[ys:ye, xs:xe] == label).astype(np.uint8)
        skel = _skeleton(component)
        if int(skel.sum()) == 0:
            continue  # 骨架为空（如整幅实例）：无端点可计
        endpoints, _ = _endpoint_branch_counts(skel)
        total += max(0, endpoints - 2)
    return int(total)


def split_instances(
    mask: np.ndarray | None,
    min_area: int = 120,
    method: str = "cc",
) -> np.ndarray:
    """实例切分，返回 int32 标签图（0 = 背景）。

    method 两种口径（默认 "cc"，向后兼容）：

    - ``"cc"``：等价于 8 连通 connectedComponents + min_area 过滤，
      小连通域直接置 0（并入背景），标签 id 保留 cv2 原编号。
    - ``"watershed"``：骨架连通域作种子（骨架复用 clinical._skeleton），
      ``skimage.segmentation.watershed(elevation=距离变换, markers=种子标签,
      mask=掩膜)`` 切分。**硬保证：在无小实例被并入背景的前提下，
      输出 (labels>0) 与掩膜二值部分完全一致**——skimage watershed 带 mask
      参数时所有 mask 内可达像素都会被标记（因此不用 cv2.watershed，
      它会产生 -1 边界并破坏该保证）。min_area 过滤后的小实例
      **并入面积最大的相邻实例**（8 邻域接触）；无相邻实例时并入背景，
      此时 (labels>0) 为掩膜的严格子集。

    空输入行为（先记 debug 日志，不抛异常）：

    - 输入为 None 或无法规整为二维掩膜（非法输入）→ 无形状可循，
      返回 ``(0, 0)`` 空 int32 标签图；
    - 输入可规整为二维但为空掩膜（全 0）→ 与输入同形状的全 0 int32 标签图。

    未知 method 抛 ValueError（调用方编码错误，非数据问题）。
    """
    m = _as_binary2d(mask)
    if m is None:
        logger.debug("split_instances：掩膜非法，返回 (0,0) 空标签图")
        return np.zeros((0, 0), dtype=np.int32)
    if int(m.sum()) == 0:
        logger.debug("split_instances：掩膜为空，返回全 0 标签图")
        return np.zeros(m.shape, dtype=np.int32)

    method_norm = str(method).strip().lower()
    if method_norm == "cc":
        return _split_cc(m, min_area)
    if method_norm == "watershed":
        return _split_watershed(m, min_area)
    raise ValueError(f"未知 method: {method!r}（可选 'cc' / 'watershed'）")


def _split_cc(mask: np.ndarray, min_area: int) -> np.ndarray:
    """connectedComponents + min_area 过滤（小实例并入背景）。"""
    threshold = max(1, int(min_area))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    labels = labels.astype(np.int32)
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) < threshold:
            labels[labels == label] = 0
    return labels


def _split_watershed(mask: np.ndarray, min_area: int) -> np.ndarray:
    """骨架连通域作种子 + 距离变换作高程的 watershed 切分。"""
    skeleton = _skeleton(mask)
    if int(skeleton.sum()) == 0:
        # 骨架为空（如掩膜铺满全图时腐蚀不变）：退回 cc 口径并记录
        logger.debug("split_instances(watershed)：骨架为空，退回 cc 切分")
        return _split_cc(mask, min_area)

    num_seeds, seed_labels = cv2.connectedComponents(skeleton, connectivity=8)
    elevation = cv2.distanceTransform(mask, cv2.DIST_L2, 3).astype(np.float32)
    labels = watershed(
        elevation,
        markers=seed_labels.astype(np.int32),
        mask=mask.astype(bool),
    ).astype(np.int32)
    return _merge_small_instances(labels, min_area)


def _merge_small_instances(labels: np.ndarray, min_area: int) -> np.ndarray:
    """把面积 < min_area 的实例并入面积最大的 8 邻域相邻实例；无邻居则并入背景。"""
    threshold = max(1, int(min_area))
    for _ in range(8):  # 防御性上限；单轮后仅剩跨轮连锁合并的极端情形
        areas = np.bincount(labels.ravel())
        small = [
            lab for lab in range(1, areas.size)
            if 0 < areas[lab] < threshold
        ]
        if not small:
            break
        merged_any = False
        for lab in small:
            if areas[lab] == 0 or areas[lab] >= threshold:
                continue  # 本轮已被其他合并改动
            component = (labels == lab).astype(np.uint8)
            dilated = cv2.dilate(component, _KERNEL_3X3_UINT8)
            touching = labels[(dilated > 0) & (labels != lab)]
            instances = touching[touching > 0]
            if instances.size > 0:
                uniq = np.unique(instances)
                target = int(uniq[int(np.argmax([areas[int(v)] for v in uniq]))])
            else:
                target = 0  # 无相邻实例 → 并入背景
            labels[labels == lab] = target
            areas[target] += areas[lab]
            areas[lab] = 0
            merged_any = True
        if not merged_any:
            break
    return labels

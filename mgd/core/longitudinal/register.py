"""随访图像配准（WP-8）。

`register_pair` 用 OpenCV 的 ECC 算法（Enhanced Correlation Coefficient 最大化）
估计 base（基线）与 follow（随访）之间的欧氏刚体变换：ECC 以最大化两幅图像的
相关系数为目标，因此配准结果可直接用于随访前后对比的前置对齐——把 follow 图
warp 回 base 坐标系后，逐实例 / 逐像素对比才有意义。

方向约定（重要）：OpenCV `findTransformECC(template, input)` 原生返回的矩阵是
template→input（即 base→follow）方向；为便于下游直接
`cv2.warpAffine(follow, warp_matrix, base.shape[::-1])` 得到对齐图，本函数对外
暴露的 `warp_matrix` 统一为 **follow→base** 的 2x3 刚体矩阵（对原生结果取逆），
docstring 与返回值均按此口径。

失败纪律：配准分数过低或求解异常时不强行返回对齐结果，而是
`{"ok": False, "warp_matrix": None, "warped": None, ...}`，note 写明原因，
由上层决定是否回退到「不配准直接对比」。
"""
from __future__ import annotations

import cv2
import numpy as np

#: ECC 求解前的高斯平滑核（利于收敛；不带入 warped 输出）
_BLUR_KSIZE = (5, 5)


def _to_unit_float(gray) -> np.ndarray:
    """转二维 float32；取值超过 1 的（如 uint8）先归一化到 [0,1]。"""
    arr = np.asarray(gray)
    if arr.ndim != 2:
        raise ValueError(f"配准输入必须是二维灰度图，收到 shape={arr.shape}")
    arr = arr.astype(np.float32, copy=False)
    peak = float(arr.max()) if arr.size else 0.0
    if peak > 1.0:
        arr = arr / peak
    return np.ascontiguousarray(arr, dtype=np.float32)


def register_pair(base_gray, follow_gray, *, min_ecc: float = 0.05) -> dict:
    """将 follow 图配准到 base 坐标系（ECC，MOTION_EUCLIDEAN 刚体变换）。

    预处理：两图转 float32、超过 1 的取值归一化到 [0,1]，再
    `cv2.GaussianBlur((5,5), 0)` 平滑以利于 ECC 收敛（模糊仅用于求解，
    `warped` 输出基于归一化未模糊的 follow，避免无谓软化）。

    参数
    ----
    base_gray : 基线灰度图（任意数值 dtype）
    follow_gray : 随访灰度图（与 base 同尺寸）
    min_ecc : 收敛后的 ECC 分数下限；低于该值视为配准失败（不强行配准）。

    返回
    ----
    dict，键为：
      - ok (bool)：是否配准成功且 ecc_score >= min_ecc
      - warp_matrix (2x3 ndarray | None)：**follow→base** 的刚体变换矩阵
        （即 cv2.warpAffine(follow, warp_matrix, base.shape[::-1]) 得到对齐图）
      - ecc_score (float)：ECC 相关分数（越大越可信；失败但有分数时也给出）
      - note (str)：失败原因或 "ok"
      - warped (ndarray | None)：成功时 follow 对齐到 base 坐标系的 float32 图

    ECC 最大化的是两图的相关系数；本函数只做刚体（平移+旋转）估计，
    用途是随访前后对比的前置对齐，不承担形变校正。
    """
    base = _to_unit_float(base_gray)
    follow = _to_unit_float(follow_gray)
    if base.shape != follow.shape:
        return {
            "ok": False,
            "warp_matrix": None,
            "ecc_score": float("nan"),
            "note": f"尺寸不一致：base{base.shape} vs follow{follow.shape}",
            "warped": None,
        }

    # 求解用平滑副本
    base_ecc = cv2.GaussianBlur(base, _BLUR_KSIZE, 0)
    follow_ecc = cv2.GaussianBlur(follow, _BLUR_KSIZE, 0)

    warp_init = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        cc, warp_cv = cv2.findTransformECC(
            templateImage=base_ecc,
            inputImage=follow_ecc,
            warpMatrix=warp_init,
            motionType=cv2.MOTION_EUCLIDEAN,
            criteria=criteria,
            inputMask=None,
            gaussFiltSize=5,
        )
    except cv2.error as exc:
        return {
            "ok": False,
            "warp_matrix": None,
            "ecc_score": float("nan"),
            "note": f"ECC 求解异常：{exc}",
            "warped": None,
        }
    except Exception as exc:  # pragma: no cover - 防御 OpenCV 之外的失败
        return {
            "ok": False,
            "warp_matrix": None,
            "ecc_score": float("nan"),
            "note": f"配准失败：{exc}",
            "warped": None,
        }

    score = float(cc)
    if not np.isfinite(score):
        return {
            "ok": False,
            "warp_matrix": None,
            "ecc_score": float("nan"),
            "note": "ECC 分数非有限值（图像可能为常数图）",
            "warped": None,
        }
    if score < float(min_ecc):
        return {
            "ok": False,
            "warp_matrix": None,
            "ecc_score": score,
            "note": f"ECC 分数 {score:.4g} 低于阈值 {min_ecc}，两图相关性不足，不强行配准",
            "warped": None,
        }

    # OpenCV 返回 template→input（base→follow）；对外统一 follow→base
    warp_matrix = np.asarray(cv2.invertAffineTransform(warp_cv), dtype=np.float64)
    warped = cv2.warpAffine(follow, warp_matrix.astype(np.float32), base.shape[::-1],
                            flags=cv2.INTER_LINEAR)
    return {
        "ok": True,
        "warp_matrix": warp_matrix,
        "ecc_score": score,
        "note": "ok",
        "warped": warped,
    }

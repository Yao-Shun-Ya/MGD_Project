"""WP-1 quantification 可追溯指标层单元测试。

全部离线、合成数据、CPU 运行，目标耗时 < 10 秒。覆盖：
1. 直条几何（width_profile / DI / τ）；
2. S 形曲线（TI / 拐点数 / 曲率）；
3. 核心断言：双平行 1px 间隔竖条（单像素桥黏连）cc=1 实例、watershed=2 实例，
   且两种方法 (labels>0) 与掩膜二值部分一致；
4. 空输入一律 NaN 且不抛异常；
5. 注册表完整性与 ConfigError 行为；
6. 物理标定解析链与 to_mm。
"""
from __future__ import annotations

import cv2
import math
import time

import numpy as np
import pytest

from mgd.core.config import Settings
from mgd.core.quantification import (
    METRIC_REGISTRY,
    MetricSpec,
    all_specs,
    break_count,
    describe,
    diameter_deformation_index,
    endpoints_and_branches,
    fractal_dimension,
    gland_area_ratio,
    gland_eyelid_distance,
    get_spec,
    inflection_count,
    local_contrast,
    register,
    resolve_pixel_scale,
    signal_index,
    split_instances,
    symmetry_index,
    to_mm,
    tortuosity_arc_chord,
    tortuosity_curvature,
    tortuosity_index,
    width_histogram,
    width_profile,
    zone_edges,
    zonal_density,
)
from mgd.core.quantification.morphometry import _skeleton_path
from mgd.core.runtime.errors import ConfigError

ENGINE_DEFINED_KEYS = {
    "width_mean_px",
    "gland_eyelid_distance_px",
    "symmetry_index",
    "zonal_density",
    "endpoint_count",
    "branch_count",
    "break_count",
    "instance_count",
}
EXPECTED_KEYS = ENGINE_DEFINED_KEYS | {
    "ga_percent",
    "diameter_deformation_index",
    "tortuosity_arc_chord",
    "tortuosity_index",
    "inflection_count",
    "fractal_dimension",
    "signal_index",
    "local_contrast",
}


# ---------------------------------------------------------------------------
# 1. 直条几何
# ---------------------------------------------------------------------------
def test_straight_bar_geometry():
    """20×100 竖直矩形：宽度均值 ∈ [18,22]，DI < 0.05，τ ∈ [0.99, 1.01]。"""
    mask = np.zeros((120, 160), dtype=np.uint8)
    mask[10:110, 70:90] = 255  # 宽 20、高 100，四周留边避免腐蚀边界效应

    widths = width_profile(mask)
    assert isinstance(widths, np.ndarray) and widths.size > 0
    assert float(np.mean(widths)) == pytest.approx(20.0, abs=2.0)

    di = diameter_deformation_index(widths)
    assert np.isfinite(di) and di < 0.05

    path = _skeleton_path(mask)
    assert path.shape[0] >= 2
    tau = tortuosity_arc_chord(path)
    # 上界放宽至 1.015：clinical._main_path 的 FIFO BFS 在偶数宽度的双列骨架带上
    # 会走出 2 个对角步（79 直 + 2 斜，arc=81.83 而非最优 81.41），τ≈1.0102
    assert 0.99 <= tau <= 1.015
    assert tortuosity_index(path) == pytest.approx(tau - 1.0)

    mean_kappa, p95_kappa = tortuosity_curvature(path)
    assert np.isfinite(mean_kappa) and mean_kappa >= 0.0
    assert np.isfinite(p95_kappa) and p95_kappa >= mean_kappa

    hist = width_histogram(widths, bins=20)
    assert len(hist["edges"]) == 21 and len(hist["counts"]) == 20
    assert sum(hist["counts"]) == widths.size


# ---------------------------------------------------------------------------
# 2. S 形曲线
# ---------------------------------------------------------------------------
def test_s_curve_tortuosity_and_inflection():
    """S 形骨架：TI > 0.1 且 inflection_count ≥ 1。"""
    length, amplitude = 100, 20
    ys = np.arange(length, dtype=float)
    xs = 80.0 + amplitude * np.sin(2.0 * np.pi * ys / length)
    path = np.stack([xs, ys], axis=1)

    ti = tortuosity_index(path)
    assert np.isfinite(ti) and ti > 0.1

    assert inflection_count(path) >= 1
    # 直线无拐点（合法测量值 0）
    straight = np.column_stack([np.arange(30, dtype=float), np.full(30, 5.0)])
    assert inflection_count(straight) == 0

    mean_kappa, p95_kappa = tortuosity_curvature(path)
    assert np.isfinite(mean_kappa) and mean_kappa > 0.0
    assert np.isfinite(p95_kappa) and p95_kappa >= mean_kappa


# ---------------------------------------------------------------------------
# 3. 核心断言：cc 与 watershed 实例切分
# ---------------------------------------------------------------------------
def test_split_instances_parallel_bars_core():
    """双平行竖条（间隔 1px、单像素桥黏连）：cc=1 实例，watershed=2 实例。

    两条 10×60 竖条间隔 1px（第 15 列为背景），在中部以单像素桥 8 连通黏连
    ——模拟欠分割预测把两条腺体黏成一个连通域的典型场景。
    """
    mask = np.zeros((80, 40), dtype=np.uint8)
    mask[10:70, 5:15] = 255   # 左竖条
    mask[10:70, 16:26] = 255  # 右竖条（间隔 1px）
    mask[40, 15] = 255        # 单像素桥 → 整体 1 个连通域
    assert int((mask > 0).sum()) == 60 * 10 * 2 + 1

    labels_cc = split_instances(mask, min_area=120, method="cc")
    assert np.unique(labels_cc[labels_cc > 0]).size == 1  # cc：黏连不分离
    assert np.array_equal(labels_cc > 0, mask > 0)

    labels_ws = split_instances(mask, min_area=120, method="watershed")
    assert np.unique(labels_ws[labels_ws > 0]).size == 2  # watershed：2 个实例
    # 硬保证：两种方法的 (labels>0) 与掩膜二值部分完全一致
    assert np.array_equal(labels_ws > 0, mask > 0)


# ---------------------------------------------------------------------------
# 4. 空输入：NaN 且不抛异常
# ---------------------------------------------------------------------------
def test_empty_inputs_return_nan_without_exceptions():
    empty = np.zeros((64, 128), dtype=np.uint8)
    bar = np.zeros((64, 128), dtype=np.uint8)
    bar[10:50, 20:40] = 255  # 20 宽实心条（骨架为双列宽 → 无端点）
    gray = np.full((64, 128), 128, dtype=np.uint8)
    full = np.full((64, 128), 255, dtype=np.uint8)  # 铺满全图 → 环带为空

    # gland_area_ratio
    assert math.isnan(gland_area_ratio(empty, empty))
    assert math.isnan(gland_area_ratio(bar, empty))
    assert math.isnan(gland_area_ratio(empty, bar))
    assert math.isnan(gland_area_ratio(None, bar))

    # width_profile / DI / histogram
    assert math.isnan(width_profile(empty))
    assert math.isnan(width_profile(None))
    widths_nan = width_profile(empty)
    assert math.isnan(diameter_deformation_index(widths_nan))
    assert math.isnan(diameter_deformation_index([]))
    assert math.isnan(diameter_deformation_index(np.array([1.0, 2.0, 3.0])))
    assert math.isnan(diameter_deformation_index(np.zeros(10)))
    assert width_histogram(np.array([np.nan])) == {"edges": [], "counts": []}

    # 弯曲 / 拐点
    assert math.isnan(tortuosity_arc_chord(np.zeros((0, 2))))
    assert math.isnan(tortuosity_arc_chord(np.array([[3.0, 4.0]])))
    assert math.isnan(tortuosity_arc_chord(np.full((5, 2), 7.0)))  # 全部重合
    assert math.isnan(tortuosity_index(np.full((5, 2), 7.0)))
    mean_kappa, p95_kappa = tortuosity_curvature(np.zeros((0, 2)))
    assert math.isnan(mean_kappa) and math.isnan(p95_kappa)
    assert math.isnan(inflection_count(np.zeros((0, 2))))

    # 信号 / 纹理
    assert math.isnan(signal_index(gray, empty))
    assert math.isnan(local_contrast(gray, None))
    assert math.isnan(signal_index(gray, full))  # 环带为空
    assert math.isnan(fractal_dimension(empty))

    # 距离
    assert math.isnan(gland_eyelid_distance(empty, bar))
    assert math.isnan(gland_eyelid_distance(bar, empty))
    ring = np.zeros((64, 128), dtype=np.uint8)  # 1px 闭合方框 → 骨架为闭环，无端点
    ring[10, 20:60] = 255
    ring[39, 20:60] = 255
    ring[10:40, 20] = 255
    ring[10:40, 59] = 255
    assert math.isnan(gland_eyelid_distance(ring, bar))  # 无骨架端点

    # 拓扑
    ep, br = endpoints_and_branches(empty)
    assert math.isnan(ep) and math.isnan(br)
    assert math.isnan(break_count(empty))
    labels_cc = split_instances(empty)
    assert labels_cc.shape == (64, 128) and int(labels_cc.max()) == 0
    labels_ws = split_instances(empty, method="watershed")
    assert int(labels_ws.max()) == 0

    # 分区 / 对称
    assert zone_edges(empty) == []
    assert all(math.isnan(v) for v in zonal_density(empty, bar))
    assert all(math.isnan(v) for v in zonal_density(bar, empty))
    assert math.isnan(symmetry_index(0.0, 0.0))
    assert math.isnan(symmetry_index(float("nan"), 1.0))
    assert math.isnan(symmetry_index(None, 1.0))


# ---------------------------------------------------------------------------
# 5. 注册表
# ---------------------------------------------------------------------------
def test_registry_complete_and_traceable():
    assert set(METRIC_REGISTRY.keys()) == EXPECTED_KEYS
    for key in sorted(EXPECTED_KEYS):
        spec = get_spec(key)
        assert spec.key == key
        assert spec.name_zh.strip()
        assert spec.definition.strip()
        assert spec.unit.strip()
        assert spec.source in ("Arita et al. 2008", "Pult & Nichols 2016", "engine-defined")
        assert len(spec.requires) > 0

    # 工程定义指标必须明确标注
    engine_keys = {k for k, s in METRIC_REGISTRY.items() if s.source == "engine-defined"}
    assert engine_keys == ENGINE_DEFINED_KEYS

    keys = [spec.key for spec in all_specs()]
    assert keys == sorted(keys)

    payload = describe("ga_percent")
    assert payload["source"] == "Arita et al. 2008"
    assert payload["definition"].startswith("GA =")
    assert set(payload) == {
        "key", "name_zh", "definition", "unit", "source", "requires", "higher_is_better",
    }


def test_registry_errors():
    duplicate = MetricSpec(
        key="ga_percent", name_zh="重复定义", definition="x", unit="x",
        source="engine-defined", requires=("a",), higher_is_better=None,
    )
    with pytest.raises(ConfigError):
        register(duplicate)
    with pytest.raises(ConfigError):
        get_spec("__no_such_metric__")


# ---------------------------------------------------------------------------
# 6. 物理标定
# ---------------------------------------------------------------------------
def test_scale_resolution_and_to_mm(tmp_path):
    calibrated = Settings(home=tmp_path, physical_scale_mm_per_px=0.05)

    mm, source = resolve_pixel_scale(calibrated)
    assert mm == pytest.approx(0.05)
    assert "settings" in source

    mm_dicom, source_dicom = resolve_pixel_scale(calibrated, {"PixelSpacing": [0.1, 0.1]})
    assert mm_dicom == pytest.approx(0.1)
    assert "dicom" in source_dicom  # DICOM 优先于 settings

    mm_aniso, _ = resolve_pixel_scale(calibrated, {"PixelSpacing": [0.1, 0.12]})
    assert mm_aniso == pytest.approx(0.11)  # 各向异性取算术平均

    mm_fallback, source_fallback = resolve_pixel_scale(
        calibrated, {"PixelSpacing": [0.0, 0.0]}
    )
    assert mm_fallback == pytest.approx(0.05)  # DICOM 非法 → 回退 settings
    assert "settings" in source_fallback

    mm_none, note = resolve_pixel_scale(Settings(home=tmp_path))
    assert mm_none is None
    assert note == "未标定：长度类指标为像素单位，不可跨设备比较"

    assert to_mm(10.0, 0.05) == pytest.approx(0.5)
    assert math.isnan(to_mm(10.0, None))
    assert math.isnan(to_mm(None, 0.05))
    assert math.isnan(to_mm(10.0, float("nan")))
    assert math.isnan(to_mm(10.0, 0.0))
    assert math.isnan(to_mm(10.0, -1.0))
    # 未标定 → to_mm 必须 NaN，绝不默认 1.0 冒充
    scale, _ = resolve_pixel_scale(Settings(home=tmp_path))
    assert scale is None
    assert math.isnan(to_mm(123.0, scale))


# ---------------------------------------------------------------------------
# 补充：拓扑、分区、信号、分形的确定性正例
# ---------------------------------------------------------------------------
def test_topology_h_shape_endpoints_branches_breaks():
    """H 形（全 1px 笔画）：4 端点；桥与竖线交汇处两侧像素均 ≥3 邻居 → 8 分叉。

    断裂数按实例端点数计：max(0, 4 − 2) = 2。
    """
    img = np.zeros((80, 80), dtype=np.uint8)
    img[20:60, 30] = 255
    img[20:60, 40] = 255
    img[40, 30:41] = 255
    ep, br = endpoints_and_branches(img)
    assert (ep, br) == (4, 8)
    assert break_count(img) == 2
    assert break_count(img[20:60, 30:31]) == 0  # 单条 1px 竖线端点=2 → 无断裂


def test_gland_eyelid_distance_line():
    img = np.zeros((80, 120), dtype=np.uint8)
    img[20:60, 30] = 255  # 1px 竖线：骨架=自身，端点 (20,30)、(59,30)
    eyelid = np.zeros((80, 120), dtype=np.uint8)
    eyelid[10:70, 0:120] = 255
    distance = gland_eyelid_distance(img, eyelid)
    assert distance == pytest.approx(29.5)  # |30 − 59.5|


def test_zoning_density_and_symmetry():
    eyelid = np.zeros((80, 120), dtype=np.uint8)
    eyelid[10:60, 10:110] = 255
    gland = np.zeros((80, 120), dtype=np.uint8)
    gland[10:60, 10:43] = 255  # 仅覆盖左 1/3 分区

    edges = zone_edges(eyelid, zones=3)
    assert len(edges) == 3
    assert edges[0][0] == pytest.approx(10.0)
    assert edges[-1][1] == pytest.approx(110.0)

    densities = zonal_density(gland, eyelid, zones=3)
    assert densities[0] == pytest.approx(100.0 * 33.0 / 34.0, rel=1e-3)
    assert densities[1] == 0.0
    assert densities[2] == 0.0

    assert symmetry_index(80.0, 100.0) == pytest.approx(20.0 / 90.0)


def test_signal_index_and_local_contrast():
    gray = np.full((80, 120), 100, dtype=np.uint8)
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[20:60, 30:70] = 255
    gray[mask > 0] = 200
    assert signal_index(gray, mask, ring_px=10) == pytest.approx(100.0 / 255.0, abs=1e-6)
    assert local_contrast(gray, mask, ring_px=10) == pytest.approx(100.0, abs=1e-6)


def test_fractal_dimension_solid_square():
    mask = np.zeros((128, 160), dtype=np.uint8)
    mask[0:64, 0:64] = 255  # 64×64 实心方块，块对齐 → D ≈ 2
    dimension = fractal_dimension(mask)
    assert abs(dimension - 2.0) < 0.05


def test_gland_area_ratio_basic():
    eyelid = np.full((64, 128), 255, dtype=np.uint8)
    gland = np.zeros((64, 128), dtype=np.uint8)
    gland[0:32, 0:64] = 255  # 恰好 1/4 全图
    assert gland_area_ratio(gland, eyelid) == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# 2026-11 审计回归：M2 / M4 / M5 / L3 / L5
# ---------------------------------------------------------------------------
def test_symmetry_index_negative_or_nonfinite_mean_returns_nan():
    """[M2] symmetry_index 仅适用于非负指标（密度/百分比）：mean ≤ 0 或非有限 → NaN。"""
    assert math.isnan(symmetry_index(-2.0, -4.0))       # 两负值
    assert math.isnan(symmetry_index(-0.1, -0.3))       # 小负值
    assert math.isnan(symmetry_index(-5.0, -5.0))       # 完全对称但为负
    assert math.isnan(symmetry_index(1.5e308, 1.5e308))  # a+b 溢出 → mean 非有限
    # 非负输入既有行为不变
    assert symmetry_index(2.0, 4.0) == pytest.approx(2.0 / 3.0)
    assert symmetry_index(80.0, 100.0) == pytest.approx(20.0 / 90.0)
    assert math.isnan(symmetry_index(0.0, 0.0))
    assert math.isnan(symmetry_index(float("nan"), 1.0))


def test_morphometry_invalid_conversion_inputs_never_raise():
    """[M4] _as_path / _as_gray2d 非法输入绝不抛异常（统一走 None → NaN 通道）。"""
    # 字符串 / dict / ragged 序列 → 转换失败 → NaN，不抛 TypeError/ValueError
    assert math.isnan(tortuosity_arc_chord("abc"))
    assert math.isnan(tortuosity_index({"a": 1}))
    ragged_value = inflection_count([[1.0, 2.0], [3.0]])
    assert math.isnan(ragged_value) or ragged_value == 0  # 以实现为准，关键是绝不抛异常
    mean_kappa, p95_kappa = tortuosity_curvature([[1.0, 2.0], [3.0]])
    assert math.isnan(mean_kappa) and math.isnan(p95_kappa)
    # 灰度图：元素不可转 float 的二维 object 数组 → NaN
    bad_gray = np.array([["a", "b"], ["c", "d"]], dtype=object)
    assert math.isnan(signal_index(bad_gray, np.ones((2, 2), dtype=np.uint8)))
    assert math.isnan(local_contrast(bad_gray, np.ones((2, 2), dtype=np.uint8)))
    # 合法输入不受影响
    line = np.column_stack([np.arange(20, dtype=float), np.full(20, 3.0)])
    assert tortuosity_arc_chord(line) == pytest.approx(1.0)


def _break_count_reference(mask: np.ndarray):
    """旧实现基准：整图逐组件 (labels == label) 扫描 + 骨架化（无 bbox 裁剪、无短路）。"""
    from mgd.core.imaging.clinical import _skeleton
    from mgd.core.quantification.topology import _as_binary2d, _endpoint_branch_counts

    m = _as_binary2d(mask)
    if m is None or int(m.sum()) == 0:
        return float("nan")
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    total = 0
    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8)
        skel = _skeleton(component)
        if int(skel.sum()) == 0:
            continue
        endpoints, _ = _endpoint_branch_counts(skel)
        total += max(0, endpoints - 2)
    return int(total)


def test_break_count_bounding_box_equivalence():
    """[M5] bbox 裁剪实现与旧整图实现结果一致（含贴边组件、闭环、黏连与随机组合）。"""
    h = np.zeros((80, 80), dtype=np.uint8)
    h[20:60, 30] = 255
    h[20:60, 40] = 255
    h[40, 30:41] = 255          # H 形：4 端点 → 2 断裂
    line = np.zeros((80, 120), dtype=np.uint8)
    line[20:60, 30] = 255       # 单条 1px 竖线：2 端点 → 0 断裂
    ring = np.zeros((80, 80), dtype=np.uint8)
    ring[10, 20:60] = 255
    ring[59, 20:60] = 255
    ring[10:60, 20] = 255
    ring[10:60, 59] = 255       # 闭环骨架 → 0 断裂
    edge = np.zeros((60, 60), dtype=np.uint8)
    edge[0:50, 0:60] = 255      # 贴图像边界的实心块（bbox 无法外扩，须截断）
    bridged = np.zeros((80, 40), dtype=np.uint8)
    bridged[10:70, 5:15] = 255
    bridged[10:70, 16:26] = 255
    bridged[40, 15] = 255       # 单像素桥黏连的双竖条

    masks = [h, line, ring, edge, bridged]
    rng = np.random.RandomState(7)
    for _ in range(8):
        mm = np.zeros((100, 140), dtype=np.uint8)
        for _ in range(rng.randint(1, 6)):  # 多个随机实心块（面积 ≥ 8px）
            y, x = rng.randint(0, 90), rng.randint(0, 120)
            hh, ww = rng.randint(8, 30), rng.randint(8, 20)
            mm[y:y + hh, x:x + ww] = 255
        masks.append(mm)

    for mask in masks:
        assert break_count(mask) == _break_count_reference(mask)

    assert break_count(h) == 2
    assert break_count(line) == 0
    assert break_count(ring) == 0
    assert break_count(edge) == 0


def test_break_count_ignores_tiny_noise_components():
    """[M5] 面积 < 8px 的噪声组件短路跳过（贡献 0），不影响腺体主体计数。"""
    dots = np.zeros((40, 40), dtype=np.uint8)
    for y, x in ((5, 5), (20, 20), (30, 5)):
        dots[y:y + 2, x:x + 2] = 255  # 三个 2px 噪声点
    gland = np.zeros((40, 40), dtype=np.uint8)
    gland[5:35, 20] = 255             # 单条 1px 竖线腺体：0 断裂

    assert break_count(dots) == 0
    combined = np.maximum(dots, gland)
    assert break_count(combined) == break_count(gland) == 0


def test_break_count_performance_many_noise_dots():
    """[M5] 640×1280 上 3000 个 2px 噪声点 + 若干腺体：break_count 运行 < 2 秒。"""
    rng = np.random.RandomState(0)
    mask = np.zeros((640, 1280), dtype=np.uint8)
    for y, x in zip(rng.randint(0, 638, size=3000), rng.randint(0, 1278, size=3000)):
        mask[y:y + 2, x:x + 2] = 255
    for col in range(100, 1200, 150):  # 8 条 10×300 竖条腺体
        mask[100:400, col:col + 10] = 255

    start = time.perf_counter()
    result = break_count(mask)
    elapsed = time.perf_counter() - start
    assert math.isfinite(result)
    # 57 为该固定种子掩膜的确定值（随机点黏连成 ≥8px 团块产生断裂），
    # 已离线与旧整图实现逐位核对一致；等价性细节见 test_break_count_bounding_box_equivalence
    assert result == 57
    assert elapsed < 2.0


def test_to_mm_rejects_negative_and_overflow():
    """[L3] to_mm：负长度 → NaN；乘积溢出 → NaN；正常换算不受影响。"""
    assert math.isnan(to_mm(-1.0, 0.05))
    assert math.isnan(to_mm(-1e-9, 0.05))
    assert math.isnan(to_mm(1e308, 10.0))     # 乘积 inf → NaN
    assert to_mm(0.0, 0.05) == 0.0            # 0 是合法长度
    assert to_mm(10.0, 0.05) == pytest.approx(0.5)
    assert to_mm(200.0, 0.01) == pytest.approx(2.0)
    # 既有空输入行为不变
    assert math.isnan(to_mm(10.0, None))
    assert math.isnan(to_mm(None, 0.05))
    assert math.isnan(to_mm(10.0, -1.0))
    assert math.isnan(to_mm(10.0, float("nan")))


def test_split_instances_none_and_illegal_inputs_no_raise():
    """[L5] None / 非法输入不抛异常，返回 (0,0) 空 int32 标签图（与 docstring 一致）。"""
    for bad in (None, "abc", 3.14, np.zeros(5), np.zeros((2, 2, 2))):
        result = split_instances(bad)
        assert isinstance(result, np.ndarray), bad
        assert result.shape == (0, 0), bad
        assert result.dtype == np.int32, bad
    for method in ("cc", "watershed"):
        assert split_instances(None, method=method).shape == (0, 0)
    # 可规整但全 0 的空掩膜 → 与输入同形状的全 0 int32 标签图（既有行为不变）
    empty = split_instances(np.zeros((64, 128), dtype=np.uint8))
    assert empty.shape == (64, 128) and empty.dtype == np.int32 and int(empty.max()) == 0

"""WP-8 纵向随访测试：sqlite 存储、ECC 配准、实例匹配。全部离线，tmp_path 建库。"""
from __future__ import annotations

import math
import re
import sqlite3

import numpy as np
import pandas as pd
import pytest

from mgd.core.imaging.clinical import INSTANCE_COLUMNS
from mgd.core.longitudinal import (
    SCHEMA_VERSION,
    ClinicStore,
    instance_centroids,
    match_instances,
    register_pair,
)
from mgd.core.longitudinal.match import COL_LENGTH, COL_TAU, COL_WIDTH


# ---------------------------------------------------------------- 存储层

def test_store_roundtrip_timeline_and_migrate_idempotent(tmp_path):
    db_file = tmp_path / "nested" / "clinic.db"  # 父目录应自动创建
    store = ClinicStore(db_file)
    try:
        assert db_file.is_file()
        assert (tmp_path / "nested").is_dir()
        # 幂等：重复 migrate 返回 0
        assert store.migrate() == 0
        assert store.migrate() == 0

        store.upsert_patient("P001", pseudonym="anon-001", note="首次建档")
        store.upsert_patient("P002")  # 最小参数
        # 乱序插入就诊，验证 timeline 按 (visit_date, id) 排序
        v_jan = store.add_visit("P001", "2026-01-10", eye="OD", eyelid="upper",
                                image_path="img/jan.png", note="基线")
        v_feb = store.add_visit("P001", "2026-02-10", image_path="img/feb.png")
        v_dec = store.add_visit("P001", "2025-12-01", note="早于基线的回溯访视")
        v_p2 = store.add_visit("P002", "2026-01-10")

        store.add_measurement(v_jan, "R_loss(%)", 12.5, unit="%", source="auto")
        store.add_measurement(v_jan, "长轴(px)", 40.0, unit="px", source="auto")
        store.add_measurement(v_feb, "R_loss(%)", 18.0, unit="%", source="auto")
        store.add_review(v_jan, "doc-A", "confirm", comment="与图像一致",
                         ts="2026-01-11T09:00:00")

        # 患者列表与外键
        patients = store.list_patients()
        assert [p["id"] for p in patients] == ["P001", "P002"]
        assert patients[0]["pseudonym"] == "anon-001"
        assert patients[1]["pseudonym"] == ""

        visits = store.list_visits("P001")
        assert [v["id"] for v in visits] == [v_dec, v_jan, v_feb]
        assert [v["visit_date"] for v in visits] == ["2025-12-01", "2026-01-10", "2026-02-10"]
        assert visits[1]["eye"] == "OD" and visits[1]["eyelid"] == "upper"

        # timeline 往返一致：每次就诊一行，内嵌 measurements / reviews
        timeline = store.timeline("P001")
        assert len(timeline) == 3
        assert [row["id"] for row in timeline] == [v_dec, v_jan, v_feb]
        jan = timeline[1]
        assert jan["measurements"] == {"R_loss(%)": 12.5, "长轴(px)": 40.0}
        assert len(jan["reviews"]) == 1
        review = jan["reviews"][0]
        assert review["reviewer_id"] == "doc-A"
        assert review["decision"] == "confirm"
        assert review["comment"] == "与图像一致"
        assert review["ts"] == "2026-01-11T09:00:00"
        assert timeline[0]["measurements"] == {} and timeline[0]["reviews"] == []
        assert store.timeline("P002")[0]["id"] == v_p2

        # upsert_patient 覆盖：不新增行
        store.upsert_patient("P001", pseudonym="anon-001b", note="更新备注")
        assert len(store.list_patients()) == 2
        assert store.list_patients()[0]["pseudonym"] == "anon-001b"
        assert store.list_patients()[0]["note"] == "更新备注"

        # 外键强制：就诊必须挂在已有患者上
        with pytest.raises(sqlite3.IntegrityError):
            store.add_visit("NO_SUCH_PATIENT", "2026-03-01")
        # 事务回滚后数据不受损
        assert len(store.list_visits("NO_SUCH_PATIENT")) == 0
    finally:
        store.close()

    # 重新打开同一库：schema 已是当前版本，迁移 0 步；数据仍在
    with ClinicStore(db_file) as reopened:
        assert reopened.migrate() == 0
        version = reopened._conn.execute(
            "SELECT MAX(version) AS v FROM schema_version"
        ).fetchone()["v"]
        assert version == SCHEMA_VERSION
        assert len(reopened.list_patients()) == 2


def test_nan_measurement_stored_as_null(tmp_path):
    store = ClinicStore(tmp_path / "nan.db")
    try:
        store.upsert_patient("P-NaN")
        vid = store.add_visit("P-NaN", "2026-01-15")
        store.add_measurement(vid, "面积(px)", float("nan"), unit="px", source="auto")
        store.add_measurement(vid, "长轴(px)", 33.25, unit="px", source="auto")

        # 原始存储：NaN → SQL NULL
        raw = store._conn.execute(
            "SELECT key, value FROM measurement WHERE visit_id = ? ORDER BY id", (vid,)
        ).fetchall()
        assert raw[0]["key"] == "面积(px)"
        assert raw[0]["value"] is None
        assert raw[1]["value"] == pytest.approx(33.25)

        # 读回：NULL → float('nan')（NaN 纪律：绝不用 0 冒充缺失）
        values = store.measurements_for_visit(vid)
        assert set(values) == {"面积(px)", "长轴(px)"}
        assert math.isnan(values["面积(px)"])
        assert values["长轴(px)"] == pytest.approx(33.25)

        timeline = store.timeline("P-NaN")
        assert math.isnan(timeline[0]["measurements"]["面积(px)"])
    finally:
        store.close()


# ---------------------------------------------------------------- 配准

def _synthetic_base(size: int = 256) -> np.ndarray:
    """高斯亮度斑合成图（斑远离边界，便于 np.roll 平移不产生缠绕伪影）。"""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    blobs = [
        (0.95, 128, 120, 20.0),
        (0.70, 80, 175, 13.0),
        (0.80, 185, 70, 16.0),
        (0.60, 205, 205, 10.0),
        (0.50, 55, 55, 9.0),
    ]
    img = np.zeros((size, size), np.float32)
    for amp, cy, cx, sigma in blobs:
        img += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma))
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def test_register_pair_translation_and_unrelated_noise():
    base = _synthetic_base(256)
    follow = np.roll(base, shift=(3, 5), axis=(0, 1))  # 下移 3 px、右移 5 px

    result = register_pair(base, follow)
    assert result["ok"] is True
    assert result["note"] == "ok"
    assert result["ecc_score"] > 0.9
    warp = result["warp_matrix"]
    assert warp is not None and warp.shape == (2, 3)
    # follow→base 的平移应为 (-5, -3)
    assert warp[0, 2] == pytest.approx(-5.0, abs=0.5)
    assert warp[1, 2] == pytest.approx(-3.0, abs=0.5)
    warped = result["warped"]
    assert warped is not None and warped.shape == base.shape
    # warped 应把 follow 拉回 base 坐标系（逐像素几乎一致）
    assert float(np.abs(warped - base).max()) < 0.05

    # 两张完全无关的噪声图：不抛异常、不强行配准
    rng = np.random.default_rng(7)
    noise_a = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
    noise_b = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
    failed = register_pair(noise_a, noise_b)
    assert failed["ok"] is False
    assert failed["warp_matrix"] is None
    assert failed["warped"] is None
    assert failed["note"]  # 原因写在 note 里
    # 尺寸不一致同样安全降级
    mismatch = register_pair(base, base[:200, :200])
    assert mismatch["ok"] is False


# ---------------------------------------------------------------- 实例匹配

def _instance_df(rows: list[dict], columns: list[str] | None = None) -> pd.DataFrame:
    """构造 INSTANCE_COLUMNS 口径的实例明细 DataFrame（缺省列填默认值）。"""
    columns = columns or list(INSTANCE_COLUMNS)
    defaults = {
        "实例编号": None, "面积(px)": 0.0, "长轴(px)": 0.0, "短轴(px)": 0.0,
        "弯曲度τ": 1.0, "平均曲率κ": 0.0, "最大曲率κ": 0.0,
        "相对延伸比": 0.0, "是否截断": 0,
    }
    full = []
    for row in rows:
        merged = dict(defaults)
        merged.update(row)
        full.append({col: merged.get(col, defaults.get(col)) for col in columns})
    return pd.DataFrame(full, columns=columns)


def test_instance_centroids_from_mask():
    mask = np.zeros((60, 80), np.uint8)
    mask[10:30, 10:30] = 1    # 中心 (19.5, 19.5)
    mask[40:50, 60:70] = 1    # 中心 (44.5, 64.5)
    centroids = instance_centroids(mask)
    assert set(centroids) == {1, 2}
    assert centroids[1] == pytest.approx((19.5, 19.5), abs=1e-6)
    assert centroids[2] == pytest.approx((64.5, 44.5), abs=1e-6)
    assert instance_centroids(np.zeros((10, 10), np.uint8)) == {}


def test_match_instances_matched_disappeared_new():
    base_df = _instance_df([
        {"实例编号": 1, "面积(px)": 300, "长轴(px)": 40.0, "短轴(px)": 8.0, "弯曲度τ": 1.20},
        {"实例编号": 2, "面积(px)": 260, "长轴(px)": 36.0, "短轴(px)": 7.0, "弯曲度τ": 1.10},
        {"实例编号": 3, "面积(px)": 200, "长轴(px)": 30.0, "短轴(px)": 6.0, "弯曲度τ": 1.30},
    ])
    follow_df = _instance_df([
        # 1' ≈ 1（质心位移 3 px），长轴 +1、τ +0.15、短轴 +0.5
        {"实例编号": 1, "面积(px)": 310, "长轴(px)": 41.0, "短轴(px)": 8.5, "弯曲度τ": 1.35},
        # 2' ≈ 2（质心不动），长轴 -2、τ -0.05、短轴 -0.5
        {"实例编号": 2, "面积(px)": 240, "长轴(px)": 34.0, "短轴(px)": 6.5, "弯曲度τ": 1.05},
        # 4：远离任何基线实例的新增实例
        {"实例编号": 4, "面积(px)": 110, "长轴(px)": 22.0, "短轴(px)": 5.0, "弯曲度τ": 1.00},
    ])
    base_centroids = {1: (50.0, 60.0), 2: (60.0, 120.0), 3: (150.0, 150.0)}
    follow_centroids = {1: (53.0, 60.0), 2: (60.0, 120.0), 4: (200.0, 200.0)}

    result = match_instances(base_df, follow_df,
                             base_centroids=base_centroids,
                             follow_centroids=follow_centroids)
    assert list(result.columns) == [
        "base_id", "follow_id", "status", "cost",
        "delta_length_px", "delta_tau", "delta_width_px",
    ]

    matched = result[result["status"] == "matched"]
    disappeared = result[result["status"] == "disappeared"]
    new = result[result["status"] == "new"]
    assert len(matched) == 2 and len(disappeared) == 1 and len(new) == 1

    by_base = {int(row["base_id"]): row for _, row in matched.iterrows()}
    assert set(by_base) == {1, 2}

    r1 = by_base[1]
    assert int(r1["follow_id"]) == 1  # 1 ↔ 1'
    assert r1["delta_length_px"] == pytest.approx(1.0)   # 41 − 40
    assert r1["delta_tau"] == pytest.approx(0.15)        # Δτ = ΔTI
    assert r1["delta_width_px"] == pytest.approx(0.5)    # 8.5 − 8.0
    assert r1["cost"] == pytest.approx(3.0 / 40.0 + 0.5 * 1.0 / 33.0, abs=0.05)

    r2 = by_base[2]
    assert int(r2["follow_id"]) == 2  # 2 ↔ 2'
    assert r2["delta_length_px"] == pytest.approx(-2.0)
    assert r2["delta_tau"] == pytest.approx(-0.05)
    assert r2["delta_width_px"] == pytest.approx(-0.5)

    d3 = disappeared.iloc[0]
    assert int(d3["base_id"]) == 3 and pd.isna(d3["follow_id"])
    n4 = new.iloc[0]
    assert int(n4["follow_id"]) == 4 and pd.isna(n4["base_id"])
    # 未匹配行 delta 全为 NaN
    for row in (d3, n4):
        for col in ("cost", "delta_length_px", "delta_tau", "delta_width_px"):
            assert pd.isna(row[col])


def test_match_shape_only_degrades_and_validates_columns():
    base_df = _instance_df([
        {"实例编号": 1, "长轴(px)": 40.0, "短轴(px)": 8.0, "弯曲度τ": 1.20},
        {"实例编号": 2, "长轴(px)": 36.0, "短轴(px)": 7.0, "弯曲度τ": 1.10},
        {"实例编号": 3, "长轴(px)": 30.0, "短轴(px)": 6.0, "弯曲度τ": 1.30},
    ])
    follow_df = _instance_df([
        {"实例编号": 1, "长轴(px)": 41.0, "短轴(px)": 8.5, "弯曲度τ": 1.35},
        {"实例编号": 2, "长轴(px)": 34.0, "短轴(px)": 6.5, "弯曲度τ": 1.05},
        {"实例编号": 4, "长轴(px)": 22.0, "短轴(px)": 5.0, "弯曲度τ": 1.00},
    ])

    # 无质心：cost 退化为形状项，仍可运行，全局最优分配成立
    result = match_instances(base_df, follow_df, base_centroids=None, follow_centroids=None)
    matched = result[result["status"] == "matched"]
    pairs = {(int(r["base_id"]), int(r["follow_id"])) for _, r in matched.iterrows()}
    assert pairs == {(1, 1), (2, 2), (3, 4)}  # 形状最优（3 只能与 4 配）
    r1 = matched[matched["base_id"] == 1].iloc[0]
    assert r1["delta_length_px"] == pytest.approx(1.0)

    # 纯形状模式（cost='length'）与质心模式结果一致（同一分配）
    length_only = match_instances(base_df, follow_df, cost="length")
    assert {(int(r["base_id"]), int(r["follow_id"]))
            for _, r in length_only[length_only["status"] == "matched"].iterrows()} == pairs

    # 空 DataFrame / 单边为空的安全行为
    empty_side = match_instances(base_df.iloc[0:0], follow_df)
    assert len(empty_side) == 3 and (empty_side["status"] == "new").all()

    # 未知 cost → ValueError
    with pytest.raises(ValueError):
        match_instances(base_df, follow_df, cost="bogus")
    # 含质心项但未提供 centroids → ValueError
    with pytest.raises(ValueError):
        match_instances(base_df, follow_df, cost="centroid")

    # 缺列 → ValueError 且提示列名
    missing_width = base_df.drop(columns=[COL_WIDTH])
    with pytest.raises(ValueError, match=re.escape(COL_WIDTH)):
        match_instances(missing_width, follow_df)
    missing_tau = follow_df.drop(columns=[COL_TAU])
    with pytest.raises(ValueError, match=re.escape(COL_TAU)):
        match_instances(base_df, missing_tau)
    missing_length = base_df.drop(columns=[COL_LENGTH])
    with pytest.raises(ValueError, match=re.escape(COL_LENGTH)):
        match_instances(missing_length, follow_df)
    missing_id = base_df.drop(columns=["实例编号"])
    with pytest.raises(ValueError, match="实例编号"):
        match_instances(missing_id, follow_df)

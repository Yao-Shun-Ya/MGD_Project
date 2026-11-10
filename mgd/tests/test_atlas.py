"""WP-6 atlas：8 类形态学分型证据 + 相似病例检索索引（全部离线合成数据）。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mgd.core.atlas import (
    FEATURE_NAMES,
    TYPE_RULES,
    AtlasIndex,
    case_vector,
    classify_case,
    classify_instance,
    vector_to_dict,
)
from mgd.core.imaging.clinical import INSTANCE_COLUMNS


def _rule_keys() -> list[str]:
    return [rule.key for rule in TYPE_RULES]


def _evidence_map(evidence_list) -> dict:
    return {item.key: item for item in evidence_list}


def _busy_case_df() -> pd.DataFrame:
    """四实例病例：两条截断（相对延伸比 0.42/0.45）、一条显著增粗（短轴 15）。

    短轴序列 [15, 4, 5, 6]：中位数 5.5（thickening 基准）、均值 7.5（confluence 基准）。
    """
    rows = [
        ["#1", 900, 120.0, 15.0, 1.80, 0.020, 0.050, 0.90, "否"],
        ["#2", 400, 60.0, 4.0, 1.10, 0.010, 0.030, 0.42, "是"],
        ["#3", 500, 80.0, 5.0, 1.20, 0.010, 0.040, 0.45, "是"],
        ["#4", 600, 90.0, 6.0, 1.30, 0.015, 0.040, 0.80, "否"],
    ]
    return pd.DataFrame(rows, columns=INSTANCE_COLUMNS)


BUSY_FEATURES = {
    "ga_percent": 12.0,                     # dropout: 12 < 0.5×40 = 20
    "reference_density_percent": 40.0,
    "truncation_ratio": 0.4,                # shortening(case): 0.4 ≥ 0.2
    "tortuosity_index": 0.45,               # tortuosity: 0.45 > 0.25 且拐点 4 ≥ 2
    "inflection_count": 4,
    "width_mean_px": 9.0,                   # thickening: 9.0 > 1.5×5.5 = 8.25
    "width_profile_std": 3.6,               # beading: CV = 3.6/9.0 = 0.4 > 0.3，峰数 3 ≥ 2
    "width_peaks": 3,
    "ghost_area_px": 260.0,                 # ghost: 260 ≥ 120
    "break_count": 2,                       # disruption: 2 ≥ 1
    "min_inter_instance_distance_px": 1.0,  # confluence: 1.0 < 0.5×7.5 = 3.75
}


def test_all_eight_rules_trigger_with_evidence():
    evidence = _evidence_map(classify_case(_busy_case_df(), BUSY_FEATURES))
    assert list(evidence) == _rule_keys()
    expected_values = {
        "dropout": 12.0, "shortening": 0.4, "tortuosity": 0.45, "thickening": 9.0,
        "beading": 0.4, "ghost": 260.0, "disruption": 2.0, "confluence": 1.0,
    }
    expected_thresholds = {
        "dropout": 20.0, "shortening": 0.2, "tortuosity": 0.25, "thickening": 8.25,
        "beading": 0.3, "ghost": 120.0, "disruption": 1.0, "confluence": 3.75,
    }
    for key in _rule_keys():
        item = evidence[key]
        assert item.present is True, key
        assert item.value == pytest.approx(expected_values[key]), key
        assert item.threshold == pytest.approx(expected_thresholds[key]), key
        assert item.definition, key  # 证据自带判定条件，可追溯


def test_rule_sources_are_explicitly_split():
    sources = {rule.key: rule.source for rule in TYPE_RULES}
    assert set(sources) == set(_rule_keys())
    assert sources["tortuosity"] == "Pult & Nichols 2016"
    assert all(sources[key] == "engine-defined" for key in _rule_keys() if key != "tortuosity")
    for rule in TYPE_RULES:
        assert rule.name_zh and rule.definition


def test_case_level_rules_fall_back_to_instance_rows():
    # features 缺 truncation_ratio → shortening 按 instance_df 逐行评估
    features = {key: value for key, value in BUSY_FEATURES.items() if key != "truncation_ratio"}
    item = _evidence_map(classify_case(_busy_case_df(), features))["shortening"]
    assert item.present is True
    assert item.value == pytest.approx(0.42)      # 最显著截断实例的相对延伸比
    assert item.threshold == pytest.approx(0.5)   # clinical `是否截断` 口径


def test_normal_gland_triggers_nothing():
    normal_df = pd.DataFrame(
        [
            ["#1", 800, 110.0, 5.0, 1.05, 0.008, 0.020, 0.90, "否"],
            ["#2", 780, 108.0, 5.0, 1.04, 0.008, 0.020, 0.88, "否"],
            ["#3", 820, 112.0, 5.0, 1.03, 0.008, 0.020, 0.92, "否"],
        ],
        columns=INSTANCE_COLUMNS,
    )
    features = {
        "ga_percent": 62.0,                     # 62 > 0.5×40 = 20 → 不缺失
        "reference_density_percent": 40.0,
        "truncation_ratio": 0.0,                # 0.0 < 0.2 → 不缩短
        "tortuosity_index": 0.05,               # 0.05 < 0.25 → 不迂曲
        "inflection_count": 1,
        "width_mean_px": 5.0,                   # 5.0 < 1.5×5 = 7.5 → 不增粗
        "width_profile_std": 0.4,               # CV = 0.08 < 0.3 → 不串珠
        "ghost_area_px": 0.0,                   # 0 < 120 → 无幻影
        "break_count": 0,                       # 0 < 1 → 无断裂
        "min_inter_instance_distance_px": 25.0,  # 25 > 0.5×5 = 2.5 → 不融合
    }
    evidence = _evidence_map(classify_case(normal_df, features))
    assert list(evidence) == _rule_keys()
    for key, item in evidence.items():
        assert item.present is False, key
        assert np.isfinite(item.value), key       # 输入齐全 → 实测值可见
        assert np.isfinite(item.threshold), key


def test_overrides_flip_conclusions():
    base = {"ga_percent": 30.0, "reference_density_percent": 40.0, "break_count": 2}
    default = _evidence_map(classify_case(_busy_case_df(), base))
    assert default["dropout"].present is False    # 30 < 0.5×40 = 20 不成立
    assert default["disruption"].present is True  # 2 ≥ 1

    flipped = _evidence_map(
        classify_case(_busy_case_df(), base, overrides={"dropout": 0.9, "disruption": 5.0})
    )
    assert flipped["dropout"].present is True     # 30 < 0.9×40 = 36 成立
    assert flipped["dropout"].threshold == pytest.approx(36.0)
    assert flipped["dropout"].value == pytest.approx(30.0)
    assert flipped["disruption"].present is False  # 2 ≥ 5 不成立
    assert flipped["disruption"].threshold == pytest.approx(5.0)

    with pytest.raises(ValueError):
        classify_case(_busy_case_df(), base, overrides={"not_a_rule": 1.0})


def test_classify_instance_single_instance_rules():
    instance = {
        "relative_extension_ratio": 0.30,  # shortening 实例版：< 0.5
        "width_mean_px": 12.0,             # thickening：12 > 1.5×5 = 7.5
        "width_median_px": 5.0,
        "tortuosity_index": 0.50,          # tortuosity：0.50 > 0.25 且拐点 3 ≥ 2
        "inflection_count": 3,
        "break_count": 1,                  # disruption：1 ≥ 1
    }
    evidence = _evidence_map(classify_instance(instance))
    assert list(evidence) == _rule_keys()
    assert evidence["shortening"].present is True
    assert evidence["shortening"].value == pytest.approx(0.30)
    assert evidence["thickening"].present is True
    assert evidence["thickening"].threshold == pytest.approx(7.5)
    assert evidence["tortuosity"].present is True
    assert evidence["disruption"].present is True
    for key in ("dropout", "beading", "ghost", "confluence"):
        assert evidence[key].present is False, key
        assert np.isnan(evidence[key].value), key  # 缺实例上下文 → NaN 证据

    # 兼容 df 行字典：中文列名 + 是否截断旗标
    row_like = {"短轴(px)": 14.0, "width_median_px": 5.0, "是否截断": "是"}
    row_evidence = _evidence_map(classify_instance(row_like))
    assert row_evidence["shortening"].present is True
    assert np.isnan(row_evidence["shortening"].value)  # 仅旗标 → value 如实 NaN
    assert row_evidence["thickening"].present is True
    assert row_evidence["thickening"].value == pytest.approx(14.0)

    assert all(not item.present for item in classify_instance({}))


def test_classify_case_survives_missing_columns():
    sparse = pd.DataFrame({"实例编号": ["#1"], "面积(px)": [300]})
    evidence = _evidence_map(
        classify_case(sparse, {"ga_percent": 12.0, "reference_density_percent": 40.0})
    )
    assert list(evidence) == _rule_keys()
    assert evidence["dropout"].present is True  # 输入齐全的规则照常判定
    for key in ("shortening", "tortuosity", "thickening", "beading", "ghost",
                "disruption", "confluence"):
        assert evidence[key].present is False, key
        assert np.isnan(evidence[key].value), key

    for df in (pd.DataFrame(columns=INSTANCE_COLUMNS), None):
        empty = classify_case(df, {})
        assert len(empty) == len(TYPE_RULES)
        assert all(not item.present for item in empty)
        assert all(np.isnan(item.value) for item in empty)


def test_feature_vector_roundtrip():
    assert len(FEATURE_NAMES) == 12
    vector = case_vector({"ga_percent": 10.0, "break_count": 2})
    assert vector.shape == (len(FEATURE_NAMES),)
    assert vector.dtype == np.float64
    assert vector[0] == 10.0 and vector[8] == 2.0
    assert np.isnan(vector[3])  # 缺失键 → NaN，不用 0 冒充

    restored = vector_to_dict(vector)
    assert list(restored) == list(FEATURE_NAMES)
    assert restored["ga_percent"] == 10.0
    assert np.isnan(restored["width_cv"])
    assert case_vector(restored).shape == (len(FEATURE_NAMES),)

    full = dict.fromkeys(FEATURE_NAMES, 1.0)
    assert np.isfinite(case_vector(full)).all()
    assert np.allclose(case_vector(full), np.ones(len(FEATURE_NAMES)))

    with pytest.raises(ValueError):
        vector_to_dict(np.zeros(3))


def test_atlas_index_search_order_and_zero_vector():
    dim = len(FEATURE_NAMES)
    vec_a = np.zeros(dim)
    vec_a[0] = 1.0
    vec_b = np.zeros(dim)
    vec_b[3] = 1.0
    vec_c = np.zeros(dim)
    vec_c[8] = 1.0
    index = AtlasIndex()
    index.add("case-a", vec_a, {"label": "A"})
    index.add("case-b", vec_b, {"label": "B"})
    index.add("case-c", vec_c, {"label": "C"})
    assert len(index) == 3

    hits = index.search(vec_a, k=3)
    assert hits[0]["case_id"] == "case-a"
    assert hits[0]["distance"] == pytest.approx(0.0)
    assert hits[0]["meta"] == {"label": "A"}
    # 正交向量距离并列 1.0 → 按 case_id 稳定排序
    assert [hit["case_id"] for hit in hits[1:]] == ["case-b", "case-c"]
    assert all(hit["distance"] == pytest.approx(1.0) for hit in hits[1:])
    assert len(index.search(vec_a, k=2)) == 2  # k 截断

    zero_hits = index.search(np.zeros(dim), k=3)  # 零向量查询不崩溃
    assert [hit["distance"] for hit in zero_hits] == [1.0, 1.0, 1.0]

    index.add("case-a", vec_b, {"label": "A2"})  # 重复 case_id 覆盖
    assert len(index) == 3
    assert index.search(vec_b, k=1)[0]["case_id"] == "case-a"

    with pytest.raises(ValueError):
        index.add("bad-dim", np.zeros(dim + 1), {})


def test_atlas_index_json_roundtrip(tmp_path):
    dim = len(FEATURE_NAMES)
    vec = case_vector({"ga_percent": 45.0, "break_count": 1})
    vec_nan = np.full(dim, np.nan)
    index = AtlasIndex()
    index.add("case-a", vec, {"ga_percent": 45.0})
    index.add("case-nan", vec_nan, {})

    path = index.to_json(tmp_path / "atlas.json")
    assert path == tmp_path / "atlas.json"
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["feature_names"] == list(FEATURE_NAMES)

    loaded = AtlasIndex.load(path)
    assert len(loaded) == len(index)
    assert loaded.feature_names == index.feature_names
    for query in (vec, vec_nan):
        before = index.search(query, k=2)
        after = loaded.search(query, k=2)
        assert [hit["case_id"] for hit in before] == [hit["case_id"] for hit in after]
        assert [hit["distance"] for hit in before] == pytest.approx(
            [hit["distance"] for hit in after]
        )
    assert loaded.search(vec, k=1)[0]["meta"] == {"ga_percent": 45.0}

    bad_path = tmp_path / "bad.json"
    bad_path.write_text(
        json.dumps({"version": 999, "feature_names": list(FEATURE_NAMES), "items": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        AtlasIndex.load(bad_path)


def test_rule_feature_keys_are_registered_metrics():
    from mgd.core.quantification.registry import METRIC_REGISTRY

    # 分型 / 特征使用的指标 key 必须与注册表对齐（width_cv 与 DI 公式一致）
    for key in ("ga_percent", "tortuosity_index", "inflection_count", "width_mean_px",
                "break_count", "diameter_deformation_index", "instance_count",
                "fractal_dimension", "signal_index"):
        assert key in METRIC_REGISTRY, key


def _strict_json_loads(text: str):
    """严格 JSON 解析：出现 NaN / Infinity / -Infinity token 直接抛 ValueError。"""

    def _reject(token: str):
        raise ValueError(f"非法 JSON 常量: {token}")

    return json.loads(text, parse_constant=_reject)


def test_atlas_index_json_numpy_and_nan_meta(tmp_path):
    """[H3+L1] meta 含 numpy 标量 / ndarray / 嵌套 dict / NaN / ±inf → 严格合法 JSON。"""
    dim = len(FEATURE_NAMES)
    vec = case_vector({"ga_percent": 30.0})
    meta = {
        "count": np.int64(7),
        "flag": np.bool_(True),
        "ratio": np.float64(0.5),
        "ratio32": np.float32(1.25),
        "arr": np.array([1.0, 2.0]),
        "nested": {"depth": {"ok": np.int32(3)}, "bad": float("nan")},
        "inf": float("inf"),
        "ninf": float("-inf"),
        "text": "保留中文",
    }
    index = AtlasIndex()
    index.add("case-np", vec, meta)
    path = index.to_json(tmp_path / "atlas_np.json")
    text = path.read_text(encoding="utf-8")
    payload = _strict_json_loads(text)  # 不抛即证明无 NaN / Infinity token

    item = payload["items"][0]
    assert item["meta"]["count"] == 7
    assert item["meta"]["flag"] is True
    assert item["meta"]["ratio"] == pytest.approx(0.5)
    assert item["meta"]["ratio32"] == pytest.approx(1.25)
    assert item["meta"]["arr"] == [1.0, 2.0]
    assert item["meta"]["nested"]["depth"]["ok"] == 3
    assert item["meta"]["nested"]["bad"] is None   # NaN → null
    assert item["meta"]["inf"] is None             # +inf → null
    assert item["meta"]["ninf"] is None            # -inf → null
    assert item["meta"]["text"] == "保留中文"

    # NaN 向量同样产出严格 JSON（null 在 load 时还原为 NaN）
    nan_index = AtlasIndex()
    nan_index.add("case-nan", np.full(dim, np.nan), {"x": float("nan")})
    nan_payload = _strict_json_loads(
        nan_index.to_json(tmp_path / "atlas_nan.json").read_text(encoding="utf-8")
    )
    assert nan_payload["items"][0]["vector"] == [None] * dim
    assert nan_payload["items"][0]["meta"]["x"] is None

    # 成功写盘后无 .tmp 残留；roundtrip 后检索行为一致
    assert list(tmp_path.glob("*.tmp")) == []
    loaded = AtlasIndex.load(path)
    assert loaded.search(vec, k=1)[0]["case_id"] == "case-np"


def test_atlas_index_to_json_cleans_tmp_on_failure(tmp_path):
    """[H3] 序列化失败 → 异常原样上抛，且目标目录无 .tmp 残留。"""

    class _Boom:
        def __str__(self) -> str:
            raise RuntimeError("无法序列化")

    index = AtlasIndex()
    index.add("case-boom", case_vector({"ga_percent": 1.0}), {"obj": _Boom()})
    with pytest.raises(RuntimeError):
        index.to_json(tmp_path / "boom.json")
    assert not (tmp_path / "boom.json").exists()
    assert list(tmp_path.iterdir()) == []  # 无 .tmp 残留


def test_atlas_index_add_deep_copies_meta():
    """[L2] add 深拷贝 meta：外部随后修改（含嵌套结构）不影响索引内数据。"""
    vec = case_vector({"ga_percent": 10.0})
    meta = {"nested": {"label": "A"}}
    index = AtlasIndex()
    index.add("case-a", vec, meta)

    meta["nested"]["label"] = "MUTATED"
    meta["added"] = 123
    assert index.search(vec, k=1)[0]["meta"] == {"nested": {"label": "A"}}

    # 覆盖同 case_id 后，旧 meta 的外部引用同样无法影响新存入的数据
    fresh = {"nested": {"label": "B"}}
    index.add("case-a", vec, fresh)
    fresh["nested"]["label"] = "MUTATED-2"
    assert index.search(vec, k=1)[0]["meta"] == {"nested": {"label": "B"}}


def test_atlas_index_load_rejects_wrong_vector_dimension(tmp_path):
    """[M1] load 时向量维度与索引不符 → ValueError（含 case_id 与两个维度数）。"""
    dim = len(FEATURE_NAMES)
    index = AtlasIndex()
    index.add("case-a", case_vector({"ga_percent": 5.0}), {})
    index.add("case-b", case_vector({"break_count": 1.0}), {})
    path = index.to_json(tmp_path / "ok.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    # 少一维
    payload["items"][1]["vector"] = payload["items"][1]["vector"][:-1]
    bad_fewer = tmp_path / "bad_fewer.json"
    bad_fewer.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="case-b"):
        AtlasIndex.load(bad_fewer)

    # 多两维：报错同时包含 case_id 与两个维度数
    payload["items"][1]["vector"] = [0.0] * (dim + 2)
    bad_more = tmp_path / "bad_more.json"
    bad_more.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        AtlasIndex.load(bad_more)
    message = str(excinfo.value)
    assert "case-b" in message
    assert str(dim + 2) in message and str(dim) in message

    # 正常文件不受影响
    assert len(AtlasIndex.load(path)) == 2


def test_overrides_reject_negative_and_zero_multiplicative_thresholds():
    """[M3] overrides 负阈值 → ValueError；乘法类系数（dropout/thickening/confluence）还要求 > 0。"""
    base: dict = {}
    # 乘法类：负系数与 0 系数都非法
    for rule_key in ("dropout", "thickening", "confluence"):
        with pytest.raises(ValueError):
            classify_case(_busy_case_df(), base, overrides={rule_key: -0.5})
        with pytest.raises(ValueError):
            classify_case(_busy_case_df(), base, overrides={rule_key: 0.0})
    # 非乘法类：负值非法，0 合法（如 ghost 阈值 0：面积 ≥ 0 恒真 → present）
    with pytest.raises(ValueError):
        classify_case(_busy_case_df(), base, overrides={"shortening": -1.0})
    with pytest.raises(ValueError):
        classify_case(_busy_case_df(), base, overrides={"ghost": -120.0})
    ghost_base = {"ghost_area_px": 10.0}  # 10 ≥ 0 → 阈值 0 时应触发
    assert _evidence_map(classify_case(_busy_case_df(), ghost_base, overrides={"ghost": 0.0}))[
        "ghost"
    ].present is True
    # 非法 overrides 不产生半更新：classify_case 抛错即可复现
    with pytest.raises(ValueError):
        classify_case(_busy_case_df(), base, overrides={"dropout": -0.1, "ghost": 5.0})


def test_case_vector_rejects_numpy_bool():
    """[L6] np.bool_ 视为布尔缺失 → NaN，绝不被 float() 转成 1.0 冒充测量值。"""
    assert np.isnan(case_vector({"ga_percent": np.bool_(True)})[0])
    assert np.isnan(case_vector({"ga_percent": np.bool_(False)})[0])
    # 原生 bool 既有行为不变；普通数值不受影响
    assert np.isnan(case_vector({"ga_percent": True})[0])
    assert case_vector({"ga_percent": 1.0})[0] == 1.0

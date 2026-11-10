"""WP-11 UI 构建测试：create_app 离线可构建、11 个 tab 齐全、概率伪彩说明文案存在。

全部 CPU 离线：只构建 Blocks（不 launch、不加载权重、不读随访库）。
构造成本集中在 import（torch / gradio），因此 4 个断言共用一次 create_app 构建
（module 级 fixture），保证整个文件 <10s。

注：必须在 import gradio / 构造 Blocks 之前关闭 gradio 分析，否则 Blocks
创建会拉起非守护的 version_check 线程（httpx 网络请求），离线环境下会让
进程退出阻塞 20s+。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from mgd.adapters.gradio_app.app import create_app  # noqa: E402
from mgd.core.config import Settings  # noqa: E402

NEW_TABS = ("质控", "解释性", "形态图谱", "随访", "分级")
LEGACY_TABS = ("诊断", "训练", "评估", "实验管理", "数据集", "系统状态")


@pytest.fixture(scope="module")
def ui_settings(tmp_path_factory):
    """与 conftest 的 settings fixture 同参数（64×128 / 单进程），module 级以复用构建。"""
    home = tmp_path_factory.mktemp("ui_home")
    return Settings(home=home, target_height=64, target_width=128, num_workers=0)


@pytest.fixture(scope="module")
def demo(ui_settings):
    return create_app(ui_settings)


def _tab_labels(demo) -> set[str]:
    """遍历 blocks 收集所有 gr.Tab 的标签。"""
    labels: set[str] = set()
    for block in demo.blocks.values():
        if isinstance(block, gr.Tab):
            label = getattr(block, "label", None)
            if isinstance(label, str):
                labels.add(label)
    return labels


def test_create_app_builds_blocks_offline(demo):
    assert isinstance(demo, gr.Blocks)


def test_new_tabs_present(demo):
    labels = _tab_labels(demo)
    missing = [name for name in NEW_TABS if name not in labels]
    assert not missing, f"缺少新 tab：{missing}（实际：{sorted(labels)}）"


def test_legacy_tabs_regression(demo):
    labels = _tab_labels(demo)
    missing = [name for name in LEGACY_TABS if name not in labels]
    assert not missing, f"原有 tab 被破坏：{missing}（实际：{sorted(labels)}）"


def test_prob_heatmap_disclaimer_in_source():
    source = Path(__file__).resolve().parents[1] / "adapters" / "gradio_app" / "app.py"
    text = source.read_text(encoding="utf-8")
    assert "概率伪彩（非归因）" in text
    assert "归因请看「解释性」" in text


# ---------------------------------------------------------------------------
# 集成层审计回归：H2（τ 原值）/ 图谱特征缺口 / L-6（report_fn 接线）/ L-11（患者一致性）
# ---------------------------------------------------------------------------
def _app_source() -> str:
    return (Path(__file__).resolve().parents[1] / "adapters" / "gradio_app" / "app.py").read_text(
        encoding="utf-8"
    )


def _nested_fn_source(text: str, name: str) -> str:
    """create_app 内嵌函数源码切片（到下一个同级 def 之前）。"""
    start = text.index(f"def {name}(")
    end = text.find("\n    def ", start + 1)
    return text[start:end if end != -1 else len(text)]


def test_tortuosity_index_uses_tau_minus_one_in_source():
    """H2：TI = τ − 1（τ 是弧长/弦长比，原值不是 TI，不得直接装进 tortuosity_index）。"""
    assert '"tortuosity_index", float(tortuosity.mean()) - 1.0' in _app_source()


def test_graph_feature_producers_in_source():
    """图谱特征缺口：ghost / beading / confluence 三个判据的生产者必须在 app.py。"""
    text = _app_source()
    for marker in ("ghost_area_px", "width_profile_std", "min_inter_instance_distance_px"):
        assert marker in text


def test_report_fn_wires_v240_blocks_in_source():
    """L-6：report_fn 接通 v2.4.0 扩展区块的最小静态证明 + 扩展失败降级（warning 而非中断）。"""
    body = _nested_fn_source(_app_source(), "report_fn")
    assert "collect_provenance(" in body
    assert "grade_all(" in body
    assert "classify_case(" in body
    assert "_atlas_case_features(" in body
    assert "logger.warning" in body


def test_followup_compare_fn_validates_patient_ownership_in_source():
    """L-11：followup_compare_fn 必须按当前输入患者（pid）核对所选两次就诊的归属。"""
    body = _nested_fn_source(_app_source(), "followup_compare_fn")
    assert "patient_id" in body
    assert "list_visits(pid)" in body
    assert "base_id not in visits or follow_id not in visits" in body


class _FakeResult:
    def __init__(self, prob_map, binary_mask):
        self.prob_map = prob_map
        self.binary_mask = binary_mask


def _two_gland_mask() -> np.ndarray:
    mask = np.zeros((40, 120), dtype="uint8")
    mask[5:35, 10:24] = 1
    mask[5:35, 40:54] = 1
    return mask


def test_atlas_case_features_values():
    """功能验证：TI=τ−1、ghost（无低置信带=0）、两实例最小间距=30px、宽度剖面 std 有限。"""
    import pandas as pd

    from mgd.adapters.gradio_app.app import _atlas_case_features

    binary = _two_gland_mask()
    result = _FakeResult(binary.astype("float32"), binary)
    instance_df = pd.DataFrame({
        "短轴(px)": [10.0, 20.0],
        "弯曲度τ": [1.5, 2.0],
        "平均曲率κ": [0.1, 0.2],
        "是否截断": ["否", "否"],
    })
    features = _atlas_case_features(result, instance_df, 12.5)
    assert features["tortuosity_index"] == pytest.approx(0.75)  # mean(1.5, 2.0)=1.75 → TI=0.75
    assert features["ga_percent"] == pytest.approx(12.5)
    assert features["ghost_area_px"] == pytest.approx(0.0)
    assert features["min_inter_instance_distance_px"] == pytest.approx(30.0)
    assert np.isfinite(features["width_profile_std"])


def test_ghost_area_px_counts_low_conf_outside_binary():
    """0.2<p<0.5 且不属于 binary_mask 的连通域：面积 ≥120px 计入，小块不计。"""
    from mgd.adapters.gradio_app.app import _ghost_area_px

    binary = np.zeros((60, 60), dtype="uint8")
    binary[5:15, 5:15] = 1
    prob = np.zeros((60, 60), dtype="float32")
    prob[binary > 0] = 0.9        # 确定性腺体：不在低置信带
    prob[30:40, 30:42] = 0.35     # 低置信块 10×12=120px ≥ 120 → 计入
    prob[50:52, 50:52] = 0.35     # 2×2=4px < 120 → 不计
    assert _ghost_area_px(prob, binary) == pytest.approx(120.0)


def test_ghost_area_px_invalid_inputs_are_nan():
    import math

    from mgd.adapters.gradio_app.app import _ghost_area_px

    assert math.isnan(_ghost_area_px(None, None))
    prob = np.zeros((10, 10), dtype="float32")
    mask = np.zeros((12, 10), dtype="uint8")
    assert math.isnan(_ghost_area_px(prob, mask))  # 形状不一致 → NaN（不判定）


def test_min_inter_instance_distance_px():
    import math

    from mgd.adapters.gradio_app.app import _min_inter_instance_distance_px

    assert _min_inter_instance_distance_px(_two_gland_mask()) == pytest.approx(30.0)
    single = np.zeros((20, 20), dtype="uint8")
    single[5:15, 5:15] = 1
    assert math.isnan(_min_inter_instance_distance_px(single))  # 单实例不判定
    assert math.isnan(_min_inter_instance_distance_px(np.zeros((20, 20), dtype="uint8")))


if __name__ == "__main__":
    pytest.main([__file__, "-q"])

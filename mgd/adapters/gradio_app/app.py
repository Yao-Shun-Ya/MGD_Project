"""Gradio 适配层（Windows WebUI 交付端）。

只做「控件 ↔ 核心 API」映射，不含算法细节。视觉按医学软件标准：
深色金属底、低饱和交互色、圆形径向光、克制动效；六个分区：
诊断 / 训练 / 评估 / 实验管理 / 数据集 / 系统状态。
"""
from __future__ import annotations

import re
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

import numpy as np

from ... import __version__
from ...core.atlas import TYPE_RULES, AtlasIndex, case_vector, classify_case
from ...core.config import Settings, get_settings, list_trained_models
from ...core.data.pairing import scan_dataset
from ...core.data.splits import SplitSpec, get_splits
from ...core.data.versioning import (
    backup_dataset,
    integrity_report,
    integrity_rows,
    record_version,
)
from ...core.evaluation import (
    EvalReport,
    evaluate_pairs,
    export_csv,
    export_latex,
    summary_rows,
)
from ...core.feedback import append_record, record_correction
from ...core.explain import (
    LayerCAM,
    OcclusionConfig,
    ScoreCAM,
    faithfulness_report,
    occlusion_map,
)
from ...core.grading import grade_all, grade_table, incompatibility_note
from ...core.imaging.preprocess import preprocess_array, to_gray
from ...core.inference import (
    InferenceOptions,
    ModelService,
    TTA_2X,
    TTA_4X,
    TTA_OFF,
    discover_weights,
    load_presets,
    save_preset,
)
from ...core.models.manifest import validate_weight
from ...core.models.registry import MODEL_NAMES
from ...core.longitudinal import ClinicStore, instance_centroids, register_pair
from ...core.provenance import collect_provenance
from ...core.qc import QCPolicy, assess_quality, decide, default_thresholds
from ...core.quantification import break_count, width_profile
from ...core.reporting.report import build_report
from ...core.runtime.device import describe_device, resolve_device
from ...core.runtime.doctor import run_doctor
from ...core.runtime.errors import MGDError
from ...core.runtime.logging import configure_logging, get_logger
from ...core.training.config_file import build_train_config
from ...core.training.engine import TrainConfig, train_iter
from ...core.training.experiments import (
    compare_rows,
    compare_runs,
    export_comparison_csv,
    export_comparison_latex,
    list_runs,
    metric_series,
)
from ...core.training.presets import (
    TRAIN_PROFILES,
    apply_tier,
    detect_device_text,
    detect_vram_gb,
    profile_names,
    profile_overrides,
    tier_table,
)
from ...core.training.runner import run_training

logger = get_logger("adapters.gradio")

#: 同一进程内串行化 GPU 工作，避免「训练 + 推理」并发争抢显存。
WORK_LOCK = threading.Lock()

_TTA_CHOICES = [TTA_OFF, TTA_2X, TTA_4X]
_SCOPE_CHOICES = ["Test（独立测试集）", "Val（验证集）", "Train（训练集自评）"]
_SPLIT_CHOICES = ["Holdout 70/15/15", "K 折交叉验证"]
_VRAM_CHOICES = ["auto", "6", "8", "12", "16", "24", "32", "40"]

CSS = """
:root {
  --mgd-bg: #12151a;
  --mgd-panel: #181c22;
  --mgd-panel-2: #1e242c;
  --mgd-line: #2a323c;
  --mgd-text: #d7dde4;
  --mgd-muted: #8d99a6;
  --mgd-accent: #6f8fa8;
  --mgd-accent-soft: #8fa9bd;
  --mgd-ok: #6f9c7f;
  --mgd-warn: #b09a6a;
  --mgd-fail: #b0776f;
}
body, .gradio-container, .app {
  background: radial-gradient(circle at 50% -10%, #1b222b 0%, var(--mgd-bg) 55%) !important;
  color: var(--mgd-text) !important;
  font-family: "Segoe UI", "Inter", "Noto Sans SC", "Microsoft YaHei", sans-serif !important;
}
.gradio-container { max-width: 100% !important; padding: 18px 22px 32px !important; }
.gradio-container h1, h2, h3, h4, p, span, label, div { color: var(--mgd-text); }
#mgd-header {
  display: flex; align-items: center; gap: 16px; padding: 14px 20px; margin-bottom: 12px;
  border: 1px solid var(--mgd-line); border-radius: 12px;
  background: linear-gradient(180deg, #202730 0%, #171c22 100%);
  box-shadow: inset 0 1px 0 #333c47, 0 10px 26px rgba(0,0,0,.32);
}
#mgd-brand { display:flex; align-items:center; gap:12px; }
#mgd-mark {
  width: 40px; height: 40px; border-radius: 50%;
  background: radial-gradient(circle at 50% 42%, rgba(143,169,189,.55) 0%, rgba(111,143,168,.16) 45%, rgba(18,21,26,0) 72%);
  border: 1px solid #38424e; box-shadow: 0 0 22px rgba(111,143,168,.22);
}
#mgd-title { font-size: 19px; font-weight: 650; letter-spacing: .4px; }
#mgd-sub { font-size: 12px; color: var(--mgd-muted); }
#mgd-chips { margin-left: auto; display:flex; gap:8px; flex-wrap: wrap; }
.mgd-chip {
  font-size: 12px; padding: 4px 10px; border-radius: 999px;
  border: 1px solid var(--mgd-line); background: #151a20; color: var(--mgd-muted);
  white-space: nowrap;
}
.tabs { border: none !important; }
.tab-nav button, .tabs > .tab-nav > button {
  color: var(--mgd-muted) !important; background: transparent !important;
  border: none !important; border-bottom: 2px solid transparent !important;
  font-weight: 500 !important;
}
.tab-nav button.selected, .tab-nav button[aria-selected="true"] {
  color: var(--mgd-text) !important; border-bottom-color: var(--mgd-accent) !important;
}
.block, .panel, .form, .gr-box, .gr-group {
  background: var(--mgd-panel) !important; border: 1px solid var(--mgd-line) !important;
  border-radius: 10px !important;
}
input:not([type="radio"]):not([type="checkbox"]):not([type="range"]):not([type="file"]),
textarea, select, .gr-input, .gr-dropdown {
  background: #12151a !important; color: var(--mgd-text) !important;
  border: 1px solid #303944 !important; border-radius: 8px !important;
}
input:not([type="radio"]):not([type="checkbox"]):not([type="range"]):not([type="file"]):focus,
textarea:focus, select:focus, .gr-input:focus-within {
  outline: none !important; border-color: var(--mgd-accent) !important;
  box-shadow: 0 0 0 3px rgba(111,143,168,.18) !important;
}
input[type="radio"], input[type="checkbox"] {
  accent-color: var(--mgd-accent) !important;
  background: transparent !important;
  border: 1px solid #46525f !important;
  box-shadow: none !important;
}
input[type="radio"]:checked, input[type="checkbox"]:checked {
  accent-color: var(--mgd-accent-soft) !important;
  background: var(--mgd-accent) !important;
  border-color: var(--mgd-accent) !important;
}
button {
  border-radius: 8px !important; border: 1px solid #333d49 !important;
  background: linear-gradient(180deg, #232a33 0%, #1a1f26 100%) !important;
  color: var(--mgd-text) !important; font-weight: 520 !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.05);
  transition: background .16s ease, border-color .16s ease, transform .16s ease;
}
button:hover { border-color: #46525f !important; background: linear-gradient(180deg, #2a323c, #1d232b) !important; }
button:focus-visible { outline: 2px solid var(--mgd-accent-soft) !important; outline-offset: 2px; }
button:active { transform: translateY(1px); }
button[variant="primary"], .primary {
  background: linear-gradient(180deg, #33505f 0%, #263c48 100%) !important;
  border-color: #3d5c6c !important;
}
button[variant="primary"]:hover { background: linear-gradient(180deg, #3b5b6d, #2a4350) !important; }
table { color: var(--mgd-text) !important; border-color: var(--mgd-line) !important; }
table th { background: #1c222a !important; color: #aeb9c4 !important; font-weight: 600; }
table td { border-color: #262e37 !important; }
table { overflow-x: auto !important; }
.table-wrap, .table-container, .gr-dataframe, .dataframe { overflow-x: auto !important; max-width: 100% !important; }
label span { color: var(--mgd-muted) !important; }
code, kbd, samp {
  background: #1e242c !important;
  color: var(--mgd-text) !important;
  border: 1px solid var(--mgd-line) !important;
  border-radius: 5px;
  padding: 1px 5px;
  font-family: "Cascadia Code", "Consolas", "SFMono-Regular", monospace;
  font-size: .92em;
}
pre {
  background: #0d1117 !important;
  color: var(--mgd-text) !important;
  border: 1px solid var(--mgd-line) !important;
  border-radius: 8px;
}
.mgd-section-title { font-size: 13px; letter-spacing: 1.2px; text-transform: uppercase;
  color: var(--mgd-accent-soft); margin: 4px 0 6px; }
.mgd-help { font-size: 12px; color: var(--mgd-muted); line-height: 1.7; }
.mgd-ok { color: var(--mgd-ok); }
.mgd-warn { color: var(--mgd-warn); }
.mgd-fail { color: var(--mgd-fail); }
#mgd-toast-stack {
  position: fixed; top: 84px; right: 18px; z-index: 100000;
  display: flex; flex-direction: column; gap: 8px; pointer-events: none;
}
.mgd-toast {
  display: flex; align-items: flex-start; gap: 9px;
  max-width: min(420px, calc(100vw - 36px)); padding: 10px 13px;
  border: 1px solid #3d5c6c; border-radius: 10px;
  background: rgba(24,28,34,.96); color: var(--mgd-text);
  box-shadow: 0 10px 30px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.06);
  font-size: 12px; line-height: 1.5;
  opacity: 0; transform: translateY(-6px);
  transition: opacity .18s ease, transform .18s ease;
  backdrop-filter: blur(10px);
}
.mgd-toast.show { opacity: 1; transform: translateY(0); }
.mgd-toast-dot {
  width: 7px; height: 7px; margin-top: 5px; border-radius: 50%; flex: none;
  background: var(--mgd-ok); box-shadow: 0 0 10px rgba(111,156,127,.8);
}
.mgd-toast-text { word-break: break-word; }
"""


TOAST_JS = r"""
(() => {
  if (window.__mgdToastInstalled) return;
  window.__mgdToastInstalled = true;
  const clean = (value) => String(value || "").replace(/\s+/g, " ").trim();
  const labelFor = (el) => {
    let label = clean(el.getAttribute("aria-label"));
    if (label.startsWith("range slider for ")) label = label.slice(17);
    if (label.startsWith("number input for ")) label = label.slice(17);
    if (!label) {
      const block = el.closest('.block, .form, [data-testid="block"]');
      label = clean(
        block?.querySelector('[data-testid="block-info"], .label-wrap, label')?.innerText || ""
      );
    }
    return label;
  };
  const install = () => {
    if (!document.body || document.getElementById("mgd-toast-stack")) return;
    const stack = document.createElement("div");
    stack.id = "mgd-toast-stack";
    stack.setAttribute("aria-live", "polite");
    document.body.appendChild(stack);
    let lastMessage = "";
    let lastAt = 0;
    let fieldTimer = 0;
    const show = (message) => {
      if (!message) return;
      const now = Date.now();
      if (message === lastMessage && now - lastAt < 650) return;
      lastMessage = message;
      lastAt = now;
      const node = document.createElement("div");
      node.className = "mgd-toast";
      node.setAttribute("role", "status");
      const dot = document.createElement("span");
      dot.className = "mgd-toast-dot";
      const text = document.createElement("span");
      text.className = "mgd-toast-text";
      text.textContent = message;
      node.append(dot, text);
      stack.appendChild(node);
      window.requestAnimationFrame(() => node.classList.add("show"));
      window.setTimeout(() => {
        node.classList.remove("show");
        window.setTimeout(() => node.remove(), 220);
      }, 2600);
    };
    const handleField = (el) => {
      if (!el || !el.matches("input, textarea, select") || el.type === "file") return;
      const label = labelFor(el);
      let value = clean(el.value);
      if (el.type === "checkbox") value = el.checked ? "开启" : "关闭";
      if (el.type === "radio") value = clean(el.value);
      if (!value) return;
      show("已更新" + (label ? "：" + label + " = " : "：") + value);
    };
    const notifyOption = (event) => {
      const option = event.target.closest('[role="option"]');
      if (!option) return;
      const chosen = clean(option.innerText).replace(/^✓\s*/, "");
      if (chosen) show("已选择：" + chosen);
    };
    document.addEventListener("pointerdown", notifyOption, true);
    document.addEventListener("click", (event) => {
      if (event.target.closest('[role="option"]')) {
        notifyOption(event);
        return;
      }
      const button = event.target.closest("button");
      if (!button || button.disabled) return;
      const text = clean(button.innerText);
      if (!text || text === "↺" || text.length > 48) return;
      if (text.includes("▼") || text.includes("▲")) return;
      if (["设置", "通过 API 使用", "Stop Recording"].includes(text)) return;
      show((button.getAttribute("role") === "tab" ? "已切换：" : "已提交：") + text);
    }, true);
    document.addEventListener("change", (event) => handleField(event.target), true);
    document.addEventListener("input", (event) => {
      const el = event.target;
      if (!el || !el.matches("input, textarea, select") || el.type === "file") return;
      window.clearTimeout(fieldTimer);
      fieldTimer = window.setTimeout(() => handleField(el), 180);
    }, true);
  };
  if (document.body) install();
  else document.addEventListener("DOMContentLoaded", install, { once: true });
})();
"""


def _theme(gr):
    """深色金属主题：同时设置 Gradio 主题 token，避免组件内部残留白底。"""
    try:
        base = gr.themes.Base(
            primary_hue=gr.themes.colors.slate,
            secondary_hue=gr.themes.colors.gray,
            neutral_hue=gr.themes.colors.zinc,
        )
    except Exception:
        return gr.themes.Base()
    dark_tokens = dict(
        body_background_fill="#12151a",
        background_fill_primary="#181c22",
        background_fill_secondary="#1e242c",
        block_background_fill="#181c22",
        block_label_background_fill="#181c22",
        input_background_fill="#12151a",
        body_text_color="#d7dde4",
        body_text_color_subdued="#8d99a6",
        border_color_primary="#2a323c",
        input_border_color="#303944",
        button_secondary_background_fill="#232a33",
        button_secondary_text_color="#d7dde4",
        table_even_background_fill="#181c22",
        table_odd_background_fill="#1e242c",
    )
    try:
        return base.set(**dark_tokens)
    except TypeError:
        return base


def _launch_style_kwargs(gr) -> dict:
    """Gradio 6 起 theme/css 必须传给 launch()，旧版本仍在 Blocks 构造器上。

    以 launch 签名检测，避免依赖版本号字符串比较。
    """
    import inspect

    try:
        params = inspect.signature(gr.Blocks.launch).parameters
    except (TypeError, ValueError):
        return {}
    if "css" in params:
        kwargs = {"css": CSS, "theme": _theme(gr)}
        if "head" in params:
            # Gradio 6 的 css 只作用于应用容器，视口两侧仍可能是白底；用 head 补齐全局底色
            kwargs["head"] = (
                "<style>html,body{background:#12151a !important;margin:0;}"
                "body{color:#e6e9ee;}</style>"
                "<script>" + TOAST_JS + "</script>"
            )
        return kwargs
    return {}


def _empty_metrics() -> list[list[str]]:
    return [["提示", "尚未推理"]]


def _candidate_map(settings: Settings) -> dict[str, tuple[str, Path | None]]:
    options: dict[str, tuple[str, Path | None]] = {}
    for candidate in discover_weights(settings):
        options[candidate.label] = (candidate.arch, candidate.path)
    for arch in MODEL_NAMES:
        if not any(value[0] == arch for value in options.values()):
            options[f"{arch} · 未训练"] = (arch, None)
    return options


def _resolve_selection(label: str, settings: Settings) -> tuple[str, Path | None]:
    # 未知 label 显式报错：静默回退到未训练的 MGD-Former 会让用户误以为在用已训练权重
    resolved = _candidate_map(settings).get(label)
    if resolved is None:
        raise ValueError(f"未知权重选项：{label!r}（权重目录可能已变化，请刷新页面）")
    return resolved


def _scope_key(label: str) -> str:
    text = str(label)
    if text.startswith("Val"):
        return "val"
    if text.startswith("Train"):
        return "train_self"
    return "test"


_WEIGHT_STATUS_CACHE: dict[tuple[str, str], tuple[int, object]] = {}


def _cached_weight_status(path: Path, arch: str, settings: Settings):
    """manifest 校验含 sha256，按 (路径, 架构) 缓存并只在 mtime 变化时重算（缓存有界）。"""
    key = (str(path), str(arch))
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError as exc:
        raise MGDError(f"权重文件不可访问：{path}", hint=str(exc)) from exc
    cached = _WEIGHT_STATUS_CACHE.get(key)
    if cached is not None and cached[0] == mtime_ns:
        return cached[1]
    status = validate_weight(path, arch, settings=settings)
    _WEIGHT_STATUS_CACHE[key] = (mtime_ns, status)  # 每个路径只保留最新 mtime，无界增长风险
    return status


def _fmt(value, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def _num_text(value, digits: int = 2) -> str:
    """数值 → 文本；None / NaN / Inf 一律 N/A（绝不拿 0 冒充测量值）。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(number):
        return "N/A"
    return f"{number:.{digits}f}"


def _qc_rows(report) -> list[list[str]]:
    """QualityReport → 展示表格：得分 / 触发项 / 理由 / 逐项原始量 / 阈值快照。"""
    rows = [["项", "数值"]]
    rows.append(["质控得分", f"{float(report.score):.1f} / 100"])
    rows.append(["触发缺陷", "、".join(report.flags) if report.flags else "无"])
    for index, reason in enumerate(report.reasons, start=1):
        rows.append([f"理由 {index}", str(reason)])
    for key, value in report.metrics.items():
        rows.append([
            f"原始量 · {key}",
            "N/A（不可测）" if value is None else f"{float(value):.4g}",
        ])
    for key, value in report.thresholds.items():
        rows.append([f"阈值 · {key}", f"{float(value):g}"])
    return rows


def _numeric_series(df, column: str) -> np.ndarray:
    """DataFrame 列 → 有限数值数组；列缺失 / 全不可解析返回空数组。"""
    if df is None or column not in getattr(df, "columns", []):
        return np.empty(0, dtype=np.float64)
    collected: list[float] = []
    for value in df[column].tolist():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            collected.append(number)
    return np.asarray(collected, dtype=np.float64)


def _truncation_ratio(instance_df) -> float:
    """实例表「是否截断」列 → 截断占比；列缺失 / 全空返回 NaN。"""
    if instance_df is None or "是否截断" not in getattr(instance_df, "columns", []):
        return float("nan")
    values = [str(value).strip() for value in instance_df["是否截断"].tolist()]
    if not values:
        return float("nan")
    flagged = sum(1 for value in values if value in ("是", "True", "true", "1"))
    return flagged / float(len(values))


#: ghost 判定的低置信概率带与最小连通域面积（与 atlas.typology.DEFAULT_THRESHOLDS["ghost"] 同口径）
_GHOST_PROB_LOW = 0.2
_GHOST_PROB_HIGH = 0.5
_GHOST_MIN_AREA_PX = 120


def _ghost_area_px(prob_map, binary_mask) -> float:
    """幻影腺体面积（px）：0.2<p<0.5 且不属于确定性腺体掩膜的连通域总面积。

    低置信像素先剔除 binary_mask，再做连通域标记；面积 ≥ 120 px 的连通域求和。
    输入缺失 / 形状不一致 → NaN（不判定）；没有任何低置信像素 → 0.0（真实测量值为 0）。
    """
    prob = np.asarray(prob_map)
    binary = np.asarray(binary_mask)
    if prob.ndim != 2 or binary.ndim != 2 or prob.shape != binary.shape:
        return float("nan")
    low_conf = (prob > _GHOST_PROB_LOW) & (prob < _GHOST_PROB_HIGH) & (binary == 0)
    if not low_conf.any():
        return 0.0
    import cv2

    num, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        low_conf.astype("uint8"), connectivity=8
    )
    total = 0
    for label in range(1, int(num)):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= _GHOST_MIN_AREA_PX:
            total += area
    return float(total)


def _width_profile_std(binary_mask) -> float:
    """骨架宽度剖面标准差（beading 判据的分子）；宽度序列无效 → NaN。"""
    try:
        profile = width_profile(binary_mask)
    except Exception:
        return float("nan")
    arr = np.asarray(profile, dtype=np.float64).ravel()
    if arr.size == 0:
        return float("nan")
    return float(np.std(arr))


def _min_inter_instance_distance_px(binary_mask) -> float:
    """实例质心两两最小欧氏距离（px，confluence 判据）；实例 < 2 或掩膜为空 → NaN。

    instance_df 当前无质心列（INSTANCE_COLUMNS 口径），统一从 binary_mask 用
    ``longitudinal.match.instance_centroids`` 提取质心后计算。
    """
    try:
        centroids = instance_centroids(binary_mask)
    except Exception:
        return float("nan")
    if len(centroids) < 2:
        return float("nan")
    points = np.asarray(list(centroids.values()), dtype=np.float64)
    diff = points[:, None, :] - points[None, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())


def _atlas_case_features(result, instance_df, ga_percent) -> dict:
    """最近诊断结果 → 图谱特征 dict（键名对齐 atlas.FEATURE_NAMES + 分型工程量）。

    缺失维度一律不进 dict（case_vector 补 NaN），绝不用 0 冒充测量值。
    除实例表统计外，还从 prob_map / binary_mask 生产 ghost_area_px（幻影）、
    width_profile_std（串珠）、min_inter_instance_distance_px（融合）三个
    分型判据，避免 ghost / beading / confluence 恒为 NaN。
    """
    features: dict = {}

    def put(key: str, value) -> None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        if np.isfinite(number):
            features[key] = number

    binary_mask = getattr(result, "binary_mask", None)
    put("ga_percent", ga_percent)
    put("break_count", break_count(binary_mask))
    if instance_df is not None and len(instance_df) > 0:
        put("instance_count", float(len(instance_df)))
        widths = _numeric_series(instance_df, "短轴(px)")
        if widths.size:
            mean_width = float(widths.mean())
            put("width_mean_px", mean_width)
            if widths.size > 1 and mean_width > 0:
                put("width_cv", float(widths.std() / mean_width))
        # TI = τ − 1（Pult & Nichols 弧长/弦长口径；τ 本身不是 TI，不能原值入特征）
        tortuosity = _numeric_series(instance_df, "弯曲度τ")
        if tortuosity.size:
            put("tortuosity_index", float(tortuosity.mean()) - 1.0)
        curvature = _numeric_series(instance_df, "平均曲率κ")
        if curvature.size:
            put("mean_curvature", float(curvature.mean()))
        ratio = _truncation_ratio(instance_df)
        put("truncation_ratio", ratio)
    # 图谱特征补全（ghost / beading / confluence 三条规则的生产者）
    put("ghost_area_px", _ghost_area_px(getattr(result, "prob_map", None), binary_mask))
    put("width_profile_std", _width_profile_std(binary_mask))
    put("min_inter_instance_distance_px", _min_inter_instance_distance_px(binary_mask))
    return features


def _visit_label(entry: dict) -> str:
    """就诊记录 → 下拉框标签（#id 可反解）。"""
    return f"#{entry.get('id')} · {entry.get('visit_date') or '—'} · {entry.get('eye') or '—'}"


def _visit_id_from_label(label) -> int | None:
    match = re.match(r"^#(\d+)", str(label or ""))
    return int(match.group(1)) if match else None


def _pair_ecc_note(base_visit: dict | None, follow_visit: dict | None) -> str:
    """两次就诊影像都存在时做 ECC 配准，返回展示用说明文本。"""
    base_path = str((base_visit or {}).get("image_path") or "")
    follow_path = str((follow_visit or {}).get("image_path") or "")
    if not base_path or not follow_path:
        return "两次就诊未同时登记影像路径，跳过 ECC 配准。"
    base_file, follow_file = Path(base_path), Path(follow_path)
    if not base_file.is_file() or not follow_file.is_file():
        return (
            f"影像文件缺失（{base_file.name} / {follow_file.name}），跳过 ECC 配准。"
        )
    try:
        import cv2

        base_gray = cv2.imread(str(base_file), cv2.IMREAD_GRAYSCALE)
        follow_gray = cv2.imread(str(follow_file), cv2.IMREAD_GRAYSCALE)
        if base_gray is None or follow_gray is None:
            return "影像读取失败，跳过 ECC 配准。"
        if base_gray.shape != follow_gray.shape:
            return (
                f"影像尺寸不一致（{base_gray.shape[::-1]} vs {follow_gray.shape[::-1]}），"
                "跳过 ECC 配准。"
            )
        outcome = register_pair(base_gray, follow_gray)
    except Exception as exc:
        return f"ECC 配准失败：{exc}"
    score_text = _fmt(outcome.get("ecc_score"), 4)
    if outcome.get("ok"):
        return f"**ECC 配准成功**：score={score_text}（{outcome.get('note')}）"
    return f"**ECC 配准未达可信阈值**：score={score_text}（{outcome.get('note')}）"


def _editor_ink_mask(editor_value) -> np.ndarray | None:
    """gr.ImageEditor 提交值 → 前景布尔掩膜；画布无内容 → None。

    gradio 6.16 的 ImageEditor 预处理返回 ``{"background": ndarray|None,
    "layers": [ndarray, ...], "composite": ndarray|None}``，且默认
    ``image_mode="RGBA"``，通道顺序为 R,G,B,A：
    - 画布存在透明区域（未铺不透明底图）时，笔迹 / 图层墨水的 alpha>0 即前景，
      直接取 alpha 通道（此时墨水 RGB 可能是黑的，按亮度 >0 会漏判）；
    - 底图不透明（alpha 全 255，例如医生上传了 JPG 掩膜）时退化为亮度 >0
      （黑底白前景惯例）；
    - composite 缺失时按「最后一个图层 → background」兜底（用户未触碰画布
      就提交的场景）。
    """
    if not isinstance(editor_value, dict):
        return None
    layers = [item for item in (editor_value.get("layers") or []) if item is not None]
    source = editor_value.get("composite")
    if source is None:
        source = layers[-1] if layers else editor_value.get("background")
    if source is None:
        return None
    arr = np.asarray(source)
    if arr.ndim == 2:
        return arr > 0
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return None
    if arr.shape[-1] >= 4:
        alpha = arr[..., 3]
        if bool((alpha < 255).any()):
            return alpha > 0
    return arr[..., :3].max(axis=-1) > 0


def create_app(settings: Settings | None = None):
    """构建 Gradio Blocks。仅构造，不启动。"""
    import gradio as gr
    from packaging.version import Version

    if Version(gr.__version__) < Version("6.16.0"):
        raise RuntimeError(
            "Gradio >= 6.16.0 is required: 6.11–6.15 freeze the browser on "
            "Tab switches that contain a Dataframe. Run install_dependencies.bat "
            "or pip install -U 'gradio==6.16.0'."
        )

    settings = settings or get_settings()
    configure_logging(log_file=settings.log_file)
    device_text = describe_device(resolve_device(settings.device))
    vram_gb = detect_vram_gb()
    service = ModelService(settings, allow_unverified=True)
    candidates = _candidate_map(settings)
    default_label = next(iter(candidates))
    presets = load_presets(settings)
    last_report: dict[str, EvalReport] = {}
    last_result: dict = {}
    last_compare: dict[str, list] = {}

    # ================= 诊断 =================
    def infer_fn(image, selection, tta, tile_infer, tile_size, overlap, threshold, min_area,
                 smooth_on, sigma, roi_only, auto_escalate, gradcam):
        if image is None:
            return (None, None, None, None, None, None, None, _empty_metrics(),
                    "**请先上传红外影像（或拖入 640×1280 灰度图）。**")
        arch, path = _resolve_selection(selection, settings)
        options = InferenceOptions(
            tile_infer=bool(tile_infer), tile_size=int(tile_size), tile_overlap=float(overlap),
            threshold=float(threshold), min_area=int(min_area), smooth_on=bool(smooth_on),
            smooth_sigma=float(sigma), roi_only=bool(roi_only), tta=str(tta),
            auto_escalate_threshold=bool(auto_escalate), gradcam=bool(gradcam),
        )
        try:
            import torch

            started = time.perf_counter()
            with WORK_LOCK:
                # reset 必须在锁内：排在后面的请求在锁外 reset 会清掉前一次的峰值统计
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                result = service.predict(image, arch=arch, options=options, weights_path=path)
            elapsed = time.perf_counter() - started
            peak = 0.0
            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        except MGDError as exc:
            return (None, None, None, None, None, None, None, _empty_metrics(), exc.user_message())
        except Exception as exc:
            return (None, None, None, None, None, None, None, _empty_metrics(), f"推理失败：{exc}")

        last_result.clear()
        last_result.update({"result": result, "arch": arch, "path": path, "options": options})
        extra = result.extra
        status = [
            f"**设备**：{extra['device']}　**模型**：{extra['arch']}　"
            f"**任务**：{'双任务（腺体+眼睑）' if extra.get('dual_task') else '单任务（旧版权重兼容）'}",
            f"**阈值**：用户 {result.requested_threshold:.2f} / 实际 {result.applied_threshold:.2f}"
            + ("（已自动抬升）" if result.threshold_escalated else "（未改动）"),
            f"**耗时**：{elapsed * 1000:.0f} ms / 例　"
            f"**显存峰值**：{(f'{peak:.3f} GB（本次推理，不含已加载模型）' if peak else 'CPU 模式，不适用')}",
            f"**分块参数**：{'开启' if tile_infer else '关闭'}　"
            f"{int(tile_size)}×{int(tile_size) * 2}　重叠 {float(overlap):.2f}　"
            f"TTA {tta}",
        ]
        if path is None:
            status.append("**权重状态**：<span class='mgd-warn'>未训练（随机初始化）</span> —— 结果只能用于界面验证，不可用于论文或临床判断。")
        else:
            weight_status = _cached_weight_status(path, arch, settings)
            if weight_status.verified:
                status.append(f"**权重状态**：<span class='mgd-ok'>已验证</span>（{Path(path).name}）")
            else:
                status.append(
                    f"**权重状态**：<span class='mgd-warn'>verified=false</span> —— {weight_status.reason}"
                )
        status.append(
            f"**AI 置信度**：{extra['confidence']['score']:.1f}/100　"
            f"模糊区 {extra['confidence']['ambiguous_ratio']:.2%}"
        )
        status.append("**口径**：形态学量化基于像素未标定的比例，仅作参考。")
        if result.warning:
            status.append(f"<span class='mgd-warn'>⚠ {result.warning}</span>")
        return (
            result.overlay, result.heatmap, result.contour, result.skeleton,
            result.instances, extra.get("uncertainty_image"), extra.get("gradcam"),
            result.metrics, "\n\n".join(status),
        )

    def report_fn():
        if not last_result:
            return None, "请先完成一次分析。"
        payload = last_result["result"]
        options = last_result.get("options") or InferenceOptions()

        # v2.4.0 扩展区块（provenance / 分级 / 形态分型）：任何一个计算失败 →
        # warning 并丢弃该区块，报告主功能永不因扩展失败而挂；quality / explain
        # 属于质控与解释性页签的独立状态，这里不传（None → 区块不渲染）。
        provenance = grading = typology = None
        try:
            provenance = collect_provenance(
                model_name=last_result["arch"],
                threshold=float(options.threshold),
                tile_size=int(options.tile_size) if options.tile_infer else None,
                tile_overlap=float(options.tile_overlap) if options.tile_infer else None,
                tta=str(options.tta),
                weights_path=last_result.get("path"),
            )
        except Exception as exc:
            logger.warning("报告 Provenance 区块生成失败：%s", exc)
        extra = getattr(payload, "extra", None) or {}
        clinical = extra.get("clinical")
        try:
            loss_value = float(
                getattr(getattr(clinical, "rloss", None), "r_loss_percent", float("nan"))
            )
        except (TypeError, ValueError):
            loss_value = float("nan")
        if clinical is not None and np.isfinite(loss_value):
            try:
                grading = grade_all(loss_value)
            except Exception as exc:
                logger.warning("报告分级区块生成失败：%s", exc)
        instance_df = extra.get("instance_df")
        if instance_df is not None and len(instance_df) > 0:
            try:
                typology = classify_case(
                    instance_df, _atlas_case_features(payload, instance_df, loss_value)
                )
            except Exception as exc:
                logger.warning("报告形态分型区块生成失败：%s", exc)
        try:
            bundle = build_report(
                payload,
                model_name=last_result["arch"],
                patient_id="case",
                grading=grading,
                typology=typology,
                provenance=provenance,
            )
        except Exception as exc:
            return None, f"报告生成失败：{exc}"
        return str(bundle.html_path), (
            f"HTML：{bundle.html_path.name}　CSV：{bundle.csv_path.name}　{bundle.pdf_note}"
        )

    # ================= 训练 =================
    def preset_fill(profile_name):
        """套用中层预设：把预设值填进表单，用户仍可继续改。"""
        overrides = profile_overrides(profile_name)
        train = overrides.get("train", {})
        label = TRAIN_PROFILES.get(profile_name, {}).get("label", profile_name)
        note = (
            f"**已套用 {label}**：{TRAIN_PROFILES.get(profile_name, {}).get('description', '')}\n\n"
            "> 表单值仍可修改；点「开始训练」时以表单为准。"
        )
        return (
            int(train.get("epochs", 20)),
            float(train.get("lr", 1e-4)),
            train.get("scheduler", "Cosine Annealing"),
            note,
        )

    def train_fn(selection, profile_name, vram, epochs, batch_size, lr, optimizer, scheduler,
                 loss_name, split_mode, fold, test_ratio, dual_task, eyelid_weight, tile_mode,
                 tile_height, augmentation, amp, val_ratio, pretrained, deep_supervision,
                 grad_accum, channels_last, cldice_weight, ema_decay, resume_from, seed,
                 publish_best, export_report):
        arch, _ = _resolve_selection(selection, settings)
        overrides = {
            "run": {
                "vram_tier": str(vram), "amp": bool(amp), "seed": int(seed),
                "grad_accum_steps": int(grad_accum), "channels_last": bool(channels_last),
            },
            "data": {
                "split_mode": "kfold" if str(split_mode).startswith("K") else "holdout",
                "fold": int(fold), "test_ratio": float(test_ratio), "val_ratio": float(val_ratio),
                "tile_mode": bool(tile_mode), "tile_height": int(tile_height),
                "augmentation": bool(augmentation),
            },
            "model": {
                "name": arch, "dual_task": bool(dual_task), "eyelid_weight": float(eyelid_weight),
                "pretrained": str(pretrained), "deep_supervision": bool(deep_supervision),
                "cldice_weight": float(cldice_weight), "ema_decay": float(ema_decay),
            },
            "train": {
                "epochs": int(epochs), "batch_size": int(batch_size), "lr": float(lr),
                "optimizer": optimizer, "scheduler": scheduler, "loss": loss_name,
                "resume_from": str(resume_from).strip() or None,
            },
            "artifacts": {
                "publish_best": bool(publish_best),
                "export_reference_report": bool(export_report),
            },
        }
        try:
            config, meta = build_train_config(overrides=overrides)
            config, tier_info = apply_tier(config, vram, vram_gb)
            config.validate()
        except (MGDError, ValueError) as exc:
            yield f"<span class='mgd-fail'>配置错误：{exc}</span>", [], None, ""
            return

        rows: list[list] = []
        lines = [
            f"**模型**：{arch}　**显存档**：{tier_info['tier']} GB（{tier_info['status']}）",
            f"**分块**：{tier_info['tile']}　**batch**：{tier_info['batch_size']}×累积 "
            f"{tier_info['grad_accum_steps']}　**AMP**：{'开' if tier_info['amp'] else '关'}",
            f"**档位依据**：{tier_info['reason']}",
        ]
        if tier_info["status"] == "估算":
            lines.append("> 该档位为估算值，尚未在本机实测；首轮请关注显存峰值。")

        queue: Queue = Queue()
        box: dict = {}

        def worker():
            try:
                # 训练与推理/评估共用 GPU：必须持同一把 WORK_LOCK，否则三者并发争抢显存
                with WORK_LOCK:
                    result, context = run_training(
                        config, meta, settings, on_event=lambda event: queue.put(("event", event))
                    )
                    box["result"] = result
                    box["context"] = context
                queue.put(("done", None))
            except Exception as exc:
                box["error"] = exc
                queue.put(("error", None))
            finally:
                try:
                    # 训练结束后释放模型缓存：旧 mtime 的模型不再复用，显存归还
                    service.release()
                except Exception:
                    pass
                queue.put(("end", None))

        threading.Thread(target=worker, daemon=True).start()

        while True:
            try:
                kind, payload = queue.get(timeout=0.2)
            except Empty:
                # 让出控制权，保持界面可交互
                yield "\n\n".join(lines), rows, None, ""
                continue
            if kind == "end":
                break
            if kind == "error":
                lines.append(f"<span class='mgd-fail'>训练失败：{box.get('error')}</span>")
                yield "\n\n".join(lines), rows, None, ""
                break
            if kind != "event":
                continue
            event = payload
            if event["event"] == "start":
                lines.append(
                    f"**设备**：{event['device']}　训练 {event['train_samples']} / 验证 {event['val_samples']}"
                )
                lines.extend(f"- {item}" for item in event["warnings"])
                yield "\n\n".join(lines), rows, None, ""
            elif event["event"] == "epoch":
                rows.append([
                    event["epoch"],
                    _fmt(event["loss"]),
                    _fmt(event["dice"]),
                    _fmt(event["val_dice"]),
                    _fmt(event.get("val_eyelid_dice")),
                    f"{event['lr']:.2e}",
                    _fmt(event["best"]),
                    _fmt(event.get("samples_per_second"), 2),
                ])
                curve = _curve_image(
                    [row[0] for row in rows], [row[3] for row in rows],
                    [row[2] for row in rows],
                )
                yield "\n\n".join(lines), rows, curve, ""
            elif event["event"] == "done":
                result = event["result"]
                lines.append(
                    f"**完成**：{result.epochs_run} 轮，最佳验证指标 {_fmt(result.best_metric)}"
                    f"（epoch {result.best_epoch}）"
                )
                lines.append(f"**吞吐**：{_fmt(result.samples_per_second, 2)} 样本/秒　"
                             f"**显存峰值**：{_fmt(result.peak_vram_gb, 3)} GB（含模型）")
                lines.append(f"**评估口径**：{result.scope_label}")
                lines.extend(f"- {item}" for item in result.warnings)
        context = box.get("context")
        run_dir = str(context.run_dir) if context else ""
        yield "\n\n".join(lines), rows, _curve_image(
            [row[0] for row in rows], [row[3] for row in rows], [row[2] for row in rows]
        ), run_dir

    def _curve_image(epochs, val_dice, train_dice):
        if not epochs:
            return None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axis = plt.subplots(figsize=(5.2, 2.8), dpi=110)
            fig.patch.set_facecolor("#181c22")
            axis.set_facecolor("#181c22")
            axis.plot(epochs, [_safe_float(v) for v in train_dice], color="#8fa9bd", linewidth=1.2,
                      label="train dice")
            axis.plot(epochs, [_safe_float(v) for v in val_dice], color="#6f9c7f", linewidth=1.6,
                      label="val dice")
            axis.tick_params(colors="#8d99a6", labelsize=8)
            for spine in axis.spines.values():
                spine.set_color("#2a323c")
            axis.grid(color="#232a33", linewidth=.6)
            axis.legend(facecolor="#181c22", edgecolor="#2a323c", labelcolor="#d7dde4", fontsize=8)
            axis.set_xlabel("epoch", color="#8d99a6", fontsize=8)
            fig.tight_layout()
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            plt.close(fig)
            return image
        except Exception:
            return None

    def _safe_float(value) -> float:
        try:
            result = float(value)
            return result if np.isfinite(result) else float("nan")
        except (TypeError, ValueError):
            return float("nan")

    # ================= 评估 =================
    def eval_fn(selection, scope_label):
        try:
            arch, path = _resolve_selection(selection, settings)
        except ValueError as exc:
            return _empty_metrics(), _empty_metrics(), str(exc)
        scope = _scope_key(scope_label)
        try:
            index = scan_dataset(settings)
            split = get_splits(index.pairs, SplitSpec(), settings)
        except MGDError as exc:
            return _empty_metrics(), _empty_metrics(), exc.user_message()
        pairs = {"test": split.test, "val": split.val, "train_self": split.train}[scope]
        if not pairs:
            return _empty_metrics(), _empty_metrics(), f"{scope_label} 为空，无法评估。"
        try:
            with WORK_LOCK:
                report = evaluate_pairs(pairs, arch=arch, weights_path=path, settings=settings,
                                        scope=scope)
        except MGDError as exc:
            return _empty_metrics(), _empty_metrics(), exc.user_message()
        last_report["report"] = report
        per_sample = [
            [item.stem, _fmt(item.metrics.dice), _fmt(item.metrics.iou), _fmt(item.metrics.hd95),
             _fmt(item.metrics.assd), _fmt(item.metrics.precision), _fmt(item.metrics.recall),
             _fmt(item.cldice), _fmt(item.r_loss_pred, 2),
             item.meiboscore_pred if item.meiboscore_pred >= 0 else "N/A",
             " | ".join(item.metrics.errors) or "-"]
            for item in report.per_sample
        ]
        note = [f"**评估口径**：{report.scope_label}", f"**权重**：{Path(report.weights).name}"]
        if report.is_self_evaluated:
            note.append("> <span class='mgd-warn'>训练集自评，不代表泛化性能，不能直接写进论文表格。</span>")
        note.extend(f"- {item}" for item in report.errors[:3])
        return summary_rows(report), per_sample, "\n".join(note)

    def export_fn(kind):
        report = last_report.get("report")
        if report is None:
            return "请先完成一次评估。"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = (export_csv if kind == "csv" else export_latex)(
            report, settings.output_dir / f"eval_{stamp}.{kind if kind == 'csv' else 'tex'}"
        )
        return f"已导出：{path}"

    # ================= 实验管理 =================
    def refresh_runs():
        summaries = list_runs(settings)
        labels = [f"{item['run_id']} · {item['arch'] or '?'}" for item in summaries]
        return gr.update(choices=labels, value=[label for label in labels[:3]]), (
            f"共 {len(summaries)} 个 run。目录：{settings.output_dir / 'runs'}"
        )

    def compare_fn(labels):
        ids = [str(label).split(" · ")[0] for label in (labels or [])]
        summaries = compare_runs(ids, settings)
        last_compare["summaries"] = summaries
        rows = compare_rows(summaries)
        return rows, _compare_curves(summaries), f"对比 {len(summaries)} 个 run"

    def _compare_curves(summaries):
        if not summaries:
            return None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axis = plt.subplots(figsize=(6.0, 3.0), dpi=110)
            fig.patch.set_facecolor("#181c22")
            axis.set_facecolor("#181c22")
            palette = ["#8fa9bd", "#6f9c7f", "#b09a6a", "#b0776f", "#9d8fb5", "#7fa3a8"]
            for index, item in enumerate(summaries[:6]):
                records = metric_series(item["path"])
                if not records:
                    continue
                epochs = [record.get("epoch") for record in records]
                values = [_safe_float(record.get("val_dice")) for record in records]
                axis.plot(epochs, values, linewidth=1.4, color=palette[index % len(palette)],
                          label=item["run_id"][:22])
            axis.tick_params(colors="#8d99a6", labelsize=8)
            for spine in axis.spines.values():
                spine.set_color("#2a323c")
            axis.grid(color="#232a33", linewidth=.6)
            axis.legend(facecolor="#181c22", edgecolor="#2a323c", labelcolor="#d7dde4", fontsize=7)
            axis.set_xlabel("epoch", color="#8d99a6", fontsize=8)
            axis.set_ylabel("val dice", color="#8d99a6", fontsize=8)
            fig.tight_layout()
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            plt.close(fig)
            return image
        except Exception:
            return None

    def export_compare(kind):
        summaries = last_compare.get("summaries") or []
        if not summaries:
            return "请先对比至少一个 run。"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = settings.output_dir / f"runs_compare_{stamp}.{kind if kind == 'csv' else 'tex'}"
        path = (export_comparison_csv if kind == "csv" else export_comparison_latex)(summaries, target)
        return f"已导出：{path}"

    # ================= 数据集 =================
    def dataset_check():
        try:
            report = integrity_report(settings)
        except Exception as exc:
            return _empty_metrics(), f"校验失败：{exc}"
        detail = [
            f"**目录**：`{settings.images_dir}` / `{settings.gland_masks_dir}` / `{settings.eyelid_masks_dir}`",
            f"**划分**：{report['split']}",
        ]
        if report["missing_gland_masks"]:
            detail.append("**缺腺体掩膜**：" + ", ".join(report["missing_gland_masks"][:10]))
        if report["missing_eyelid_masks"]:
            detail.append("**缺眼睑掩膜**：" + ", ".join(report["missing_eyelid_masks"][:10]))
        if report["duplicate_stems"]:
            detail.append("**重名**：" + ", ".join(report["duplicate_stems"][:10]))
        if report["naming_anomalies"]:
            detail.append("**命名异常**：" + "；".join(report["naming_anomalies"][:8]))
        if report["ok"]:
            detail.append("<span class='mgd-ok'>完整性校验通过。</span>")
        else:
            detail.append("<span class='mgd-fail'>存在缺失/重复，写版本记录会被拒绝。</span>")
        return integrity_rows(report), "\n\n".join(detail)

    def dataset_version(source, device):
        try:
            payload = record_version(settings, source=source or "本地采集", device=device or "未记录")
        except (ValueError, OSError) as exc:
            return f"<span class='mgd-fail'>未写入：{exc}</span>"
        return (
            "**版本已记录**\n\n"
            f"- 配对：{payload['usable_pairs']}\n"
            f"- 哈希：`{payload['hash']['digest'][:20]}…`（{payload['hash']['mode']}）\n"
            f"- 划分签名：`{(payload['split_signature'] or '')[:16]}…`\n"
            f"- 文件：`{settings.data_dir / 'dataset_version.json'}`"
        )

    def dataset_backup():
        try:
            target = backup_dataset(settings)
        except OSError as exc:
            return f"<span class='mgd-fail'>备份失败：{exc}</span>"
        return f"已备份到 `{target}`（只复制，不覆盖原文件）"

    # ================= 系统状态 =================
    def system_report():
        try:
            report = run_doctor(settings)
            return report.as_text()
        except Exception as exc:
            return f"自检失败：{exc}"

    def weights_table():
        rows = [["模型", "权重文件", "状态", "verified", "原因"]]
        for label, (arch, path) in _candidate_map(settings).items():
            if path is None:
                rows.append([arch, "—", "未训练", "否", "没有可用权重，界面用随机初始化验证流程"])
                continue
            status = _cached_weight_status(path, arch, settings)
            rows.append([arch, Path(path).name, status.status,
                         "是" if status.verified else "否", status.reason])
        return rows

    def tiers_view():
        rows = [["档位", "分块", "batch", "梯度累积", "AMP", "channels_last", "深监督", "TTA",
                 "状态", "依据", "显存峰值", "吞吐"]]
        for item in tier_table():
            rows.append([
                item["档位"], item["分块"], item["batch"], item["梯度累积"], item["AMP"],
                item["channels_last"], item["深监督"], item["TTA"], item["状态"], item["依据"],
                item["显存峰值(GB)"], item["吞吐(样本/秒)"],
            ])
        return rows

    # ================= 质控（WP-11，纯 CPU，不碰模型） =================
    def qc_fn(image, eyelid_mask):
        placeholder = [["—", "—"]]
        if image is None:
            gr.Warning("请先上传待检灰度图")
            return placeholder, ""
        try:
            gray = to_gray(np.asarray(image))
            mask = None
            if eyelid_mask is not None:
                mask = to_gray(np.asarray(eyelid_mask))
                if mask.shape != gray.shape:
                    raise ValueError(
                        f"眼睑掩膜尺寸 {mask.shape[::-1]} 与灰度图 {gray.shape[::-1]} 不一致，"
                        "请上传同尺寸掩膜"
                    )
                mask = (mask > 0).astype(np.uint8)
            thresholds = default_thresholds(settings.qc_policy)
            report = assess_quality(gray, eyelid_mask=mask, thresholds=thresholds)
            decision = decide(report, QCPolicy())
        except Exception as exc:
            gr.Warning(f"质控失败：{exc}")
            return placeholder, ""
        action = (
            f"**建议动作**：`{str(decision['action']).upper()}` —— {decision['reason']}"
            f"　（阈值组：qc_policy={settings.qc_policy}）"
        )
        return _qc_rows(report), action

    # ================= 解释性（WP-11，GPU 回调，必须持 WORK_LOCK） =================
    def explain_fn(method, window, stride, max_windows):
        placeholder = [["指标", "数值"]]
        if not last_result:
            gr.Warning("请先在「诊断」页完成一次分析")
            return None, placeholder, "尚无最近诊断。"
        try:
            import torch

            result = last_result["result"]
            arch = last_result["arch"]
            weights_path = last_result.get("path")
            options = last_result.get("options") or InferenceOptions()
            started = time.perf_counter()
            with WORK_LOCK:
                gray = to_gray(result.image_rgb)
                pre = preprocess_array(gray, settings)
                tensor = torch.from_numpy(np.asarray(pre.model_input))[
                    None, None, :, :
                ].float().to(service.device)
                prob_map = np.asarray(result.prob_map)
                if str(method) == "遮挡敏感度":

                    def predict_fn(img):
                        probability, _ = service.probability(
                            img, arch=arch, options=options, weights_path=weights_path
                        )
                        return probability

                    config = OcclusionConfig(
                        window=max(1, int(window)),
                        stride=max(1, int(stride)),
                        max_windows=max(1, int(max_windows)),
                    )
                    payload = occlusion_map(predict_fn, pre.resized, config)
                    attribution = np.asarray(payload["map"], dtype=np.float32)
                    note = (
                        f"遮挡敏感度（窗口 {payload['windows']} 个，stride 缩放 "
                        f"{payload['scale']:.2f}；{payload['note']}）"
                    )
                else:
                    model = service.load(arch, weights_path)
                    if str(method) == "Score-CAM":
                        attribution = np.asarray(
                            ScoreCAM(model, max_channels=32).generate(tensor), dtype=np.float32
                        )
                        note = "Score-CAM（激活掩膜前向打分，max_channels=32）"
                    else:
                        attribution = np.asarray(
                            LayerCAM(model).generate(tensor), dtype=np.float32
                        )
                        note = "LayerCAM（逐通道 relu(梯度) 空间加权）"
                if attribution.shape != prob_map.shape:
                    import cv2

                    attribution = cv2.resize(
                        attribution, (prob_map.shape[1], prob_map.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    ).astype(np.float32)
                faith = faithfulness_report(prob_map, attribution)
            elapsed = time.perf_counter() - started
        except Exception as exc:
            gr.Warning(f"归因生成失败：{exc}")
            return None, placeholder, "归因失败，详见提示。"
        rows = [
            ["deletion AUC（越低越好）", _fmt(faith["deletion_auc"])],
            ["insertion AUC（越高越好）", _fmt(faith["insertion_auc"])],
            ["road（insertion − deletion）", _fmt(faith["road"])],
            ["随机对照 deletion AUC", _fmt(faith["random_deletion_auc"])],
            ["随机对照 insertion AUC", _fmt(faith["random_insertion_auc"])],
            ["verdict", str(faith["verdict"])],
        ]
        status = f"**方法**：{note}　**耗时**：{elapsed * 1000:.0f} ms"
        return attribution, rows, status

    # ================= 形态图谱（WP-11，纯 CPU：掩膜/表格计算，不碰模型） =================
    def atlas_fn():
        type_placeholder = [["分型", "触发", "实测值", "阈值", "判定依据"]]
        similar_placeholder = [["病例", "余弦距离", "元数据"]]
        if not last_result:
            gr.Warning("请先在「诊断」页完成一次分析")
            return type_placeholder, similar_placeholder, ""
        try:
            result = last_result["result"]
            extra = result.extra
            instance_df = extra.get("instance_df")
            clinical = extra.get("clinical")
            ga_percent = getattr(getattr(clinical, "rloss", None), "r_loss_percent", float("nan"))
            case_features = _atlas_case_features(result, instance_df, ga_percent)
            evidence = classify_case(instance_df, case_features)
            names = {rule.key: rule.name_zh for rule in TYPE_RULES}
            type_rows = []
            for item in evidence:
                value_text = _num_text(item.value)
                if item.unit and value_text != "N/A":
                    value_text = f"{value_text} {item.unit}"
                type_rows.append([
                    names.get(item.key, item.key),
                    "是" if item.present else "否",
                    value_text,
                    _num_text(item.threshold),
                    item.definition,
                ])
            query = case_vector(case_features)
            index_path = settings.atlas_index_path
            if index_path is not None and Path(index_path).is_file():
                index = AtlasIndex.load(index_path)
                hits = index.search(query, k=5)
                if hits:
                    similar_rows = [
                        [str(hit.get("case_id", "—")), _fmt(hit.get("distance"), 4),
                         str(hit.get("meta") or "—")]
                        for hit in hits
                    ]
                else:
                    similar_rows = [["（索引为空，无已登记病例）", "—", "—"]]
                note = f"图谱索引：`{Path(index_path)}`"
            else:
                similar_rows = [[f"图谱索引为空（{index_path}）", "—", "—"]]
                note = "未找到图谱索引文件，只能展示分型证据；检索需先构建 AtlasIndex。"
        except Exception as exc:
            gr.Warning(f"形态分型失败：{exc}")
            return type_placeholder, similar_placeholder, ""
        return type_rows, similar_rows, note

    # ================= 随访（WP-11，sqlite 读取，不碰模型） =================
    def timeline_fn(patient_id):
        empty_update = gr.update(choices=[], value=None)
        placeholder = [["就诊 ID", "日期", "眼别", "眼睑", "影像", "备注", "测量项数"]]
        pid = str(patient_id or "").strip()
        if not pid:
            gr.Warning("请输入患者 ID")
            return placeholder, empty_update, empty_update, ""
        db_path = settings.db_path
        if db_path is None or not Path(db_path).is_file():
            return placeholder, empty_update, empty_update, (
                f"暂无随访数据库（将在首次写入时创建：{db_path}）"
            )
        try:
            with ClinicStore(db_path) as store:
                entries = store.timeline(pid)
        except Exception as exc:
            gr.Warning(f"时间线查询失败：{exc}")
            return placeholder, empty_update, empty_update, ""
        if not entries:
            return placeholder, empty_update, empty_update, f"患者 {pid} 暂无就诊记录。"
        rows = []
        labels = []
        for entry in entries:
            rows.append([
                f"#{entry.get('id')}",
                str(entry.get("visit_date") or "—"),
                str(entry.get("eye") or "—"),
                str(entry.get("eyelid") or "—"),
                str(entry.get("image_path") or "—"),
                str(entry.get("note") or "—"),
                str(len(entry.get("measurements") or {})),
            ])
            labels.append(_visit_label(entry))
        base_update = gr.update(choices=labels, value=labels[-2] if len(labels) >= 2 else labels[0])
        follow_update = gr.update(choices=labels, value=labels[-1])
        return rows, base_update, follow_update, f"共 {len(entries)} 次就诊；已默认选中最近两次。"

    def followup_compare_fn(patient_id, base_label, follow_label):
        placeholder = [["测量项", "基线", "随访", "差值（随访 − 基线）"]]
        pid = str(patient_id or "").strip()
        base_id = _visit_id_from_label(base_label)
        follow_id = _visit_id_from_label(follow_label)
        if not pid or base_id is None or follow_id is None:
            gr.Warning("请先查询时间线并选择两次就诊")
            return placeholder, ""
        db_path = settings.db_path
        if db_path is None or not Path(db_path).is_file():
            gr.Warning(f"暂无随访数据库（将在首次写入时创建：{db_path}）")
            return placeholder, ""
        try:
            with ClinicStore(db_path) as store:
                # 患者一致性校验（L-11）：下拉框可能残留上一位患者的就诊标签，
                # 两次所选就诊都必须属于当前输入患者的时间线，否则警告并返回占位。
                visits = {int(entry["id"]): entry for entry in store.list_visits(pid)}
                if base_id not in visits or follow_id not in visits:
                    gr.Warning("所选就诊不属于当前患者的时间线，请重新查询时间线后再对比")
                    return placeholder, ""
                base_measurements = store.measurements_for_visit(base_id)
                follow_measurements = store.measurements_for_visit(follow_id)
        except Exception as exc:
            gr.Warning(f"随访对比失败：{exc}")
            return placeholder, ""
        rows = []
        for key in sorted(set(base_measurements) | set(follow_measurements)):
            base_value = base_measurements.get(key, float("nan"))
            follow_value = follow_measurements.get(key, float("nan"))
            if np.isfinite(base_value) and np.isfinite(follow_value):
                delta_text = f"{follow_value - base_value:+.4g}"
            else:
                delta_text = "N/A"
            rows.append([key, _num_text(base_value), _num_text(follow_value), delta_text])
        if not rows:
            rows = [["（两次就诊均无测量记录）", "—", "—", "—"]]
        return rows, _pair_ecc_note(visits.get(base_id), visits.get(follow_id))

    # ================= 分级（WP-11，纯计算零依赖） =================
    def grade_fn(loss_percent):
        try:
            value = float(loss_percent)
        except (TypeError, ValueError):
            gr.Warning("请输入有效的 R_loss 百分比（0-100 的数字）")
            return [["—", "—"]]
        if not np.isfinite(value) or value < 0 or value > 100:
            gr.Warning("R_loss 应为 0-100 之间的数字")
            return [["—", "—"]]
        return grade_table(value)[1:]  # 表头行已在组件 headers 中

    # ================= 医生修正回流（WP-10 接线；纯 numpy/CPU，不持 GPU 串行锁） =================
    # 审计由 governance 接线覆盖（inference 事件）
    def load_model_mask_fn():
        """把最近一次诊断的模型掩膜铺进修正画布（透明底 + 可编辑白前景图层）。"""
        if not last_result:
            gr.Warning("请先在「诊断」页完成一次分析")
            return None
        try:
            model_mask = np.asarray(last_result["result"].binary_mask)
            layer = np.zeros((model_mask.shape[0], model_mask.shape[1], 4), dtype="uint8")
            layer[model_mask > 0] = (255, 255, 255, 255)
            background = np.zeros_like(layer)
        except Exception as exc:
            gr.Warning(f"读取上次模型掩膜失败：{exc}")
            return None
        # 图层可被医生用橡皮擦除（删除）或笔刷补画（新增）；透明底撑起画布尺寸
        return {"background": background, "layers": [layer], "composite": layer}

    def submit_correction_fn(editor_value, reviewer_id_text, note_text):
        """把画布上的修正掩膜落盘：record_correction + append_record。"""
        if not last_result:
            gr.Warning("请先在「诊断」页完成一次分析")
            return "尚无最近诊断：修正未记录。"
        try:
            ink = _editor_ink_mask(editor_value)
            if ink is None or not bool(ink.any()):
                gr.Warning("请先在画布上绘制修正掩膜（白色笔迹；可先点「载入上次模型掩膜」）")
                return "画布为空：修正未记录。"
            result = last_result["result"]
            model_mask = np.asarray(result.binary_mask)
            edited = ink.astype("uint8")
            if edited.shape != model_mask.shape:
                import cv2

                edited = cv2.resize(
                    edited, (model_mask.shape[1], model_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            case_id = f"ui_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            record = record_correction(
                case_id, model_mask, edited,
                reviewer_id=str(reviewer_id_text or "").strip() or "unknown",
                note=str(note_text or ""),
            )
            row = append_record(record, settings, model_mask=model_mask, edited_mask=edited)
        except Exception as exc:
            gr.Warning(f"修正记录失败：{exc}")
            return "修正未保存，详见提示。"
        gr.Info(f"修正已记录：+{record.added_px} / −{record.removed_px}")
        return (
            f"**修正已记录**（{row['case_id']}）　新增 {record.added_px} px / "
            f"删除 {record.removed_px} px　方向 {record.direction}　"
            f"dice_before {_num_text(record.dice_before)}　"
            f"dice(model↔edit) {_num_text(record.dice_after_vs_edit)}"
        )

    # ================= 界面 =================
    style_kwargs = _launch_style_kwargs(gr)
    blocks_kwargs = {} if "css" in style_kwargs else {"css": CSS, "theme": _theme(gr)}
    with gr.Blocks(title=f"腺而易见 MGD 智能辅助诊疗平台 v{__version__}", **blocks_kwargs) as demo:
        gr.HTML(
            f"""
            <div id="mgd-header">
              <div id="mgd-brand"><div id="mgd-mark"></div>
                <div><div id="mgd-title">腺而易见 · MGD 智能辅助诊疗平台</div>
                     <div id="mgd-sub">睑板腺红外影像分析与学术训练工作台</div></div>
              </div>
              <div id="mgd-chips">
                <span class="mgd-chip">v{__version__}</span>
                <span class="mgd-chip">{device_text}</span>
                <span class="mgd-chip">已训练：{', '.join(list_trained_models(settings)) or '无'}</span>
              </div>
            </div>
            """
        )

        with gr.Tabs():
            # -------- 诊断 -------------
            with gr.Tab("诊断"):
                gr.Markdown(
                    "<div class='mgd-help'>上传 640×1280 红外影像后开始分析。"
                    "结果区会显式标注模型状态（含 verified=false 的原因）、耗时、显存峰值与分块参数；"
                    "未验证权重只能用于流程验证。</div>"
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        image_input = gr.Image(type="numpy", label="红外影像")
                        model_dropdown = gr.Dropdown(choices=sorted(candidates), value=default_label,
                                                     label="模型 / 权重")
                        tta_radio = gr.Radio(
                            choices=_TTA_CHOICES, value=TTA_OFF,
                            label="TTA（推理期增强）",
                            info="只在推理阶段生效；训练期数据增强在“训练 → 高级参数”里。",
                        )
                        with gr.Accordion("后处理参数（高级）", open=False):
                            tile_infer = gr.Checkbox(value=True, label="分块推理")
                            tile_size = gr.Slider(64, 640, value=384, step=32, label="分块高度")
                            overlap = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="分块重叠")
                            threshold = gr.Slider(0.05, 0.95, value=0.70, step=0.05, label="分割阈值")
                            min_area = gr.Slider(1, 2000, value=120, step=1, label="最小连通域面积")
                            smooth_on = gr.Checkbox(value=True, label="概率图平滑")
                            sigma = gr.Slider(0.0, 3.0, value=1.2, step=0.1, label="平滑 σ")
                            roi_only = gr.Checkbox(value=True, label="仅眼睑 ROI")
                            auto_escalate = gr.Checkbox(value=False, label="允许自动抬升阈值（会如实标注）")
                            gradcam_checkbox = gr.Checkbox(value=False, label="True Grad-CAM（额外反传）")
                        with gr.Row():
                            preset_dropdown = gr.Dropdown(choices=sorted(presets),
                                                          value=sorted(presets)[0], label="推理预设")
                            apply_button = gr.Button("应用预设")
                        preset_name = gr.Textbox(label="保存为新预设", placeholder="输入名称")
                        save_button = gr.Button("保存预设")
                        preset_status = gr.Markdown("")
                        infer_button = gr.Button("开始分析", variant="primary")
                    with gr.Column(scale=5):
                        status_output = gr.Markdown("等待上传影像…")
                        overlay_output = gr.Image(label="分割叠加", type="numpy")
                        with gr.Tabs():
                            with gr.Tab("概率热力图"):
                                gr.Markdown(
                                    "<div class='mgd-help'>概率伪彩（非归因）；归因请看「解释性」。</div>"
                                )
                                heatmap_output = gr.Image(label="概率热力图（非 Grad-CAM）", type="numpy")
                            with gr.Tab("True Grad-CAM"):
                                gradcam_output = gr.Image(label="True Grad-CAM", type="numpy")
                            with gr.Tab("不确定性"):
                                uncertainty_output = gr.Image(label="像素熵不确定性", type="numpy")
                            with gr.Tab("轮廓"):
                                contour_output = gr.Image(label="轮廓", type="numpy")
                            with gr.Tab("骨架"):
                                skeleton_output = gr.Image(label="骨架", type="numpy")
                            with gr.Tab("实例"):
                                instance_output = gr.Image(label="腺体实例", type="numpy")
                        metrics_output = gr.Dataframe(headers=["指标", "数值"], value=_empty_metrics(),
                                                      interactive=False, label="量化指标（表格可横向滚动）")
                        report_button = gr.Button("生成 HTML 临床报告")
                        report_file = gr.File(label="报告下载（HTML / CSV）")
                        report_status = gr.Markdown("")

                # -------- 医生修正回流（WP-10 接线；纯 numpy/CPU，不持 GPU 串行锁） --------
                with gr.Group():
                    gr.Markdown("<div class='mgd-section-title'>医生修正回流</div>")
                    gr.Markdown(
                        "<div class='mgd-help'>在画布上用白色笔刷勾出修正后的腺体掩膜"
                        "（笔刷直径可在画布工具栏调整）；可先点「载入上次模型掩膜」把模型输出铺成"
                        "可编辑图层，再用笔刷补画 / 橡皮擦除。提交只落盘统计与掩膜哈希，"
                        "不内嵌掩膜像素。</div>"
                    )
                    correction_editor = gr.ImageEditor(
                        type="numpy",
                        label="修正掩膜画布",
                        brush=gr.Brush(default_color="#ffffff", colors=["#ffffff"],
                                       color_mode="fixed"),
                        height=320,
                    )
                    with gr.Row():
                        correction_reviewer = gr.Textbox(label="评审人 ID", placeholder="如 dr_wang")
                        correction_note = gr.Textbox(label="备注（可选）", placeholder="修正原因")
                        load_mask_button = gr.Button("载入上次模型掩膜")
                        submit_correction_button = gr.Button("提交修正", variant="primary")
                    correction_status = gr.Markdown("")
                    gr.Markdown(
                        "<div class='mgd-help'>修正数据仅本地留存于 <code>db/corrections.jsonl</code>；"
                        "导出训练候选用 <code>mgd-feedback export</code>。</div>"
                    )

                infer_button.click(
                    infer_fn,
                    inputs=[image_input, model_dropdown, tta_radio, tile_infer, tile_size, overlap,
                            threshold, min_area, smooth_on, sigma, roi_only, auto_escalate,
                            gradcam_checkbox],
                    outputs=[overlay_output, heatmap_output, contour_output, skeleton_output,
                             instance_output, uncertainty_output, gradcam_output, metrics_output,
                             status_output],
                )
                report_button.click(report_fn, outputs=[report_file, report_status])
                apply_button.click(
                    _preset_apply_handler(settings, load_presets),
                    inputs=[preset_dropdown],
                    outputs=[tile_infer, tile_size, overlap, threshold, min_area, smooth_on, sigma,
                             roi_only, tta_radio, gradcam_checkbox, preset_status],
                )
                save_button.click(
                    _preset_save_handler(settings, load_presets, save_preset),
                    inputs=[preset_name, tile_infer, tile_size, overlap, threshold, min_area,
                            smooth_on, sigma, roi_only, tta_radio, gradcam_checkbox],
                    outputs=[preset_dropdown, preset_status],
                )
                load_mask_button.click(load_model_mask_fn, outputs=[correction_editor])
                submit_correction_button.click(
                    submit_correction_fn,
                    inputs=[correction_editor, correction_reviewer, correction_note],
                    outputs=[correction_status],
                )

            # -------- 训练 -------------
            with gr.Tab("训练"):
                gr.Markdown(
                    "<div class='mgd-help'>三层入口共用同一参数 schema：YAML + CLI <code>--set a.b=c</code>、"
                    "命令行预设（学术 / 快速试跑 / 低显存）、本页表单。普通用户无需写 YAML；"
                    "高级参数在下方折叠区可展开修改。启动时会自动探测显存并推荐档位。</div>"
                )
                vram_recommend = "未探测到 CUDA（CPU 环境）" if vram_gb is None else f"{vram_gb:.1f} GB"
                gr.Markdown(
                    f"<div class='mgd-help'>当前设备：<b>{detect_device_text()}</b>　"
                    f"探测显存：<b>{vram_recommend}</b>　推荐档位由 <code>apply_tier(auto)</code> 以安全边界取整。</div>"
                )
                with gr.Row():
                    train_model = gr.Dropdown(choices=sorted(candidates), value=default_label,
                                              label="模型 / 权重（MGD-Former 为默认主干）")
                    profile_dropdown = gr.Dropdown(choices=profile_names(), value="academic",
                                                   label="预设（中层入口）")
                    apply_profile = gr.Button("套用预设")
                    vram_dropdown = gr.Dropdown(choices=_VRAM_CHOICES, value="auto", label="显存档位")
                profile_note = gr.Markdown("")
                with gr.Accordion("基础参数（默认值即可直接跑）", open=True):
                    with gr.Row():
                        epochs_number = gr.Number(value=20, precision=0, label="总轮数")
                        batch_number = gr.Number(value=1, precision=0, label="batch_size")
                        lr_number = gr.Number(value=1e-4, label="学习率")
                        optimizer_dropdown = gr.Dropdown(choices=["AdamW", "Adam", "SGD"], value="AdamW",
                                                         label="优化器")
                        scheduler_dropdown = gr.Dropdown(
                            choices=["ReduceLROnPlateau (默认)", "Cosine Annealing", "Step Decay",
                                     "CosineAnnealingWarmRestarts"],
                            value="Cosine Annealing", label="学习率调度")
                    with gr.Row():
                        split_mode = gr.Dropdown(choices=_SPLIT_CHOICES, value="Holdout 70/15/15",
                                                 label="划分模式")
                        fold_number = gr.Number(value=0, precision=0, label="K 折 fold")
                        val_ratio = gr.Slider(0.0, 0.4, value=0.15, step=0.05, label="验证集比例")
                        test_ratio = gr.Slider(0.0, 0.4, value=0.15, step=0.05, label="测试集比例")
                        dual_task = gr.Checkbox(value=True, label="双任务（腺体+眼睑）")
                with gr.Accordion("高级参数（可选，与 CLI/YAML 完全一致）", open=False):
                    with gr.Row():
                        train_tile = gr.Checkbox(value=True, label="分块训练")
                        train_tile_size = gr.Slider(64, 640, value=384, step=32, label="分块高度")
                        augmentation = gr.Checkbox(value=True, label="训练期数据增强")
                        amp_checkbox = gr.Checkbox(value=True, label="AMP（仅 CUDA 生效）")
                        pretrained_dropdown = gr.Dropdown(choices=["auto", "none", "ade20k", "imagenet"],
                                                          value="auto", label="编码器预训练")
                        deep_supervision = gr.Checkbox(value=True, label="深监督")
                    with gr.Row():
                        grad_accum = gr.Slider(1, 32, value=1, step=1, label="梯度累积步数")
                        channels_last = gr.Checkbox(value=True, label="channels_last")
                        cldice_weight = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="clDice 权重")
                        ema_decay = gr.Slider(0.9, 1.0, value=0.999, step=0.001, label="EMA 衰减")
                        seed_number = gr.Number(value=42, precision=0, label="随机种子")
                    with gr.Row():
                        eyelid_weight = gr.Slider(0.0, 1.0, value=0.25, step=0.05, label="眼睑损失权重")
                        loss_dropdown = gr.Dropdown(
                            choices=["MGD 复合损失", "Dice+CE 组合损失", "Dice Loss", "BCE (pos_weight)",
                                     "Focal Loss", "Tversky Loss", "FocalTversky Loss",
                                     "SoftclDice + Dice", "AnatomicalJointLoss (解剖联合)"],
                            value="MGD 复合损失", label="损失函数")
                        resume_text = gr.Textbox(value="", label="断点续训（run 目录或 .ckpt，留空=新 run）")
                        publish_best = gr.Checkbox(value=False, label="发布 best 到模型目录")
                        export_report = gr.Checkbox(value=True, label="生成 reference_report")
                train_button = gr.Button("开始训练", variant="primary")
                train_status = gr.Markdown("未开始。失败时错误会内联显示在本行下方。")
                history_table = gr.Dataframe(
                    headers=["epoch", "loss", "train_dice", "val_dice", "val_eyelid_dice", "lr",
                             "best", "样本/秒"],
                    interactive=False, label="训练指标流水",
                )
                train_curve = gr.Image(label="验证曲线（train / val dice）", type="numpy")
                run_dir_box = gr.Textbox(label="run 目录（配置快照 / metrics.jsonl / checkpoints）",
                                         interactive=False)
                apply_profile.click(
                    preset_fill, inputs=[profile_dropdown],
                    outputs=[epochs_number, lr_number, scheduler_dropdown, profile_note],
                )
                _bind_train_button(
                    train_button, train_fn,
                    inputs=[train_model, profile_dropdown, vram_dropdown, epochs_number, batch_number,
                            lr_number, optimizer_dropdown, scheduler_dropdown, loss_dropdown,
                            split_mode, fold_number, test_ratio, dual_task, eyelid_weight,
                            train_tile, train_tile_size, augmentation, amp_checkbox, val_ratio,
                            pretrained_dropdown, deep_supervision, grad_accum, channels_last,
                            cldice_weight, ema_decay, resume_text, seed_number, publish_best,
                            export_report],
                    outputs=[train_status, history_table, train_curve, run_dir_box],
                )

            # -------- 评估 -------------
            with gr.Tab("评估"):
                gr.Markdown(
                    "<div class='mgd-help'>复用持久化划分（<code>dataset_splits.json</code>，seed=42）。"
                    "默认 Test 集；训练集自评会明确标注，不能直接写进论文。"
                    "导出 CSV / LaTeX 时会保留逐样本指标，便于 bootstrap 与配对检验。</div>"
                )
                with gr.Row():
                    eval_model = gr.Dropdown(choices=sorted(candidates), value=default_label,
                                             label="模型 / 权重")
                    eval_scope = gr.Dropdown(choices=_SCOPE_CHOICES, value=_SCOPE_CHOICES[0],
                                             label="测试集")
                    eval_button = gr.Button("开始评估", variant="primary")
                eval_note = gr.Markdown("")
                eval_summary = gr.Dataframe(headers=["指标", "数值"], value=_empty_metrics(),
                                            interactive=False, label="汇总")
                eval_per_sample = gr.Dataframe(
                    headers=["样本", "Dice", "IoU", "HD95", "ASSD", "Precision", "Recall",
                             "clDice", "R_loss(%)", "Meiboscore", "错误"],
                    interactive=False, label="逐样本（可横向滚动）",
                )
                with gr.Row():
                    export_csv_button = gr.Button("导出 CSV")
                    export_tex_button = gr.Button("导出 LaTeX")
                export_status = gr.Markdown("")
                eval_button.click(eval_fn, inputs=[eval_model, eval_scope],
                                  outputs=[eval_summary, eval_per_sample, eval_note])
                export_csv_button.click(lambda: export_fn("csv"), outputs=export_status)
                export_tex_button.click(lambda: export_fn("latex"), outputs=export_status)

            # -------- 实验管理 -------------
            with gr.Tab("实验管理"):
                gr.Markdown(
                    "<div class='mgd-help'>每个 run 有唯一 ID、配置快照、代码版本、metrics.jsonl 与产物目录。"
                    "勾选多个 run 可横向对比并导出论文用 CSV / LaTeX；缺 <code>split_signature</code> 的 run 不能进论文表。</div>"
                )
                with gr.Row():
                    refresh_button = gr.Button("刷新 run 列表")
                    compare_button = gr.Button("横向对比", variant="primary")
                runs_status = gr.Markdown("")
                run_selector = gr.Dropdown(choices=[], value=[], multiselect=True,
                                           label="选择 run（默认最近 3 个）")
                compare_table = gr.Dataframe(
                    headers=["run_id", "arch", "epochs", "best_metric", "best_epoch", "val_dice",
                             "显存峰值", "样本/秒", "耗时(s)", "split_signature", "已验证",
                             "创建时间"],
                    interactive=False, label="run 对比",
                )
                compare_curve = gr.Image(label="验证 Dice 曲线", type="numpy")
                with gr.Row():
                    compare_csv = gr.Button("导出对比 CSV")
                    compare_tex = gr.Button("导出对比 LaTeX")
                compare_status = gr.Markdown("")
                refresh_button.click(refresh_runs, outputs=[run_selector, runs_status])
                compare_button.click(compare_fn, inputs=[run_selector],
                                     outputs=[compare_table, compare_curve, compare_status])
                compare_csv.click(lambda: export_compare("csv"), outputs=compare_status)
                compare_tex.click(lambda: export_compare("latex"), outputs=compare_status)

            # -------- 数据集 -------------
            with gr.Tab("数据集"):
                gr.Markdown(
                    "<div class='mgd-help'>版本化记录哈希清单、样本计数、划分签名与来源/设备；"
                    "任何写操作（版本记录、备份）前都会先校验完整性。"
                    "本平台绝不覆盖用户影像、掩膜与权重；备份只复制。</div>"
                )
                with gr.Row():
                    dataset_check_button = gr.Button("完整性校验", variant="primary")
                    dataset_version_button = gr.Button("记录数据集版本")
                    dataset_backup_button = gr.Button("备份数据集（只复制）")
                dataset_table = gr.Dataframe(headers=["项", "数量", "状态"], interactive=False,
                                             label="完整性")
                dataset_note = gr.Markdown("尚未校验。")
                with gr.Row():
                    source_text = gr.Textbox(value="本地采集", label="来源")
                    device_text_box = gr.Textbox(value="未记录", label="采集设备")
                dataset_version_note = gr.Markdown("")
                dataset_backup_note = gr.Markdown("")
                dataset_check_button.click(dataset_check, outputs=[dataset_table, dataset_note])
                dataset_version_button.click(dataset_version, inputs=[source_text, device_text_box],
                                             outputs=dataset_version_note)
                dataset_backup_button.click(dataset_backup, outputs=dataset_backup_note)

            # -------- 系统状态 -------------
            with gr.Tab("系统状态"):
                gr.Markdown(
                    "<div class='mgd-help'>换机器第一件事：点「运行环境自检」。"
                    "报告含 Python / torch / 设备 / 依赖 / 数据 / 权重 manifest / 磁盘 / 写权限；"
                    "失败项会给出可执行的处理建议。</div>"
                )
                doctor_button = gr.Button("运行环境自检", variant="primary")
                doctor_output = gr.Textbox(label="mgd doctor", lines=16, interactive=False)
                gr.Markdown("<div class='mgd-section-title'>权重状态</div>")
                weights_button = gr.Button("刷新权重状态")
                weights_table_box = gr.Dataframe(
                    headers=["模型", "权重文件", "状态", "verified", "原因"],
                    interactive=False, label="权重 manifest",
                )
                gr.Markdown("<div class='mgd-section-title'>显存档位表</div>")
                tiers_table = gr.Dataframe(
                    headers=["档位", "分块", "batch", "梯度累积", "AMP", "channels_last", "深监督",
                             "TTA", "状态", "依据", "显存峰值", "吞吐"],
                    interactive=False, label="显存档位（实测/估算均标注依据）",
                )
                doctor_button.click(system_report, outputs=doctor_output)
                weights_button.click(weights_table, outputs=weights_table_box)
                demo.load(weights_table, outputs=weights_table_box)
                demo.load(tiers_view, outputs=tiers_table)

            # -------- 质控 -------------
            with gr.Tab("质控"):
                gr.Markdown(
                    "<div class='mgd-help'>采集端质量门（WP-4）：对单帧灰度图做模糊 / 曝光 / 反光 / "
                    "翻转不全-过度翻转 / 离焦 / 睫毛遮挡六类判定并给出 0-100 得分与处置建议。"
                    "本页纯 CPU 计算，不加载模型；眼睑掩膜缺失时相关项记为「不可测」，绝不瞎猜。</div>"
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        qc_image = gr.Image(type="numpy", label="采集灰度图")
                        qc_eyelid = gr.Image(type="numpy", label="眼睑掩膜（可选，按 >0 二值化）")
                        qc_button = gr.Button("运行质控", variant="primary")
                    with gr.Column(scale=3):
                        qc_table = gr.Dataframe(
                            headers=["项", "数值"], value=[["—", "—"]],
                            interactive=False, label="质控报告（score / flags / 原始量 / 阈值）",
                        )
                        qc_action = gr.Markdown("")
                qc_button.click(qc_fn, inputs=[qc_image, qc_eyelid],
                                outputs=[qc_table, qc_action])

            # -------- 解释性 -------------
            with gr.Tab("解释性"):
                gr.Markdown(
                    "<div class='mgd-help'>真归因 + 忠实度证据（WP-3）：LayerCAM / Score-CAM 依赖"
                    "「诊断」页的最近一次分析（复用当时的权重与推理参数）；遮挡敏感度对概率图滑窗"
                    "扰动打分，与模型解耦。忠实度表给出 deletion / insertion AUC 与随机归因对照，"
                    "防止「看起来像热力图」的伪归因混入交付。本页回调持 GPU 串行锁。</div>"
                )
                explain_method = gr.Radio(
                    choices=["LayerCAM", "Score-CAM", "遮挡敏感度"], value="LayerCAM",
                    label="归因方法",
                )
                with gr.Accordion("遮挡敏感度参数（仅该方法使用）", open=False):
                    occl_window = gr.Slider(8, 256, value=64, step=8, label="窗口边长（px）")
                    occl_stride = gr.Slider(4, 128, value=32, step=4, label="步长（px）")
                    occl_max_windows = gr.Number(
                        value=256, precision=0, label="窗口数上限",
                        info="超出时步长自动翻倍以控制耗时",
                    )
                explain_button = gr.Button("生成归因", variant="primary")
                attribution_output = gr.Image(label="归因图（[0,1]）", type="numpy")
                faithfulness_table = gr.Dataframe(
                    headers=["指标", "数值"], interactive=False, label="忠实度证据",
                )
                explain_note = gr.Markdown("")
                explain_button.click(
                    explain_fn,
                    inputs=[explain_method, occl_window, occl_stride, occl_max_windows],
                    outputs=[attribution_output, faithfulness_table, explain_note],
                )

            # -------- 形态图谱 -------------
            with gr.Tab("形态图谱"):
                gr.Markdown(
                    "<div class='mgd-help'>对「诊断」页最近一次分析做 8 类形态学分型"
                    "（dropout / shortening / tortuosity / thickening / beading / ghost / "
                    "disruption / confluence），逐条给出实测值、阈值与判定依据；"
                    "并按固定顺序特征向量在图谱索引中检索相似病例 Top-5。本页纯 CPU 计算。</div>"
                )
                atlas_button = gr.Button("对最近诊断做分型", variant="primary")
                atlas_type_table = gr.Dataframe(
                    headers=["分型", "触发", "实测值", "阈值", "判定依据"], interactive=False,
                    label="8 类分型证据",
                )
                atlas_similar_table = gr.Dataframe(
                    headers=["病例", "余弦距离", "元数据"], interactive=False,
                    label="相似病例 Top-5",
                )
                atlas_note = gr.Markdown("")
                atlas_button.click(atlas_fn,
                                   outputs=[atlas_type_table, atlas_similar_table, atlas_note])

            # -------- 随访 -------------
            with gr.Tab("随访"):
                gr.Markdown(
                    "<div class='mgd-help'>纵向随访（WP-8）：按患者 ID 查询就诊时间线，"
                    "选两次就诊对比测量差值；两次就诊都登记了本地影像文件时自动做 ECC 刚体配准并"
                    "报告相关分数。数据库在首次写入时创建，本页只读、不碰模型。</div>"
                )
                with gr.Row():
                    patient_text = gr.Textbox(label="患者 ID", placeholder="如 P001")
                    timeline_button = gr.Button("查询时间线", variant="primary")
                timeline_note = gr.Markdown("")
                timeline_table = gr.Dataframe(
                    headers=["就诊 ID", "日期", "眼别", "眼睑", "影像", "备注", "测量项数"],
                    interactive=False, label="就诊时间线",
                )
                with gr.Row():
                    base_visit = gr.Dropdown(choices=[], label="基线就诊")
                    follow_visit = gr.Dropdown(choices=[], label="随访就诊")
                    compare_button = gr.Button("对比", variant="primary")
                delta_table = gr.Dataframe(
                    headers=["测量项", "基线", "随访", "差值（随访 − 基线）"],
                    interactive=False, label="Δ 指标",
                )
                compare_note = gr.Markdown("")
                timeline_button.click(timeline_fn, inputs=[patient_text],
                                      outputs=[timeline_table, base_visit, follow_visit,
                                               timeline_note])
                compare_button.click(followup_compare_fn,
                                     inputs=[patient_text, base_visit, follow_visit],
                                     outputs=[delta_table, compare_note])

            # -------- 分级 -------------
            with gr.Tab("分级"):
                grade_headers = list(grade_table(0.0)[0])
                gr.Markdown(
                    "<div class='mgd-help'>双分级体系对照（WP-2）：同一 R_loss 分别按 "
                    "Arita Meiboscore 与 Pult Meiboscale 分级。两套体系切点不同，级别不可直接互换，"
                    "互斥声明见下方固定文本。纯计算，零依赖。</div>"
                )
                with gr.Row():
                    loss_number = gr.Number(
                        value=30.0, label="R_loss 腺体缺失率（%）",
                        info="0-100；可从「诊断」页指标表读取",
                    )
                    grade_button = gr.Button("对照分级", variant="primary")
                grade_table_box = gr.Dataframe(
                    headers=grade_headers, value=[["—", "—"]],
                    interactive=False, label="分级对照",
                )
                gr.Markdown(incompatibility_note())
                grade_button.click(grade_fn, inputs=[loss_number], outputs=[grade_table_box])

        demo._mgd_launch_style = style_kwargs  # launch() 取用（Gradio 6）
        return demo


# ----------------- 辅助：把推理预设的 handler 与训练按钮绑定拆出 -----------------
def _bind_train_button(button, handler, inputs, outputs):
    button.click(handler, inputs=inputs, outputs=outputs)


def _preset_apply_handler(settings, loader):
    def apply(name):
        try:
            presets = loader(settings)
        except MGDError as exc:
            return (False, 384, 0.25, 0.70, 120, True, 1.2, True, TTA_OFF, False, exc.user_message())
        values = presets.get(name)
        if not values:
            return (False, 384, 0.25, 0.70, 120, True, 1.2, True, TTA_OFF, False, "预设不存在")
        return (
            bool(values.get("tile_infer", True)), int(values.get("tile_size", 384)),
            float(values.get("tile_overlap", 0.25)), float(values.get("threshold", 0.70)),
            int(values.get("min_area", 120)), bool(values.get("smooth_on", True)),
            float(values.get("smooth_sigma", 1.2)), bool(values.get("roi_only", True)),
            str(values.get("tta", TTA_OFF)), bool(values.get("gradcam", False)),
            f"已应用预设：{name}",
        )

    return apply


def _preset_save_handler(settings, loader, writer):
    import gradio as gr

    def save(name, tile_infer, tile_size, overlap, threshold, min_area, smooth_on, sigma,
             roi_only, tta, gradcam):
        if not str(name).strip():
            return gr.update(), "请输入预设名称"
        try:
            path = writer(str(name), {
                "tile_infer": bool(tile_infer), "tile_size": int(tile_size),
                "tile_overlap": float(overlap), "threshold": float(threshold),
                "min_area": int(min_area), "smooth_on": bool(smooth_on),
                "smooth_sigma": float(sigma), "roi_only": bool(roi_only), "tta": str(tta),
                "gradcam": bool(gradcam),
            }, settings)
        except MGDError as exc:
            return gr.update(), exc.user_message()
        return gr.update(choices=sorted(loader(settings)), value=str(name)), f"已保存：{path}"

    return save


def launch(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    app = create_app(settings)
    try:
        app.queue(default_concurrency_limit=1)
    except (AttributeError, TypeError):
        try:
            app.queue(concurrency_count=1)
        except (AttributeError, TypeError):
            app.queue()
    auth = ((("mgd", settings.auth_token),) if settings.auth_token else None)
    if auth is None:
        host_text = str(settings.host or "")
        if host_text not in ("127.0.0.1", "localhost", "::1"):
            # 不阻断启动（Docker 容器依赖 0.0.0.0），但必须让操作者看到风险
            print(
                f"[安全警告] auth_token 为空且绑定在 {host_text}：WebUI 对外无认证。"
                "请在局域网隔离/防火墙保护下使用，或用 --auth 设置访问口令。",
                flush=True,
            )
    style_kwargs = getattr(app, "_mgd_launch_style", {})
    app.launch(
        server_name=settings.host,
        server_port=settings.port,
        share=False,
        auth=auth,
        show_error=True,
        inbrowser=False,
        **style_kwargs,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="腺而易见 WebUI")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--auth", default=None, help="访问口令（会写入 Settings.auth_token）")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)
    settings = get_settings(data_dir=args.data_dir) if args.data_dir else get_settings()
    import dataclasses

    if args.host:
        settings = dataclasses.replace(settings, host=args.host)
    if args.port:
        settings = dataclasses.replace(settings, port=int(args.port))
    if args.auth:
        settings = dataclasses.replace(settings, auth_token=args.auth)
    launch(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

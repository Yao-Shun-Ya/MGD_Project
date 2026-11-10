"""g06 向后兼容 API 面：逐项 import + 存在性断言 + InferenceOptions 12 字段 +
INSTANCE_COLUMNS 冻结清单 + meiboscore 六点行为快照 + 训练参数旧键。"""
from __future__ import annotations

import dataclasses

import _common

EXPECTED_OPTION_FIELDS = [
    "tile_infer", "tile_size", "tile_overlap", "threshold", "min_area", "smooth_on",
    "smooth_sigma", "roi_only", "tta", "auto_escalate_threshold", "gradcam", "review_ratio",
]
FROZEN_INSTANCE_COLUMNS = [
    "实例编号", "面积(px)", "长轴(px)", "短轴(px)",
    "弯曲度τ", "平均曲率κ", "最大曲率κ", "相对延伸比", "是否截断",
]
MEIBOSCORE_SNAPSHOT = [(4.9, 0), (5.1, 1), (33.33, 1), (33.35, 2), (66.6, 2), (66.67, 3)]
EXPECTED_EXCEPTIONS = [
    "ConfigError", "DataIntegrityError", "DeviceError", "ModelNotFoundError",
    "ManifestError", "WeightLoadError", "InferenceError", "TrainingError",
    "InteropDependencyError",
]
OLD_SPEC_KEYS = ["train.epochs", "model.name", "run.seed", "data.split_mode",
                 "eval.threshold", "eval.tta", "train.batch_size", "artifacts.publish_best"]


def run() -> dict:
    gate = _common.Gate("g06_api_surface")

    from mgd.core import config
    for fn in ("get_settings", "model_best_path", "checkpoint_path",
               "canonical_weight_candidates", "list_trained_models"):
        gate.check(callable(getattr(config, fn, None)), f"config.{fn} 存在且可调用",
                   "该函数是向后兼容面，删除即破坏外部调用方；如需重构保留薄壳转发")

    from mgd.core.inference import service
    for sym in ("InferenceOptions", "InferenceResult", "ModelService", "normalize_tta"):
        gate.check(hasattr(service, sym), f"inference.service.{sym} 存在")
    field_names = [f.name for f in dataclasses.fields(service.InferenceOptions)]
    gate.check(field_names == EXPECTED_OPTION_FIELDS,
               f"InferenceOptions 字段与冻结清单逐项相等（12 个，实际 {field_names}）",
               "字段顺序/名称变化会破坏旧 UI/CLI 序列化；新增字段只能追加在尾部且带默认值")
    for name in EXPECTED_OPTION_FIELDS:
        gate.check(name in field_names, f"InferenceOptions.{name} 存在")

    from mgd.core.imaging import clinical
    for sym in ("INSTANCE_COLUMNS", "meiboscore_from_rloss", "r_loss", "instance_metrics",
                "calculate_clinical_metrics", "cldice_score"):
        gate.check(hasattr(clinical, sym), f"imaging.clinical.{sym} 存在")
    gate.check(clinical.INSTANCE_COLUMNS == FROZEN_INSTANCE_COLUMNS,
               "INSTANCE_COLUMNS 与冻结 9 列清单逐项相等",
               f"冻结清单 = {FROZEN_INSTANCE_COLUMNS}；改动需同步 atlas/longitudinal/report")
    for value, expected in MEIBOSCORE_SNAPSHOT:
        actual = clinical.meiboscore_from_rloss(value)
        gate.check(actual == expected,
                   f"meiboscore_from_rloss({value}) == {expected}（实际 {actual}）",
                   "分级边界口径被改动；Arita 切点为 ≤5 / 100/3 / 200/3")

    from mgd.core.evaluation import stats
    for sym in ("METRIC_KEYS", "bootstrap_ci", "wilcoxon_pairs", "reference_report_payload"):
        gate.check(hasattr(stats, sym), f"evaluation.stats.{sym} 存在")
    gate.check(len(stats.METRIC_KEYS) == 8, f"METRIC_KEYS 含 8 个指标（实际 {len(stats.METRIC_KEYS)}）")

    from mgd.core.reporting import report
    gate.check(callable(getattr(report, "build_report", None)), "reporting.report.build_report 存在")

    from mgd.core.training.engine import TrainConfig
    gate.check(dataclasses.is_dataclass(TrainConfig), "training.engine.TrainConfig 存在且为 dataclass")
    from mgd.core.models import registry
    for sym in ("create_model", "unwrap_outputs", "normalize_model_name", "MODEL_NAMES"):
        gate.check(hasattr(registry, sym), f"models.registry.{sym} 存在")

    from mgd.core.runtime import errors
    gate.check(hasattr(errors, "MGDError"), "runtime.errors.MGDError 基类存在")
    for name in EXPECTED_EXCEPTIONS:
        cls = getattr(errors, name, None)
        gate.check(isinstance(cls, type) and issubclass(cls, errors.MGDError),
                   f"runtime.errors.{name} 存在且是 MGDError 子类")
    gate.check(callable(getattr(errors.MGDError, "user_message", None)),
               "MGDError.user_message 可调用（外壳统一提示入口）")

    from mgd.core.training import specs
    for name in OLD_SPEC_KEYS:
        gate.check(name in specs.SPEC_BY_NAME, f"训练参数旧键 {name} 仍在 SPEC_BY_NAME",
                   "旧键是 YAML/CLI/UI 三层契约；移除=破坏既有配置文件")

    gate.note("快照含 config 5 函数、service 4 符号+12 字段、clinical 6 符号+9 列+6 点行为、"
              "stats 4 符号、registry 4 符号、9 异常类、8 旧参数键")
    return gate.result()


if __name__ == "__main__":
    _common.run_as_main(run)

"""g09 specs ↔ TrainConfig 对齐：唯一参数源与引擎字段/默认值/默认 YAML 三方一致。"""
from __future__ import annotations

import _common

if True:
    import yaml
    from mgd.core.training import specs
    from mgd.core.training.engine import TrainConfig
    from mgd.core.runtime.errors import ConfigError

NEW_FIELD_DEFAULTS = [
    ("infer.qc_policy", "qc_policy"),
    ("infer.explain_method", "explain_method"),
    ("eval.conformal_alpha", "conformal_alpha"),
    ("data.physical_scale_mm_per_px", "physical_scale_mm_per_px"),
]
NEW_SCHEMA_KEYS = [("data", "physical_scale_mm_per_px"), ("infer", "qc_policy"),
                   ("infer", "explain_method"), ("eval", "conformal_alpha")]
DEFAULT_YAML = _common.PROJECT_ROOT / "mgd/configs/train_default.yaml"


def run() -> dict:
    gate = _common.Gate("g09_specs_alignment")

    gate.check("infer" in specs.SECTIONS, f"SECTIONS 含 'infer'（实际 {specs.SECTIONS}）",
               "v2.4.0 诊断中心级推理参数必须挂在 infer 分节")

    pseudo_fields: list[str] = []
    for spec in specs.SPECS:
        if hasattr(TrainConfig, spec.field):
            gate.check(True, f"{spec.name} → TrainConfig.{spec.field} 存在")
        elif spec.field.startswith("_"):
            pseudo_fields.append(spec.name)
        else:
            gate.check(False, f"{spec.name} → TrainConfig.{spec.field} 不存在",
                       "specs 是唯一参数源：要么在 engine.TrainConfig 补字段，要么删 spec")
    gate.check(pseudo_fields == ["run.output_dir"],
               f"下划线伪字段只有 run.output_dir（实际 {pseudo_fields}）",
               "伪字段由 config_file.py 特判处理，新增伪字段必须同步该处逻辑")
    config_file_src = (_common.PROJECT_ROOT / "mgd/core/training/config_file.py").read_text(
        encoding="utf-8", errors="replace")
    gate.check('== "_output_dir"' in config_file_src or "_output_dir" in config_file_src,
               "伪字段 _output_dir 在 config_file.py 有消费点（否则 YAML 键会静默失效）")

    config_obj = TrainConfig()
    for spec_name, field in NEW_FIELD_DEFAULTS:
        spec = specs.SPEC_BY_NAME.get(spec_name)
        if not gate.check(spec is not None, f"SPEC_BY_NAME 含 {spec_name}"):
            continue
        actual = getattr(config_obj, field)
        gate.check(actual == spec.default,
                   f"{spec_name} 默认值一致（TrainConfig={actual!r} vs spec={spec.default!r}）",
                   "两处默认值必须逐字一致（v2.4.0 对齐纪律），改一处必须同步另一处")

    gate.check(DEFAULT_YAML.is_file(), f"默认配置文件存在：{DEFAULT_YAML.name}")
    if DEFAULT_YAML.is_file():
        payload = yaml.safe_load(DEFAULT_YAML.read_text(encoding="utf-8"))
        gate.check(isinstance(payload, dict) and payload, "train_default.yaml 可加载为非空映射")
        try:
            specs.validate_payload(payload)
            gate.check(True, "train_default.yaml 通过 validate_payload（无未知分节/键、类型范围合法）")
        except ConfigError as exc:
            gate.check(False, "train_default.yaml 通过 validate_payload", f"ConfigError: {exc}")
        unknown_sections = [key for key in payload if key not in specs.SCHEMA]
        gate.check(not unknown_sections, f"YAML 分节全部在 SCHEMA 内（多余 {unknown_sections}）")

    for section, key in NEW_SCHEMA_KEYS:
        gate.check(key in specs.SCHEMA.get(section, {}),
                   f"SCHEMA[{section}] 含新键 {key}",
                   "新键必须在 specs.py 登记，否则 YAML/--set/UI 三层入口都不认识它")

    gate.note(f"specs 共 {len(specs.SPECS)} 条；逐条做了 TrainConfig 字段存在性检查；"
              "伪字段豁免 1 条（run.output_dir → config_file.py 特判）")
    return gate.result()


if __name__ == "__main__":
    _common.run_as_main(run)

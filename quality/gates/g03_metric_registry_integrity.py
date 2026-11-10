"""g03 指标注册表完整性：≥16 个 key，definition/unit/source 非空，source 在词表内，
key 全小写蛇形，describe() 返回字段齐全的 dict（指标可追溯纪律的静态面）。"""
from __future__ import annotations

import re

import _common

if True:  # 门禁以脚本运行时 _common 已把 PROJECT_ROOT 放进 sys.path
    from mgd.core.quantification import registry as metric_registry

KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
EXPECTED_FIELDS = {"key", "name_zh", "definition", "unit", "source", "requires", "higher_is_better"}
EXPECTED_SOURCES = ("Arita et al. 2008", "Pult & Nichols 2016", "engine-defined")


def run() -> dict:
    gate = _common.Gate("g03_metric_registry_integrity")
    reg = metric_registry.METRIC_REGISTRY
    keys = sorted(reg)

    gate.check(metric_registry.ALLOWED_SOURCES == EXPECTED_SOURCES,
               "source 词表与冻结口径一致（Arita 2008 / Pult & Nichols 2016 / engine-defined）",
               "词表在 mgd/core/quantification/registry.py 的 ALLOWED_SOURCES，改动需过评审")
    gate.check(len(keys) >= 16, f"注册指标数 ≥ 16（实际 {len(keys)}）",
               "新增指标请在 registry.py 以 MetricSpec 注册，并补 definition 与文献出处")
    gate.check(len(set(keys)) == len(keys), "注册表 key 无重复")

    for key in keys:
        spec = reg[key]
        gate.check(bool(KEY_RE.match(key)), f"指标 {key} key 为小写蛇形",
                   "改名 key 并同步 quantification/morphometry.py 等产出点")
        gate.check(bool(str(spec.name_zh or "").strip()), f"指标 {key} name_zh 非空")
        gate.check(bool(str(spec.definition or "").strip()), f"指标 {key} definition 非空",
                   "definition 必须写清公式——报告里的每个数字都要能追到公式")
        gate.check(bool(str(spec.unit or "").strip()), f"指标 {key} unit 非空")
        gate.check(spec.source in metric_registry.ALLOWED_SOURCES,
                   f"指标 {key} source={spec.source!r} 在词表内",
                   "source 只允许 ALLOWED_SOURCES 三类之一")
        gate.check(isinstance(spec.requires, tuple), f"指标 {key} requires 为 tuple")

        described = metric_registry.describe(key)
        gate.check(isinstance(described, dict) and EXPECTED_FIELDS <= set(described),
                   f"指标 {key} describe() 返回 dict 且 7 字段齐全",
                   "describe() 由 asdict 派生；若缺字段请检查 MetricSpec 定义")
        gate.check(described.get("key") == key, f"指标 {key} describe().key 自洽")

    sources_used = sorted({spec.source for spec in reg.values()})
    gate.note(f"{len(keys)} 个指标；实际使用的 source：{sources_used}")
    return gate.result()


if __name__ == "__main__":
    _common.run_as_main(run)

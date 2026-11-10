"""g10 文档同步门禁：版本唯一源、pyproject 动态版本 + 可选组 + 新脚本、README/冻结基线/
已知局限/ADR 与 v2.4.0 对齐。文档未就绪时本门禁 FAIL 属预期（门禁即备忘录）。"""
from __future__ import annotations

import re

import _common

if True:
    import mgd

PYPROJECT = _common.PROJECT_ROOT / "pyproject.toml"
README = _common.PROJECT_ROOT / "README.md"
FREEZE = _common.PROJECT_ROOT / "docs/engineering/FREEZE_BASELINE.md"
LIMITS = _common.PROJECT_ROOT / "docs/engineering/KNOWN_LIMITATIONS.md"
ADR = _common.PROJECT_ROOT / "docs/engineering/ARCHITECTURE_DECISION.md"
LIMIT_KEYWORDS = ["概率伪彩", "工程默认", "engine-defined", "像素单位", "跨设备泛化"]


def read(path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def versions(text: str) -> list[tuple[int, int, int]]:
    """提取文档中带 v 前缀的版本号（如 v2.5.0）；依赖版本（16.2.3 等）不计入。"""
    return [(int(a), int(b), int(c)) for a, b, c in re.findall(r"\bv(\d+)\.(\d+)\.(\d+)\b", text)]


def _pyproject_dependencies(text: str) -> list[str]:
    """取出 [project] dependencies 数组里的依赖串（保持顺序）。"""
    block = text.split("dependencies = [", 1)[-1].split("]", 1)[0]
    return [line.strip().rstrip(",").strip().strip('"').strip("'")
            for line in block.splitlines() if line.strip().startswith('"')]


def _requirement_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")]


def run() -> dict:
    gate = _common.Gate("g10_docs_sync")

    # 版本一致性：唯一源只允许是 mgd.__version__，文档必须跟随它而不是领先于它。
    current_text = mgd.__version__
    parts = current_text.split(".")
    gate.check(len(parts) == 3 and all(part.isdigit() for part in parts),
               f"mgd.__version__ 为 X.Y.Z 形式（实际 {current_text!r}）",
               "版本唯一源在 mgd/__init__.py；pyproject 经 attr 动态读取")
    current = tuple(int(part) for part in parts)

    pyproject = read(PYPROJECT)
    gate.check('version = { attr = "mgd.__version__" }' in pyproject,
               "pyproject 含动态版本 attr = mgd.__version__",
               "在 [tool.setuptools.dynamic] 写 version = { attr = \"mgd.__version__\" }")
    gate.check("pydicom" in pyproject and "highdicom" in pyproject,
               "pyproject 含 dicom optional group（pydicom + highdicom）",
               "[project.optional-dependencies] 增加 dicom = [\"pydicom>=…\", \"highdicom>=…\"]")

    # 依赖同源：安装脚本读 mgd/requirements/base.txt，pip install -e . 读 pyproject；
    # 两处不一致会造成「脚本装一套、打包声明另一套」，因此逐条比对。
    base_req = read(_common.PROJECT_ROOT / "mgd/requirements/base.txt")
    py_deps = _pyproject_dependencies(pyproject)
    req_deps = _requirement_lines(base_req)
    gate.check(py_deps == req_deps,
               f"pyproject dependencies 与 mgd/requirements/base.txt 逐条一致（{len(py_deps)} 条）",
               f"不一致：pyproject={py_deps} / base.txt={req_deps}")

    for script in ("mgd-qc", "mgd-atlas", "mgd-followup", "mgd-report"):
        gate.check(f"{script} = " in pyproject, f"pyproject [project.scripts] 含 {script}",
                   f"补 {script} = \"mgd.adapters.cli.<module>:main\" 并重装包（pip install -e .）")

    readme = read(README)
    gate.check(current_text in readme, f"README.md 含当前版本 {current_text}",
               "README 未跟随版本：更新状态章节、徽章与增量说明")
    for word in ("mgd-qc", "mgd-atlas", "mgd-followup", "mgd-report"):
        gate.check(word in readme, f"README.md 含 '{word}'",
                   f"在 README 快速开始/CLI 章节补 {word} 用法示例")

    freeze = read(FREEZE)
    gate.check(current_text in freeze, f"docs/engineering/FREEZE_BASELINE.md 含当前版本 {current_text}",
               "冻结基线未跟随版本：补当前版本的冻结章节（新增子包清单 + 纪律条款）")

    limits = read(LIMITS)
    adr = read(ADR)

    # 防回归：文档不得声称比代码更新的版本（曾出现文档写 v2.5.0、代码仍是 2.4.0）。
    ahead: list[str] = []
    for name, text in (("README.md", readme), ("FREEZE_BASELINE.md", freeze),
                       ("KNOWN_LIMITATIONS.md", limits), ("ARCHITECTURE_DECISION.md", adr)):
        higher = [v for v in versions(text) if v > current]
        if higher:
            ahead.append(f"{name} 出现 v{'.'.join(map(str, max(higher)))}")
    gate.check(not ahead,
               f"文档无高于当前版本 {current_text} 的版本号（违规 {len(ahead)}）",
               "；".join(ahead) + " → 先升 mgd/__init__.py 再改文档，或把文档改回当前版本")

    # 防回归：「现为 X.Y.Z」这类当前版本陈述必须等于 mgd.__version__。
    declared = [tuple(int(part) for part in text.split(".")) for text in
                re.findall(r"现为\s*v?(\d+\.\d+\.\d+)", freeze + readme + limits)]
    stale = [v for v in declared if v != current]
    gate.check(not stale,
               f"文档「现为 X.Y.Z」与当前版本一致（不一致 {len(stale)}）",
               "；".join(f"现为 {'.'.join(map(str, v))}" for v in stale) + f" → 应为 {current_text}")

    hits = [kw for kw in LIMIT_KEYWORDS if kw in limits]
    gate.check(len(hits) >= 5,
               f"KNOWN_LIMITATIONS.md 含 ≥5 条 v2.4.0 诚实条目（命中关键词 {len(hits)}/5：{hits}）",
               "逐条补齐关键词覆盖：概率伪彩 / 工程默认 / engine-defined / 像素单位 / 跨设备泛化")

    for adr_id in ("ADR-002", "ADR-003", "ADR-004"):
        gate.check(adr_id in adr, f"ARCHITECTURE_DECISION.md 含 {adr_id}",
                   f"补 {adr_id} 决议（分型阈值 / 双分级互斥 / 物理标定等 v2.4.0 决议）")

    gate.note("检查面：版本来源/形式 / 文档版本跟随与防超前（新增）/ pyproject 动态版本+dicom组+4脚本 / "
              "README 版本+4 CLI / 冻结基线 / 已知局限 5 关键词 / ADR-002..004")
    return gate.result()


if __name__ == "__main__":
    _common.run_as_main(run)

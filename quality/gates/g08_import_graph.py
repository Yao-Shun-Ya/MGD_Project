"""g08 导入图方向断言（AST 建图）：
① core 不 → adapters；② v2.4.0 新子包顶层无 torch；③ interop 顶层无 pydicom/highdicom；
④ adapters → core 正向边真实存在；⑤ reporting 不 import gradio（任意层级）。"""
from __future__ import annotations

import ast

import _common

NEW_SUBPKGS = ["mgd/core/quantification", "mgd/core/grading", "mgd/core/atlas", "mgd/core/qc",
               "mgd/core/conformal", "mgd/core/longitudinal", "mgd/core/interop",
               "mgd/core/governance", "mgd/core/feedback"]


def run() -> dict:
    gate = _common.Gate("g08_import_graph")
    core_to_adapters: list[str] = []
    torch_top: dict[str, list[str]] = {pkg: [] for pkg in NEW_SUBPKGS}
    interop_dicom_top: list[str] = []
    reporting_gradio: list[str] = []
    adapters_to_core: list[str] = []
    files = list(_common.iter_py_files("mgd", exclude_tests=True))

    for path in files:
        tree, _lines = _common.load_module_ast(path)
        if tree is None:
            continue
        rel = _common.rel_path(path)
        package = _common.package_of(path)
        module = ".".join(package)

        for _node, target, _level, lineno in _common.import_pairs(tree, package):
            # ① core → adapters（任意层级都算破坏分层）
            if module.startswith("mgd.core") and _common.starts_with_mod(target, "mgd.adapters"):
                core_to_adapters.append(f"{rel}:{lineno} → {target}")
            # ⑤ reporting → gradio（任意层级）
            if module.startswith("mgd.core.reporting") and _common.starts_with_mod(target, "gradio"):
                reporting_gradio.append(f"{rel}:{lineno} → {target}")
            # ④ adapters → core 正向边
            if module.startswith("mgd.adapters") and _common.starts_with_mod(target, "mgd.core"):
                adapters_to_core.append(f"{rel}:{lineno} → {target}")

        for node in _common.top_level_imports(tree):
            targets = ([a.name for a in node.names] if isinstance(node, ast.Import)
                       else [_common.resolve_import(package, node.module, node.level or 0)])
            for target in targets:
                # ② 新子包顶层 torch
                for pkg in NEW_SUBPKGS:
                    if module.startswith(pkg + ".") and (target == "torch" or target.startswith("torch.")):
                        torch_top[pkg].append(f"{rel}:{node.lineno} 顶层 import {target}")
                # ③ interop 顶层 dicom 库
                if module.startswith("mgd.core.interop.") and any(
                        _common.starts_with_mod(target, m) for m in ("pydicom", "highdicom")):
                    interop_dicom_top.append(f"{rel}:{node.lineno} 顶层 import {target}")

    gate.check(len(files) >= 100, f"导入图覆盖 mgd/ 全部源文件 ≥ 100（实际 {len(files)}）",
               "文件数异常说明仓库结构变化，先更新扫描根")
    gate.check(not core_to_adapters, f"core/** 无 → adapters 边（违规 {len(core_to_adapters)}）",
               "；".join(core_to_adapters[:4]) + " → 分层方向只能 adapters → core")
    for pkg, hits in torch_top.items():
        gate.check(not hits, f"{pkg} 顶层无 torch",
                   "；".join(hits[:3]) + " → 新子包保持纯 numpy，torch 留在既有重型模块")
    gate.check(not interop_dicom_top, "interop 顶层无 pydicom/highdicom（可选依赖延迟导入）",
               "；".join(interop_dicom_top[:4]) + " → 函数体内 try-import + InteropDependencyError")
    gate.check(bool(adapters_to_core),
               f"adapters → core 正向边真实存在（{len(adapters_to_core)} 条，抽样 "
               f"{adapters_to_core[:2]}）",
               "若为 0 说明 adapters 层与 core 断开，CLI/UI 已失效")
    gate.check(not reporting_gradio, "reporting 不 import gradio（报告层保持无 UI 依赖）",
               "；".join(reporting_gradio[:3]) + " → 报告输出 HTML/CSV 字符串，UI 渲染交给 adapters")

    edges = len(adapters_to_core)
    gate.note(f"建图 {len(files)} 文件；adapters→core 边 {edges} 条；5 类方向断言全部执行")
    return gate.result()


if __name__ == "__main__":
    _common.run_as_main(run)

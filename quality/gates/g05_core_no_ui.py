"""g05 分层纪律（core 无 UI / 无重量级依赖上浮）：
① core 顶层不得 import gradio / pydicom / highdicom；
② matplotlib 顶层只允许 plotting.py 与 visualize.py；
③ 任何层不得 import mgd.adapters；④ v2.4.0 新子包顶层无 torch。"""
from __future__ import annotations

import ast

import _common

CORE_DIR = "mgd/core"
UI_MODULES = ("gradio",)
DICOM_MODULES = ("pydicom", "highdicom")
MPL_ALLOWED = {"mgd/core/evaluation/plotting.py", "mgd/core/imaging/visualize.py"}
#: v2.4.0 诊断中心级新子包（provenance 为单文件模块）——顶层禁止 torch
STRICT_NO_TORCH = [
    "mgd/core/quantification", "mgd/core/grading", "mgd/core/atlas", "mgd/core/qc",
    "mgd/core/conformal", "mgd/core/longitudinal", "mgd/core/interop",
    "mgd/core/governance", "mgd/core/feedback", "mgd/core/provenance.py",
]


def top_import_targets(tree: ast.AST, package: list[str]) -> list[tuple[str, int]]:
    """顶层 import 的绝对目标模块与行号。"""
    out = []
    for node in _common.top_level_imports(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            out.append((_common.resolve_import(package, node.module, node.level or 0),
                        node.lineno))
    return out


def run() -> dict:
    gate = _common.Gate("g05_core_no_ui")
    ui_hits: list[str] = []
    dicom_hits: list[str] = []
    mpl_hits: list[str] = []
    adapter_hits: list[str] = []
    torch_hits: dict[str, list[str]] = {rel: [] for rel in STRICT_NO_TORCH}
    files = list(_common.iter_py_files(CORE_DIR, exclude_tests=True))

    for path in files:
        tree, _lines = _common.load_module_ast(path)
        if tree is None:
            continue
        rel = _common.rel_path(path)
        package = _common.package_of(path)
        for target, lineno in top_import_targets(tree, package):
            for mod in UI_MODULES:
                if _common.starts_with_mod(target, mod):
                    ui_hits.append(f"{rel}:{lineno} 顶层 import {target}")
            for mod in DICOM_MODULES:
                if _common.starts_with_mod(target, mod):
                    dicom_hits.append(f"{rel}:{lineno} 顶层 import {target}")
            if target.startswith("matplotlib") and rel not in MPL_ALLOWED:
                mpl_hits.append(f"{rel}:{lineno} 顶层 import {target}")
        # mgd.adapters 反向依赖：任何层级（函数内延迟导入也算）
        for _node, target, _level, lineno in _common.import_pairs(tree, package):
            if _common.starts_with_mod(target, "mgd.adapters"):
                adapter_hits.append(f"{rel}:{lineno} import {target}")
        # 严格子包顶层 torch
        for scope in STRICT_NO_TORCH:
            in_scope = (rel == scope) if scope.endswith(".py") else rel.startswith(scope + "/")
            if in_scope:
                for target, lineno in top_import_targets(tree, package):
                    if target == "torch" or target.startswith("torch."):
                        torch_hits[scope].append(f"{rel}:{lineno} 顶层 import {target}")

    gate.check(len(files) >= 80, f"core 扫描覆盖 ≥ 80 个文件（实际 {len(files)}）",
               "core 目录结构变化时同步更新本门禁的扫描范围")
    gate.check(not ui_hits, "core 顶层无 gradio（UI 只住在 adapters/gradio_app）",
               "；".join(ui_hits[:4]) + " → 把 import 挪到 adapters 层，core 返回纯数据")
    gate.check(not dicom_hits, "core 顶层无 pydicom/highdicom（可选依赖必须延迟导入）",
               "；".join(dicom_hits[:4]) + " → 参考 interop/dicom_seg.py 的函数内 try-import 模式")
    gate.check(not mpl_hits, f"matplotlib 顶层只出现在 plotting.py / visualize.py（违规 {len(mpl_hits)}）",
               "；".join(mpl_hits[:4]) + " → 绘图移入允许文件，或改为函数内延迟导入")
    gate.check(not adapter_hits, "core 任何层不 import mgd.adapters（分层单向：adapters → core）",
               "；".join(adapter_hits[:4]) + " → 用回调/数据对象解耦，禁止 core 反向调用外壳")

    for scope, hits in torch_hits.items():
        gate.check(not hits, f"{scope} 顶层无 torch（v2.4.0 新子包纪律）",
                   "；".join(hits[:3]) + " → torch 只允许 models/training/explain/inference/imaging/"
                   "runtime 等既有重型模块使用；新子包保持纯 numpy")

    scanned_torch = sum(len(v) for v in torch_hits.values())
    gate.note(f"core 文件 {len(files)} 个；gradio/dicom/adapters 违规 0；顶层 torch 违规 {scanned_torch}")
    return gate.result()


if __name__ == "__main__":
    _common.run_as_main(run)

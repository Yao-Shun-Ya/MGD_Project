"""g11 禁径扫描：v2.4.0 新模块不得硬编码 models_weights / outputs_mgd / _transfer /
data/images 等物理路径字符串——路径必须经 Settings 属性访问（可测试、可迁移）。"""
from __future__ import annotations

import ast

import _common

FORBIDDEN = ["models_weights", "outputs_mgd", "_transfer", "data/images", "data\\images"]
TARGETS = [
    "mgd/core/quantification", "mgd/core/grading", "mgd/core/atlas", "mgd/core/qc",
    "mgd/core/conformal", "mgd/core/longitudinal", "mgd/core/interop",
    "mgd/core/governance", "mgd/core/feedback", "mgd/core/provenance.py",
    "mgd/adapters/cli/qc_cli.py", "mgd/adapters/cli/atlas_cli.py",
    "mgd/adapters/cli/followup_cli.py", "mgd/adapters/cli/report_cli.py",
]


def string_constants(tree: ast.AST) -> list[tuple[int, str]]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append((getattr(node, "lineno", 0), node.value))
    return out


def run() -> dict:
    gate = _common.Gate("g11_forbidden_paths")
    scanned_files = 0
    violations: list[str] = []

    for target in TARGETS:
        found_in_file: list[str] = []
        for path in _common.iter_py_files(target, exclude_tests=True):
            scanned_files += 1
            tree, _lines = _common.load_module_ast(path)
            if tree is None:
                continue
            rel = _common.rel_path(path)
            for lineno, text in string_constants(tree):
                for bad in FORBIDDEN:
                    if bad in text:
                        found_in_file.append(
                            f"{rel}:{lineno} 含 {bad!r}（…{text[max(0, text.find(bad) - 20):text.find(bad) + 30]!r}…）")
        gate.check(not found_in_file,
                   f"{target} 无禁径字符串（{len(found_in_file)} 处）",
                   ("；".join(found_in_file[:3]) + " → 改用 get_settings().model_dir / "
                    "output_dir / data_dir 等属性拼路径，使测试可用 MGD_HOME 重定向")
                   if found_in_file else "合规")

    gate.check(scanned_files >= 14, f"禁径扫描覆盖 ≥ 14 个文件（实际 {scanned_files}）",
               "新模块加入交付时，把路径加进 TARGETS 清单")
    gate.note(f"扫描 {scanned_files} 个 v2.4.0 新模块的字符串字面量；"
              f"禁径词表：{FORBIDDEN}；tests/ 与既有模块按规豁免")
    return gate.result()


if __name__ == "__main__":
    _common.run_as_main(run)

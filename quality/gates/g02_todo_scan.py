"""g02 未完成标记扫描：mgd/（排除 tests/）中 TODO/FIXME/XXX/HACK/pass#待/NotImplementedError 必须为 0。

AST + 正则双通道：NotImplementedError 另做 AST raise 校验，正则兜底注释/字符串形态。
"""
from __future__ import annotations

import ast
import re

import _common

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("TODO", re.compile(r"\bTODO\b")),
    ("FIXME", re.compile(r"\bFIXME\b")),
    ("XXX", re.compile(r"\bXXX\b")),
    ("HACK", re.compile(r"\bHACK\b")),
    ("pass # 待", re.compile(r"pass\s*#\s*待")),
    ("NotImplementedError", re.compile(r"NotImplementedError")),
]


def ast_not_implemented(tree: ast.AST) -> list[int]:
    """AST 级定位 `raise NotImplementedError`（或表达式形式调用）的行号。"""
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc is not None:
            target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if isinstance(target, ast.Name) and target.id == "NotImplementedError":
                lines.append(node.lineno)
    return lines


def run() -> dict:
    gate = _common.Gate("g02_todo_scan")
    hits: dict[str, list[str]] = {label: [] for label, _ in PATTERNS}
    scanned = 0
    parsed = 0

    for path in _common.iter_py_files("mgd", exclude_tests=True):
        scanned += 1
        tree, lines = _common.load_module_ast(path)
        rel = _common.rel_path(path)
        if tree is not None:
            parsed += 1
            for lineno in ast_not_implemented(tree):
                hits["NotImplementedError"].append(f"{rel}:{lineno} AST raise NotImplementedError")
        for label, pattern in PATTERNS:
            for offset, text in enumerate(lines):
                if pattern.search(text):
                    entry = f"{rel}:{offset + 1} {text.strip()[:90]}"
                    if entry not in hits[label]:
                        hits[label].append(entry)

    gate.check(scanned >= 80, f"扫描覆盖 mgd/（排除 tests）源文件 ≥ 80（实际 {scanned}）",
               "若远小于该值说明枚举失败，检查 _common.iter_py_files")
    gate.check(parsed == scanned, f"全部扫描文件可被 AST 解析（{parsed}/{scanned}）",
               "存在 SyntaxError 的文件会先被 g01 拦截，此处不应出现")
    for label, _ in PATTERNS:
        found = hits[label]
        gate.check(not found, f"{label} 标记数为 0（实际 {len(found)}）",
                   "；".join(found[:4])[:300] + " → 补完实现或删除占位，禁止带病交付")
    total = sum(len(v) for v in hits.values())
    gate.note(f"扫描 {scanned} 个文件 × 6 类标记，命中 {total} 处；tests/ 目录按规豁免")
    return gate.result()


if __name__ == "__main__":
    _common.run_as_main(run)

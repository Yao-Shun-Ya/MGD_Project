"""质量守卫共享工具（下划线开头，不参与 gNN 门禁发现）。

约定：
- 每个门禁 gNN_*.py 暴露 ``run() -> dict``（ok/name/details/checks/failed）；
- 门禁以独立子进程运行，stdout 最后一行输出 ``GATE_RESULT_JSON: {...}``；
- 本模块提供：项目根定位、AST 文件扫描、导入解析、结果累积器与统一输出。
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULT_MARKER = "GATE_RESULT_JSON: "
DETAILS_LIMIT = 2000

#: json 序列化前的净化函数命名豁免（g04 / 冗余豁免清单）
SANITIZER_RE = re.compile(r"_sanitize|_jsonable|_json_safe|_to_jsonable|_clean|_finite")


class Gate:
    """检查累积器：check() 逐条记账，result() 输出统一结构。"""

    def __init__(self, name: str):
        self.name = name
        self.checks = 0
        self.failed: list[str] = []
        self.notes: list[str] = []

    def check(self, ok: bool, label: str, fix: str = "") -> bool:
        """登记一项检查；失败时把「可操作修法」并入失败清单。"""
        self.checks += 1
        if not ok:
            self.failed.append(label if not fix else f"{label} → 修法：{fix}")
        return bool(ok)

    def note(self, text: str) -> None:
        self.notes.append(text)

    def result(self, details: str = "") -> dict:
        body = details or "；".join(self.notes)
        if self.failed:
            head = (body + " ｜｜ " if body else "") + "失败项："
            body = head + " ∥ ".join(self.failed)
        body = body or ("全部通过" if not self.failed else "存在失败项")
        return {
            "ok": not self.failed,
            "name": self.name,
            "details": body[:DETAILS_LIMIT],
            "checks": self.checks,
            "failed": self.failed[:20],
        }


def run_as_main(run_fn) -> None:
    """门禁 __main__ 统一出口：打印单行结果 JSON（供 run_gates.py 解析）。"""
    payload = run_fn()
    payload["details"] = str(payload.get("details", ""))[:DETAILS_LIMIT]
    print(RESULT_MARKER + json.dumps(payload, ensure_ascii=False))


# --------------------------------------------------------------------------- 文件与 AST


def iter_py_files(rel: str = "mgd", *, exclude_tests: bool = True):
    """按路径枚举项目内 .py 文件（稳定排序，剔除缓存与可选 tests）。"""
    base = PROJECT_ROOT / rel
    if base.is_file():
        yield base
        return
    for path in sorted(base.rglob("*.py")):
        parts = path.relative_to(PROJECT_ROOT).parts
        if "__pycache__" in parts:
            continue
        if exclude_tests and "tests" in parts:
            continue
        yield path


def rel_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix().replace("\\", "/")


def load_module_ast(path: Path):
    """返回 (ast_tree | None, source_lines)；解析失败不抛出（由调用方记账）。"""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None, source.splitlines()
    return tree, source.splitlines()


def package_of(path: Path) -> list[str]:
    """文件路径 → 包成分列表（mgd/core/config.py → ['mgd','core','config']）。"""
    return list(path.resolve().relative_to(PROJECT_ROOT).with_suffix("").parts)


def resolve_import(package: list[str], module: str | None, level: int) -> str:
    """解析 import 目标为绝对模块名；level=0 表示绝对导入。

    ``package`` 是含模块名的路径成分（如 ['mgd','core','config']）。
    相对层级语义：level=1 → 所在包（去掉模块名一层），每升一级再多剥一层。
    即 base = package[:len(package) - level]。
    """
    if level <= 0:
        return module or ""
    if level >= len(package):
        return module or ""
    base = package[: len(package) - level]
    return ".".join(base + ([module] if module else []))


def top_level_imports(tree: ast.AST) -> list[ast.stmt]:
    """模块顶层（含顶层 try/if/with 内、但不含函数/类体内）的 import 语句。"""
    found: list[ast.stmt] = []
    stack = list(ast.iter_child_nodes(tree))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found.append(node)
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return found


def all_imports(tree: ast.AST) -> list[tuple[ast.stmt, str, int]]:
    """文件内全部 import（任意层级）：[(节点, 目标模块, 相对层级)]。"""
    out: list[tuple[ast.stmt, str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((node, alias.name, 0))
        elif isinstance(node, ast.ImportFrom):
            out.append((node, node.module, node.level or 0))
    return out


def import_pairs(tree: ast.AST, package: list[str]):
    """全部 import 解析为 (节点, 绝对目标模块, 层级, 行号)。"""
    pairs = []
    for node, target, level in all_imports(tree):
        pairs.append((node, resolve_import(package, target, level), level, node.lineno))
    return pairs


def starts_with_mod(target: str, mod: str) -> bool:
    return target == mod or target.startswith(mod + ".")


def fn_source_segment(lines: list[str], node) -> str:
    """取节点（函数）的源码片段，用于豁免判定。"""
    if node is None or not lines:
        return ""
    start = max(0, int(getattr(node, "lineno", 1)) - 1)
    end = min(len(lines), int(getattr(node, "end_lineno", start + 1)))
    return "\n".join(lines[start:end])

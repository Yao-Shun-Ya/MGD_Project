"""g04 NaN 纪律（静态面）：mgd/core/{quantification,qc,grading,atlas,conformal,feedback,reporting}
① 禁止裸吞异常（except [Exception|BaseException]: pass）；② NaN 纪律静态证据；
③ json.dumps/dump 调用点必须 allow_nan=False 或就近有 sanitize 豁免。"""
from __future__ import annotations

import ast

import _common

SCAN_DIRS = [
    "mgd/core/quantification", "mgd/core/qc", "mgd/core/grading", "mgd/core/atlas",
    "mgd/core/conformal", "mgd/core/feedback", "mgd/core/reporting",
    # 堆料：治理/互操作/随访/血缘同样产生 JSON，一并纳入 json 严格性检查
    "mgd/core/governance", "mgd/core/interop", "mgd/core/longitudinal", "mgd/core/provenance.py",
]
BARE_TYPES = {"Exception", "BaseException"}
SWALLOW_RE = None  # 由 AST 判定，不用正则误伤


def except_pass_sites(tree: ast.AST) -> list[int]:
    """裸吞位点：handler 类型为空或 Exception/BaseException，且 body 全是 pass。"""
    sites: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        types = set()
        if node.type is None:
            types.add("")
        elif isinstance(node.type, ast.Name):
            types.add(node.type.id)
        elif isinstance(node.type, ast.Tuple):
            types |= {elt.id for elt in node.type.elts if isinstance(elt, ast.Name)}
        if not (types <= BARE_TYPES or "" in types):
            continue
        if node.body and all(isinstance(stmt, ast.Pass) for stmt in node.body):
            sites.append(node.lineno)
    return sites


def parent_map(tree: ast.AST) -> dict:
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def enclosing_function(parents: dict, node):
    cur = parents.get(node)
    while cur is not None and not isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
        cur = parents.get(cur)
    return cur


def is_float_nan_call(node) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "float" and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant) and node.args[0].value == "nan")


def nan_raise_evidence(tree: ast.AST) -> bool:
    """NaN 拒绝式证据：raise ValueError(...含 NaN...)（qc/quality.py 的纪律形态）。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            for arg in node.exc.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and "NaN" in arg.value:
                    return True
    return False


def run() -> dict:
    gate = _common.Gate("g04_nan_discipline")
    swallow: list[str] = []
    json_sites: list[tuple[str, int, bool, str]] = []  # (文件, 行, 是否合规, 原因)
    scanned = 0

    for rel in SCAN_DIRS:
        for path in _common.iter_py_files(rel, exclude_tests=True):
            scanned += 1
            tree, lines = _common.load_module_ast(path)
            if tree is None:
                continue
            file_rel = _common.rel_path(path)
            for lineno in except_pass_sites(tree):
                swallow.append(f"{file_rel}:{lineno}")
            parents = parent_map(tree)
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "json"
                        and node.func.attr in ("dumps", "dump")):
                    continue
                allow = any(kw.arg == "allow_nan" and isinstance(kw.value, ast.Constant)
                            and kw.value.value is False for kw in node.keywords)
                if allow:
                    json_sites.append((file_rel, node.lineno, True, "allow_nan=False"))
                    continue
                func = enclosing_function(parents, node)
                seg = _common.fn_source_segment(lines, func)
                if _common.SANITIZER_RE.search(seg):
                    json_sites.append((file_rel, node.lineno, True, "所在函数含 sanitize 豁免"))
                    continue
                lo, hi = max(0, node.lineno - 6), min(len(lines), node.lineno + 5)
                if _common.SANITIZER_RE.search("\n".join(lines[lo:hi])):
                    json_sites.append((file_rel, node.lineno, True, "上下文 5 行内 sanitize"))
                else:
                    json_sites.append((file_rel, node.lineno, False,
                                       "无 allow_nan=False 且无 sanitize → NaN 会写出非法 JSON"))

    gate.check(scanned >= 30, f"NaN 纪律扫描覆盖 ≥ 30 个文件（实际 {scanned}）",
               "子包缺失时检查 SCAN_DIRS 是否与 mgd/core 结构一致")
    gate.check(not swallow, f"扫描范围内无 except裸吞（实际 {len(swallow)} 处：{swallow[:6]}）",
               "把 `except Exception: pass` 改为显式记录 logger.warning 或向上抛出；"
               "静默吞异常违反 NaN/错误纪律")

    morph = _common.PROJECT_ROOT / "mgd/core/quantification/morphometry.py"
    mtree, _ = _common.load_module_ast(morph)
    m_nan = mtree is not None and any(
        is_float_nan_call(node) for node in ast.walk(mtree))
    gate.check(m_nan, "quantification/morphometry.py 含 float(\"nan\")（空输入返回 NaN 的静态证据）",
               "空掩膜路径必须返回 NaN 而不是 0（NaN 纪律，见模块 docstring）")

    quality = _common.PROJECT_ROOT / "mgd/core/qc/quality.py"
    qtree, _ = _common.load_module_ast(quality)
    q_has_literal = qtree is not None and any(
        is_float_nan_call(node) for node in ast.walk(qtree))
    q_rejects = qtree is not None and nan_raise_evidence(qtree)
    gate.check(q_has_literal or q_rejects,
               "qc/quality.py 含 NaN 纪律证据（float(\"nan\") 字面量或 raise ValueError(...NaN...) 拒绝式）",
               "quality.py 以「NaN/Inf 输入必须抛 ValueError 拒绝」落实纪律；"
               "若两种证据都不存在，检查输入校验是否被删")

    bad_sites = [f"{f}:{ln}（{why}）" for f, ln, ok, why in json_sites if not ok]
    gate.check(not bad_sites, f"全部 json.dumps/dump 调用点严格 JSON 合规（共 {len(json_sites)} 处）",
               ("；".join(bad_sites[:4]) + " → 给调用点加 allow_nan=False，或写盘前过 "
                "_sanitize/_jsonable 类净化函数") if bad_sites else "合规")

    ok_count = sum(1 for site in json_sites if site[2])
    gate.note(f"裸吞扫描 {scanned} 文件；json 调用点 {len(json_sites)} 处（合规 {ok_count}，"
              f"豁免依据：allow_nan=False / 函数级 sanitize）")
    return gate.result()


if __name__ == "__main__":
    _common.run_as_main(run)

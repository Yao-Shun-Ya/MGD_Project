"""g12 可选依赖纪律：interop 顶层无 pydicom/highdicom；缺依赖时 export_seg/export_sr
真实调用抛 InteropDependencyError（消息含安装命令）；该异常是 MGDError 子类。

双环境设计（v2.5.0 接线更新）：无论当前环境是否安装 dicom extra 都可验证——
- 未安装：直接调用即触发缺依赖路径（最真实的降级验证）；
- 已安装：monkeypatch sys.modules 强制缺失后调用（模拟降级），并额外断言
  「依赖已装时输入校验路径正常」（source_ds=None → TypeError，证明依赖检查
  先于输入校验的顺序契约未被倒置）；另加 AST 顺序检查（try-import 在校验之前）。
"""
from __future__ import annotations

import ast
import sys
from types import ModuleType

import numpy as np

import _common

if True:
    from mgd.core.interop import dicom_seg, dicom_sr
    from mgd.core.runtime.errors import InteropDependencyError, MGDError

INSTALL_HINT = 'pip install "mgd-seg[dicom]"'
TOP_LEVEL_FORBIDDEN = ("pydicom", "highdicom")


def top_level_targets(tree) -> list[str]:
    package = ["mgd", "core", "interop"]
    out: list[str] = []
    for node in _common.top_level_imports(tree):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        else:
            out.append(_common.resolve_import(package, node.module, node.level or 0))
    return out


def expect_dependency_error(gate: _common.Gate, call, label: str, fix: str) -> None:
    try:
        call()
    except InteropDependencyError as exc:
        message = str(exc)
        hint = str(getattr(exc, "hint", "") or "")
        gate.check(INSTALL_HINT in message or INSTALL_HINT in hint,
                   label + "，且消息含安装命令", fix + f"；实际消息：{message[:160]}")
    except Exception as exc:  # noqa: BLE001 —— 门禁需要区分异常类型
        gate.check(False, label + "（抛了错误类型的异常）",
                   fix + f"；实际抛出 {type(exc).__name__}: {exc}"[:220])
    else:
        gate.check(False, label, fix + "；未抛异常（依赖已安装或校验顺序被改动）")


def _force_missing_deps() -> None:
    """进程内模拟未安装：sys.modules 塞 None 使 import pydicom/highdicom 抛 ImportError。"""
    for name in ("pydicom", "highdicom", "pydicom.dataset", "pydicom.sr", "highdicom.seg",
                 "highdicom.sr", "highdicom.content", "highdicom.enum"):
        sys.modules[name] = None


def _restore_deps(saved: dict[str, ModuleType | None]) -> None:
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def import_guard_precedes_validation(filename: str, gate: _common.Gate) -> None:
    """AST 顺序检查：源码中「依赖守卫（InteropDependencyError）」必须出现在
    「source_ds 输入校验（TypeError）」之前——顺序倒置会让缺依赖用户看到
    无关的 TypeError 而不是安装提示。"""
    path = _common.PROJECT_ROOT / "mgd/core/interop" / filename
    text = path.read_text(encoding="utf-8")
    guard_pos = text.find("InteropDependencyError(")
    validation_pos = text.find("必须是 pydicom.dataset.Dataset")
    gate.check(guard_pos != -1, f"{filename} 存在依赖守卫", "缺失 InteropDependencyError 守卫")
    if "source_ds" in text and "必须是 pydicom.dataset.Dataset" in text:
        gate.check(validation_pos != -1, f"{filename} 存在 source_ds 校验",
                   "缺失输入校验（行为回归）")
        gate.check(0 < guard_pos < validation_pos,
                   f"{filename} 依赖守卫先于输入校验（顺序契约）",
                   "把 try-import 块移到所有输入校验之前")


def run() -> dict:
    gate = _common.Gate("g12_optional_dep_discipline")

    for filename in ("dicom_seg.py", "dicom_sr.py"):
        tree, _lines = _common.load_module_ast(_common.PROJECT_ROOT / "mgd/core/interop" / filename)
        tops = top_level_targets(tree) if tree is not None else ["<unparsable>"]
        bad = [t for t in tops if any(_common.starts_with_mod(t, m) for m in TOP_LEVEL_FORBIDDEN)]
        gate.check(not bad, f"interop/{filename} 顶层无 pydicom/highdicom（违规 {bad}）",
                   "可选依赖只能在函数体内 try-import，参考 dicom_seg.export_seg 的写法")
        import_guard_precedes_validation(filename, gate)

    pydicom_installed = True
    try:
        import pydicom  # noqa: F401
    except ImportError:
        pydicom_installed = False

    mask = np.zeros((8, 8), dtype=np.uint8)

    if not pydicom_installed:
        # 未安装：最真实的降级路径——直接调用即触发
        gate.note("当前环境未安装 pydicom：走真实缺依赖路径")
        expect_dependency_error(
            gate,
            lambda: dicom_seg.export_seg(mask, source_ds=None,
                                         out_path=_common.PROJECT_ROOT / "outputs/_g12_seg.dcm"),
            "export_seg 缺依赖抛 InteropDependencyError", "依赖检查必须在输入校验之前")
        expect_dependency_error(
            gate,
            lambda: dicom_sr.export_sr(measurements={}, source_ds=None,
                                       out_path=_common.PROJECT_ROOT / "outputs/_g12_sr.dcm"),
            "export_sr 缺依赖抛 InteropDependencyError", "依赖检查必须在输入校验之前")
    else:
        # 已安装：monkeypatch 模拟缺依赖 + 真实环境行为断言（双保险）
        gate.note("当前环境已安装 pydicom/highdicom：monkeypatch 模拟缺依赖 + 校验真实输入路径")
        saved = {name: sys.modules.get(name) for name in
                 ("pydicom", "highdicom", "pydicom.dataset", "pydicom.sr", "highdicom.seg",
                  "highdicom.sr", "highdicom.content", "highdicom.enum")}
        try:
            _force_missing_deps()
            expect_dependency_error(
                gate,
                lambda: dicom_seg.export_seg(mask, source_ds=None,
                                             out_path=_common.PROJECT_ROOT / "outputs/_g12_seg.dcm"),
                "export_seg 模拟缺依赖抛 InteropDependencyError", "依赖检查必须在输入校验之前")
            expect_dependency_error(
                gate,
                lambda: dicom_sr.export_sr(measurements={}, source_ds=None,
                                           out_path=_common.PROJECT_ROOT / "outputs/_g12_sr.dcm"),
                "export_sr 模拟缺依赖抛 InteropDependencyError", "依赖检查必须在输入校验之前")
        finally:
            _restore_deps(saved)
        # 依赖已装的真实行为：source_ds=None → 类型校验路径（证明守卫顺序未被倒置）
        try:
            dicom_seg.export_seg(mask, source_ds=None,
                                 out_path=_common.PROJECT_ROOT / "outputs/_g12_seg.dcm")
            gate.check(False, "export_seg（依赖已装）source_ds=None 抛 TypeError",
                       "输入校验丢失（行为回归）")
        except InteropDependencyError:
            gate.check(False, "export_seg（依赖已装）source_ds=None 抛 TypeError",
                       "依赖守卫误触发（sys.modules 未恢复干净）")
        except TypeError as exc:
            gate.check("pydicom.dataset.Dataset" in str(exc),
                       "export_seg（依赖已装）source_ds=None 抛 TypeError",
                       f"消息不符：{exc}")
        try:
            dicom_sr.export_sr(measurements={}, source_ds=None,
                               out_path=_common.PROJECT_ROOT / "outputs/_g12_sr.dcm")
            gate.check(False, "export_sr（依赖已装）source_ds=None 抛 TypeError",
                       "输入校验丢失（行为回归）")
        except InteropDependencyError:
            gate.check(False, "export_sr（依赖已装）source_ds=None 抛 TypeError",
                       "依赖守卫误触发（sys.modules 未恢复干净）")
        except TypeError as exc:
            gate.check("pydicom.dataset.Dataset" in str(exc),
                       "export_sr（依赖已装）source_ds=None 抛 TypeError",
                       f"消息不符：{exc}")

    gate.check(issubclass(InteropDependencyError, MGDError),
               "InteropDependencyError 是 MGDError 子类（外壳统一捕获）",
               "新异常类必须挂在 MGDError 体系下")

    gate.note("缺依赖路径验证与已装路径验证双环境覆盖；真实 SEG/SR 往返由 "
              "tests/test_interop_roundtrip.py 承担")
    return gate.result()


if __name__ == "__main__":
    _common.run_as_main(run)

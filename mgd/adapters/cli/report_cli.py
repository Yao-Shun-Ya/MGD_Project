"""测量报告 CLI：`mgd-report --input <json> [--html] [--csv] [--sr]`。

输入 JSON：{"patient_id": str, "loss_percent": float|null, "measurements": {"ga_percent": x, ...}}
- --html：渲染带指标定义/单位/来源的测量摘要 HTML（quantification.describe）+ 双体系分级对照
- --csv：键值 CSV
- --sr：DICOM SR 导出（需要 pydicom/highdicom 与 --dicom 源影像；缺依赖时输出安装提示，退出码 3）
- --sr 可与 --html / --csv 组合：SR 成功后继续渲染 HTML/CSV；SR 失败则立即返回错误码，
  此时 HTML/CSV 不生成（一次调用要么全成功、要么以第一个失败为准）。
退出码：0 成功；3 可选依赖缺失；2 输入/运行错误（含 RBAC 拒绝：当前角色无
export_report 权限——拒绝先落 access_denied 审计再返回）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from ...core.config import get_settings
from ...core.governance.hooks import require_or_audit
from ...core.grading.report import grade_rows, incompatibility_rows
from ...core.quantification.registry import all_specs, describe
from ...core.runtime.errors import InteropDependencyError, MGDError
from ...core.runtime.logging import configure_logging

#: 与 core.interop.dicom_sr 相同口径的安装提示（缺 pydicom / highdicom → 退出码 3）
_SR_INSTALL_HINT = 'pip install "mgd-seg[dicom]"'


def _reconfigure_stdio() -> None:
    """管道/重定向输出统一 UTF-8（M-2）：GBK 控制台代码页下打印中文不再崩溃。

    pytest 捕获缓冲等非 TextIOWrapper 流没有 reconfigure，静默保持原样。
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _clean(value) -> float:
    number = float(value)
    return number if number == number else float("nan")


def _metric_text(value) -> str:
    """HTML 展示文本：NaN / Inf 一律 N/A（L-1），绝不让 `nan`/`inf` 落进页面。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{number:.4f}" if math.isfinite(number) else "N/A"


def _sr_measurements(measurements: dict) -> dict:
    """SR 汇总标量（export_sr 必需的三项）。

    meiboscore 的 0 是合法等级（正常），必须显式判 None 而非 truthiness——
    旧的 `int(value or -1)` 会把 0 级吞成 -1（未评估）（M-4）。
    缺 key → -1（未评估）；NaN → int(NaN) 抛 ValueError，由调用方映射退出码 2。
    """
    meiboscore = measurements.get("meiboscore")
    return {
        "ga_percent": measurements.get("ga_percent", float("nan")),
        "instance_count": int(measurements.get("instance_count", 0) or 0),
        "meiboscore": -1 if meiboscore is None else int(meiboscore),
    }


def _read_source_dicom(path: str):
    """--sr 源影像读取；pydicom 缺失 → InteropDependencyError（与 export_sr 同一口径）。

    dcmread 必须发生在 export_sr 之前（source_ds 是其入参），因此这里的延迟导入
    也要把 ImportError 显式转成 InteropDependencyError，由 main() 统一映射为
    退出码 3 + 安装提示（H-1：绝不裸 traceback）；highdicom 缺失仍由 export_sr 抛出。
    """
    try:
        import pydicom
    except ImportError as exc:
        raise InteropDependencyError(
            f"DICOM SR 导出需要 pydicom / highdicom（当前未安装）。请执行：{_SR_INSTALL_HINT}",
            hint=_SR_INSTALL_HINT,
        ) from exc
    return pydicom.dcmread(path)


def main(argv: list[str] | None = None) -> int:
    _reconfigure_stdio()
    parser = argparse.ArgumentParser(description="腺而易见测量报告（HTML/CSV/DICOM SR）")
    parser.add_argument("--input", required=True, help="测量 JSON 路径")
    parser.add_argument("--html", action="store_true", help="输出 HTML 摘要")
    parser.add_argument("--csv", action="store_true", help="输出 CSV")
    parser.add_argument("--sr", default=None, metavar="OUT.dcm",
                        help="导出 DICOM SR（需 --dicom 与 pydicom/highdicom）")
    parser.add_argument("--dicom", default=None, help="SR 的源影像 Dataset 路径")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出生成路径与指标")
    parser.add_argument("--out", default=None, help="输出目录（默认 output_dir/reports）")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)
    settings = get_settings(data_dir=args.data_dir) if args.data_dir else get_settings()
    configure_logging(log_file=settings.log_file)

    input_path = Path(args.input).expanduser()
    if not input_path.is_file():
        print(f"[失败] 输入 JSON 不存在：{input_path}", file=sys.stderr)
        return 2
    if not (args.html or args.csv or args.sr):
        print("[失败] 至少选择 --html / --csv / --sr 之一", file=sys.stderr)
        return 2

    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        measurements = {key: _clean(value) for key, value in (payload.get("measurements") or {}).items()}
        loss_percent = payload.get("loss_percent")
        loss_percent = _clean(loss_percent) if loss_percent is not None else float("nan")
        patient_id = str(payload.get("patient_id") or "case")
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"[失败] 输入 JSON 无效：{exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out).expanduser() if args.out else settings.output_dir / "reports"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_id = "".join(ch for ch in patient_id if ch.isalnum() or ch in "-_") or "case"

    # ---- 治理门 + DICOM SR：失败立即返回错误码；成功后继续渲染 --html / --csv（M-5） ----
    sr_path: Path | None = None
    try:
        # RBAC 治理门（报告导出）：放在输出目录创建等任何副作用之前；拒绝时
        # access_denied 已先行落审计，ConfigError（MGDError 子类）由本函数既有的
        # except (MGDError, ValueError, OSError) 分支接住 → user_message + 退出码 2。
        require_or_audit(settings, "export_report", entry="mgd-report")
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.sr:
            if not args.dicom:
                print("[失败] --sr 需要 --dicom <源影像路径>", file=sys.stderr)
                return 2
            source_ds = _read_source_dicom(args.dicom)
            from ...core.interop.dicom_sr import export_sr

            sr_path = export_sr(
                _sr_measurements(measurements),
                source_ds,
                out_dir / f"{safe_id}_{stamp}.sr.dcm",
            )
            print(f"SR 已导出：{sr_path}")
    except InteropDependencyError as exc:
        print(f"[依赖缺失] {exc.user_message()}", file=sys.stderr)
        return 3
    except (MGDError, ValueError, OSError) as exc:
        print(f"[失败] {getattr(exc, 'user_message', lambda: str(exc))()}", file=sys.stderr)
        return 2

    # ---- HTML / CSV（纯标准库渲染，不依赖 InferenceResult） ----
    html_path = out_dir / f"{safe_id}_{stamp}_summary.html"
    csv_path = out_dir / f"{safe_id}_{stamp}_summary.csv"

    if args.csv:
        rows = [["key", "value", "unit", "definition", "source"]]
        for spec in all_specs():
            value = measurements.get(spec.key)
            if value is None:
                continue
            shown = "N/A" if value != value else f"{value:.4f}"
            rows.append([spec.key, shown, spec.unit, spec.definition, spec.source])
        if loss_percent == loss_percent:
            rows.append(["loss_percent", f"{loss_percent:.4f}", "%", "腺体缺失率", "Arita et al. 2008"])
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
            import csv

            csv.writer(handle).writerows(rows)

    if args.html:
        import html as html_module

        def esc(text) -> str:
            return html_module.escape(str(text))

        metric_rows = "".join(
            f"<tr><td>{esc(spec.key)}</td><td>{_metric_text(measurements[spec.key])}</td>"
            f"<td>{esc(spec.unit)}</td><td>{esc(spec.definition)}</td><td>{esc(spec.source)}</td></tr>"
            for spec in all_specs() if spec.key in measurements
        )
        grade_table_html = "".join(
            "<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>"
            for row in (grade_rows(loss_percent) + incompatibility_rows(loss_percent))
            if loss_percent == loss_percent
        ) or "<tr><td colspan='2'>缺失率未提供，跳过分级对照</td></tr>"
        doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/><title>MGD 测量摘要 · {esc(patient_id)}</title>
<style>body {{ font-family: -apple-system, "Segoe UI", sans-serif; margin: 24px; color: #1b2634; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
th, td {{ border-bottom: 1px solid #dfe6ee; padding: 6px 10px; font-size: 14px; text-align: left; }}
th {{ background: #eef3f8; }} h2 {{ border-left: 4px solid #0b3d6b; padding-left: 10px; }}</style></head>
<body>
<h1>MGD 测量摘要 · {esc(patient_id)}</h1>
<h2>定量指标（带定义/单位/来源）</h2>
<table><tr><th>key</th><th>值</th><th>单位</th><th>定义</th><th>来源</th></tr>{metric_rows}</table>
<h2>分级（双体系对照）</h2>
<table>{grade_table_html}</table>
<p>生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}　源：{esc(input_path.name)}</p>
</body></html>"""
        html_path.write_text(doc, encoding="utf-8")

    written = [str(path) for path in (
        sr_path, html_path if args.html else None, csv_path if args.csv else None
    ) if path]
    if args.json:
        print(json.dumps({
            "written": written,
            "measurements": {key: (None if value != value else value) for key, value in measurements.items()},
            "loss_percent": None if loss_percent != loss_percent else loss_percent,
        }, ensure_ascii=False, indent=2))
    else:
        print("已生成：" + "、".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

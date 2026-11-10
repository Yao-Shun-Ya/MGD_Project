"""单病例 HTML 报告（Phase 6）。

- 图像以 base64 内嵌，单个 HTML 文件可直接发给临床/评审；
- 同步导出 CSV（汇总 + 实例明细）；
- PDF：优先 weasyprint（未安装时退化为浏览器打印），绝不假装已生成；
- 脱敏（v2.5.0）：``settings.deid_salt`` 非空且病例号非空时，病例号一律以
  ``pseudonymize`` 伪名展示并用于产物文件名——原始 ID 不进 HTML / 文件名
  （CSV 本就没有病例号字段），HTML 头部注明「已脱敏（伪名）」。
"""
from __future__ import annotations

import base64
import csv
import html
import io
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ..config import get_settings
from ..governance.deid import pseudonymize
from ..inference.service import InferenceResult


@dataclass
class ReportBundle:
    html_path: Path
    csv_path: Path
    pdf_path: Path | None = None
    pdf_note: str = ""


def _b64(image: np.ndarray | None) -> str:
    if image is None:
        return ""
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array * (255 if array.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    elif array.shape[2] == 3:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".png", array)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def _table(rows: list[list[str]], header: tuple[str, str] | None = ("指标", "数值")) -> str:
    head = ""
    if header:
        head = "<thead><tr>" + "".join(f"<th>{item}</th>" for item in header) + "</tr></thead>"
    body = "".join(
        "<tr>" + "".join(f"<td>{str(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table>{head}<tbody>{body}</tbody></table>"


def _image_card(title: str, data_uri: str) -> str:
    if not data_uri:
        return ""
    return f"<div class='card'><h3>{title}</h3><img src='{data_uri}' alt='{title}'/></div>"


def _kv_table(rows: list[list[str]]) -> str:
    """两列键值表（沿用报告内联风格）。"""
    if not rows:
        return ""
    cells = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>" for k, v in rows)
    return f"<table>{cells}</table>"


def _format_metric(value) -> str:
    """NaN 纪律的展示层：NaN / ±Inf → 'N/A（未计算）'，不用 0 冒充（L-5）。"""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A（未计算）" if value is None else html.escape(str(value))
    return "N/A（未计算）" if not math.isfinite(numeric) else f"{numeric:.4f}"


def _extended_sections(quality, grading, typology, provenance, explain) -> str:
    """v2.4.0 五个可选区块；全部为 None → 返回空串（不得出现空表头）。

    顺序：Provenance → 质控 → 分级（两套对照）→ 形态分型 → 解释性（含忠实度 verdict）。
    """
    blocks: list[str] = []

    if provenance is not None:
        data = provenance.to_dict() if hasattr(provenance, "to_dict") else dict(provenance)
        rows = [
            ["软件版本", data.get("software_version")],
            ["模型", data.get("model_name")],
            ["权重 sha256", data.get("weights_sha256") or "未提供"],
            ["权重验证状态", "verified" if data.get("weights_verified") else
             ("未通过验证（verified=false）" if data.get("weights_verified") is not None else "未校验")],
            ["阈值", data.get("threshold")],
            ["分块", f"{data.get('tile_size')}×2 / overlap {data.get('tile_overlap')}" if data.get("tile_size") else "未记录"],
            ["TTA", data.get("tta")],
            ["校准 ID", data.get("calibration_id") or "未使用"],
            ["生成时间", data.get("created_at")],
        ]
        blocks.append("<h2>血缘 Provenance</h2>" + _kv_table(rows))

    if quality is not None:
        rows = [["质控评分", f"{quality.score:.1f} / 100"]]
        rows.append(["触发旗标", "、".join(quality.flags) if quality.flags else "无"])
        for reason in quality.reasons:
            rows.append(["判据说明", reason])
        for key, value in sorted(quality.metrics.items()):
            rows.append([f"原始量 {key}", _format_metric(value)])
        blocks.append("<h2>采集质控</h2>" + _kv_table(rows))

    if grading:
        rows = [
            ["Arita 睑板腺评分", str(grading.get("arita_meiboscore", "N/A"))],
            ["Pult Meiboscale", str(grading.get("pult_meiboscale", "N/A"))],
            ["不可互换提示", "两套切点不同（1/3,2/3 vs 25/50/75），数值不可直接互换"],
        ]
        blocks.append("<h2>分级（双体系对照）</h2>" + _kv_table(rows))

    if typology:
        header = "".join(f"<th>{html.escape(col)}</th>" for col in ("分型", "触发", "实测值", "阈值", "判定依据"))
        body_rows = []
        for evidence in typology:
            value = getattr(evidence, "value", float("nan"))
            body_rows.append(
                "<tr><td>{key}</td><td>{present}</td><td>{value}</td><td>{threshold}</td><td>{definition}</td></tr>".format(
                    key=html.escape(str(getattr(evidence, "key", ""))),
                    present="是" if getattr(evidence, "present", False) else "否",
                    value=_format_metric(value),
                    threshold=_format_metric(getattr(evidence, "threshold", float("nan"))),
                    definition=html.escape(str(getattr(evidence, "definition", ""))),
                )
            )
        blocks.append(
            "<h2>形态分型证据</h2><table><tr>" + header + "</tr>" + "".join(body_rows) + "</table>"
        )

    if explain:
        verdict = str(explain.get("verdict", ""))
        # 不做内层转义：_kv_table 已统一 html.escape，双重转义会把 `<` 渲染成 `&amp;lt;`（L-2）
        verdict_text = {"pass": "通过（归因与概率结构一致）", "fail": "未通过（归因可信度不足，仅供排查参考）"}.get(
            verdict, verdict or "未评估"
        )
        rows = [["忠实度 verdict", verdict_text]]
        for key, label in (
            ("deletion_auc", "Deletion AUC（越低越忠实）"),
            ("insertion_auc", "Insertion AUC（越高越忠实）"),
            ("road", "ROAD 简化代理"),
            ("random_deletion_auc", "随机对照 Deletion AUC"),
            ("random_insertion_auc", "随机对照 Insertion AUC"),
        ):
            if key in explain:
                rows.append([label, _format_metric(explain.get(key))])
        blocks.append("<h2>解释性（归因忠实度）</h2>" + _kv_table(rows))

    return "\n  ".join(blocks)


def _extended_csv_rows(result, quality, grading) -> list[list[str]]:
    """CSV 追加行：r_loss/ga/di/ti/si/grade_arita/grade_pult/qc_score/qc_flags（有数据才写）。"""
    rows: list[list[str]] = []
    extra = getattr(result, "extra", {}) or {}
    clinical = extra.get("clinical")
    if clinical is not None and getattr(clinical, "rloss", None) is not None:
        rows.append(["r_loss", _format_metric(clinical.rloss.r_loss_percent)])
    quant = extra.get("quant_metrics") or {}
    for key, label in (
        ("ga_percent", "ga"),
        ("diameter_deformation_index", "di"),
        ("tortuosity_index", "ti"),
        ("signal_index", "si"),
    ):
        if key in quant:
            rows.append([label, _format_metric(quant.get(key))])
    if grading:
        rows.append(["grade_arita", str(grading.get("arita_meiboscore", "N/A"))])
        rows.append(["grade_pult", str(grading.get("pult_meiboscale", "N/A"))])
    if quality is not None:
        rows.append(["qc_score", f"{quality.score:.1f}"])
        rows.append(["qc_flags", "、".join(quality.flags) if quality.flags else "无"])
    return rows


def build_report(
    result: InferenceResult,
    *,
    model_name: str,
    patient_id: str = "",
    output_dir: str | Path | None = None,
    quality=None,
    grading: dict | None = None,
    typology=None,
    provenance=None,
    explain: dict | None = None,
) -> ReportBundle:
    """生成单病例自包含报告（HTML + CSV，PDF 可选）。

    v2.4.0 增量（全部可选、None 时完全不渲染对应区块，不出现空表头）：
    - quality: qc.quality.QualityReport → 「采集质控」区块
    - grading: dict[str, int]（grading.grade_all 输出）→ 「分级（双体系对照）」区块
    - typology: list[atlas.typology.TypeEvidence] → 「形态分型证据」区块
    - provenance: core.provenance.Provenance → 「血缘 Provenance」区块
    - explain: dict（explain.faithfulness.faithfulness_report 输出）→ 「解释性（含忠实度 verdict）」区块
    HTML 区块顺序：Provenance → 质控 → 分级 → 定量指标 → 形态分型 → 解释性 → 图像。
    CSV 追加行：r_loss/ga/di/ti/si/grade_arita/grade_pult/qc_score/qc_flags（有数据才写）。

    脱敏（v2.5.0）：``settings.deid_salt`` 非空且 ``patient_id`` 非空时，病例号在
    HTML 头部与产物文件名（HTML/CSV/PDF）中一律使用 ``pseudonymize(patient_id, salt)``
    的稳定伪名，并注明「已脱敏（伪名）」；原始 ID 不落盘。不传 salt（既有调用 /
    既有测试）行为与旧版完全一致。
    """
    settings = get_settings()
    directory = Path(output_dir) if output_dir else settings.output_dir / "reports"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    salt = getattr(settings, "deid_salt", None)
    salted = bool(salt) and bool(str(patient_id).strip())
    if salted:
        display_id = pseudonymize(str(patient_id), str(salt))
    else:
        display_id = str(patient_id)
    safe_id = "".join(ch for ch in (display_id or "case") if ch.isalnum() or ch in "-_") or "case"
    html_path = directory / f"{safe_id}_{stamp}.html"
    csv_path = directory / f"{safe_id}_{stamp}.csv"

    extra = result.extra or {}
    confidence = extra.get("confidence") or {}
    warnings = [item for item in [result.warning] if item]
    if confidence.get("need_review"):
        warnings.append("AI 置信度触发人工复核阈值")

    images = "".join(
        [
            _image_card("原始影像", _b64(result.image_rgb)),
            _image_card("腺体分割叠加", _b64(result.overlay)),
            _image_card("概率热力图", _b64(result.heatmap)),
            _image_card("边界轮廓", _b64(result.contour)),
            _image_card("骨架/中心线", _b64(result.skeleton)),
            _image_card("实例标注", _b64(result.instances)),
            _image_card("True Grad-CAM", _b64(extra.get("gradcam"))),
            _image_card("像素不确定性", _b64(extra.get("uncertainty_image"))),
        ]
    )
    warning_html = (
        "<div class='warn'><strong>⚠ 复核提示</strong><ul>"
        + "".join(f"<li>{item}</li>" for item in warnings)
        + "</ul></div>"
        if warnings
        else "<div class='ok'>未触发复核阈值</div>"
    )

    instance_df = extra.get("instance_df")
    instance_html = ""
    if instance_df is not None and len(instance_df) > 0:
        instance_html = "<h2>腺体实例明细</h2>" + instance_df.to_html(index=False, classes="data", border=0)

    # 先转义再插值：函数下方会定义局部变量 html（HTML 文本），届时模块名已被遮蔽。
    # 脱敏生效时原始 ID 不出现在报告任何位置；头部注明「已脱敏（伪名）」。
    patient_html = html.escape(display_id or "未填写")
    if salted:
        patient_html += " · 已脱敏（伪名）"
    model_html = html.escape(str(model_name))
    extended_html = _extended_sections(quality, grading, typology, provenance, explain)

    page_html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>MGD 智能诊断报告</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; margin: 0; background: #f4f6f9; color: #1b2634; }}
.header {{ background: #0b3d6b; color: #fff; padding: 28px 40px; }}
.header h1 {{ margin: 0; font-size: 26px; letter-spacing: 1px; }}
.header p {{ margin: 8px 0 0; opacity: .85; }}
.container {{ max-width: 1080px; margin: 24px auto; padding: 0 20px; }}
.card {{ display: inline-block; vertical-align: top; width: 48%; margin: 1%; background: #fff; border-radius: 10px; padding: 12px; box-shadow: 0 2px 10px rgba(0,0,0,.06); }}
.card img {{ width: 100%; border-radius: 6px; }}
h2 {{ font-size: 18px; margin: 26px 0 10px; border-left: 4px solid #0b3d6b; padding-left: 10px; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; }}
th, td {{ padding: 8px 12px; border-bottom: 1px solid #e6ebf1; font-size: 14px; text-align: left; }}
th {{ background: #eef3f8; }}
.warn {{ background: #fff4e5; border-left: 4px solid #d9822b; padding: 14px 18px; border-radius: 6px; margin-top: 16px; }}
.ok {{ background: #eaf7ef; border-left: 4px solid #2f9e44; padding: 14px 18px; border-radius: 6px; margin-top: 16px; }}
.footer {{ text-align: center; color: #6b7785; font-size: 12px; padding: 24px; }}
</style></head><body>
<div class="header">
  <h1>MGD 睑板腺智能诊断报告</h1>
  <p>病例号：{patient_html} &nbsp;|&nbsp; 模型：{model_html} &nbsp;|&nbsp; 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
<div class="container">
  {warning_html}
  <h2>量化指标</h2>
  {_table(result.metrics)}
  {extended_html}
  <h2>影像与可解释性</h2>
  {images}
  {instance_html}
  <div class="footer">本报告由 AI 辅助生成，仅用于科研与辅助诊断，最终诊断需由执业医师确认。</div>
</div></body></html>"""

    html_path.write_text(page_html, encoding="utf-8")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["指标", "数值"])
        writer.writerows(result.metrics)
        writer.writerow([])
        writer.writerow(["置信度评分", confidence.get("score")])
        writer.writerow(["模糊区占比", confidence.get("ambiguous_ratio")])
        # v2.4.0 追加行（有数据才写，不用空占位）
        for key, value in _extended_csv_rows(result, quality, grading):
            writer.writerow([key, value])
        if instance_df is not None and len(instance_df) > 0:
            writer.writerow([])
            writer.writerow(list(instance_df.columns))
            for row in instance_df.itertuples(index=False):
                writer.writerow(list(row))

    pdf_path: Path | None = None
    pdf_note = "未安装 weasyprint；可用浏览器打开 HTML 后『打印为 PDF』"
    try:
        from weasyprint import HTML  # type: ignore

        pdf_path = directory / f"{safe_id}_{stamp}.pdf"
        HTML(string=page_html).write_pdf(str(pdf_path))
        pdf_note = "已通过 weasyprint 生成"
    except Exception as exc:  # 可选依赖缺失不应阻断报告
        pdf_note = f"PDF 未生成（{type(exc).__name__}），可用浏览器打印 HTML"

    return ReportBundle(html_path=html_path, csv_path=csv_path, pdf_path=pdf_path, pdf_note=pdf_note)

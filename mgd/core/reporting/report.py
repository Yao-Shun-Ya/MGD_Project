"""单病例 HTML 报告（Phase 6）。

- 图像以 base64 内嵌，单个 HTML 文件可直接发给临床/评审；
- 同步导出 CSV（汇总 + 实例明细）；
- PDF：优先 weasyprint（未安装时退化为浏览器打印），绝不假装已生成。
"""
from __future__ import annotations

import base64
import csv
import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ..config import get_settings
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


def build_report(
    result: InferenceResult,
    *,
    model_name: str,
    patient_id: str = "",
    output_dir: str | Path | None = None,
) -> ReportBundle:
    settings = get_settings()
    directory = Path(output_dir) if output_dir else settings.output_dir / "reports"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_id = "".join(ch for ch in (patient_id or "case") if ch.isalnum() or ch in "-_") or "case"
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

    html = f"""<!DOCTYPE html>
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
  <p>病例号：{patient_id or "未填写"} &nbsp;|&nbsp; 模型：{model_name} &nbsp;|&nbsp; 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
<div class="container">
  {warning_html}
  <h2>量化指标</h2>
  {_table(result.metrics)}
  <h2>影像与可解释性</h2>
  {images}
  {instance_html}
  <div class="footer">本报告由 AI 辅助生成，仅用于科研与辅助诊断，最终诊断需由执业医师确认。</div>
</div></body></html>"""

    html_path.write_text(html, encoding="utf-8")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["指标", "数值"])
        writer.writerows(result.metrics)
        writer.writerow([])
        writer.writerow(["置信度评分", confidence.get("score")])
        writer.writerow(["模糊区占比", confidence.get("ambiguous_ratio")])
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
        HTML(string=html).write_pdf(str(pdf_path))
        pdf_note = "已通过 weasyprint 生成"
    except Exception as exc:  # 可选依赖缺失不应阻断报告
        pdf_note = f"PDF 未生成（{type(exc).__name__}），可用浏览器打印 HTML"

    return ReportBundle(html_path=html_path, csv_path=csv_path, pdf_path=pdf_path, pdf_note=pdf_note)

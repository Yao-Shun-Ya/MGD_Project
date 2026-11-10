"""g15 报告区块：旧调用无 5 个新区块且无 N/A 占位；全参数调用五区块齐全（血缘/质控/
分级/分型/解释性）；构造最小合成 InferenceResult（64×128），输出写临时目录。"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import _common

if True:
    from mgd.core.inference.service import InferenceResult
    from mgd.core.provenance import collect_provenance
    from mgd.core.qc.quality import QualityReport
    from mgd.core.atlas.typology import TypeEvidence
    from mgd.core.reporting.report import build_report

BLOCK_MARKERS = {
    "provenance": "<h2>血缘 Provenance</h2>",
    "quality": "<h2>采集质控</h2>",
    "grading": "<h2>分级（双体系对照）</h2>",
    "typology": "<h2>形态分型证据</h2>",
    "explain": "<h2>解释性（归因忠实度）</h2>",
}
NA_PLACEHOLDER = "N/A（未计算）"


def make_result() -> InferenceResult:
    shape = (64, 128)
    rng = np.random.default_rng(7)
    gray = (rng.random(shape) * 255).astype(np.uint8)
    color = np.stack([gray, gray, gray], axis=2)
    return InferenceResult(
        image_rgb=color, prob_map=gray.astype(np.float32) / 255.0,
        binary_mask=(gray > 128).astype(np.uint8), overlay=color, heatmap=color,
        contour=(gray > 200).astype(np.uint8), skeleton=(gray > 200).astype(np.uint8),
        instances=(gray > 200).astype(np.uint8),
        metrics=[["腺体缺失率 R_loss (%)", "12.30"], ["Meiboscore", "1"]],
        requested_threshold=0.7, applied_threshold=0.7, threshold_escalated=False,
        warning=None, extra={},
    )


def run() -> dict:
    gate = _common.Gate("g15_report_blocks")
    tmp = Path(tempfile.mkdtemp(prefix="g15_"))

    bundle = build_report(make_result(), model_name="MGD-Former", patient_id="G15OLD",
                          output_dir=tmp / "old")
    gate.check(bundle.html_path.is_file(), "旧调用（无扩展参数）成功产出 HTML",
               f"build_report 写盘失败：{bundle.html_path}")
    old_html = bundle.html_path.read_text(encoding="utf-8", errors="replace")

    for key, marker in BLOCK_MARKERS.items():
        gate.check(marker not in old_html, f"旧调用 HTML 无「{marker[4:-5]}」区块",
                   "None 参数必须完全不渲染区块（不得出现空表头）")
    gate.check(NA_PLACEHOLDER not in old_html,
               "旧调用 HTML 无 N/A（未计算）占位（未传参区块不渲染任何行）",
               "占位符出现说明某个扩展区块被空对象触发")

    provenance = collect_provenance(model_name="MGD-Former", threshold=0.7,
                                    tile_size=384, tile_overlap=0.25, tta="off")
    quality = QualityReport(score=95.0, flags=[], reasons=[],
                            metrics={"blur": 123.4, "specular_ratio": 0.002})
    typology = [TypeEvidence(key="dropout", present=True, value=12.0, threshold=20.0,
                             unit="%", definition="GA < 0.5 × 参考密度")]
    explain = {"verdict": "pass", "deletion_auc": 0.21, "insertion_auc": 0.63}
    grading = {"arita_meiboscore": 1, "pult_meiboscale": 2}

    full = build_report(make_result(), model_name="MGD-Former", patient_id="G15FULL",
                        output_dir=tmp / "full", quality=quality, grading=grading,
                        typology=typology, provenance=provenance, explain=explain)
    gate.check(full.html_path.is_file(), "全参数调用成功产出 HTML")
    full_html = full.html_path.read_text(encoding="utf-8", errors="replace")
    for key, marker in BLOCK_MARKERS.items():
        gate.check(marker in full_html, f"全参数 HTML 含「{marker[4:-5]}」区块",
                   f"扩展参数 {key} 传入后未渲染：检查 _extended_sections 分支")
    gate.check(full.csv_path.is_file(), "全参数调用产出 CSV（v2.4.0 追加行）")

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    gate.note("合成 64×128 影像 + 最小 Provenance/QualityReport/TypeEvidence/explain；"
              "旧调用与全参数调用双快照对照")
    return gate.result()


if __name__ == "__main__":
    _common.run_as_main(run)

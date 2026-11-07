"""True Grad-CAM、不确定性、置信度与 HTML 报告。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from mgd.core.imaging.visualize import TrueGradCAM, confidence_score, pixel_entropy, tta_variance
from mgd.core.inference.service import InferenceResult
from mgd.core.reporting.report import build_report


def _toy_model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 1, 1),
    )


def test_gradcam_shape_and_range():
    model = _toy_model().eval()
    tensor = torch.randn(1, 1, 64, 128, requires_grad=True)
    cam = TrueGradCAM(model).generate(tensor)
    assert cam.shape == (64, 128)
    assert 0.0 <= float(cam.min()) and float(cam.max()) <= 1.0


def test_entropy_and_tta_variance_bounds():
    prob = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    assert pixel_entropy(prob)[0, 1] == 1.0
    assert pixel_entropy(prob)[0, 0] < 1e-3
    stack = np.stack([np.zeros((2, 2)), np.ones((2, 2))]).astype(np.float32)
    variance = tta_variance(stack)
    assert variance.shape == (2, 2)
    assert variance.max() <= 0.25 + 1e-6


def test_confidence_flags_ambiguous_pixels():
    confident = np.zeros((10, 10), dtype=np.float32)
    confident[:] = 0.99
    assert confidence_score(confident)["need_review"] is False
    ambiguous = np.full((10, 10), 0.5, dtype=np.float32)
    result = confidence_score(ambiguous)
    assert result["need_review"] is True
    assert result["score"] < 10.0


def _fake_result() -> InferenceResult:
    h, w = 32, 64
    rows = [["腺体缺失率 R_loss (%)", "12.50"], ["Meiboscore 分级", "1（轻度 MGD）"],
            ["检测腺体数量", "3"]]
    df = pd.DataFrame([
        {"实例编号": "#1", "面积(px)": 120, "长轴(px)": 40.0, "短轴(px)": 8.0,
         "弯曲度τ": 1.02, "平均曲率κ": 0.01, "最大曲率κ": 0.03, "相对延伸比": 0.7, "是否截断": "否"}
    ])
    return InferenceResult(
        image_rgb=np.zeros((h, w, 3), np.uint8),
        prob_map=np.zeros((h, w), np.float32),
        binary_mask=np.zeros((h, w), np.float32),
        overlay=np.zeros((h, w, 3), np.uint8),
        heatmap=np.zeros((h, w, 3), np.uint8),
        contour=np.zeros((h, w, 3), np.uint8),
        skeleton=np.zeros((h, w, 3), np.uint8),
        instances=np.zeros((h, w, 3), np.uint8),
        metrics=rows,
        requested_threshold=0.7,
        applied_threshold=0.7,
        threshold_escalated=False,
        warning="模糊区占比超过阈值，建议人工复核",
        extra={
            "confidence": {"score": 55.0, "ambiguous_ratio": 0.3, "need_review": True,
                           "mean_entropy": 0.4},
            "instance_df": df,
            "gradcam": np.zeros((h, w, 3), np.uint8),
            "uncertainty_image": np.zeros((h, w, 3), np.uint8),
            "dual_task": True,
        },
    )


def test_html_report_is_self_contained(tmp_path):
    bundle = build_report(_fake_result(), model_name="MGD-PriorNet", patient_id="case-001",
                          output_dir=tmp_path)
    html = bundle.html_path.read_text(encoding="utf-8")
    assert "data:image/png;base64," in html
    assert "MGD 睑板腺智能诊断报告" in html
    assert "腺体实例明细" in html
    assert "复核提示" in html
    assert bundle.csv_path.is_file()
    assert "实例编号" in bundle.csv_path.read_text(encoding="utf-8-sig")

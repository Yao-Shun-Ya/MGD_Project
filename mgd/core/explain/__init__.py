"""WP-3 可解释性包：真归因 + 忠实度证据，替代「概率伪彩当热力图」。

四个层次的归因能力：
- layers.LayerCAM / layers.ScoreCAM：基于 hook 的逐层归因（梯度加权 / 掩膜前向打分）；
- occlusion.occlusion_map：滑窗遮挡重要度（predict_fn 注入，与模型完全解耦）；
- attribution.instance_attribution：实例粒度遮挡归因表；
- faithfulness.deletion_auc / insertion_auc / road_score / faithfulness_report：
  归因质量的可检验证据（deletion/insertion AUC + 随机归因对照），防止
  「看起来像热力图」的伪归因混入交付。
"""
from __future__ import annotations

from .attribution import INSTANCE_COLUMNS, instance_attribution
from .faithfulness import deletion_auc, faithfulness_report, insertion_auc, road_score
from .layers import LayerCAM, ScoreCAM
from .occlusion import OcclusionConfig, occlusion_map

__all__ = [
    "LayerCAM",
    "ScoreCAM",
    "OcclusionConfig",
    "occlusion_map",
    "instance_attribution",
    "INSTANCE_COLUMNS",
    "deletion_auc",
    "insertion_auc",
    "road_score",
    "faithfulness_report",
]

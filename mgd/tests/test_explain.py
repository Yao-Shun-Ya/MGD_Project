"""WP-3 explain 包测试：微型模型 + CPU + 离线，总耗时 <10s。

覆盖：
1. LayerCAM 输出形状/范围 + 对亮方块区域有响应（真归因定位）；
2. ScoreCAM 形状/范围（自动探测 + 显式层 + max_channels 批量化）；
3. occlusion_map 高值区域重要度 > 平坦区域；max_windows 超限 stride 翻倍；
4. 忠实度防回归门：真实归因 pass、固定 seed 随机归因 fail（两条都必须成立）；
5. hook 清理：generate 前后 model 的 hook 计数不变（含异常路径）；
6. 审计回归：CAM NaN/全零守卫、调用方梯度保护、全正 CAM 峰值归一化、
   fp16 输入、忠实度输入有限性/范围校验、OcclusionConfig 浮点规范化。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mgd.core.explain import (
    LayerCAM,
    OcclusionConfig,
    ScoreCAM,
    deletion_auc,
    faithfulness_report,
    instance_attribution,
    insertion_auc,
    occlusion_map,
    road_score,
)

#: 亮方块在 32x32 输入中的位置
_SQUARE = (slice(12, 20), slice(12, 20))


def _hook_count(model: torch.nn.Module) -> int:
    """统计模型全部子模块上的 hook 数量（forward / pre / backward / full backward）。

    torch 对部分 hook 字典是懒初始化的，统一用 getattr 兜底空字典。
    """
    hook_attrs = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_full_backward_hooks")
    total = 0
    for module in model.modules():
        for attr in hook_attrs:
            total += len(getattr(module, attr, {}))
    return total


def _fake_model() -> nn.Sequential:
    """手工构造权重（无训练）：8x8 亮方块经「均值滤波 → 中心加权」产生高响应。

    conv1 = 逐通道均值滤波（通道系数递增，保证激活通道不完全相同）；
    conv2 = 中心权重 2.0/8 主导、其余为 0，偏置 0；Sigmoid 输出单通道概率。
    """
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 1, 3, padding=1),
        nn.Sigmoid(),
    )
    with torch.no_grad():
        model[0].weight.zero_()
        for c in range(8):
            model[0].weight[c, 0] = (0.6 + 0.1 * c) / 9.0
        model[0].bias.zero_()
        model[2].weight.zero_()
        model[2].weight[:, :, 1, 1] = 2.0 / 8.0
        model[2].bias.zero_()
    return model


def _square_tensor() -> torch.Tensor:
    """1x1x32x32：背景 0.0，中心 8x8 亮方块 1.0。"""
    image = np.zeros((32, 32), dtype=np.float32)
    image[_SQUARE] = 1.0
    return torch.from_numpy(image)[None, None].float()


def _square_response(cam: np.ndarray) -> tuple[float, float]:
    mask = np.zeros((32, 32), dtype=bool)
    mask[_SQUARE] = True
    return float(cam[mask].mean()), float(cam[~mask].mean())


# --------------------------------------------------------------------------- #
# 1. LayerCAM
# --------------------------------------------------------------------------- #

def test_layercam_localizes_bright_square_with_full_resolution_shape():
    model = _fake_model()
    tensor = _square_tensor()
    cam = LayerCAM(model).generate(tensor, channel=0)
    assert cam.shape == (32, 32)  # 全分辨率：与输入空间形状一致
    assert cam.dtype == np.float32
    assert float(cam.min()) >= 0.0 and float(cam.max()) <= 1.0
    inside, outside = _square_response(cam)
    assert inside - outside > 0.1  # 对亮方块区域确有响应


def test_layercam_explicit_target_layer_and_out_size():
    model = _fake_model()
    tensor = _square_tensor()
    cam = LayerCAM(model, target_layers=[model[0]]).generate(tensor, channel=0, out_size=(16, 16))
    assert cam.shape == (16, 16)
    assert float(cam.min()) >= 0.0 and float(cam.max()) <= 1.0
    mask = np.zeros((16, 16), dtype=bool)
    mask[6:10, 6:10] = True  # 8x8 方块在 16x16 中的位置
    assert float(cam[mask].mean()) > float(cam[~mask].mean())


def test_layercam_rejects_non_single_batch():
    with pytest.raises(ValueError):
        LayerCAM(_fake_model()).generate(torch.zeros(2, 1, 32, 32))


# --------------------------------------------------------------------------- #
# 2. ScoreCAM
# --------------------------------------------------------------------------- #

def test_scorecam_auto_probe_localizes_and_respects_range():
    model = _fake_model()
    cam = ScoreCAM(model, max_channels=4).generate(_square_tensor(), channel=0)
    assert cam.shape == (32, 32)
    assert cam.dtype == np.float32
    assert float(cam.min()) >= 0.0 and float(cam.max()) <= 1.0
    inside, outside = _square_response(cam)
    assert inside - outside > 0.1


def test_scorecam_explicit_layer_batches_all_candidate_masks():
    model = _fake_model()
    # 显式第一层卷积（8 通道）+ max_channels=4 → 4 个掩膜拼一个 batch 一次前向
    cam = ScoreCAM(model, target_layer=model[0], max_channels=4).generate(_square_tensor(), channel=0)
    assert cam.shape == (32, 32)
    assert float(cam.min()) >= 0.0 and float(cam.max()) <= 1.0
    inside, outside = _square_response(cam)
    assert inside - outside > 0.1


def test_scorecam_rejects_invalid_channel():
    with pytest.raises(ValueError):
        ScoreCAM(_fake_model(), max_channels=4).generate(_square_tensor(), channel=9)


# --------------------------------------------------------------------------- #
# 3. occlusion_map
# --------------------------------------------------------------------------- #

def test_occlusion_map_prefers_high_value_region_over_flat_region():
    image = np.full((64, 128), 0.2, dtype=np.float32)
    image[8:24, 16:48] = 0.9  # 高值块，其余平坦

    result = occlusion_map(lambda img: img, image, OcclusionConfig(window=32, stride=16, fill="mean"))
    prob_map = result["map"]
    assert prob_map.shape == image.shape and prob_map.dtype == np.float32
    assert float(prob_map.min()) >= 0.0 and float(prob_map.max()) <= 1.0
    block = float(prob_map[10:22, 20:44].mean())
    flat = float(prob_map[40:60, 80:120].mean())
    assert block > flat + 0.1
    assert result["windows"] == 3 * 7  # ceil 布点：3 x 7 个窗口
    assert result["scale"] == 1.0
    assert result["note"] == ""


def test_occlusion_map_zero_fill_still_ranks_block_first():
    image = np.full((64, 128), 0.2, dtype=np.float32)
    image[8:24, 16:48] = 0.9
    result = occlusion_map(lambda img: img, image, OcclusionConfig(window=32, stride=16, fill="zero"))
    prob_map = result["map"]
    assert float(prob_map[10:22, 20:44].mean()) > float(prob_map[40:60, 80:120].mean()) + 0.1


def test_occlusion_map_doubles_stride_when_exceeding_max_windows():
    image = np.full((64, 128), 0.5, dtype=np.float32)
    config = OcclusionConfig(window=16, stride=8, max_windows=50)
    # 8 步长时 7 x 15 = 105 个窗口 > 50；翻倍到 16 → 4 x 8 = 32 个，合规
    result = occlusion_map(lambda img: img, image, config)
    assert result["windows"] == 32
    assert result["scale"] == 2.0
    assert "stride" in result["note"] and "max_windows" in result["note"]


def test_occlusion_config_rejects_unknown_fill():
    with pytest.raises(ValueError):
        OcclusionConfig(fill="noise")


# --------------------------------------------------------------------------- #
# 4. 忠实度防回归门
# --------------------------------------------------------------------------- #

def _bimodal_prob_map() -> np.ndarray:
    """64x64 合成双峰概率图：两个高概率方块 + 零背景。"""
    prob = np.zeros((64, 64), dtype=np.float32)
    prob[4:28, 4:28] = 0.95
    prob[36:60, 36:60] = 0.85
    return prob


def test_faithfulness_gate_real_attribution_passes():
    prob = _bimodal_prob_map()
    report = faithfulness_report(prob, prob, random_seed=42)
    assert report["verdict"] == "pass"
    assert report["deletion_auc"] <= report["random_deletion_auc"] - 0.05
    assert report["insertion_auc"] >= report["random_insertion_auc"] + 0.05
    assert report["road"] == pytest.approx(report["insertion_auc"] - report["deletion_auc"])


def test_faithfulness_gate_random_attribution_fails():
    prob = _bimodal_prob_map()
    noise = np.random.default_rng(42).uniform(size=prob.shape).astype(np.float32)
    report = faithfulness_report(prob, noise, random_seed=42)
    # 随机归因不可能同时显著优于自身的随机对照 → fail（确定性判据）
    assert report["verdict"] == "fail"


def test_faithfulness_auc_directionality_and_shape_check():
    prob = _bimodal_prob_map()
    noise = np.random.default_rng(42).uniform(size=prob.shape).astype(np.float32)
    # 方向性：真实归因 deletion 更低、insertion 更高
    assert deletion_auc(prob, prob) < deletion_auc(prob, noise)
    assert insertion_auc(prob, prob) > insertion_auc(prob, noise)
    # road（简化代理）对真实归因更高
    assert road_score(prob, prob) > road_score(prob, noise)
    with pytest.raises(ValueError):
        deletion_auc(prob, np.zeros((8, 8), dtype=np.float32))


# --------------------------------------------------------------------------- #
# 5. hook 清理
# --------------------------------------------------------------------------- #

def test_hooks_cleaned_after_generate_and_on_failure_paths():
    model = _fake_model()
    tensor = _square_tensor()
    before = _hook_count(model)
    assert before == 0  # 假模型自身不带 hook

    LayerCAM(model).generate(tensor, channel=0)
    assert _hook_count(model) == before
    ScoreCAM(model, max_channels=4).generate(tensor, channel=0)
    assert _hook_count(model) == before

    # 异常路径：非法通道在 hook 挂载之后抛出，finally 必须清理干净
    with pytest.raises(ValueError):
        LayerCAM(model).generate(tensor, channel=9)
    assert _hook_count(model) == before
    with pytest.raises(ValueError):
        ScoreCAM(model, max_channels=4).generate(tensor, channel=9)
    assert _hook_count(model) == before


class _Bypass(nn.Module):
    """输出与卷积分支无关（×0.0）：制造「梯度全零」的探测失败场景。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 1:2] * 0.0


def test_layer_probe_failure_raises_honest_runtime_error():
    model = nn.Sequential(nn.Conv2d(1, 2, 3, padding=1), _Bypass())
    tensor = torch.zeros(1, 1, 16, 16)
    before = _hook_count(model)
    with pytest.raises(RuntimeError) as excinfo:
        LayerCAM(model).generate(tensor)
    assert "未找到有梯度响应的卷积层" in str(excinfo.value)
    with pytest.raises(RuntimeError) as excinfo:
        ScoreCAM(model, max_channels=4).generate(tensor)
    assert "未找到有梯度响应的卷积层" in str(excinfo.value)
    assert _hook_count(model) == before  # 失败路径同样不留 hook


# --------------------------------------------------------------------------- #
# 6. instance_attribution
# --------------------------------------------------------------------------- #

def test_instance_attribution_ranks_by_probability_drop():
    image = np.full((32, 32), 0.1, dtype=np.float32)
    image[2:10, 2:10] = 0.9  # 实例 1：亮，遮挡掉分大
    image[20:28, 20:28] = 0.4  # 实例 2：中
    instances = np.zeros((32, 32), dtype=np.int32)
    instances[2:10, 2:10] = 1
    instances[20:28, 20:28] = 2

    frame = instance_attribution(lambda img: img, image, instances)
    assert list(frame.columns) == ["实例编号", "原始概率均值", "遮挡后概率下降均值", "归因排名"]
    row1 = frame[frame["实例编号"] == 1].iloc[0]
    row2 = frame[frame["实例编号"] == 2].iloc[0]
    assert float(row1["原始概率均值"]) == pytest.approx(0.9)
    assert int(row1["归因排名"]) == 1 and int(row2["归因排名"]) == 2
    assert float(row1["遮挡后概率下降均值"]) > float(row2["遮挡后概率下降均值"]) > 0


def test_instance_attribution_truncates_by_area_beyond_max_instances():
    image = np.full((32, 32), 0.1, dtype=np.float32)
    image[2:10, 2:10] = 0.9
    image[20:28, 20:28] = 0.4
    instances = np.zeros((32, 32), dtype=np.int32)
    instances[2:10, 2:10] = 1
    instances[20:28, 20:28] = 2
    instances[14:18, 14:18] = 3  # 16 px 小实例，面积最小

    frame = instance_attribution(lambda img: img, image, instances, max_instances=2)
    assert len(frame) == 2
    assert set(frame["实例编号"]) == {1, 2}  # 按面积取前 2（64 px > 16 px）
    assert sorted(int(r) for r in frame["归因排名"]) == [1, 2]

    empty = instance_attribution(lambda img: img, image, np.zeros_like(instances))
    assert len(empty) == 0
    assert list(empty.columns) == ["实例编号", "原始概率均值", "遮挡后概率下降均值", "归因排名"]


# --------------------------------------------------------------------------- #
# 7. 审计回归：NaN 守卫 / 调用方梯度保护 / 归一化 / fp16 / 输入校验
# --------------------------------------------------------------------------- #

def _nan_weight_model() -> nn.Sequential:
    """把第二层卷积一个权重置 NaN：前向产生全 NaN 输出（训练发散的微缩场景）。"""
    model = _fake_model()
    with torch.no_grad():
        model[2].weight[0, 0, 1, 1] = float("nan")
    return model


def test_layercam_nan_weight_raises_and_cleans_hooks():
    # 修复前：cam.max()=NaN，NaN <= 0.0 为 False → 全 NaN CAM 被静默返回
    model = _nan_weight_model()
    before = _hook_count(model)
    with pytest.raises(RuntimeError, match="NaN/Inf 或全零"):
        LayerCAM(model).generate(_square_tensor(), channel=0)
    assert _hook_count(model) == before  # 失败路径同样不留 hook


def test_scorecam_nan_weight_raises_and_cleans_hooks():
    model = _nan_weight_model()
    before = _hook_count(model)
    with pytest.raises(RuntimeError, match="NaN/Inf 或全零"):
        ScoreCAM(model, max_channels=4).generate(_square_tensor(), channel=0)
    assert _hook_count(model) == before


def test_grad_pass_preserves_caller_parameter_gradients():
    # 预置：conv1.weight.grad = 全零张量（backward 会真实产生非零梯度），
    # conv2.bias.grad = None（backward 会填充）。generate 后必须原样还原。
    model = _fake_model()
    zeros_grad = torch.zeros_like(model[0].weight)
    model[0].weight.grad = zeros_grad
    model[2].bias.grad = None

    cam = LayerCAM(model).generate(_square_tensor(), channel=0)
    assert cam.shape == (32, 32)  # generate 本身正常完成

    # 原来有 grad 的：还原原张量（值不被 backward 梯度污染）
    assert model[0].weight.grad is zeros_grad
    assert float(model[0].weight.grad.abs().max()) == 0.0
    # 原来没有 grad 的：仍为 None
    assert model[0].bias.grad is None
    assert model[2].bias.grad is None


def test_layercam_all_positive_cam_normalizes_peak_to_one():
    # 全正 CAM 场景：全 1 输入 + 全正卷积核 → 所选层激活/梯度处处为正，
    # ReLU 后 CAM 全正（raw min/max ≈ 0.21）。旧公式 (max-min)/(max+eps)
    # 峰值仅 ~0.79；新公式 (max-min)/((max-min)+eps) 峰值 ≈ 1.0。
    model = nn.Sequential(
        nn.Conv2d(1, 1, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(1, 1, 3, padding=1),
        nn.Sigmoid(),
    )
    with torch.no_grad():
        model[0].weight.fill_(1.0 / 9.0)
        model[0].bias.zero_()
        model[2].weight.fill_(1.0 / 9.0)
        model[2].bias.zero_()
    cam = LayerCAM(model, target_layers=[model[0]]).generate(torch.ones(1, 1, 32, 32), channel=0)
    assert abs(float(cam.max()) - 1.0) <= 0.05
    assert float(cam.min()) >= 0.0


def test_scorecam_half_precision_input_returns_float32_cam():
    # torch 2.5.1 CPU 支持 fp16 Conv2d / bilinear interpolate，直接走真实路径
    # （本机已验证可行）；修复前 float32 掩膜栈把 masked_batch 提升为 fp32，
    # 与 fp16 权重 dtype 不匹配，forward 处直接报错。
    model = _fake_model().half()
    tensor = _square_tensor().half()
    cam = ScoreCAM(model, target_layer=model[0], max_channels=4).generate(tensor, channel=0)
    assert cam.shape == (32, 32)
    assert cam.dtype == np.float32
    assert float(cam.min()) >= 0.0 and float(cam.max()) <= 1.0
    inside, outside = _square_response(cam)
    assert inside - outside > 0.1


def test_faithfulness_rejects_nan_inf_and_uint8_prob_map():
    prob = _bimodal_prob_map()
    attr = np.clip(prob, 0.0, 1.0)

    prob_nan = prob.copy()
    prob_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="prob_map 含 NaN"):
        deletion_auc(prob_nan, attr)

    prob_inf = prob.copy()
    prob_inf[0, 0] = np.inf
    with pytest.raises(ValueError, match="prob_map 含 NaN"):
        insertion_auc(prob_inf, attr)

    attr_nan = attr.copy()
    attr_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="attribution 含 NaN"):
        road_score(prob, attr_nan)

    # uint8 掩膜（{0,255}）误当概率图：超出 [0,1] 被范围校验拒绝
    prob_u8 = (prob > 0.5).astype(np.uint8) * 255
    with pytest.raises(ValueError, match="uint8 掩膜"):
        faithfulness_report(prob_u8, attr)

    # 既有合法用例不回归：[0,1] 概率图照常计算
    assert 0.0 <= deletion_auc(prob, attr) <= 1.0
    assert 0.0 <= insertion_auc(prob, attr) <= 1.0


def test_occlusion_config_coerces_float_window_and_stride():
    # 浮点入参 int 截断写回：校验值 == 生效值（window=8.9 不会按 8.9 参与运算）
    config = OcclusionConfig(window=8.9, stride=7.9)
    assert config.window == 8 and isinstance(config.window, int)
    assert config.stride == 7 and isinstance(config.stride, int)

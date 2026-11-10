"""conformal ↔ 评估链路接线测试：pipeline 全链 + stats 接口 + CLI 源码级断言。

全部离线（合成数据、tmp_path、无网络、无权重）——不做 CLI 端到端（真实权重
不存在），eval_cli 仅做参数解析与源码级接线断言。
"""
from __future__ import annotations

import json
import math

import pytest

from mgd.adapters.cli import eval_cli
from mgd.core.conformal.calibrator import conformal_upper_quantile
from mgd.core.conformal.pipeline import (
    attach_calibration_to_report,
    fit_metric_calibration,
    load_latest_metric_calibration,
    save_metric_calibration,
)
from mgd.core.evaluation.evaluator import EvalReport, EvalSampleResult
from mgd.core.evaluation.stats import reference_report_payload
from mgd.core.imaging.metrics import MetricResult

ARCH = "MGD-Former"


# ---------------------------------------------------------------------------
# duck EvalReport：与 evaluator.EvalReport 同构的最小对象（pipeline 只用这些属性）
# ---------------------------------------------------------------------------

class _DuckMetrics:
    def __init__(self, dice):
        self.dice = dice


class _DuckSample:
    def __init__(self, dice=None, r_pred=None, r_gt=None):
        self.metrics = _DuckMetrics(dice)
        self.r_loss_pred = r_pred
        self.r_loss_gt = r_gt


class _DuckReport:
    def __init__(self, samples, arch=ARCH, scope="test"):
        self.per_sample = samples
        self.arch = arch
        self.scope = scope


def _big_report(n: int = 40) -> _DuckReport:
    """n=40 的确定性合成报告：Dice 递减、R_loss 残差线性增长。"""
    samples = []
    for i in range(n):
        dice = 0.95 - 0.004 * i  # 0.95 → 0.794，全有限
        r_pred = 10.0 + 0.1 * i
        r_gt = r_pred - (0.2 + 0.01 * i)  # 残差 0.2 → 0.59
        samples.append(_DuckSample(dice=dice, r_pred=r_pred, r_gt=r_gt))
    return _DuckReport(samples)


# ---------------------------------------------------------------------------
# fit_metric_calibration：两口径拟合 / 稳定 id / NaN 剔除 / 样本不足降级
# ---------------------------------------------------------------------------

def test_fit_produces_both_active_blocks():
    payload = fit_metric_calibration(_big_report(40), alpha=0.10)
    dice, rloss = payload["dice_coverage"], payload["rloss_interval"]
    assert dice["skipped_reason"] is None and dice["bound"] is not None
    assert rloss["skipped_reason"] is None and rloss["q"] is not None
    assert dice["n"] == 40 and rloss["n"] == 40
    assert payload["n_dice"] == 40 and payload["n_rloss"] == 40
    assert payload["alpha"] == pytest.approx(0.10)
    assert dice["coverage"] == pytest.approx(0.90)
    assert payload["format_version"] == 1
    assert payload["arch"] == ARCH and payload["scope"] == "test"
    assert len(payload["calibration_id"]) == 12
    int(payload["calibration_id"], 16)  # 12 位十六进制（sha256 前缀）


def test_calibration_id_stable_for_identical_input():
    first = fit_metric_calibration(_big_report(40), alpha=0.10)
    second = fit_metric_calibration(_big_report(40), alpha=0.10)
    assert first["calibration_id"] == second["calibration_id"]


def test_dice_bound_and_rloss_q_match_conformal_math():
    payload = fit_metric_calibration(_big_report(40), alpha=0.10)
    bound = payload["dice_coverage"]["bound"]
    assert 0.0 < bound <= 1.0
    dices = [0.95 - 0.004 * i for i in range(40)]
    expected_bound = 1.0 - conformal_upper_quantile([1.0 - d for d in dices], 0.10)
    assert bound == pytest.approx(expected_bound)

    q = payload["rloss_interval"]["q"]
    assert q >= 0.0
    residuals = [0.2 + 0.01 * i for i in range(40)]
    assert q == pytest.approx(conformal_upper_quantile(residuals, 0.10))


def test_fit_ignores_nan_samples_per_scope():
    samples = [_DuckSample(dice=0.9 - 0.01 * i, r_pred=10.0 + i, r_gt=10.0 + i - 0.5)
               for i in range(40)]
    samples[0].metrics.dice = float("nan")   # dice 口径剔除 1 例
    samples[1].r_loss_pred = float("nan")    # rloss 口径剔除 1 例
    samples[2].r_loss_gt = None              # 缺失值同样剔除
    payload = fit_metric_calibration(_DuckReport(samples), alpha=0.10)
    assert payload["n_dice"] == 39 and payload["n_rloss"] == 38
    assert payload["dice_coverage"]["n"] == 39
    assert payload["rloss_interval"]["n"] == 38
    assert payload["dice_coverage"]["skipped_reason"] is None
    assert payload["rloss_interval"]["skipped_reason"] is None


def test_fit_with_insufficient_samples_skips_both_blocks():
    payload = fit_metric_calibration(_big_report(5), alpha=0.10)
    dice, rloss = payload["dice_coverage"], payload["rloss_interval"]
    assert dice["bound"] is None and rloss["q"] is None
    assert dice["skipped_reason"] and "不足" in dice["skipped_reason"]
    assert "n=5" in dice["skipped_reason"] and "1/alpha" in dice["skipped_reason"]
    assert rloss["skipped_reason"] and "n=5" in rloss["skipped_reason"]
    assert dice["n"] == 5 and rloss["n"] == 5


def test_fit_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        fit_metric_calibration(_big_report(40), alpha=0.9)
    with pytest.raises(ValueError):
        fit_metric_calibration(_big_report(40), alpha=0.0)


# ---------------------------------------------------------------------------
# attach_calibration_to_report
# ---------------------------------------------------------------------------

def test_attach_sets_conformal_and_extra_calibration_id():
    calibration = fit_metric_calibration(_big_report(40), alpha=0.10)
    payload: dict = {"metrics": {"dice": 0.9}, "extra": {"split_mode": "holdout"}}
    attached = attach_calibration_to_report(payload, calibration)
    assert attached is payload  # 就地修改并返回同一 dict
    assert payload["conformal"]["calibration_id"] == calibration["calibration_id"]
    assert payload["extra"]["calibration_id"] == calibration["calibration_id"]
    assert payload["extra"]["split_mode"] == "holdout"  # 已有 extra 被更新而非覆盖


def test_attach_creates_missing_extra():
    calibration = fit_metric_calibration(_big_report(40), alpha=0.10)
    payload: dict = {"metrics": {}}
    attach_calibration_to_report(payload, calibration)
    assert payload["extra"]["calibration_id"] == calibration["calibration_id"]


def test_attach_none_leaves_payload_untouched():
    payload: dict = {"metrics": {}}
    attach_calibration_to_report(payload, None)
    assert "conformal" not in payload
    assert "extra" not in payload


# ---------------------------------------------------------------------------
# save / load_latest：持久化与“损坏不抛错”
# ---------------------------------------------------------------------------

def test_save_writes_file_into_settings_calibration_dir(settings):
    calibration = fit_metric_calibration(_big_report(40), alpha=0.10)
    path = save_metric_calibration(calibration, settings)
    expected = settings.calibration_dir / f"cal_{ARCH}_test_{calibration['calibration_id']}.json"
    assert path == expected and path.is_file()
    restored = json.loads(path.read_text(encoding="utf-8"))
    assert restored["calibration_id"] == calibration["calibration_id"]
    assert restored["dice_coverage"]["bound"] == pytest.approx(calibration["dice_coverage"]["bound"])
    assert restored["version"] == 1 and restored["name"] == path.stem


def test_load_latest_returns_newest_matching_arch(settings):
    old = fit_metric_calibration(_big_report(40), alpha=0.10)
    save_metric_calibration(old, settings)
    latest = load_latest_metric_calibration(ARCH, settings)
    assert latest is not None
    assert latest["calibration_id"] == old["calibration_id"]
    # 其他 arch 的文件不被误取
    other = fit_metric_calibration(_big_report(40), alpha=0.20)
    other = dict(other, arch="UNet")
    save_metric_calibration(other, settings)
    latest = load_latest_metric_calibration(ARCH, settings)
    assert latest["calibration_id"] == old["calibration_id"]


def test_load_latest_prefers_larger_created_at(settings):
    # 手工落两份不同 created_at 的文件（save_calibration 会覆盖 created_at，
    # 故直接写盘控制时间戳），验证“取 created_at 最新”的排序语义。
    settings.calibration_dir.mkdir(parents=True, exist_ok=True)
    old_payload = {"calibration_id": "aaa000000000", "created_at": "2020-01-01T00:00:00+00:00"}
    new_payload = {"calibration_id": "bbb000000000", "created_at": "2030-01-01T00:00:00+00:00"}
    (settings.calibration_dir / f"cal_{ARCH}_test_2020.json").write_text(
        json.dumps(old_payload), encoding="utf-8")
    (settings.calibration_dir / f"cal_{ARCH}_test_2030.json").write_text(
        json.dumps(new_payload), encoding="utf-8")
    latest = load_latest_metric_calibration(ARCH, settings)
    assert latest["calibration_id"] == "bbb000000000"


def test_load_latest_survives_missing_or_corrupt_directory(settings):
    assert load_latest_metric_calibration(ARCH, settings) is None  # 目录不存在
    settings.calibration_dir.mkdir(parents=True, exist_ok=True)
    (settings.calibration_dir / f"cal_{ARCH}_test_dead.json").write_text(
        "not json {{{", encoding="utf-8")
    (settings.calibration_dir / f"cal_{ARCH}_test_list.json").write_text(
        "[1, 2, 3]", encoding="utf-8")
    (settings.calibration_dir / f"cal_{ARCH}_test_nodate.json").write_text(
        json.dumps({"calibration_id": "x"}), encoding="utf-8")
    assert load_latest_metric_calibration(ARCH, settings) is None  # 全部损坏 → None

    good = fit_metric_calibration(_big_report(40), alpha=0.10)
    save_metric_calibration(good, settings)
    latest = load_latest_metric_calibration(ARCH, settings)
    assert latest["calibration_id"] == good["calibration_id"]  # 损坏文件被跳过不影响好文件


# ---------------------------------------------------------------------------
# stats.reference_report_payload：conformal 关键字 + 严格 JSON 可序列化
# ---------------------------------------------------------------------------

def _real_report(n: int = 12) -> EvalReport:
    samples = []
    for i in range(n):
        samples.append(
            EvalSampleResult(
                f"case_{i:02d}",
                MetricResult(dice=0.90 - 0.005 * i, iou=0.60, hd95=3.0, assd=1.0,
                             precision=0.9, recall=0.8),
                eyelid_dice=0.88,
                cldice=0.85,
                r_loss_pred=10.0 + 0.2 * i,
                r_loss_gt=10.0 + 0.2 * i - 0.4,
            )
        )
    summary = {key: 0.5 + 0.01 * i for i, key in
               enumerate(("dice", "iou", "hd95", "assd", "precision", "recall"))}
    return EvalReport(arch=ARCH, weights="w.pth", scope="test", scope_label="独立测试集",
                      samples=n, summary=summary, per_sample=samples, device="CPU")


def _reject_constant(value: str):
    raise AssertionError(f"JSON 含非有限常量：{value}")


def test_reference_payload_with_conformal_is_strict_json(settings):
    report = _real_report(12)
    calibration = fit_metric_calibration(report, alpha=0.10)
    payload = reference_report_payload(report, split_signature="sig123",
                                       extra={"run_id": "r1"}, conformal=calibration)
    assert payload["conformal"]["calibration_id"] == calibration["calibration_id"]
    assert payload["extra"]["calibration_id"] == calibration["calibration_id"]
    assert payload["extra"]["run_id"] == "r1"
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False)  # NaN/Inf 直接抛错
    roundtripped = json.loads(text, parse_constant=_reject_constant)  # 双保险
    assert roundtripped["conformal"]["rloss_interval"]["q"] >= 0.0


def test_reference_payload_without_conformal_unchanged():
    payload = reference_report_payload(_real_report(12), split_signature="sig123")
    assert "conformal" not in payload  # 旧调用零变化
    assert "extra" not in payload


# ---------------------------------------------------------------------------
# eval_cli：不做端到端（真实权重不存在），做参数解析 + 源码级接线断言
# ---------------------------------------------------------------------------

def test_eval_cli_parses_conformal_flags():
    args = eval_cli.build_parser().parse_args([])
    assert args.conformal_alpha == pytest.approx(0.10)
    assert args.no_conformal is False
    args = eval_cli.build_parser().parse_args(["--conformal-alpha", "0.2", "--no-conformal"])
    assert args.conformal_alpha == pytest.approx(0.2)
    assert args.no_conformal is True


def test_eval_cli_source_wired_to_pipeline():
    source = open(eval_cli.__file__, encoding="utf-8").read()
    for token in ("fit_metric_calibration", "save_metric_calibration",
                  "attach_calibration_to_report", "--no-conformal", "--conformal-alpha",
                  "print_conformal_lines"):
        assert token in source


def test_print_conformal_lines_active_and_skipped(capsys):
    active = {"alpha": 0.10,
              "dice_coverage": {"bound": 0.81, "skipped_reason": None},
              "rloss_interval": {"q": 0.62, "skipped_reason": None}}
    eval_cli.print_conformal_lines(active)
    out = capsys.readouterr().out
    assert "Dice 覆盖下界（1-α=90%）：≥0.81" in out
    assert "R_loss 预测区间：±0.62 pp" in out

    skipped = {"alpha": 0.10,
               "dice_coverage": {"bound": None, "skipped_reason": "有效样本 n=5 不足 1/alpha=10"},
               "rloss_interval": {"q": None, "skipped_reason": "有效样本 n=5 不足 1/alpha=10"}}
    eval_cli.print_conformal_lines(skipped)
    out = capsys.readouterr().out
    assert "不可用（有效样本 n=5 不足 1/alpha=10）" in out
    assert "≥" not in out and "±" not in out


# ---------------------------------------------------------------------------
# pipeline 全链：fit → save → load_latest → attach → reference payload
# ---------------------------------------------------------------------------

def test_full_pipeline_chain_end_to_end(settings):
    report = _real_report(15)
    calibration = fit_metric_calibration(report, alpha=0.10)
    path = save_metric_calibration(calibration, settings)
    loaded = load_latest_metric_calibration(ARCH, settings)
    assert loaded is not None and loaded["calibration_id"] == calibration["calibration_id"]
    assert math.isfinite(loaded["dice_coverage"]["bound"])

    payload = reference_report_payload(
        report, split_signature="sig", assignment_signature="asig",
        extra={"split_mode": "holdout", "fold": 0},
    )
    attach_calibration_to_report(payload, loaded)
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    restored = json.loads(text, parse_constant=_reject_constant)
    assert restored["conformal"]["dice_coverage"]["bound"] == pytest.approx(
        calibration["dice_coverage"]["bound"])
    assert restored["conformal"]["rloss_interval"]["q"] == pytest.approx(
        calibration["rloss_interval"]["q"])
    assert restored["extra"]["calibration_id"] == calibration["calibration_id"]
    assert path.is_file()  # 落盘文件与报告中的 conformal 同源

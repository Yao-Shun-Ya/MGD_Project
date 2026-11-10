"""治理接线（governance wiring）测试：审计事件 / RBAC / conformal 注入 / 报告脱敏 / CLI 门。

全部离线（tmp_path、无网络、无真实权重）。分三层：
1. hooks 单元测试：audit_event / require_or_audit 的成功、上下文合并、非阻塞与严格语义；
2. 源码级接线断言：service / runner / verify / versioning / 三个 CLI 均接入了 hooks；
3. 行为级测试：attach_diagnostic_conformal 注入、build_report 脱敏、CLI RBAC 拒绝
   端到端（拒绝 → access_denied 落审计 → 既有 except 接住 ConfigError → 退出码 2）。

说明：service / reporting 的 import 会连带 torch（~秒级），统一放在用到的测试函数
内部延迟执行；hooks / 源码断言层零重依赖。
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from mgd.adapters.cli import atlas_cli, eval_cli, report_cli
from mgd.core.config import Settings, set_settings
from mgd.core.governance.audit import AuditLog
from mgd.core.governance.deid import pseudonymize
from mgd.core.governance.hooks import audit_event, audit_log_path, require_or_audit
from mgd.core.runtime.errors import ConfigError
from mgd.core.runtime.logging import bind_context, clear_context

#: 全部治理事件的接线清单（event, strict）——hooks 层必须能构造每一种
WIRED_EVENTS: list[tuple[str, bool]] = [
    ("inference", False),
    ("inference_failed", False),
    ("training_done", False),
    ("training_failed", False),
    ("weight_verify", True),
    ("dataset_backup", False),
    ("evaluation", False),
    ("access_denied", True),
]


# ---------------------------------------------------------------- 工具


def _read_audit(settings: Settings) -> list[dict]:
    path = audit_log_path(settings)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _broken_audit_log(monkeypatch: pytest.MonkeyPatch) -> None:
    """把 hooks 命名空间里的 AuditLog 换成 append 必炸的替身（磁盘满 / 权限拒绝的代理）。"""

    class _BrokenAuditLog:
        def __init__(self, path) -> None:
            self.path = path

        def append(self, event, actor, payload):
            raise OSError("disk full（模拟审计写入失败）")

    monkeypatch.setattr("mgd.core.governance.hooks.AuditLog", _BrokenAuditLog)


def _deid_inference_result():
    """最小 InferenceResult（真实类型，供 build_report / attach 函数测试）。"""
    from mgd.core.inference.service import InferenceResult

    rgb = np.zeros((8, 16, 3), np.uint8)
    zero = np.zeros((8, 16), np.float32)
    return InferenceResult(
        image_rgb=rgb,
        prob_map=zero,
        binary_mask=zero,
        overlay=rgb,
        heatmap=rgb,
        contour=rgb,
        skeleton=rgb,
        instances=rgb,
        metrics=[["腺体缺失率 R_loss (%)", "12.50"]],
        requested_threshold=0.7,
        applied_threshold=0.7,
        threshold_escalated=False,
        extra={"confidence": {"score": 55.0, "ambiguous_ratio": 0.1, "need_review": False}},
    )


# ---------------------------------------------------------------- hooks：audit_event


def test_audit_event_appends_to_home_logs_with_context(tmp_path):
    """audit_event 落在 home/logs/audit.jsonl；payload 自动携带日志上下文。"""
    settings = Settings(home=tmp_path)
    bind_context(case_id="c1", run_id="r9", stage="inference", arch="UNet")
    try:
        record = audit_event(settings, "inference", actor="system", payload={"applied_threshold": 0.7})
    finally:
        clear_context()

    assert record is not None
    assert record["event"] == "inference" and record["actor"] == "system"
    payload = record["payload"]
    assert payload["case_id"] == "c1", "case_id 必须从日志上下文自动进入 payload"
    assert payload["run_id"] == "r9"
    assert payload["stage"] == "inference" and payload["arch"] == "UNet"
    assert payload["applied_threshold"] == 0.7, "显式 payload 与上下文合并且不互相覆盖"

    path = audit_log_path(settings)
    assert path == tmp_path / "logs" / "audit.jsonl"
    assert path.is_file()
    ok, msg = AuditLog(path).verify()
    assert (ok, msg) == (True, "ok"), "hooks 写出的记录必须保持哈希链完整"


def test_audit_event_explicit_payload_overrides_context(tmp_path):
    """显式 payload 键优先于日志上下文（上下文只补缺）。"""
    settings = Settings(home=tmp_path)
    bind_context(case_id="ctx-id")
    try:
        record = audit_event(settings, "inference", payload={"case_id": "explicit-id", "note": "x"})
    finally:
        clear_context()
    assert record["payload"]["case_id"] == "explicit-id"
    assert record["payload"]["note"] == "x"
    assert "run_id" not in record["payload"], "未绑定的上下文字段不得出现"


def test_audit_event_nonblocking_failure_returns_none_and_logs_error(tmp_path, caplog, monkeypatch):
    """strict=False：审计写入失败 → 返回 None + ERROR 日志（带堆栈摘要），不抛异常。"""
    settings = Settings(home=tmp_path)
    _broken_audit_log(monkeypatch)

    mgd_logger = logging.getLogger("mgd")
    restore_propagate = mgd_logger.propagate
    mgd_logger.propagate = True  # 防御：此前某测试 configure_logging 关掉了传播
    try:
        with caplog.at_level(logging.ERROR, logger="mgd.governance.hooks"):
            result = audit_event(settings, "training_done", payload={"run_id": "r1"})
    finally:
        mgd_logger.propagate = restore_propagate

    assert result is None, "非阻塞语义：业务拿到 None 而不是异常"
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors and "写入失败" in errors[0].getMessage()
    assert errors[0].exc_info is not None, "必须带堆栈摘要（可定位审计失败原因）"


def test_audit_event_strict_failure_raises(tmp_path, monkeypatch):
    """strict=True：审计失败必须向上抛（治理关键事件不允许无声失败）。"""
    settings = Settings(home=tmp_path)
    _broken_audit_log(monkeypatch)
    with pytest.raises(OSError, match="disk full"):
        audit_event(settings, "weight_verify", payload={"weights": "a.pth"}, strict=True)


def test_all_wired_events_constructible_via_hooks(tmp_path):
    """全部 8 种治理事件都能经 hooks 写入且保持链完整（事件名表 = 接线表）。"""
    settings = Settings(home=tmp_path)
    for event, strict in WIRED_EVENTS:
        record = audit_event(
            settings, event, actor="wiring-test", payload={"probe": event}, strict=strict
        )
        assert record is not None and record["event"] == event

    path = audit_log_path(settings)
    records = _read_audit(settings)
    assert [r["event"] for r in records] == [event for event, _ in WIRED_EVENTS]
    assert all(r["actor"] == "wiring-test" for r in records)
    ok, msg = AuditLog(path).verify()
    assert (ok, msg) == (True, "ok")


# ---------------------------------------------------------------- hooks：require_or_audit


def test_require_or_audit_allows_admin_without_audit_entry(tmp_path):
    """admin 全权放行；放行路径不产生审计噪音。"""
    settings = Settings(home=tmp_path, role="admin")
    assert require_or_audit(settings, "delete_case", entry="mgd-eval") is None
    assert require_or_audit(settings, "manage_audit", entry="x") is None
    assert require_or_audit(settings, "run_inference", entry="y") is None
    assert _read_audit(settings) == []


def test_require_or_audit_denied_writes_access_denied_then_raises(tmp_path, monkeypatch):
    """tech + delete_case → access_denied 审计先落地，ConfigError 随后抛出。"""
    settings = Settings(home=tmp_path, role="tech")
    seen_events: list[str] = []
    original_append = AuditLog.append

    def spy_append(self, event, actor, payload):
        seen_events.append(event)
        return original_append(self, event, actor, payload)

    monkeypatch.setattr(AuditLog, "append", spy_append)

    with pytest.raises(ConfigError, match="无权执行 delete_case"):
        require_or_audit(settings, "delete_case", entry="mgd-eval")

    assert seen_events == ["access_denied"], "拒绝审计必须在异常逃出前完成"
    records = _read_audit(settings)
    assert [r["event"] for r in records] == ["access_denied"]
    payload = records[0]["payload"]
    assert payload["role"] == "tech" and payload["action"] == "delete_case"
    assert payload["entry"] == "mgd-eval"
    ok, msg = AuditLog(audit_log_path(settings)).verify()
    assert (ok, msg) == (True, "ok")


def test_require_or_audit_actor_recorded(tmp_path):
    """access_denied 记录审计主体 actor（默认 local，可显式指定）。"""
    settings = Settings(home=tmp_path, role="tech")
    with pytest.raises(ConfigError):
        require_or_audit(settings, "manage_users", entry="console", actor="tech01")
    records = _read_audit(settings)
    assert records[0]["actor"] == "tech01"


# ---------------------------------------------------------------- attach_diagnostic_conformal


def test_attach_diagnostic_conformal_none_leaves_result_untouched():
    """calibration=None → extra 无 conformal 键、metrics 不加行（无校准 = 零痕迹）。"""
    from mgd.core.inference.service import attach_diagnostic_conformal

    result = _deid_inference_result()
    before_rows = list(result.metrics)
    attach_diagnostic_conformal(result, None)
    attach_diagnostic_conformal(result, {})  # 空字典同样视为无校准
    assert "conformal" not in result.extra
    assert result.metrics == before_rows


def test_attach_diagnostic_conformal_active_adds_extra_and_metrics_row():
    """有效校准 → extra['conformal'] 三字段 + metrics 追加 Dice 覆盖下界行。"""
    from mgd.core.inference.service import attach_diagnostic_conformal

    result = _deid_inference_result()
    calibration = {
        "calibration_id": "abcd1234ef56",
        "alpha": 0.10,
        "dice_coverage": {"bound": 0.8123, "coverage": 0.9, "n": 40},
        "rloss_interval": {"q": 0.62, "coverage": 0.9, "n": 40},
    }
    attach_diagnostic_conformal(result, calibration)
    assert result.extra["conformal"] == {
        "dice_lower_bound": pytest.approx(0.8123),
        "coverage": pytest.approx(0.9),
        "calibration_id": "abcd1234ef56",
    }
    assert result.metrics[-1] == ["Dice 覆盖下界(1-α)", "≥0.81 (校准 id abcd1234ef56)"]


def test_attach_diagnostic_conformal_skipped_bound_writes_extra_without_row():
    """口径被 skipped（bound=None）→ extra 仍写（诚实标记），但不产生 ≥nan 的指标行。"""
    from mgd.core.inference.service import attach_diagnostic_conformal

    result = _deid_inference_result()
    calibration = {
        "calibration_id": "id0000000001",
        "dice_coverage": {"bound": None, "skipped_reason": "有效样本 n=5 不足 1/alpha=10"},
    }
    attach_diagnostic_conformal(result, calibration)
    assert result.extra["conformal"]["dice_lower_bound"] is None
    assert result.extra["conformal"]["calibration_id"] == "id0000000001"
    assert not any("Dice 覆盖下界" in row[0] for row in result.metrics)


# ---------------------------------------------------------------- 源码级接线断言


def _source_of(module) -> str:
    return Path(module.__file__).read_text(encoding="utf-8")


def test_source_wiring_inference_service():
    """service 层：conformal 延迟加载 + 双审计事件 + 注入函数均在源码中。"""
    from mgd.core.inference import service

    source = _source_of(service)
    for token in (
        "load_latest_metric_calibration",  # 延迟导入的 conformal 读取
        "attach_diagnostic_conformal",  # 可测的注入函数
        "audit_event",
        '"inference"',  # 成功事件
        '"inference_failed"',  # 失败事件
    ):
        assert token in source, f"service.py 缺少接线标记 {token}"
    assert "from ..conformal.pipeline import load_latest_metric_calibration" in source, (
        "conformal 必须延迟导入（函数内），不得进顶层 import"
    )


def test_source_wiring_runner_events():
    from mgd.core.training import runner

    source = _source_of(runner)
    for token in ("audit_event", '"training_done"', '"training_failed"'):
        assert token in source, f"runner.py 缺少接线标记 {token}"


def test_source_wiring_verify_strict_event():
    from mgd.core.models import verify

    source = _source_of(verify)
    for token in ("audit_event", '"weight_verify"', "strict=True"):
        assert token in source, f"verify.py 缺少接线标记 {token}（权重治理必须 strict）"


def test_source_wiring_versioning_event():
    from mgd.core.data import versioning

    source = _source_of(versioning)
    for token in ("audit_event", '"dataset_backup"'):
        assert token in source, f"versioning.py 缺少接线标记 {token}"


def test_source_wiring_eval_cli_rbac_and_evaluation():
    source = _source_of(eval_cli)
    for token in ("require_or_audit", '"run_inference"', "audit_event", '"evaluation"'):
        assert token in source, f"eval_cli.py 缺少接线标记 {token}"


def test_source_wiring_report_cli_rbac():
    source = _source_of(report_cli)
    for token in ("require_or_audit", '"export_report"'):
        assert token in source, f"report_cli.py 缺少接线标记 {token}"


def test_source_wiring_atlas_cli_rbac():
    source = _source_of(atlas_cli)
    for token in ("require_or_audit", '"run_inference"'):
        assert token in source, f"atlas_cli.py 缺少接线标记 {token}"


# ---------------------------------------------------------------- CLI RBAC 端到端
#
# 说明：当前权限矩阵把 run_inference / export_report 授予了全部四个角色（含 tech），
# 且 Settings.__post_init__ 拒绝未知角色，因此「合法配置下的 CLI 拒绝」在现矩阵下
# 不可达。端到端拒绝路径改为在治理接缝（hooks.require）处注入 ConfigError——
# 这恰好验证「CLI 既有 except 分支接住 ConfigError → user_message + 退出码 2」
# 这一接线契约本身；矩阵收紧后同一契约自动生效，无需改测试。


def _deny_all_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    """让 hooks.require 一律拒绝（模拟矩阵收紧后的无权角色）。"""

    def deny(role, action):
        raise ConfigError(f"角色 {role} 无权执行 {action}")

    monkeypatch.setattr("mgd.core.governance.hooks.require", deny)


def test_eval_cli_denied_exit_2_and_audited(tmp_path, capsys, monkeypatch):
    """评估被拒 → access_denied 落审计，eval_cli 既有 except MGDError 接住 → 退出码 2。"""
    settings = Settings(home=tmp_path, role="tech")
    set_settings(settings)
    _deny_all_actions(monkeypatch)
    code = eval_cli.main([])
    assert code == 2
    err = capsys.readouterr().err
    assert "无权执行 run_inference" in err and "Traceback" not in err
    records = _read_audit(settings)
    assert [r["event"] for r in records] == ["access_denied"]
    assert records[0]["payload"]["entry"] == "mgd-eval"


def test_report_cli_denied_exit_2_and_audited(tmp_path, capsys, monkeypatch):
    """报告导出被拒 → 拒绝路径不产生任何产物文件。"""
    settings = Settings(home=tmp_path, role="tech")
    set_settings(settings)
    _deny_all_actions(monkeypatch)
    input_path = tmp_path / "m.json"
    input_path.write_text(
        json.dumps({"patient_id": "P1", "loss_percent": 30.0,
                    "measurements": {"ga_percent": 12.5}}),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    code = report_cli.main(["--input", str(input_path), "--html", "--out", str(out_dir)])
    assert code == 2
    err = capsys.readouterr().err
    assert "无权执行 export_report" in err and "Traceback" not in err
    records = _read_audit(settings)
    assert [r["event"] for r in records] == ["access_denied"]
    assert records[0]["payload"]["entry"] == "mgd-report"
    assert not list(out_dir.glob("*_summary.html")), "拒绝时不得产出报告文件"


def test_atlas_cli_denied_exit_2_and_audited(tmp_path, capsys, monkeypatch):
    """图谱 classify 被拒 → access_denied 落审计 + 退出码 2。"""
    settings = Settings(home=tmp_path, role="tech")
    set_settings(settings)
    _deny_all_actions(monkeypatch)
    mask_path = tmp_path / "mask.png"
    mask_path.write_bytes(b"fake-mask-bytes")
    code = atlas_cli.main(["classify", "--mask", str(mask_path)])
    assert code == 2
    err = capsys.readouterr().err
    assert "无权执行 run_inference" in err and "Traceback" not in err
    records = _read_audit(settings)
    assert [r["event"] for r in records] == ["access_denied"]
    assert records[0]["payload"]["entry"] == "mgd-atlas"


def test_report_cli_allowed_for_default_doctor(tmp_path, capsys):
    """默认 doctor 角色：export_report 放行，产物正常生成、无拒绝审计。"""
    settings = Settings(home=tmp_path)  # 默认 role="doctor"
    set_settings(settings)
    input_path = tmp_path / "m.json"
    input_path.write_text(
        json.dumps({"patient_id": "P1", "loss_percent": 30.0,
                    "measurements": {"ga_percent": 12.5}}),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    assert report_cli.main(["--input", str(input_path), "--html", "--out", str(out_dir)]) == 0
    assert list(out_dir.glob("*_summary.html")), "放行路径产物必须正常生成"
    assert _read_audit(settings) == [], "放行不产生审计噪音"


def test_atlas_describe_not_gated(tmp_path, capsys):
    """describe 是纯文档动作，任何角色（含 tech）都不设治理门、不产生审计。"""
    settings = Settings(home=tmp_path, role="tech")
    set_settings(settings)
    assert atlas_cli.main(["describe"]) == 0
    assert "dropout" in capsys.readouterr().out
    assert _read_audit(settings) == []


# ---------------------------------------------------------------- build_report 脱敏


def test_report_with_salt_uses_pseudonym_everywhere(tmp_path):
    """deid_salt 生效：文件名 / HTML 头部 / 全文都不出现原始 ID，只有稳定伪名。"""
    from mgd.core.reporting.report import build_report

    settings = Settings(home=tmp_path, deid_salt="s1")
    set_settings(settings)
    bundle = build_report(
        _deid_inference_result(), model_name="UNet", patient_id="P001",
        output_dir=tmp_path / "r",
    )
    expected = pseudonymize("P001", "s1")
    assert expected and expected != "P001"
    assert bundle.html_path.name.startswith(f"{expected}_"), "文件名必须用伪名"
    assert bundle.csv_path.name.startswith(f"{expected}_")

    html_text = bundle.html_path.read_text(encoding="utf-8")
    assert "已脱敏（伪名）" in html_text, "HTML 头部必须注明脱敏"
    assert expected in html_text
    assert "P001" not in html_text and "P001" not in bundle.html_path.name

    csv_text = bundle.csv_path.read_text(encoding="utf-8-sig")
    assert "P001" not in csv_text, "原始 ID 不得落入 CSV"


def test_report_without_salt_keeps_original_behavior(tmp_path):
    """不传 salt（既有调用）→ 行为与旧版一致：原文显示、无脱敏标注。"""
    from mgd.core.reporting.report import build_report

    settings = Settings(home=tmp_path)
    set_settings(settings)
    bundle = build_report(
        _deid_inference_result(), model_name="UNet", patient_id="P001",
        output_dir=tmp_path / "r",
    )
    assert bundle.html_path.name.startswith("P001_")
    html_text = bundle.html_path.read_text(encoding="utf-8")
    assert "P001" in html_text
    assert "已脱敏" not in html_text


def test_report_with_salt_but_empty_patient_id_unchanged(tmp_path):
    """salt 生效但病例号为空 → 无可脱敏对象，保持 'case'/'未填写' 旧行为。"""
    from mgd.core.reporting.report import build_report

    settings = Settings(home=tmp_path, deid_salt="s1")
    set_settings(settings)
    bundle = build_report(
        _deid_inference_result(), model_name="UNet", patient_id="",
        output_dir=tmp_path / "r",
    )
    assert bundle.html_path.name.startswith("case_")
    html_text = bundle.html_path.read_text(encoding="utf-8")
    assert "未填写" in html_text
    assert "已脱敏（伪名）" not in html_text

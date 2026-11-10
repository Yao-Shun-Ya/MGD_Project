"""日志系统 v2 契约测试：上下文绑定 / 三路输出 / 轮转 / sanitize / mgd-logs CLI。

全部离线（tmp_path），不联网、不真训、不真推理。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mgd.adapters.cli import logs_cli
from mgd.core.runtime import logging as runtime_logging
from mgd.core.runtime.errors import InferenceError
from mgd.core.runtime.logging import (
    LogContext,
    bind_context,
    clear_context,
    configure_logging,
    get_log_context,
    get_logger,
    reset_logging,
)


def _reject_constant(name: str):
    raise ValueError(f"严格 JSON 不允许常量 {name}")


def _strict_loads(text: str):
    return json.loads(text, parse_constant=_reject_constant)


def _strict_lines(path: Path) -> list[dict]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [_strict_loads(line) for line in lines]


def _ctx(record: dict) -> dict:
    return record.get("ctx") or {}


# ---------------------------------------------------------------------------
# 结构化上下文
# ---------------------------------------------------------------------------
def test_bind_context_merges_only_given_fields():
    bind_context(run_id="r1")
    ctx = get_log_context()
    assert ctx.run_id == "r1" and ctx.stage is None
    bind_context(stage="train")
    ctx = get_log_context()
    assert ctx.run_id == "r1" and ctx.stage == "train"
    bind_context(case_id=None)  # None = 未给定，不覆盖
    assert get_log_context().case_id is None


def test_bind_context_custom_keys_go_to_extra():
    bind_context(job="export", attempt=2)
    assert get_log_context().extra == (("job", "export"), ("attempt", "2"))
    bind_context(attempt=3)  # 同名 extra 键覆盖
    assert get_log_context().extra == (("job", "export"), ("attempt", "3"))


def test_clear_context_partial_and_full():
    bind_context(run_id="r", case_id="c", stage="s", job="j")
    clear_context(["case_id"])
    ctx = get_log_context()
    assert ctx.run_id == "r" and ctx.case_id is None and ctx.stage == "s"
    assert ctx.extra == (("job", "j"),)
    clear_context(["job"])
    assert get_log_context().extra == ()
    clear_context()
    assert get_log_context() == LogContext()


def test_reset_logging_clears_contextvars():
    bind_context(run_id="leaky")
    reset_logging()
    assert get_log_context() == LogContext()


# ---------------------------------------------------------------------------
# 三路输出与严格 JSON
# ---------------------------------------------------------------------------
def test_events_jsonl_strict_fields_and_error_structure(tmp_path):
    configure_logging(level="INFO", log_file=tmp_path / "mgd.log")
    log = get_logger("unit.events")
    bind_context(run_id="run42", case_id="abc123", stage="unit", custom="x")
    log.info("normal event")
    try:
        raise InferenceError("boom", hint="check weights")
    except InferenceError:
        log.error("inference failed", exc_info=True)
    clear_context()

    records = _strict_lines(tmp_path / "mgd_events.jsonl")
    assert [r["level"] for r in records] == ["INFO", "ERROR"]
    info, err = records
    for field in ("ts", "level", "logger", "module", "func", "line", "msg", "ctx"):
        assert field in info
    assert info["logger"] == "mgd.unit.events" and info["msg"] == "normal event"
    assert info["ctx"] == {"run_id": "run42", "case_id": "abc123", "stage": "unit", "custom": "x"}
    assert isinstance(info["line"], int)
    assert err["error"] == {"type": "InferenceError", "hint": "check weights"}
    assert "Traceback" in err["exc"] and "boom" in err["exc"]
    # errors.jsonl 只含 WARNING 及以上（同结构）
    error_records = _strict_lines(tmp_path / "errors.jsonl")
    assert [r["msg"] for r in error_records] == ["inference failed"]
    assert error_records[0]["level"] == "ERROR"


def test_mgderror_detected_from_msg_and_args_patterns(tmp_path):
    """调用方习惯不变：MGDError 作为 %s 参数 / 作为 msg 本身，都能结构化出 error.hint。"""
    configure_logging(level="INFO", log_file=tmp_path / "mgd.log")
    log = get_logger("unit.err")
    log.warning("加载失败：%s", InferenceError("bad weight", hint="verify first"))
    log.error(InferenceError("direct error", hint="direct hint"))

    records = _strict_lines(tmp_path / "mgd_events.jsonl")
    assert [r["error"]["hint"] for r in records] == ["verify first", "direct hint"]
    assert [r["error"]["type"] for r in records] == ["InferenceError", "InferenceError"]
    assert all("exc" not in r for r in records)  # 无 exc_info 时不写 exc 字段


def test_nan_and_inf_args_sanitized_no_nan_token(tmp_path):
    configure_logging(level="INFO", log_file=tmp_path / "mgd.log")
    log = get_logger("unit.nan")
    log.warning("payload %s", {"v": float("inf"), "n": float("nan")})
    log.info("val=%s", float("nan"))

    # 严格解析（parse_constant 遇裸 NaN/Infinity 即抛错）；非有限值只能是消息文本或 null
    records = _strict_lines(tmp_path / "mgd_events.jsonl")
    assert records[0]["msg"] == "payload {'v': inf, 'n': nan}"
    assert records[1]["msg"] == "val=nan"


def test_configure_logging_is_idempotent(tmp_path):
    log_file = tmp_path / "mgd.log"
    configure_logging(level="INFO", log_file=log_file)
    configure_logging(level="INFO", log_file=log_file)
    log = get_logger("unit.idem")
    for i in range(3):
        log.info(f"line {i}")
    assert len(_strict_lines(tmp_path / "mgd_events.jsonl")) == 3
    console_handlers = [
        h for h in get_logger().handlers if type(h).__name__ == "StreamHandler"
    ]
    assert len(console_handlers) == 1


def test_console_format_with_context_suffix(tmp_path, capsys):
    configure_logging(level="INFO")
    log = get_logger("unit.console")
    bind_context(run_id="r1", case_id="c2", stage="s3", team="ops")
    log.info("hello world")
    out = capsys.readouterr().out
    assert "hello world [run=r1 case=c2 stage=s3 team=ops]" in out
    clear_context()
    log.info("plain message")
    out = capsys.readouterr().out
    assert "plain message" in out and "[" not in out


def test_console_error_appends_exception_summary(tmp_path, capsys):
    configure_logging(level="INFO")
    log = get_logger("unit.console")
    try:
        raise InferenceError("显存不足", hint="调小分块")
    except InferenceError:
        log.error("推理失败", exc_info=True)
    out = capsys.readouterr().out
    assert "推理失败" in out and "InferenceError: 显存不足" in out
    # INFO 级别不附加 exc 摘要（也无 exc_info），行内不出现 traceback 摘要
    log.info("normal info")
    assert "Traceback" not in capsys.readouterr().out


def test_rotation_creates_backup_file(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_logging, "_ROTATE_MAX_BYTES", 256)
    configure_logging(level="INFO", log_file=tmp_path / "mgd.log")
    log = get_logger("unit.rot")
    for i in range(15):
        log.info("rotate filler message %02d %s", i, "x" * 40)
    assert (tmp_path / "mgd_events.jsonl").is_file()
    assert (tmp_path / "mgd_events.jsonl.1").is_file()  # 触发了轮转备份


# ---------------------------------------------------------------------------
# 接线（源码级断言：不真训/不真推理）
# ---------------------------------------------------------------------------
def _source(relative: str) -> str:
    from mgd.core import config

    return (Path(config.PACKAGE_DIR) / relative).read_text(encoding="utf-8")


def test_predict_source_binds_inference_context():
    text = _source("core/inference/service.py")
    assert 'bind_context(stage="inference", case_id=_case_id_from_image(image))' in text
    assert 'clear_context(["case_id", "stage"])' in text
    assert text.index('bind_context(stage="inference"') < text.index('clear_context(["case_id", "stage"])')


def test_training_sources_bind_run_context():
    engine = _source("core/training/engine.py")
    assert 'bind_context(run_id=run_dir.name if run_dir else "norun", stage="train", arch=arch)' in engine
    assert 'bind_context(stage=f"train.e{epoch}")' in engine
    assert 'clear_context(["run_id", "stage", "arch"])' in engine
    runner = _source("core/training/runner.py")
    assert "bind_context(run_id=context.run_id)" in runner
    assert 'clear_context(["run_id"])' in runner


def test_evaluator_source_binds_eval_context():
    text = _source("core/evaluation/evaluator.py")
    assert 'bind_context(stage="eval", scope=str(scope), arch=str(arch))' in text
    assert 'clear_context(["stage", "scope", "arch"])' in text


# ---------------------------------------------------------------------------
# mgd-logs CLI
# ---------------------------------------------------------------------------
def _seed_fake_logs(tmp_path: Path) -> None:
    events = [
        {"ts": "2026-11-10T09:00:00.000+00:00", "level": "INFO", "logger": "mgd.inference",
         "module": "service.py", "func": "predict", "line": 1, "msg": "inference ok",
         "ctx": {"run_id": "run_a", "case_id": "cafebabe1234", "stage": "inference"}},
        {"ts": "2026-11-10T10:00:00.000+00:00", "level": "ERROR", "logger": "mgd.training",
         "module": "engine.py", "func": "train_iter", "line": 2, "msg": "训练失败",
         "ctx": {"run_id": "run_a", "stage": "train.e3"},
         "error": {"type": "TrainingError", "hint": "降低 batch_size"}, "exc": "Traceback ..."},
        {"ts": "2026-11-10T11:00:00.000+00:00", "level": "WARNING", "logger": "mgd.evaluation",
         "module": "evaluator.py", "func": "evaluate_pairs", "line": 3, "msg": "评估告警",
         "ctx": {"stage": "eval", "scope": "test"}, "duration": 12.5},
    ]
    (tmp_path / "mgd_events.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in events) + "\n", encoding="utf-8")
    (tmp_path / "errors.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in events[1:]) + "\n", encoding="utf-8")


def test_logs_cli_tail_filters_and_errors_only(tmp_path, capsys):
    _seed_fake_logs(tmp_path)
    assert logs_cli.main(["tail", "--log-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "inference ok" in out and "训练失败" in out and "[run=run_a case=cafebabe1234" in out

    assert logs_cli.main(["tail", "--log-dir", str(tmp_path), "--last", "2"]) == 0
    out = capsys.readouterr().out
    assert "inference ok" not in out and "训练失败" in out and "评估告警" in out

    assert logs_cli.main(["tail", "--log-dir", str(tmp_path), "--level", "ERROR"]) == 0
    out = capsys.readouterr().out
    assert "训练失败" in out and "inference ok" not in out and "评估告警" not in out

    assert logs_cli.main(["tail", "--log-dir", str(tmp_path), "--errors-only"]) == 0
    out = capsys.readouterr().out
    assert "训练失败" in out and "评估告警" in out and "inference ok" not in out


def test_logs_cli_search_case_run_grep(tmp_path, capsys):
    _seed_fake_logs(tmp_path)
    assert logs_cli.main(["search", "--case", "cafebabe1234", "--log-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "inference ok" in out and "共 1 条命中" in out

    assert logs_cli.main(["search", "--run", "run_a", "--log-dir", str(tmp_path)]) == 0
    assert "共 2 条命中" in capsys.readouterr().out

    assert logs_cli.main(["search", "--grep", "训练", "--log-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "训练失败" in out and "共 1 条命中" in out


def test_logs_cli_summary_counts_latest_error_slowest(tmp_path, capsys):
    _seed_fake_logs(tmp_path)
    assert logs_cli.main(["summary", "--hours", "48", "--log-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "共 3 条" in out
    assert "ERROR=1" in out and "WARNING=1" in out and "INFO=1" in out
    assert "mgd.inference=1" in out and "train.e3" in out
    assert "最近错误" in out and "训练失败" in out and "TrainingError" in out
    assert "最慢记录" in out and "12.500s" in out


def test_logs_cli_json_output_is_strict(tmp_path, capsys):
    _seed_fake_logs(tmp_path)
    assert logs_cli.main(["tail", "--log-dir", str(tmp_path), "--json"]) == 0
    records = _strict_lines_from_stdout(capsys.readouterr().out)
    assert [r["level"] for r in records] == ["INFO", "ERROR", "WARNING"]
    assert logs_cli.main(["summary", "--log-dir", str(tmp_path), "--json"]) == 0
    payload = _strict_loads(capsys.readouterr().out)
    assert payload["by_level"] == {"INFO": 1, "ERROR": 1, "WARNING": 1}
    assert payload["latest_error"]["msg"] == "训练失败"
    assert payload["slowest_duration"] == pytest.approx(12.5)


def _strict_lines_from_stdout(text: str) -> list[dict]:
    return [_strict_loads(line) for line in text.splitlines() if line.strip()]


def test_logs_cli_corrupt_lines_skipped_with_warning(tmp_path, capsys):
    good = json.dumps(
        {"ts": "2026-11-10T09:00:00.000+00:00", "level": "INFO", "logger": "mgd.x",
         "module": "x.py", "func": "f", "line": 1, "msg": "good line", "ctx": {}},
        ensure_ascii=False)
    (tmp_path / "mgd_events.jsonl").write_text(f"{good}\nnot-json-garbage\n{{broken\n", encoding="utf-8")
    assert logs_cli.main(["tail", "--log-dir", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert "good line" in captured.out
    assert "跳过 2 行损坏" in captured.err


def test_logs_cli_missing_files_and_dir_exit_zero(tmp_path, capsys):
    assert logs_cli.main(["tail", "--log-dir", str(tmp_path)]) == 0
    assert "不存在" in capsys.readouterr().out
    ghost = tmp_path / "ghost_logs"
    assert logs_cli.main(["files", "--log-dir", str(ghost)]) == 0
    assert "不存在" in capsys.readouterr().out
    capsys.readouterr()


def test_logs_cli_files_lists_sizes(tmp_path, capsys):
    _seed_fake_logs(tmp_path)
    assert logs_cli.main(["files", "--log-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "mgd_events.jsonl" in out and "errors.jsonl" in out and "合计" in out

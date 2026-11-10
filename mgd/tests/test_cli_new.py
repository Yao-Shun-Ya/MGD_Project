"""WP-12 验收：4 个新 CLI 的 --help 与错误码（不联网、不碰真实数据）。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mgd.adapters.cli import atlas_cli, followup_cli, logs_cli, qc_cli, report_cli

NEW_CLIS = (
    ("mgd-qc", qc_cli),
    ("mgd-atlas", atlas_cli),
    ("mgd-followup", followup_cli),
    ("mgd-report", report_cli),
)


@pytest.mark.parametrize("name,module", NEW_CLIS, ids=[name for name, _ in NEW_CLIS])
def test_help_exits_zero(name, module, capsys):
    # argparse 的 --help 以 SystemExit(0) 结束——退出码 0 即任务书的 "--help 返回 0"
    with pytest.raises(SystemExit) as excinfo:
        module.main(["--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "--help" not in captured.err


def test_invalid_args_exit_nonzero(capsys):
    """argparse 缺参 → SystemExit(2)；手动校验分支 → 返回 2。任务书的「非 0」均满足。"""
    with pytest.raises(SystemExit) as excinfo:
        qc_cli.main([])          # 缺 --image 必填参数
    assert excinfo.value.code == 2
    capsys.readouterr()
    with pytest.raises(SystemExit) as excinfo:
        report_cli.main([])      # 缺 --input 必填参数
    assert excinfo.value.code == 2
    capsys.readouterr()
    assert report_cli.main(["--input", "nope.json"]) == 2  # 只给 input 不选输出
    capsys.readouterr()
    with pytest.raises(SystemExit) as excinfo:
        followup_cli.main(["nope"])          # 未知 action
    assert excinfo.value.code == 2
    capsys.readouterr()


def test_missing_file_inputs_exit_nonzero(tmp_path, capsys):
    assert qc_cli.main(["--image", str(tmp_path / "ghost.png")]) == 2
    capsys.readouterr()
    assert report_cli.main(["--input", str(tmp_path / "ghost.json"), "--html"]) == 2
    capsys.readouterr()
    assert atlas_cli.main(["classify", "--mask", str(tmp_path / "ghost.png")]) == 2
    capsys.readouterr()


def test_qc_reject_exit_code(tmp_path, capsys):
    """质控 reject 语义：多缺陷叠加把分数压到 40 以下 → 退出码 1（评估成功但建议拒收）。

    全黑图触发 blur(-25)+underexposed(-20)=55 → review；再叠加 specular(-15) 与
    极小眼睑 ROI 触发 poor_eversion(-20) → 20 分 < reject_below(40) → reject。
    """
    import numpy as np
    from PIL import Image

    image = np.zeros((128, 256), dtype="uint8")
    image[10:40, 10:50] = 255          # 高光块（1200px ≈ 3.7% > 1%）→ specular
    image_path = tmp_path / "bad.png"
    Image.fromarray(image).save(image_path)
    eyelid_path = tmp_path / "tiny.png"
    eyelid = np.zeros((128, 256), dtype="uint8")
    eyelid[60:62, 100:104] = 255       # 面积占比 ~0.0002 → poor_eversion
    Image.fromarray(eyelid).save(eyelid_path)
    code = qc_cli.main(["--image", str(image_path), "--eyelid", str(eyelid_path), "--json"])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] == "reject"
    assert {"underexposed", "blur", "specular", "poor_eversion"} <= set(payload["flags"])


# ---------------------------------------------------------------------------
# WP-12.1 回归修复（集成层审计）：M-1 / M-2 / M-3 / M-6 / L-7
# ---------------------------------------------------------------------------
def _strict_json_loads(text: str):
    """严格 JSON 解析：出现裸 NaN / Infinity 常量即抛错（M-3 的断言风格）。"""

    def _reject(name: str):
        raise ValueError(f"严格 JSON 不允许常量 {name}")

    return json.loads(text, parse_constant=_reject)


def test_followup_compare_json_visit_ids(tmp_path, capsys):
    """M-1：compare 的就诊行键是 `id`（不是 visit_id），JSON 分支不得输出 null。"""
    db = tmp_path / "clinic.db"
    for date, value in (("2026-01-01", 10.0), ("2026-02-01", 25.0)):
        assert followup_cli.main([
            "add-measurement", "--patient", "P1", "--date", date,
            "--key", "ga_percent", "--value", str(value), "--db", str(db),
        ]) == 0
        capsys.readouterr()
    assert followup_cli.main(["compare", "--patient", "P1", "--json", "--db", str(db)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["base_visit"] is not None and payload["follow_visit"] is not None
    assert payload["base_visit"] != payload["follow_visit"]
    delta_row = next(row for row in payload["rows"] if row[0] == "ga_percent")
    assert float(delta_row[3]) == pytest.approx(15.0)  # 25 − 10
    # 同类键名回归：timeline 文本分支同样必须用 `id`（原 `row['visit_id']` 会 KeyError 裸崩）
    assert followup_cli.main(["timeline", "--patient", "P1", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "就诊 #" in out and "ga_percent" in out


def test_followup_readonly_missing_db_no_implicit_create(tmp_path, capsys):
    """L-7：timeline / compare 只读动作在库不存在时返回 2，且绝不隐式建库。"""
    db = tmp_path / "missing.db"
    assert followup_cli.main(["timeline", "--patient", "P1", "--db", str(db)]) == 2
    assert "不存在" in capsys.readouterr().err
    assert not db.exists()
    assert followup_cli.main(["compare", "--patient", "P1", "--db", str(db)]) == 2
    capsys.readouterr()
    assert not db.exists()
    # init 保持可创建语义
    assert followup_cli.main(["init", "--db", str(db)]) == 0
    capsys.readouterr()
    assert db.is_file()


def test_followup_corrupt_db_returns_2_not_traceback(tmp_path, capsys):
    """M-6：垃圾 bytes 的 .db（sqlite3.DatabaseError）→ 友好提示 + 退出码 2。"""
    db = tmp_path / "broken.db"
    db.write_bytes(b"this is not a sqlite database" * 100)
    assert followup_cli.main(["timeline", "--patient", "P1", "--db", str(db)]) == 2
    err = capsys.readouterr().err
    assert "[失败]" in err and "Traceback" not in err


def test_atlas_classify_zero_mask_json_is_strict(tmp_path, capsys):
    """M-3：全零掩膜 classify --json 输出严格 JSON（NaN/Inf 一律 null，无裸 NaN）。"""
    import numpy as np
    from PIL import Image

    mask_path = tmp_path / "zero.png"
    Image.fromarray(np.zeros((64, 128), dtype="uint8")).save(mask_path)
    assert atlas_cli.main(["classify", "--mask", str(mask_path), "--json"]) == 0
    payload = _strict_json_loads(capsys.readouterr().out)
    assert isinstance(payload, list) and len(payload) == 8
    by_key = {row["key"]: row for row in payload}
    # 全零掩膜：GA 无眼睑参考 / 宽度中位无效等 → 缺失项必须是 null 而不是裸 NaN
    assert by_key["dropout"]["value"] is None
    assert by_key["confluence"]["threshold"] is None
    for row in payload:
        assert row["value"] is None or isinstance(row["value"], (int, float))
        assert row["threshold"] is None or isinstance(row["threshold"], (int, float))


def test_atlas_describe_survives_gbk_pipe(tmp_path):
    """M-2：GBK 管道（PYTHONIOENCODING=gbk + -X utf8=0）下跑 CLI → 退出码 0，
    且输出以 UTF-8 可读（修复前 stdout 按控制台 GBK 编码，UTF-8 侧读到乱码）。"""
    import os
    import subprocess
    import sys

    project_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONIOENCODING="gbk", GRADIO_ANALYTICS_ENABLED="False",
               PYTHONPATH=str(project_root))
    proc = subprocess.run(
        [sys.executable, "-X", "utf8=0", "-m", "mgd.adapters.cli.atlas_cli", "describe"],
        cwd=str(project_root), env=env, capture_output=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")
    text = proc.stdout.decode("utf-8")
    assert "dropout" in text
    assert "腺体缺失" in text  # GBK 字节按 UTF-8 解码只会得到乱码，匹配不上即回归


# ---------------------------------------------------------------------------
# WP-13：mgd-logs 检索 CLI 的 --help 与错误码（同 WP-12 惯例）
# ---------------------------------------------------------------------------
def test_logs_cli_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        logs_cli.main(["--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "--help" not in captured.err


def test_logs_cli_unknown_subcommand_exits_two(capsys):
    with pytest.raises(SystemExit) as excinfo:
        logs_cli.main(["nope"])
    assert excinfo.value.code == 2
    capsys.readouterr()


def test_logs_cli_missing_required_args_exit_two(capsys):
    """缺子命令 → 2；search 缺 --case/--run/--grep 之一 → 2。"""
    with pytest.raises(SystemExit) as excinfo:
        logs_cli.main([])
    assert excinfo.value.code == 2
    capsys.readouterr()
    with pytest.raises(SystemExit) as excinfo:
        logs_cli.main(["search"])
    assert excinfo.value.code == 2
    capsys.readouterr()


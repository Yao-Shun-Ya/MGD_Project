"""WP-10 修正回流接线测试：JSONL 持久化 / feedback CLI / 诊断 tab UI 接线。

全部离线、tmp_path、CPU、目标 < 10 秒。覆盖：
- record_to_row：NaN→None、direction 字段、掩膜 sha256 稳定（含 8MB 分块正确性）；
- append_record：建文件、严格 JSON（parse_constant 拒绝 NaN/Infinity）、
  尾行换行守卫（无换行残片不粘连）、deid_salt 假名化（同盐稳定 / 换盐改变 /
  无盐明文保留）；
- load_records：last 截尾、损坏行跳过并计数告警；
- summarize_file / export_from_file：同口径汇总、排序、截断、空 ValueError；
- feedback CLI：record 全链（tmp 写 PNG → main([...]) → JSONL 行正确）、
  缺文件 → 退出码 2、--json、list/summary/export、--help → SystemExit(0)；
- app.py / corrections.py 源码级防回归（gr.ImageEditor、submit_correction_fn、
  append_record 接线、pseudonymize 延迟导入）。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from mgd.adapters.cli import feedback_cli  # noqa: E402
from mgd.core.config import Settings  # noqa: E402
from mgd.core.feedback import (  # noqa: E402
    append_record,
    corrections_path,
    export_from_file,
    load_records,
    record_correction,
    record_to_row,
    summarize_file,
)
from mgd.core.feedback import corrections as feedback_corrections  # noqa: E402

HEIGHT, WIDTH = 64, 128


def _blank(height: int = 16, width: int = 16) -> np.ndarray:
    return np.zeros((height, width), dtype=np.uint8)


def _block(top: int, left: int, height: int, width: int,
           canvas_h: int = 16, canvas_w: int = 16) -> np.ndarray:
    mask = _blank(canvas_h, canvas_w)
    mask[top:top + height, left:left + width] = 255
    return mask


def _tmp_settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(home=tmp_path, target_height=HEIGHT, target_width=WIDTH,
                    num_workers=0, **overrides)


def _loads_strict(text: str):
    """json.loads 且拒绝 NaN/Infinity 字面量（严格 JSON 校验）。"""
    def reject(token: str) -> None:
        raise ValueError(f"非严格 JSON 常量：{token}")
    return json.loads(text, parse_constant=reject)


def _write_png(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    return path


# ------------------------------------------------------- record_to_row
def test_record_to_row_nan_to_none_and_direction() -> None:
    rec = record_correction("c1", _blank(), _block(0, 0, 4, 5), ts="2026-01-01T00:00:00+00:00")
    row = record_to_row(rec)
    assert row["dice_before"] is None                    # NaN → None（严格 JSON）
    assert row["dice_after_vs_edit"] == 0.0              # model 全空 → Dice=0.0
    assert row["added_px"] == 20 and row["removed_px"] == 0
    assert row["direction"] == "add_only"                # direction 落进行内
    assert set(row) == {"case_id", "dice_before", "dice_after_vs_edit", "added_px",
                        "removed_px", "ts", "reviewer_id", "note", "direction",
                        "model_mask_sha256", "edited_mask_sha256"}


def test_record_to_row_mask_sha256_stable_and_discriminating() -> None:
    mask = _block(0, 0, 4, 4)
    other = _block(0, 0, 5, 5)
    rec = record_correction("c", mask, other)
    sha_model = feedback_corrections._mask_sha256(mask)
    sha_edited = feedback_corrections._mask_sha256(other)
    row = record_to_row(rec, model_mask_sha256=sha_model, edited_mask_sha256=sha_edited)
    assert row["model_mask_sha256"] == sha_model
    assert row["edited_mask_sha256"] == sha_edited
    assert sha_model != sha_edited                       # 内容不同 → 哈希不同
    assert sha_model == feedback_corrections._mask_sha256(mask.copy())  # 同内容 → 稳定
    assert len(sha_model) == 64


def test_mask_sha256_chunking_matches_whole_bytes() -> None:
    """大于 8MB 的掩膜按行分块喂入，结果必须与整段 tobytes() 一次性哈希一致。"""
    big = np.zeros((3000, 3000), dtype="uint8")          # 9 MB > 8MB/块
    big[10:2000, 10:2000] = 255
    import hashlib

    expected = hashlib.sha256(big.tobytes()).hexdigest()
    assert feedback_corrections._mask_sha256(big) == expected


# ------------------------------------------------------- append_record
def test_append_record_creates_file_and_strict_json(tmp_path) -> None:
    settings = _tmp_settings(tmp_path)
    rec = record_correction("case1", _block(0, 0, 4, 4), _block(0, 0, 2, 4),
                            ts="2026-01-01T00:00:00+00:00", reviewer_id="dr_a", note="n1")
    row = append_record(rec, settings, model_mask=rec.model_mask, edited_mask=rec.edited_mask)
    path = corrections_path(settings)
    assert path == settings.home / "db" / "corrections.jsonl"
    assert path.exists()
    raw = path.read_text(encoding="utf-8")
    assert raw.endswith("\n")                            # 行以换行结尾
    lines = raw.splitlines()
    assert len(lines) == 1
    parsed = _loads_strict(lines[0])                     # NaN/Infinity → ValueError
    assert parsed["case_id"] == "case1"                  # 无盐 → 明文保留
    assert parsed["reviewer_id"] == "dr_a" and parsed["note"] == "n1"
    assert parsed["model_mask_sha256"] and parsed["edited_mask_sha256"]
    assert parsed == row


def test_append_record_guards_missing_trailing_newline(tmp_path) -> None:
    """预先写无换行尾巴的损坏残片，再 append 不粘连：残片被跳过、新行独立可读。"""
    settings = _tmp_settings(tmp_path)
    path = corrections_path(settings)
    path.parent.mkdir(parents=True)
    path.write_text('{"case_id": "半行损坏残片"', encoding="utf-8")  # 无换行
    rec = record_correction("case2", _blank(), _block(1, 1, 2, 2))
    append_record(rec, settings)
    rows = load_records(settings)
    assert [r["case_id"] for r in rows] == ["case2"]


def test_append_record_deid_pseudonym_stable(tmp_path) -> None:
    settings = _tmp_settings(tmp_path, deid_salt="salt-a")
    append_record(record_correction("P001", _blank(), _block(0, 0, 2, 2),
                                    ts="2026-01-01T00:00:00+00:00"), settings)
    row = append_record(record_correction("P001", _blank(), _block(0, 0, 2, 2),
                                          ts="2026-01-02T00:00:00+00:00"), settings)
    from mgd.core.governance.deid import pseudonymize

    assert row["case_id"] == pseudonymize("P001", "salt-a")
    rows = load_records(settings)
    assert len({r["case_id"] for r in rows}) == 1        # 同盐同 ID → 伪名稳定可关联
    assert all(r["case_id"] != "P001" for r in rows)     # 明文不落盘


def test_append_record_deid_salt_changes_pseudonym(tmp_path) -> None:
    rec = record_correction("P001", _blank(), _block(0, 0, 2, 2))
    row_a = append_record(rec, _tmp_settings(tmp_path / "a", deid_salt="salt-a"))
    row_b = append_record(rec, _tmp_settings(tmp_path / "b", deid_salt="salt-b"))
    assert row_a["case_id"] != "P001" and row_b["case_id"] != "P001"
    assert row_a["case_id"] != row_b["case_id"]          # 换盐 → 不同伪名


# ------------------------------------------------------- load_records
def test_load_records_last_truncation(tmp_path) -> None:
    settings = _tmp_settings(tmp_path)
    for index in range(5):
        append_record(record_correction(f"c{index}", _blank(), _block(0, 0, 1, 1)), settings)
    assert len(load_records(settings)) == 5
    tail = load_records(settings, last=2)
    assert [r["case_id"] for r in tail] == ["c3", "c4"]  # 最新在后、截尾
    assert load_records(settings, last=0) == load_records(settings)  # 0 = 全部
    assert load_records(_tmp_settings(tmp_path / "empty")) == []     # 文件不存在 → 空


def test_load_records_skips_corrupt_lines_with_warning(tmp_path, monkeypatch) -> None:
    settings = _tmp_settings(tmp_path)
    path = corrections_path(settings)
    path.parent.mkdir(parents=True)
    good = json.dumps(record_to_row(record_correction("ok", _blank(), _block(0, 0, 2, 2))),
                      ensure_ascii=False)
    path.write_text(f"{good}\n不是 JSON\n[1, 2]\n\n{good}\n", encoding="utf-8")
    seen: list[str] = []
    monkeypatch.setattr(
        feedback_corrections.logger, "warning",
        lambda message, *args: seen.append(str(message) % args if args else str(message)),
    )
    rows = load_records(settings)
    assert len(rows) == 2                                # 两条损坏行 + 空行被跳过
    assert all(r["case_id"] == "ok" for r in rows)
    assert sum("已跳过" in text for text in seen) == 2   # 损坏行逐条 warning 计数


# ------------------------------------------- summarize_file / export_from_file
def test_summarize_file_matches_summarize_semantics(tmp_path) -> None:
    settings = _tmp_settings(tmp_path)
    append_record(record_correction("a", _blank(), _block(0, 0, 3, 3)), settings)  # add_only, +9
    append_record(record_correction("b", _block(0, 0, 2, 2), _blank()), settings)  # remove_only, −4
    summary = summarize_file(settings)
    assert summary["n"] == 2
    assert summary["direction_counts"]["add_only"] == 1
    assert summary["direction_counts"]["remove_only"] == 1
    assert summary["net_add_cases"] == 1
    assert math.isclose(summary["mean_added_px"], 4.5)
    assert math.isnan(summary["mean_dice_before"])       # 无参考 → NaN 均值被忽略


def test_summarize_file_empty_is_zero_structure(tmp_path) -> None:
    summary = summarize_file(_tmp_settings(tmp_path))    # 文件不存在也不抛异常
    assert summary["n"] == 0 and summary["net_add_cases"] == 0
    assert math.isnan(summary["mean_added_px"])


def test_export_from_file_rank_truncate_and_row_format(tmp_path) -> None:
    settings = _tmp_settings(tmp_path)
    append_record(record_correction("small", _blank(), _block(0, 0, 1, 1),
                                    ts="2026-01-01T00:00:00+00:00"), settings)
    append_record(record_correction("big", _blank(), _block(0, 0, 5, 5),
                                    ts="2026-01-02T00:00:00+00:00",
                                    reviewer_id="dr_wang", note="大修"), settings)
    append_record(record_correction("mid", _block(0, 0, 3, 3), _blank(),
                                    ts="2026-01-03T00:00:00+00:00"), settings)
    out = export_from_file(settings, tmp_path / "out", max_items=2)
    assert out == tmp_path / "out" / "correction_candidates.jsonl"
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["case_id"] for r in rows] == ["big", "mid"]            # 25 > 9 > 1 降序截断
    assert rows[0]["note"] == "大修" and rows[0]["reviewer_id"] == "dr_wang"
    assert rows[0]["dice_before"] is None
    # 行格式与 export_candidates 完全一致（8 字段投影，无 direction / 哈希）
    assert set(rows[0]) == {"case_id", "added_px", "removed_px", "dice_before",
                            "dice_after_vs_edit", "ts", "reviewer_id", "note"}
    leftovers = [p.name for p in (tmp_path / "out").iterdir() if p.name != out.name]
    assert leftovers == []                               # 原子写无残留


def test_export_from_file_empty_raises(tmp_path) -> None:
    settings = _tmp_settings(tmp_path)
    with pytest.raises(ValueError):
        export_from_file(settings, tmp_path / "out")     # 文件不存在 → ValueError
    assert not (tmp_path / "out").exists()               # 未落任何文件


# ------------------------------------------------------- feedback CLI
@pytest.fixture
def cli_settings(tmp_path, monkeypatch):
    """把 CLI 的 get_settings 指到 tmp home（离线、不污染全局单例）。"""
    settings = _tmp_settings(tmp_path)
    monkeypatch.setattr(feedback_cli, "get_settings", lambda **overrides: settings)
    return settings


def test_cli_record_end_to_end(tmp_path, cli_settings) -> None:
    settings = cli_settings
    model = _write_png(tmp_path / "m.png", _block(0, 0, 8, 8, HEIGHT, WIDTH))          # 64 px
    edited = _write_png(tmp_path / "e.png",
                        _block(0, 0, 8, 4, HEIGHT, WIDTH) | _block(0, 100, 8, 8, HEIGHT, WIDTH))
    code = feedback_cli.main(["record", "--case", "CASE9", "--model-mask", str(model),
                              "--edited-mask", str(edited), "--reviewer", "dr_a", "--note", "首例"])
    assert code == 0
    rows = load_records(settings)
    assert len(rows) == 1
    row = rows[0]
    assert row["case_id"] == "CASE9"
    assert row["reviewer_id"] == "dr_a" and row["note"] == "首例"
    assert row["added_px"] == 64 and row["removed_px"] == 32     # 保留 32 + 新增 64
    assert row["direction"] == "mixed"
    assert row["model_mask_sha256"] and row["edited_mask_sha256"]


def test_cli_record_missing_file_exit_2(tmp_path, cli_settings, capsys) -> None:
    code = feedback_cli.main(["record", "--case", "X",
                              "--model-mask", str(tmp_path / "nope.png"),
                              "--edited-mask", str(tmp_path / "e.png")])
    assert code == 2
    assert "不存在" in capsys.readouterr().err


def test_cli_record_json_output(tmp_path, cli_settings, capsys) -> None:
    model = _write_png(tmp_path / "m.png", _block(0, 0, 4, 4, HEIGHT, WIDTH))
    edited = _write_png(tmp_path / "e.png", _block(0, 0, 4, 4, HEIGHT, WIDTH))
    code = feedback_cli.main(["record", "--case", "J1", "--model-mask", str(model),
                              "--edited-mask", str(edited), "--json"])
    assert code == 0
    payload = _loads_strict(capsys.readouterr().out)
    assert payload["case_id"] == "J1"
    assert payload["dice_after_vs_edit"] == 1.0
    assert payload["dice_before"] is None               # 无参考 → NaN → null
    assert payload["direction"] == "none"


def test_cli_record_deid_pseudonym(tmp_path, monkeypatch) -> None:
    settings = _tmp_settings(tmp_path, deid_salt="cli-salt")
    monkeypatch.setattr(feedback_cli, "get_settings", lambda **overrides: settings)
    model = _write_png(tmp_path / "m.png", _block(0, 0, 4, 4, HEIGHT, WIDTH))
    edited = _write_png(tmp_path / "e.png", _block(0, 0, 4, 2, HEIGHT, WIDTH))
    assert feedback_cli.main(["record", "--case", "SECRET1", "--model-mask", str(model),
                              "--edited-mask", str(edited)]) == 0
    rows = load_records(settings)
    from mgd.core.governance.deid import pseudonymize

    assert len(rows) == 1 and rows[0]["case_id"] != "SECRET1"
    assert rows[0]["case_id"] == pseudonymize("SECRET1", "cli-salt")


def test_cli_list_summary_export(tmp_path, cli_settings, capsys) -> None:
    settings = cli_settings
    model = _write_png(tmp_path / "m.png", _blank(HEIGHT, WIDTH))
    edited = _write_png(tmp_path / "e.png", _block(0, 0, 2, 2, HEIGHT, WIDTH))
    assert feedback_cli.main(["record", "--case", "L1", "--model-mask", str(model),
                              "--edited-mask", str(edited)]) == 0
    assert feedback_cli.main(["record", "--case", "L2", "--model-mask", str(model),
                              "--edited-mask", str(edited)]) == 0
    capsys.readouterr()                                  # 排空 record 确认行

    assert feedback_cli.main(["list", "--last", "1"]) == 0
    listed = capsys.readouterr().out
    assert "L2" in listed and "L1" not in listed         # last 截尾（人读模式）

    assert feedback_cli.main(["list", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["case_id"] for r in rows] == ["L1", "L2"]

    assert feedback_cli.main(["summary", "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["n"] == 2 and summary["direction_counts"]["add_only"] == 2
    assert summary["mean_dice_before"] is None          # NaN → null（严格 JSON）

    assert feedback_cli.main(["summary"]) == 0          # 人读模式不崩溃
    capsys.readouterr()

    out_dir = tmp_path / "exported"
    assert feedback_cli.main(["export", "--out", str(out_dir), "--max", "1", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 1
    exported = [json.loads(line)
                for line in Path(payload["path"]).read_text(encoding="utf-8").splitlines()]
    assert len(exported) == 1 and exported[0]["case_id"] in ("L1", "L2")


def test_cli_help_system_exit_zero(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        feedback_cli.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "record" in out and "summary" in out and "export" in out


# ------------------------------------------------- app.py / corrections.py 防回归
def test_app_source_contains_correction_wiring() -> None:
    text = (ROOT / "adapters" / "gradio_app" / "app.py").read_text(encoding="utf-8")
    assert "gr.ImageEditor(" in text                     # 修正画布存在
    assert "def submit_correction_fn(" in text           # 提交回调存在
    assert "append_record(" in text and "record_correction(" in text   # 持久化接线
    assert "载入上次模型掩膜" in text
    assert "db/corrections.jsonl" in text                # 留存位置说明
    assert "审计由 governance 接线覆盖" in text           # 审计交由治理代理的注释
    body = text[text.index("def submit_correction_fn("):text.index("# ================= 界面")]
    assert "WORK_LOCK" not in body                       # 修正回调不持 GPU 锁
    assert "audit_event" not in body                     # 不自行触发审计


def test_corrections_source_lazy_pseudonymize_import() -> None:
    text = (ROOT / "core" / "feedback" / "corrections.py").read_text(encoding="utf-8")
    assert "from ..governance.deid import pseudonymize" in text
    # 延迟导入：出现在函数体内（corrections_path 定义之后），不在模块顶层 import 区
    assert text.index("from ..governance.deid import pseudonymize") > text.index(
        "def corrections_path")


def test_editor_ink_mask_extraction() -> None:
    """RGBA 透明画布取 alpha；不透明底图取亮度；composite 缺失按图层兜底。"""
    from mgd.adapters.gradio_app.app import _editor_ink_mask

    rgba = np.zeros((8, 8, 4), dtype="uint8")            # 透明画布 + 白笔迹
    rgba[2:4, 2:6] = (255, 255, 255, 255)
    ink = _editor_ink_mask({"background": None, "layers": [], "composite": rgba})
    assert ink is not None and int(ink.sum()) == 8

    opaque = np.zeros((8, 8, 4), dtype="uint8")          # 不透明黑底 + 白前景
    opaque[..., 3] = 255
    opaque[1:3, 1:3, :3] = 255
    assert int(_editor_ink_mask({"background": None, "layers": [], "composite": opaque}).sum()) == 4

    fallback = _editor_ink_mask({"background": None, "layers": [rgba], "composite": None})
    assert int(fallback.sum()) == 8                      # 最后一个图层兜底

    gray = np.zeros((6, 6), dtype="uint8")
    gray[0:2, 0:2] = 255
    assert int(_editor_ink_mask({"background": gray, "layers": [], "composite": gray}).sum()) == 4

    assert _editor_ink_mask({"background": None, "layers": [], "composite": None}) is None
    assert _editor_ink_mask(None) is None


if __name__ == "__main__":
    pytest.main([__file__, "-q"])

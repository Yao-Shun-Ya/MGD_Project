"""WP-7 governance 单元测试（离线 / tmp_path）。

覆盖：审计哈希链、脱敏（EXIF / DICOM 标签 / 假名化）、RBAC、双人复核与仲裁，
以及既有实现 mgd/core/provenance.py 的回归测试（to_dict / collect_provenance）。

说明：provenance 依赖 inference.service（间接引入 torch/monai），为控制整文件耗时，
其 import 放在对应测试函数内部延迟执行，并排在文件末尾。
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from mgd.core.governance.audit import GENESIS, AuditLog, canonical_json
from mgd.core.governance.deid import DEID_TAGS, pseudonymize, scrub_dicom_tags, scrub_image_exif
from mgd.core.governance.rbac import ACTIONS, PERMISSIONS, can, require
from mgd.core.governance.review import REVIEW_DECISIONS, ReviewLog, ReviewRecord
from mgd.core.runtime.errors import ConfigError

# ---------------------------------------------------------------- 审计日志


def test_audit_append_and_hash_chain(tmp_path):
    log = AuditLog(tmp_path / "nested" / "audit.jsonl")  # 父目录自动创建
    r1 = log.append("inference.run", "tech01", {"case": "c1"})
    r2 = log.append("report.export", "doctor01", {"case": "c1", "备注": "中文不转义"})
    r3 = log.append("case.delete", "chief01", {"case": "c2"})

    assert (tmp_path / "nested" / "audit.jsonl").exists()
    assert len(log) == 3
    assert [r["seq"] for r in (r1, r2, r3)] == [1, 2, 3]
    assert r1["prev_hash"] == GENESIS
    assert r2["prev_hash"] == r1["hash"]
    assert r3["prev_hash"] == r2["hash"]

    # 记录结构完整
    for rec in (r1, r2, r3):
        assert set(rec) == {"seq", "ts", "event", "actor", "payload", "prev_hash", "hash"}
        ts = datetime.fromisoformat(rec["ts"])
        assert ts.utcoffset() == timedelta(0), "ts 必须是 ISO UTC"
        assert isinstance(ts.tzinfo, timezone) or ts.utcoffset() == timedelta(0)

    # hash 可手算复核：sha256(prev_hash + canonical_json(去掉 hash 的记录))
    for rec, prev in ((r1, GENESIS), (r2, r1["hash"]), (r3, r2["hash"])):
        body = {k: v for k, v in rec.items() if k != "hash"}
        expect = hashlib.sha256((prev + canonical_json(body)).encode("utf-8")).hexdigest()
        assert rec["hash"] == expect

    ok, msg = log.verify()
    assert (ok, msg) == (True, "ok")


def test_audit_detect_tampered_payload(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    for i in range(3):
        log.append("event", "actor", {"i": i})

    path = tmp_path / "audit.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[1])
    tampered["payload"]["i"] = 999
    lines[1] = json.dumps(tampered, ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, msg = log.verify()
    assert ok is False
    assert "seq=2" in msg, f"断点消息应包含被篡改记录的 seq：{msg}"
    assert msg.startswith("chain broken at seq=2")


def test_audit_corrupt_line_skipped_and_seq_resumes(tmp_path, caplog):
    log = AuditLog(tmp_path / "audit.jsonl")
    for i in range(5):
        log.append("e", "a", {"i": i})
    with open(tmp_path / "audit.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"seq": 6, "ts": "20')  # 半行 JSON（模拟崩溃残留）

    with caplog.at_level(logging.WARNING, logger="mgd.governance.audit"):
        assert len(log) == 5  # 损坏行不计入
        tail3 = log.tail(3)
    assert any("损坏" in rec.getMessage() for rec in caplog.records)

    # seq 从现有最大 seq + 1 续写，新记录直接衔接最后一条完整记录
    after = log.append("after", "a", {"i": 5})
    assert after["seq"] == 6
    records = log.tail(2)
    assert [r["seq"] for r in records] == [5, 6]
    assert records[1]["prev_hash"] == records[0]["hash"]
    assert len(log) == 6
    ok, msg = log.verify()
    assert (ok, msg) == (True, "ok"), "跳过损坏行后剩余链应完整"

    assert log.tail(50) == log.tail(10**6)
    assert log.tail(0) == []


def test_audit_empty_and_missing_file(tmp_path):
    log = AuditLog(tmp_path / "fresh.jsonl")
    assert len(log) == 0
    assert log.tail() == []
    assert log.verify() == (True, "ok")
    first = log.append("boot", "admin", {})
    assert first["seq"] == 1 and first["prev_hash"] == GENESIS
    assert log.verify() == (True, "ok")


# ---------------------------------------------------------------- 脱敏


def test_deid_tags_match_spec():
    assert DEID_TAGS == (
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "PatientSex",
        "PatientAge",
        "PatientAddress",
        "PatientTelephoneNumbers",
        "OtherPatientIDs",
        "PatientInsurancePlanCodeSequence",
        "InstitutionName",
        "ReferringPhysicianName",
        "PerformingPhysicianName",
        "OperatorName",
    )


def test_pseudonymize_stable_salt_sensitive():
    a1 = pseudonymize("P-0001", "salt-A")
    assert a1 == pseudonymize("P-0001", "salt-A")
    assert a1 != pseudonymize("P-0001", "salt-B"), "不同盐必须得到不同假名"
    assert a1 != pseudonymize("P-0002", "salt-A"), "不同 ID 必须得到不同假名"
    assert len(a1) == 16
    assert a1 == hashlib.sha256(b"salt-AP-0001").hexdigest()[:16]  # salt+id 拼接顺序锁定

    with pytest.raises(ValueError):
        pseudonymize("", "salt")
    with pytest.raises(ValueError):
        pseudonymize("P-0001", "")
    with pytest.raises(ValueError):
        pseudonymize("", "")


def test_scrub_image_exif_roundtrip():
    arr = np.zeros((9, 12, 3), dtype=np.uint8)
    arr[2:6, 3:9] = (200, 120, 30)
    img = Image.fromarray(arr)
    exif = img.getexif()
    exif[0x010F] = "MakeCorp"  # Make
    exif[0x0110] = "CameraModel-XYZ"  # Model
    src_buf = io.BytesIO()
    img.save(src_buf, format="PNG", exif=exif.tobytes())
    src_bytes = src_buf.getvalue()

    # 前置确认：源图确实带 EXIF
    with Image.open(io.BytesIO(src_bytes)) as probe:
        assert len(probe.getexif()) >= 2

    scrubbed = scrub_image_exif(src_bytes)
    with Image.open(io.BytesIO(scrubbed)) as out:
        assert out.format == "PNG"
        assert len(out.getexif()) == 0, "EXIF 必须被清除"
        out_arr = np.array(out)
    assert np.array_equal(out_arr, arr), "像素必须逐位保留"

    with pytest.raises(ValueError):
        scrub_image_exif(b"this is definitely not an image")
    with pytest.raises(ValueError):
        scrub_image_exif("not bytes")


def test_scrub_dicom_tags_clears_and_backfills():
    ds = SimpleNamespace(
        PatientName="ZHANG^SAN",
        PatientID="P-0001",
        PatientBirthDate="1980-01-01",
        PatientSex="M",
        PatientAge="",  # 已脱敏（空）-> 跳过
        PatientAddress=None,  # 空值 -> 跳过
        PatientTelephoneNumbers="13800000000",
        OtherPatientIDs=["ALT-1", "ALT-2"],
        InstitutionName="Some Hospital",
        Modality="OP",  # 非 DEID 字段
        StudyDate="20261110",  # 非 DEID 字段
    )
    out, mapping = scrub_dicom_tags(ds)
    assert out is ds
    for tag in DEID_TAGS:
        assert not getattr(ds, tag, ""), f"{tag} 应被清除或本就不存在"
    assert mapping["PatientName"] == "ZHANG^SAN"
    assert mapping["PatientID"] == "P-0001"
    assert mapping["OtherPatientIDs"] == str(["ALT-1", "ALT-2"])
    assert "PatientAge" not in mapping and "PatientAddress" not in mapping
    assert ds.Modality == "OP" and ds.StudyDate == "20261110"

    # 幂等：二次脱敏不再产生 mapping 条目
    _, mapping2 = scrub_dicom_tags(ds)
    assert mapping2 == {}

    # 传入既有 mapping 时在其上累加
    fresh = SimpleNamespace(PatientID="P-0002")
    carry = {"seed": "keep"}
    _, merged = scrub_dicom_tags(fresh, carry)
    assert carry is merged and merged["seed"] == "keep" and merged["PatientID"] == "P-0002"


# ---------------------------------------------------------------- RBAC


def test_rbac_permission_matrix():
    assert can("tech", "delete_case") is False
    assert can("admin", "delete_case") is True
    assert can("doctor", "grade_case") is True
    assert can("tech", "run_inference") is True
    assert can("doctor", "edit_mask") is True
    assert can("chief", "manage_atlas") is True
    assert can("chief", "manage_calibration") is True
    assert can("admin", "manage_users") is True
    assert can("admin", "manage_audit") is True
    assert can("doctor", "delete_case") is False
    assert can("chief", "manage_users") is False

    # 层级完备：tech ⊆ doctor ⊆ chief ⊆ admin
    assert PERMISSIONS["tech"] < PERMISSIONS["doctor"] < PERMISSIONS["chief"] < PERMISSIONS["admin"]
    assert set(ACTIONS) == set(PERMISSIONS["admin"]), "admin 必须覆盖全部固定动作词表"
    assert len(ACTIONS) == 11


def test_rbac_unknown_role_action_fail_closed():
    assert can("nobody", "view_case") is False  # 未知角色
    assert can("tech", "not_an_action") is False  # 未知动作
    assert can("", "") is False
    assert can(None, "view_case") is False  # type: ignore[arg-type]
    # 不抛异常（fail-closed），仅返回 False
    assert isinstance(can("nobody", "not_an_action"), bool)


def test_rbac_require_raises_config_error():
    require("chief", "delete_case")  # 有权：不抛
    require("admin", "manage_audit")
    with pytest.raises(ConfigError) as exc_info:
        require("tech", "manage_users")
    assert "tech" in str(exc_info.value) and "manage_users" in str(exc_info.value)
    with pytest.raises(ConfigError):
        require("ghost", "view_case")


# ---------------------------------------------------------------- 双人复核


def test_review_disagreement_triggers_arbitration(tmp_path):
    log = ReviewLog(tmp_path / "review" / "review.jsonl")
    r1 = log.record("c1", "docA", "agree", comment=" masks ok")
    r2 = log.record("c1", "docB", "disagree")
    assert isinstance(r1, ReviewRecord) and isinstance(r2, ReviewRecord)
    assert (r1.case_id, r1.reviewer_id, r1.decision) == ("c1", "docA", "agree")
    assert r1.ts and datetime.fromisoformat(r1.ts)

    assert log.decisions("c1") == {"docA": "agree", "docB": "disagree"}
    assert log.needs_arbitration("c1") is True

    entry = log.arbitrate("c1", "chiefX", "agree", comment="以主任复核为准")
    assert entry["event"] == "arbitration"
    assert entry["arbiter_id"] == "chiefX" and entry["final_decision"] == "agree"

    history = log.history("c1")
    assert [h["event"] for h in history] == ["review", "review", "arbitration"]


def test_review_consensus_needs_no_arbitration(tmp_path):
    log = ReviewLog(tmp_path / "review.jsonl")
    log.record("c1", "docA", "agree")
    log.record("c1", "docB", "agree")
    assert log.needs_arbitration("c1") is False
    with pytest.raises(ConfigError):
        log.arbitrate("c1", "chiefX", "agree")

    # 单人意见同样不触发
    log.record("c2", "docA", "disagree")
    assert log.needs_arbitration("c2") is False
    with pytest.raises(ConfigError):
        log.arbitrate("c2", "chiefX", "disagree")

    # 未知病例：空意见、无需仲裁
    assert log.decisions("missing") == {}
    assert log.needs_arbitration("missing") is False


def test_review_duplicate_record_overwrites(tmp_path):
    log = ReviewLog(tmp_path / "review.jsonl")
    log.record("c1", "docA", "agree", comment="first")
    log.record("c1", "docB", "agree")
    log.record("c1", "docA", "disagree", comment="revised")  # 同人重复 -> 覆盖

    assert log.decisions("c1") == {"docA": "disagree", "docB": "agree"}
    assert log.needs_arbitration("c1") is True

    reviews = [h for h in log.history("c1") if h["event"] == "review"]
    assert len(reviews) == 2, "旧记录应被覆盖删除而非叠加"
    doc_a = next(h for h in reviews if h["reviewer_id"] == "docA")
    assert doc_a["decision"] == "disagree" and doc_a["comment"] == "revised"


def test_review_invalid_decision_rejected(tmp_path):
    log = ReviewLog(tmp_path / "review.jsonl")
    with pytest.raises(ConfigError):
        log.record("c1", "docA", "maybe")
    assert len(log.history("c1")) == 0
    assert REVIEW_DECISIONS == ("agree", "disagree")


# ---------------------------------------------------------------- provenance（既有实现回归）


def test_provenance_to_dict_complete_and_json_serializable():
    from mgd.core.provenance import Provenance

    record = Provenance(
        software_version="2.3.0",
        model_name="MGD_UNet_v1.0.0",
        weights_sha256=None,
        weights_verified=None,
        threshold=0.5,
        tile_size=64,
        tile_overlap=0.25,
        tta="关闭",
        calibration_id=None,
        created_at="2026-11-10T00:00:00+00:00",
    )
    payload = record.to_dict()
    assert set(payload) == {
        "software_version",
        "model_name",
        "weights_sha256",
        "weights_verified",
        "threshold",
        "tile_size",
        "tile_overlap",
        "tta",
        "calibration_id",
        "created_at",
    }
    serialized = json.dumps(payload, ensure_ascii=False)  # 必须可 JSON 序列化
    assert "MGD_UNet_v1.0.0" in serialized
    assert payload["threshold"] == 0.5 and payload["tile_size"] == 64


def test_collect_provenance_weights_sha256(tmp_path):
    from mgd.core.provenance import collect_provenance

    weights = tmp_path / "MGD_UNet_v1.0.0.pth"
    payload = b"MGD-weights-bytes-0123456789abcdef"
    weights.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    record = collect_provenance(
        model_name="MGD_UNet_v1.0.0",
        threshold=0.5,
        tile_size=64,
        tile_overlap=0.25,
        weights_path=weights,
        weights_verified=True,
        calibration_id="cal-1",
    )
    assert record.weights_sha256 == expected, "sha256 必须与手算一致"
    assert record.weights_verified is True
    assert record.calibration_id == "cal-1"
    json.dumps(record.to_dict(), ensure_ascii=False)

    # 权重路径不存在：sha256 置 None，不抛异常（provenance 不阻断业务）
    missing = collect_provenance(model_name="m", threshold=0.5, weights_path=tmp_path / "ghost.pth")
    assert missing.weights_sha256 is None


def test_collect_provenance_without_weights():
    from mgd.core.provenance import collect_provenance

    record = collect_provenance(model_name="m", threshold=0.3)
    assert record.weights_sha256 is None
    assert record.weights_verified is None
    assert record.tile_size is None and record.tile_overlap is None
    assert record.calibration_id is None
    assert record.tta == "关闭"  # TTA_OFF 默认值
    json.dumps(record.to_dict(), ensure_ascii=False)

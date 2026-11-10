"""g18 JSON 严格性三连（运行时）：①AuditLog.append 含 NaN → 落盘文件过严格解析；
②AtlasIndex NaN 向量 + numpy meta → to_json 产物过严格解析；③NaN 校准参数在写端被
save_calibration/fit 抛 ValueError 拦截。parse_constant 遇 NaN/Infinity 即抛错。"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

import _common

if True:
    from mgd.core.atlas.index import AtlasIndex
    from mgd.core.conformal.calibrator import ConformalCalibrator
    from mgd.core.conformal.store import save_calibration
    from mgd.core.governance.audit import AuditLog


def strict_loads(text: str):
    """严格 JSON 解析：出现 NaN/Infinity 字面量立即抛 ValueError。"""
    def reject(name: str):
        raise ValueError(f"非严格 JSON 常量：{name}")
    return json.loads(text, parse_constant=reject)


def run() -> dict:
    gate = _common.Gate("g18_json_strictness")
    tmp = Path(tempfile.mkdtemp(prefix="g18_"))

    # ① 审计日志：payload 带 NaN/Inf 也能落成严格 JSON
    log = AuditLog(tmp / "audit.jsonl")
    log.append("g18_event", "quality-gate",
               {"value": float("nan"), "inf": float("inf"), "neg": float("-inf"),
                "nested": [1, {"x": float("nan")}]})
    lines = (tmp / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    ok_parse = True
    for line in lines:
        try:
            strict_loads(line)
        except ValueError as exc:
            ok_parse = False
            gate.check(False, "AuditLog 落盘文件通过严格 JSON 解析", f"{exc}；行：{line[:120]}")
            break
    gate.check(ok_parse and lines,
               f"AuditLog.append(NaN payload) → 文件严格 JSON（{len(lines)} 行）",
               "审计链 canonical_json 必须 _sanitize_finite + allow_nan=False")
    verified, why = log.verify()
    gate.check(verified and why == "ok", f"审计哈希链自校验通过（verify → {why}）",
               f"verify 失败原因：{why}")

    # ② 图谱索引：NaN 向量 + numpy meta → 严格 JSON + 往返还原
    index = AtlasIndex(feature_names=["f0", "f1"])
    index.add("case_nan", np.array([float("nan"), 0.5]),
              {"area": np.int64(12), "score": np.float64(0.5),
               "arr": np.array([1.0, float("nan")])})
    atlas_path = index.to_json(tmp / "atlas_index.json")
    try:
        strict_loads(atlas_path.read_text(encoding="utf-8"))
        gate.check(True, "AtlasIndex.to_json 产物通过严格 JSON 解析（NaN 向量 + numpy meta）")
    except ValueError as exc:
        gate.check(False, "AtlasIndex.to_json 产物通过严格 JSON 解析", str(exc)[:200])
    restored = AtlasIndex.load(atlas_path)
    vector = restored.search(np.array([float("nan"), 0.5]), k=1)
    gate.check(bool(vector) and vector[0]["case_id"] == "case_nan",
               "AtlasIndex.load 往返还原 NaN 向量并可检索",
               "load 必须把 None 还原为 float('nan')（读端纪律）")

    # ③ 校准参数：NaN 在写端被拦截
    try:
        save_calibration("g18_nan", {"quantile": float("nan")}, directory=tmp / "calib")
        gate.check(False, "save_calibration(NaN payload) → ValueError",
                   "store 写盘必须 allow_nan=False；NaN 写成 JSON 字面量即非法文档")
    except ValueError:
        gate.check(True, "save_calibration(NaN payload) → ValueError（写端拦截）")
    calibrator = ConformalCalibrator(alpha=0.10)
    try:
        calibrator.fit([0.1] * 30, [0.1] * 29 + [float("nan")])
        gate.check(False, "ConformalCalibrator.fit(NaN scores) → ValueError",
                   "fit 必须逐对校验有限性，NaN 会绕过一切比较")
    except ValueError:
        gate.check(True, "ConformalCalibrator.fit(NaN scores) → ValueError")

    shutil.rmtree(tmp, ignore_errors=True)
    gate.note("三连覆盖：治理审计（读端严格）/ 图谱索引（写端净化+读端还原）/"
              "共形校准（写端拒绝）——NaN 全生命周期闭环")
    return gate.result()


if __name__ == "__main__":
    _common.run_as_main(run)

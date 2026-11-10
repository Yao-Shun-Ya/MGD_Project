"""g14 Settings 新字段：db_path/calibration_dir/atlas_index_path/qc_policy/physical_scale
存在且语义正确；calibration_dir 跟随 output_dir；from_env 识别 4 个新环境变量；非法 scale 拒绝。"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import _common

if True:
    from mgd.core.config import Settings
    from mgd.core.runtime.errors import ConfigError

ENV_KEYS = ["MGD_DB_PATH", "MGD_CALIBRATION_DIR", "MGD_ATLAS_INDEX", "MGD_SCALE_MM_PER_PX"]


def run() -> dict:
    gate = _common.Gate("g14_settings_fields")
    tmp = Path(tempfile.mkdtemp(prefix="g14_"))

    settings = Settings(home=tmp)
    for field in ("db_path", "calibration_dir", "atlas_index_path", "qc_policy",
                  "physical_scale_mm_per_px"):
        gate.check(hasattr(settings, field), f"Settings 实例含字段 {field}",
                   "v2.4.0 新字段从 Settings 删除会破坏诊断中心级链路")

    gate.check(settings.db_path == tmp / "db" / "mgd_clinic.db",
               f"db_path 默认 home/db/mgd_clinic.db（实际 {settings.db_path}）")
    gate.check(settings.atlas_index_path == tmp / "db" / "atlas_index.json",
               f"atlas_index_path 默认 home/db/atlas_index.json（实际 {settings.atlas_index_path}）")
    gate.check(settings.qc_policy == "default",
               f"qc_policy 默认 'default'（实际 {settings.qc_policy!r}）")

    custom = Settings(home=tmp, output_dir=tmp / "custom")
    gate.check(custom.calibration_dir == tmp / "custom" / "calibration",
               f"calibration_dir 跟随 output_dir（实际 {custom.calibration_dir}）",
               "__post_init__ 中 calibration_dir 必须基于已解析的 output_dir 派生")

    saved = {key: os.environ.get(key) for key in ENV_KEYS + ["MGD_HOME"]}
    try:
        os.environ["MGD_HOME"] = str(tmp)
        os.environ["MGD_DB_PATH"] = str(tmp / "env_clinic.db")
        os.environ["MGD_CALIBRATION_DIR"] = str(tmp / "env_calib")
        os.environ["MGD_ATLAS_INDEX"] = str(tmp / "env_atlas.json")
        os.environ["MGD_SCALE_MM_PER_PX"] = "0.25"
        env_settings = Settings.from_env()
        gate.check(env_settings.db_path == tmp / "env_clinic.db",
                   "from_env 识别 MGD_DB_PATH")
        gate.check(env_settings.calibration_dir == tmp / "env_calib",
                   "from_env 识别 MGD_CALIBRATION_DIR")
        gate.check(env_settings.atlas_index_path == tmp / "env_atlas.json",
                   "from_env 识别 MGD_ATLAS_INDEX")
        gate.check(env_settings.physical_scale_mm_per_px == 0.25,
                   f"from_env 识别 MGD_SCALE_MM_PER_PX（实际 {env_settings.physical_scale_mm_per_px}）")
        os.environ["MGD_SCALE_MM_PER_PX"] = "abc"
        try:
            Settings.from_env()
            gate.check(False, "MGD_SCALE_MM_PER_PX 非数字 → ConfigError", "from_env 应转换失败即抛 ConfigError")
        except ConfigError:
            gate.check(True, "MGD_SCALE_MM_PER_PX 非数字 → ConfigError")
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    try:
        Settings(home=tmp, physical_scale_mm_per_px=-1)
        gate.check(False, "非法 scale（-1）→ ConfigError",
                   "__post_init__ 必须对 physical_scale_mm_per_px <= 0 抛 ConfigError")
    except ConfigError:
        gate.check(True, "非法 scale（-1）→ ConfigError")

    gate.note(f"临时 home={tmp}；env 注入与恢复均走 try/finally，不污染父进程环境")
    return gate.result()


if __name__ == "__main__":
    _common.run_as_main(run)

"""单次推理/报告的血缘记录（Provenance）。

目标：任何一份报告都能回答「这个数字是谁、用哪个权重、什么参数、什么时候算出来的」。
字段与 AGENT_TASK §WP-7 一一对应；`collect_provenance()` 负责从现有对象收集并计算
权重 sha256（流式读取，不加载权重进内存）。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .inference.service import TTA_OFF


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class Provenance:
    """一次推理/报告的完整血缘。所有字段都可序列化为 JSON。"""

    software_version: str
    model_name: str
    weights_sha256: str | None
    weights_verified: bool | None
    threshold: float
    tile_size: int | None
    tile_overlap: float | None
    tta: str
    calibration_id: str | None
    created_at: str

    def to_dict(self) -> dict:
        """JSON 安全字典（Path 等一律转字符串）。"""
        payload = asdict(self)
        for key, value in payload.items():
            if value is not None and not isinstance(value, (str, int, float, bool, list, dict)):
                payload[key] = str(value)
        return payload


def collect_provenance(
    *,
    model_name: str,
    threshold: float,
    tile_size: int | None = None,
    tile_overlap: float | None = None,
    tta: str = TTA_OFF,
    weights_path: str | Path | None = None,
    weights_verified: bool | None = None,
    calibration_id: str | None = None,
    software_version: str | None = None,
) -> Provenance:
    """从推理参数收集血缘记录。

    - `weights_sha256`：给路径时流式计算 sha256（复用 manifest 的实现）；失败置 None 并不阻断。
    - `weights_verified`：由调用方传入（来自 validate_weight 的结果）；None = 未校验。
    - `software_version`：默认取 `mgd.__version__`（版本唯一源）。
    """
    from .. import __version__

    sha256: str | None = None
    if weights_path is not None:
        try:
            from .models.manifest import sha256_file

            sha256 = sha256_file(Path(weights_path))
        except (OSError, ValueError):
            sha256 = None
    return Provenance(
        software_version=str(software_version or __version__),
        model_name=str(model_name),
        weights_sha256=sha256,
        weights_verified=None if weights_verified is None else bool(weights_verified),
        threshold=float(threshold),
        tile_size=None if tile_size is None else int(tile_size),
        tile_overlap=None if tile_overlap is None else float(tile_overlap),
        tta=str(tta),
        calibration_id=None if calibration_id is None else str(calibration_id),
        created_at=_utc_now(),
    )

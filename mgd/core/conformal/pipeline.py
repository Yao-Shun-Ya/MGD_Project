"""conformal ↔ 评估链路的桥梁：从 EvalReport 逐样本指标拟合两个临床口径的校准。

两个口径（同一 α，split conformal，Vovk et al.）：

1. **dice_coverage**（单例 Dice 覆盖下界）
   - 非一致性得分 ``s_i = 1 - dice_i``（忽略 NaN 样本；得分越小越好）。
   - 有限样本修正上分位 ``q = sorted(s)[ceil((n+1)(1-alpha)) - 1]``。
   - 输出 ``bound = 1 - q``，保证：在可交换性假设下，
     ``P(新病例 Dice >= bound) >= 1 - alpha``（对任意有限 n 精确成立）。

2. **rloss_interval**（R_loss 预测区间）
   - 残差 ``s_i = |r_loss_pred_i - r_loss_gt_i|``（两侧都有限的样本）。
   - 输出 ``q``，保证：在可交换性假设下，
     ``P(gt ∈ [pred - q, pred + q]) >= 1 - alpha``，
     即新病例的 R_loss 预测值 ± q 区间以 1-α 概率盖住真值。

统计假设（两口径相同）——**可交换性 exchangeability**：校准集得分与目标病例
得分可交换（i.i.d. 是其特例）。本引擎的校准集即持久化评估划分
（train/val/test 的独立 test/val 份），因此对同一引擎、同一数据域的新病例
保证成立；若部署分布漂移（设备/人群改变），覆盖率保证失效，须重新校准。
对 ``train_self``（训练集自评）口径，得分来自训练分布，不能解释为泛化保证。

样本量要求：每个口径的有效样本数 ``n >= 1/alpha`` 才有有限样本保证；
不足的口径在对应块写 ``skipped_reason``（不硬失败），``bound``/``q`` 为 None。

实现契约：得分构造走 :meth:`ConformalCalibrator.from_dict` 校验路径
（逐值有限性 + alpha 合法性），分位走 :meth:`ConformalCalibrator.quantile`
（再验 ``n >= 1/alpha``）——与 calibrator 模块同一套校验口径，绝不静默出区间。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from ..runtime.logging import get_logger
from .calibrator import ConformalCalibrator, validate_alpha
from .store import save_calibration

logger = get_logger("conformal.pipeline")

#: payload 结构版本；字段变更时递增
FORMAT_VERSION = 1

#: 文件名安全字符白名单（与 store._NAME_PATTERN 对齐；glob 元字符一律排除）
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]")


def _safe_component(value: object) -> str:
    """把 arch/scope 等拼进校准文件名前先做白名单净化（防御路径穿越/glob 元字符）。"""
    cleaned = _SAFE_COMPONENT.sub("_", str(value or ""))
    return cleaned or "_"


def _finite_float(value: object) -> float | None:
    """宽松取数：None/非数值/NaN/inf 一律返回 None（调用端跳过该样本）。"""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sha256_prefix(text: str, length: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _insufficient_reason(n: int, min_required: float) -> str:
    return f"有效样本 n={n} 不足 1/alpha={min_required:g}"


def fit_metric_calibration(report, *, alpha: float = 0.10) -> dict:
    """从评估报告的逐样本指标拟合两个临床口径的 conformal 校准。

    :param report: duck 类型，需有 ``per_sample`` 列表（元素含 ``.metrics.dice``、
        ``.r_loss_pred``、``.r_loss_gt``）与 ``arch``/``scope`` 属性；
        :class:`~mgd.core.evaluation.evaluator.EvalReport` 即满足。
    :param alpha: 目标失效率，∈ (0, 0.5]；0.10 → 覆盖率 1-α = 90%。
    :returns: payload dict（结构见模块 docstring / ``format_version=1``）::

        {"calibration_id": sha256 前 12 位, "alpha": α, "n_dice": n1, "n_rloss": n2,
         "dice_coverage": {"bound": float|None, "coverage": 1-α, "n": n1,
                           "skipped_reason": str|None},
         "rloss_interval": {"q": float|None, "coverage": 1-α, "n": n2,
                            "skipped_reason": str|None},
         "created_at": UTC ISO, "format_version": 1, "arch": ..., "scope": ...}

    每个口径独立判定样本量：有效样本 ``n < 1/alpha`` 的口径写
    ``skipped_reason="有效样本 n=X 不足 1/alpha=Y"`` 且 ``bound``/``q`` 为 None，
    另一口径不受影响。NaN/inf 样本在计分前剔除（不冒充有效）。

    :raises ValueError: alpha 非法（不在 (0, 0.5]）——输入校验错误应显式暴露，
        与样本不足（预期内的业务状态，写 skipped）区分。
    """
    alpha = validate_alpha(alpha)
    min_required = 1.0 / alpha
    per_sample = list(getattr(report, "per_sample", None) or [])

    dice_scores: list[float] = []
    rloss_scores: list[float] = []
    for item in per_sample:
        dice = _finite_float(getattr(getattr(item, "metrics", None), "dice", None))
        if dice is not None:
            dice_scores.append(1.0 - dice)  # 非一致性得分：Dice 越低越“不一致”
        pred = _finite_float(getattr(item, "r_loss_pred", None))
        gt = _finite_float(getattr(item, "r_loss_gt", None))
        if pred is not None and gt is not None:
            rloss_scores.append(abs(pred - gt))

    dice_block: dict = {
        "bound": None,
        "coverage": 1.0 - alpha,
        "n": len(dice_scores),
        "skipped_reason": None,
    }
    rloss_block: dict = {
        "q": None,
        "coverage": 1.0 - alpha,
        "n": len(rloss_scores),
        "skipped_reason": None,
    }

    if len(dice_scores) < min_required:
        dice_block["skipped_reason"] = _insufficient_reason(len(dice_scores), min_required)
    else:
        calibrator = ConformalCalibrator.from_dict({"alpha": alpha, "scores": dice_scores})
        dice_block["bound"] = 1.0 - calibrator.quantile()  # P(Dice >= bound) >= 1-α

    if len(rloss_scores) < min_required:
        rloss_block["skipped_reason"] = _insufficient_reason(len(rloss_scores), min_required)
    else:
        calibrator = ConformalCalibrator.from_dict({"alpha": alpha, "scores": rloss_scores})
        rloss_block["q"] = calibrator.quantile()  # R_loss 预测值 ± q，覆盖率 1-α

    payload = {
        # id 只由得分决定（与 created_at 无关）→ 同一评估重跑得到同一 id，
        # 可作为报告间的可复核指纹
        "calibration_id": _sha256_prefix(
            "|".join(f"{s:.12g}" for s in (dice_scores + rloss_scores))
        ),
        "alpha": float(alpha),
        "n_dice": len(dice_scores),
        "n_rloss": len(rloss_scores),
        "dice_coverage": dice_block,
        "rloss_interval": rloss_block,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "format_version": FORMAT_VERSION,
        "arch": str(getattr(report, "arch", "") or ""),
        "scope": str(getattr(report, "scope", "") or ""),
    }
    skipped = [
        name for name, block in (("dice_coverage", dice_block), ("rloss_interval", rloss_block))
        if block["skipped_reason"]
    ]
    logger.debug(
        "metric 校准完成：calibration_id=%s, n_dice=%d, n_rloss=%d%s",
        payload["calibration_id"], len(dice_scores), len(rloss_scores),
        f"，skipped：{', '.join(skipped)}" if skipped else "",
    )
    return payload


def save_metric_calibration(payload: dict, settings) -> Path:
    """把 :func:`fit_metric_calibration` 的 payload 持久化到 ``settings.calibration_dir``。

    文件名 ``cal_{arch}_{scope}_{calibration_id}.json``；arch/scope 经白名单净化，
    保证合法且 glob 安全。外层 ``version``/``created_at``/``name`` 由
    :func:`~mgd.core.conformal.store.save_calibration` 统一覆盖（store 元数据可信）。

    :raises ValueError: payload 缺 ``calibration_id`` 或名称非法。
    """
    calibration_id = str(payload.get("calibration_id") or "")
    if not calibration_id:
        raise ValueError("payload 缺少 calibration_id：请先运行 fit_metric_calibration")
    name = f"cal_{_safe_component(payload.get('arch'))}_{_safe_component(payload.get('scope'))}_{calibration_id}"
    path = save_calibration(name, payload, directory=settings.calibration_dir)
    logger.debug("metric 校准已保存：%s", path)
    return path


def load_latest_metric_calibration(arch: str, settings) -> dict | None:
    """取 ``settings.calibration_dir`` 下 ``cal_{arch}_*.json`` 中 ``created_at`` 最新的一份。

    只读、永不抛错：目录缺失 / 无匹配文件 / 文件损坏（非 JSON、顶层非对象、
    缺 ``created_at``）一律跳过或返回 None——读取失败应让调用方走“无校准”
    分支，而不是让评估/推理中断。同秒并列时按文件名字典序取最大（确定性）。
    """
    directory = Path(getattr(settings, "calibration_dir", "") or "")
    if not str(directory):
        return None
    prefix = f"cal_{_safe_component(arch)}_"
    if not directory.is_dir():
        return None
    best_key: tuple[str, str] | None = None
    best_payload: dict | None = None
    for path in sorted(directory.glob(f"{prefix}*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue  # 损坏文件跳过，不阻断
        if not isinstance(payload, dict):
            continue
        created_at = payload.get("created_at")
        if not isinstance(created_at, str) or not created_at:
            continue
        key = (created_at, path.name)
        if best_key is None or key > best_key:
            best_key, best_payload = key, payload
    return best_payload


def attach_calibration_to_report(payload: dict, calibration: dict | None) -> dict:
    """把校准 payload 挂到报告 payload 上（就地修改并返回同一 dict）。

    - ``calibration`` 非 None：写 ``payload["conformal"] = calibration``，并把
      ``payload["extra"]["calibration_id"]`` 设为其 id（``extra`` 已存在则更新，
      不存在则创建；不改动调用方传入的 extra 原对象以外的结构）。
    - ``calibration`` 为 None：不做任何修改（报告不含 conformal 键）——
      对应 ``--no-conformal`` 或校准失败的降级路径。
    """
    if calibration is None:
        return payload
    payload["conformal"] = calibration
    extra = payload.get("extra")
    if isinstance(extra, dict):
        extra["calibration_id"] = calibration.get("calibration_id")
    else:
        payload["extra"] = {"calibration_id": calibration.get("calibration_id")}
    return payload

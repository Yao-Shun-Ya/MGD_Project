"""形态图谱 CLI：`mgd-atlas describe|classify|search`。

- describe：列出 8 类形态分型规则（定义/阈值/来源）
- classify：对单张腺体掩膜做分型（可选眼睑掩膜提供 GA 参考）
- search：把掩膜特征向量化后在图谱索引中检索相似病例
--json 输出为严格 JSON：NaN/Inf 一律清洗为 null（json.dumps 默认会写裸 NaN，
很多解析器直接拒绝）。退出码：0 成功；2 输入/运行错误。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2

from ...core.atlas.index import AtlasIndex
from ...core.atlas.typology import TYPE_RULES, classify_case
from ...core.atlas.features import case_vector
from ...core.config import get_settings
from ...core.governance.hooks import require_or_audit
from ...core.imaging.clinical import instance_metrics
from ...core.imaging.preprocess import load_gray, resize_gray
from ...core.quantification.morphometry import gland_area_ratio
from ...core.quantification.topology import break_count, split_instances
from ...core.runtime.errors import MGDError
from ...core.runtime.logging import configure_logging


def _load_mask(path: Path, settings) -> "cv2.typing.MatLike":
    mask = load_gray(path)
    if mask.shape[:2] != settings.target_size:
        mask = resize_gray(mask, settings.target_size, cv2.INTER_NEAREST)
    return (mask > 0).astype("uint8")


def _json_safe(value):
    """递归清洗为严格 JSON 可序列化：NaN / ±Inf → None（M-3）。"""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _dump_json(payload) -> None:
    """严格 JSON 输出：先清洗非有限浮点，再 dumps（default=str 兜底罕见类型）。"""
    print(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, default=str))


def _reconfigure_stdio() -> None:
    """管道/重定向输出统一 UTF-8（M-2）：GBK 控制台代码页下打印中文不再崩溃。

    pytest 捕获缓冲等非 TextIOWrapper 流没有 reconfigure，静默保持原样。
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def main(argv: list[str] | None = None) -> int:
    _reconfigure_stdio()
    parser = argparse.ArgumentParser(description="腺而易见形态图谱（8 类分型 + 相似病例检索）")
    parser.add_argument("action", choices=["describe", "classify", "search"])
    parser.add_argument("--mask", default=None, help="腺体掩膜路径（classify/search 必填）")
    parser.add_argument("--eyelid", default=None, help="可选眼睑掩膜（提供后计算 GA 与实例表）")
    parser.add_argument("--reference-density", type=float, default=None,
                        help="dropout 判定的同区参考密度（%%），缺省则 dropout 项 NaN 不判定")
    parser.add_argument("--min-distance", type=float, default=None,
                        help="confluence 判定的实例最小间距（px），缺省则该项 NaN 不判定")
    parser.add_argument("--k", type=int, default=5, help="search 返回条数")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)
    settings = get_settings(data_dir=args.data_dir) if args.data_dir else get_settings()
    configure_logging(log_file=settings.log_file)

    try:
        if args.action == "describe":
            rows = [
                {"key": rule.key, "name_zh": rule.name_zh, "definition": rule.definition, "source": rule.source}
                for rule in TYPE_RULES
            ]
            if args.json:
                _dump_json(rows)
            else:
                for row in rows:
                    print(f"[{row['key']}] {row['name_zh']}　({row['source']})\n    {row['definition']}")
            return 0

        if not args.mask:
            print("[失败] classify / search 需要 --mask <腺体掩膜路径>", file=sys.stderr)
            return 2
        mask_path = Path(args.mask).expanduser()
        if not mask_path.is_file():
            print(f"[失败] 掩膜不存在：{mask_path}", file=sys.stderr)
            return 2
        # RBAC 治理门：classify / search 属推理类动作（describe 纯文档不设限），
        # 放在参数校验之后、任何特征计算之前。拒绝时 access_denied 已先行落审计，
        # ConfigError（MGDError 子类）由本函数既有 except (MGDError, ValueError, OSError)
        # 接住 → user_message + 退出码 2。
        require_or_audit(settings, "run_inference", entry="mgd-atlas")
        mask = _load_mask(mask_path, settings)
        eyelid = None
        if args.eyelid:
            eyelid_path = Path(args.eyelid).expanduser()
            if not eyelid_path.is_file():
                print(f"[失败] 眼睑掩膜不存在：{eyelid_path}", file=sys.stderr)
                return 2
            eyelid = _load_mask(eyelid_path, settings)

        instance_df = instance_metrics(mask, eyelid)
        case_features: dict = {
            "ga_percent": gland_area_ratio(mask, eyelid) if eyelid is not None else float("nan"),
            "break_count": break_count(mask),
            "instance_count": int(len(instance_df)),
        }
        if args.reference_density is not None:
            case_features["reference_density_percent"] = float(args.reference_density)
        if args.min_distance is not None:
            case_features["min_inter_instance_distance_px"] = float(args.min_distance)

        if args.action == "classify":
            evidence = classify_case(instance_df, case_features)
            rows = [
                {"key": item.key, "present": bool(item.present), "value": item.value,
                 "threshold": item.threshold, "definition": item.definition}
                for item in evidence
            ]
            if args.json:
                _dump_json(rows)
            else:
                for row in rows:
                    state = "触发" if row["present"] else "未触发"
                    print(f"[{state}] {row['key']:12s} 值={row['value']:.4f} 阈值={row['threshold']:.4f}")
            return 0

        # search
        index_path = settings.atlas_index_path
        if not Path(index_path).is_file():
            print(f"[失败] 图谱索引不存在：{index_path}（先用 WebUI「形态图谱」页或编程接口构建）",
                  file=sys.stderr)
            return 2
        index = AtlasIndex.load(index_path)
        vector = case_vector({**case_features, "min_inter_instance_distance_px":
                              case_features.get("min_inter_instance_distance_px", float("nan"))})
        hits = index.search(vector, k=max(1, int(args.k)))
        if args.json:
            _dump_json(hits)
        else:
            for hit in hits:
                print(f"{hit['distance']:.4f}  {hit['case_id']}")
        return 0
    except (MGDError, ValueError, OSError) as exc:
        print(f"[失败] {getattr(exc, 'user_message', lambda: str(exc))()}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

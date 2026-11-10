"""采集质控 CLI：`mgd-qc --image <path> [--eyelid <path>]`。

评估单张影像的采集质量并给出建议动作（accept / review / reject）。
退出码：0 = 评估成功（accept/review）；1 = 评估成功但结论为 reject；2 = 输入/运行错误。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from ...core.config import get_settings
from ...core.imaging.preprocess import load_gray, resize_gray
from ...core.qc.policy import QCPolicy, decide
from ...core.qc.quality import assess_quality, default_thresholds
from ...core.runtime.errors import MGDError
from ...core.runtime.logging import configure_logging


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
    parser = argparse.ArgumentParser(description="腺而易见采集质控（模糊/曝光/反光/离焦/翻转/睫毛启发式）")
    parser.add_argument("--image", required=True, help="待检影像路径")
    parser.add_argument("--eyelid", default=None, help="可选眼睑掩膜路径（缺失时翻焦/睫毛项标 None 不计分）")
    parser.add_argument("--policy", default=None, help="质控策略名（默认取 Settings.qc_policy）")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)
    settings = get_settings(data_dir=args.data_dir) if args.data_dir else get_settings()
    configure_logging(log_file=settings.log_file)

    image_path = Path(args.image).expanduser()
    if not image_path.is_file():
        print(f"[失败] 影像不存在：{image_path}", file=sys.stderr)
        return 2
    eyelid_path = Path(args.eyelid).expanduser() if args.eyelid else None
    if eyelid_path is not None and not eyelid_path.is_file():
        print(f"[失败] 眼睑掩膜不存在：{eyelid_path}", file=sys.stderr)
        return 2

    try:
        gray = load_gray(image_path)
        if gray.shape[:2] != settings.target_size:
            gray = resize_gray(gray, settings.target_size, cv2.INTER_LINEAR)
        eyelid = None
        if eyelid_path is not None:
            raw = load_gray(eyelid_path)
            if raw.shape[:2] != settings.target_size:
                raw = resize_gray(raw, settings.target_size, cv2.INTER_NEAREST)
            eyelid = (raw > 0).astype("uint8")
        report = assess_quality(gray, eyelid_mask=eyelid, thresholds=default_thresholds(
            args.policy or settings.qc_policy))
        verdict = decide(report, QCPolicy())
    except (MGDError, ValueError, OSError) as exc:
        print(f"[失败] {getattr(exc, 'user_message', lambda: str(exc))()}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({
            "score": report.score,
            "flags": report.flags,
            "reasons": report.reasons,
            "metrics": report.metrics,
            "action": verdict["action"],
            "action_reason": verdict["reason"],
        }, ensure_ascii=False, indent=2))
    else:
        print(f"质控评分：{report.score:.1f} / 100　建议：{verdict['action']}（{verdict['reason']}）")
        print(f"旗标：{'、'.join(report.flags) if report.flags else '无'}")
        for reason in report.reasons:
            print(f"  - {reason}")
        for key, value in sorted(report.metrics.items()):
            shown = "N/A（不可测）" if value is None or value != value else f"{value:.4f}"
            print(f"  {key:24s} {shown}")
    return 1 if verdict["action"] == "reject" else 0


if __name__ == "__main__":
    raise SystemExit(main())

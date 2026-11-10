"""学术训练流水线：阶段编排、批量排队、失败重试、断点续训、聚合报告。

阶段：pretrain / train / finetune / eval / ablation / report
- 真实实现：train、finetune（= 预训练权重 + 微调学习率）、eval、ablation（= 多配置训练）、report；
- pretrain 在离线环境不可用，会如实标记 skipped 并写原因，不假装跑了。
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..config import Settings, get_settings
from ..data.pairing import scan_dataset
from ..data.splits import SplitSpec, get_splits, stems_signature
from ..evaluation.evaluator import evaluate_pairs
from ..evaluation.stats import bootstrap_ci
from ..inference.service import InferenceOptions
from ..runtime.logging import get_logger
from ..training.runner import read_run, run_training

logger = get_logger("workflow.pipeline")

STAGES = ("pretrain", "train", "finetune", "eval", "ablation", "report")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_state(path: Path) -> dict:
    """读取流水线状态；损坏文件回退为空状态并保留告警，避免 KeyError。"""
    empty = {"entries": {}, "runs": []}
    if not path.is_file():
        return empty
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("流水线状态文件损坏，已回退为空状态：%s（%s）", path, exc)
        return empty
    if not isinstance(payload, dict):
        logger.warning("流水线状态格式异常（非对象），已回退为空状态：%s", path)
        return empty
    payload.setdefault("entries", {})
    payload.setdefault("runs", [])
    if not isinstance(payload["entries"], dict):
        logger.warning("流水线状态 entries 字段损坏，已重置：%s", path)
        payload["entries"] = {}
    if not isinstance(payload["runs"], list):
        logger.warning("流水线状态 runs 字段损坏，已重置：%s", path)
        payload["runs"] = []
    return payload


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 原子写：断电/中断不能丢断点续跑状态
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def run_pipeline(
    stages: list[str],
    configs: list[tuple[str, Any, dict]],
    settings: Settings | None = None,
    *,
    retries: int = 1,
    state_path: str | Path | None = None,
    resume: bool = True,
    on_event: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """顺序执行配置 × 阶段。返回聚合结果（含每个 run 的 split_signature）。"""
    settings = settings or get_settings()
    # 阶段合法性在入口一次性校验：避免跑到最后一个阶段才发现拼写错误
    for stage in stages:
        normalized = stage.strip().lower()
        if normalized not in STAGES:
            raise ValueError(f"未知阶段 {stage!r}，可选：{', '.join(STAGES)}")
    state_file = Path(state_path) if state_path else settings.output_dir / "pipeline_state.json"
    state = _load_state(state_file) if resume else {"entries": {}, "runs": []}
    results: list[dict[str, Any]] = []

    for stage in stages:
        normalized = stage.strip().lower()
        if normalized == "pretrain":
            entry = {"stage": stage, "label": "self-supervised", "status": "skipped",
                     "reason": "自监督预训练需要大规模无标注语料与额外算力，本环境离线不可用"}
            results.append(entry)
            if on_event:
                on_event({"event": "stage", **entry})
            continue

        for label, config, meta in configs:
            key = f"{normalized}::{label}"
            stored = state.get("entries", {}).get(key)
            if resume and isinstance(stored, dict) and stored.get("status") == "finished":
                results.append({**stored, "skipped_resume": True})
                continue

            attempt = 0
            while True:
                attempt += 1
                entry: dict[str, Any] = {"stage": normalized, "label": label, "attempt": attempt,
                                         "started_at": _now()}
                try:
                    if normalized in ("train", "finetune", "ablation"):
                        result, context = run_training(config, meta, settings, on_event=None)
                        entry.update({
                            "status": "finished",
                            "run_id": context.run_id,
                            "run_dir": str(context.run_dir),
                            "best_metric": result.best_metric,
                            "peak_vram_gb": result.peak_vram_gb,
                            "samples_per_second": result.samples_per_second,
                            "pretrained_status": result.pretrained_status,
                        })
                    elif normalized == "eval":
                        entry.update(_eval_entry(config, settings))
                    else:  # report
                        entry.update({"status": "finished", "report": _write_report(state, settings)})
                except Exception as exc:
                    entry.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                    if attempt <= retries:
                        if on_event:
                            on_event({"event": "retry", "label": label, "stage": normalized,
                                      "attempt": attempt, "error": str(exc)})
                        time.sleep(min(5.0, 0.5 * attempt))
                        continue
                break

            entry["finished_at"] = _now()
            state["entries"][key] = entry
            if entry.get("run_id"):
                state["runs"].append({"stage": normalized, "label": label, **entry})
            _save_state(state_file, state)
            results.append(entry)
            if on_event:
                on_event({"event": "stage", **entry})

    return {
        "stages": stages,
        "results": results,
        "state_path": str(state_file),
        "started_at": results[0].get("started_at") if results else _now(),
        "finished_at": _now(),
    }


def _eval_entry(config, settings: Settings) -> dict[str, Any]:
    index = scan_dataset(settings)
    split = get_splits(index.pairs, SplitSpec(mode=str(config.split_mode).lower(),
                                              n_folds=int(config.n_folds), fold=int(config.fold)),
                       settings)
    pairs = split.test or split.val
    if not pairs:
        return {"status": "skipped", "reason": "无独立测试/验证集"}
    report = evaluate_pairs(
        pairs, arch=config.model_name, settings=settings,
        options=InferenceOptions(threshold=float(config.eval_threshold),
                                 tile_size=int(config.eval_tile_size),
                                 tile_overlap=float(config.eval_tile_overlap),
                                 tta=str(config.eval_tta)),
        scope="test" if split.test else "val",
    )
    values = [item.metrics.dice for item in report.per_sample]
    return {
        "status": "finished",
        "samples": report.samples,
        "dice": bootstrap_ci(values)["mean"],
        "split_signature": stems_signature([pair.stem for pair in index.pairs]),
        "summary": dict(report.summary),
    }


def _collect_run_dirs(state: dict, settings: Settings) -> list[dict[str, Any]]:
    """state 中登记的 run 优先；state 为空或漏记时回退扫描 output_dir/runs。

    只跑 `--stages report` 时 state["runs"] 为空，报告仍应汇总磁盘上已有的 run。
    """
    collected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in state.get("runs") or []:
        run_id = str(item.get("run_id") or "")
        run_dir = str(item.get("run_dir") or "")
        if not run_id or not run_dir:
            continue
        collected.append({"run_id": run_id, "run_dir": run_dir,
                          "label": item.get("label"), "stage": item.get("stage"),
                          "best_metric": item.get("best_metric")})
        seen.add(run_id)
    runs_root = Path(settings.output_dir) / "runs"
    if runs_root.is_dir():
        for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
            if run_dir.name in seen or not (run_dir / "run_result.json").is_file():
                continue  # 跳过中断/未完成目录，避免污染论文汇总
            collected.append({"run_id": run_dir.name, "run_dir": str(run_dir),
                              "label": "disk-scan", "stage": "train", "best_metric": None})
    return collected


def _write_report(state: dict, settings: Settings) -> str:
    """聚合已完成 run；强制携带 split_signature，缺签名条目会被列出。"""
    runs = []
    for item in _collect_run_dirs(state, settings):
        bundle = read_run(item["run_dir"])
        result = bundle.get("run_result.json") or {}
        reference = bundle.get("reference_report.json") or {}
        runs.append({
            "run_id": item.get("run_id"),
            "label": item.get("label"),
            "stage": item.get("stage"),
            "arch": result.get("arch"),
            "best_metric": item.get("best_metric") if item.get("best_metric") is not None
            else result.get("best_metric"),
            "split_signature": reference.get("split_signature") or item.get("split_signature"),
            "verified": bool(reference.get("valid")),
        })
    missing = [item["run_id"] for item in runs if not item["split_signature"]]
    payload = {
        "format_version": 1,
        "created_at": _now(),
        "runs": runs,
        "missing_split_signature": missing,
        "note": "报告只汇总产物；缺 split_signature 的 run 不能用于论文表格",
    }
    target = settings.output_dir / "pipeline_report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(target)

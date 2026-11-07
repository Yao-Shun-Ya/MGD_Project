"""一次训练 run 的编排：目录、快照、日志、metrics、续训、验证口径报告。

CLI / WebUI / 流水线都调用这里，保证三者产物结构完全一致。
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

from ..config import Settings, get_settings
from ..data.pairing import scan_dataset
from ..data.splits import SplitSpec, get_splits, stems_signature
from ..evaluation.evaluator import evaluate_pairs
from ..evaluation.stats import reference_report_payload, write_reference_report
from ..inference.service import InferenceOptions
from ..models.registry import normalize_model_name
from ..runtime.errors import MGDError
from ..runtime.logging import get_logger
from .config_file import snapshot_payload, write_snapshot
from .engine import TrainingResult, train_iter
from .run import (
    RunContext,
    append_metric,
    config_fingerprint,
    create_run,
    resolve_resume,
    write_env_snapshot,
    write_result,
)

logger = get_logger("training.runner")

EventHandler = Callable[[dict], None]


def _attach_file_log(path: Path) -> logging.Handler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger("mgd").addHandler(handler)
    return handler


def run_training(
    config,
    meta: dict,
    settings: Settings | None = None,
    on_event: EventHandler | None = None,
) -> tuple[TrainingResult, RunContext]:
    """执行一次完整训练并返回 (结果, run 上下文)。"""
    settings = settings or get_settings()
    arch = normalize_model_name(config.model_name)
    fingerprint = config_fingerprint(snapshot_payload(config, meta))
    context = create_run(
        settings,
        arch,
        name=config.run_name,
        output_dir=meta.get("output_dir") or config.run_dir,
        fingerprint=fingerprint,
    )
    run_dir = context.run_dir
    resume = resolve_resume(config.resume_from, context.checkpoints_dir) if config.resume_from else None
    effective = replace(config, run_dir=str(run_dir), resume_from=str(resume) if resume else None)
    meta = dict(meta)
    meta["run_id"] = context.run_id

    write_snapshot(context.config_path, effective, meta)
    write_env_snapshot(context.env_path, settings, device_text=effective.device)
    handler = _attach_file_log(context.log_path)

    started = time.perf_counter()
    result: TrainingResult | None = None
    try:
        for event in train_iter(effective, settings):
            if event.get("event") == "epoch":
                append_metric(context.metrics_path, event)
            if event.get("event") == "done":
                result = event["result"]
            if on_event is not None:
                on_event(event)
    finally:
        logging.getLogger("mgd").removeHandler(handler)

    if result is None:
        raise MGDError("训练未产生结果")
    result.run_dir = str(run_dir)
    result_dict = dict(result.__dict__)
    result_dict["run_id"] = context.run_id
    result_dict["wall_seconds"] = round(time.perf_counter() - started, 3)
    write_result(context.result_path, result_dict)

    if effective.export_reference_report and result.best_weights:
        _export_reference(context, effective, settings)

    return result, context


def _export_reference(context: RunContext, config, settings: Settings) -> None:
    """训练结束后在独立测试集上生成 reference_report.json（带 split_signature）。"""
    try:
        index = scan_dataset(settings)
        spec = SplitSpec(
            mode=str(config.split_mode).lower(),
            val_ratio=float(config.val_ratio),
            test_ratio=float(config.test_ratio),
            seed=int(config.seed),
            n_folds=int(config.n_folds),
            fold=int(config.fold),
        )
        split = get_splits(index.pairs, spec, settings)
        pairs = split.test or split.val
        if not pairs:
            logger.warning("没有独立测试集/验证集，跳过 reference_report")
            return
        best = next(iter(sorted(context.checkpoints_dir.glob("*_best.pth"))), None)
        if best is None:
            logger.warning("run 目录中没有 best 权重，跳过 reference_report")
            return
        options = InferenceOptions(
            tile_infer=True,
            tile_size=int(config.eval_tile_size),
            tile_overlap=float(config.eval_tile_overlap),
            threshold=float(config.eval_threshold),
            tta=str(config.eval_tta),
        )
        report = evaluate_pairs(
            pairs,
            arch=config.model_name,
            weights_path=str(best),
            settings=settings,
            options=options,
            scope="test" if split.test else "val",
        )
        payload = reference_report_payload(
            report,
            split_signature=stems_signature([pair.stem for pair in index.pairs]),
            data_signature=stems_signature([pair.stem for pair in index.pairs]),
            n_bootstrap=1000,
            extra={
                "run_id": context.run_id,
                "evaluated_weights": best.name,
                "note": "训练产物为草稿（verified=false），需 mgd-verify --evaluate 才可转正",
            },
        )
        write_reference_report(context.reference_path, payload)
    except Exception as exc:  # 报告失败不能让训练结果丢失
        logger.warning("生成 reference_report 失败：%s", exc)


def read_run(run_dir: str | Path) -> dict:
    """读取 run 目录摘要（实验管理用）。"""
    directory = Path(run_dir)
    payload: dict = {"run_id": directory.name, "run_dir": str(directory)}
    for name in ("config.yaml", "env.json", "run_result.json", "reference_report.json"):
        path = directory / name
        if path.is_file():
            try:
                if name.endswith(".json"):
                    payload[name] = json.loads(path.read_text(encoding="utf-8"))
                else:
                    payload[name] = path.read_text(encoding="utf-8")
            except (OSError, json.JSONDecodeError):
                payload[name] = None
    metrics_path = directory / "metrics.jsonl"
    records: list[dict] = []
    if metrics_path.is_file():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    payload["metrics"] = records
    return payload

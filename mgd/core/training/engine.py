"""训练引擎：完整状态检查点、正确的 LR 日志顺序、严格数据集划分。"""
from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import torch
from monai.inferers import sliding_window_inference

from ... import __version__
from ..config import Settings, checkpoint_path, get_settings, model_best_path
from ..data.dataset import build_dataloader, build_dataset
from ..data.pairing import Pair, scan_dataset
from ..data.splits import SplitSpec, get_splits, stems_signature
from ..data.transforms import build_eval_transforms, build_train_transforms
from ..models.manifest import build_manifest, write_manifest
from ..models.registry import create_model, load_pretrained, normalize_model_name, unwrap_outputs
from ..runtime.device import describe_device, empty_cache, resolve_device
from ..runtime.errors import MGDError, TrainingError
from ..runtime.logging import get_logger
from .checkpoint import apply_checkpoint, load_checkpoint, save_checkpoint
from .determinism import rng_state, seed_everything
from .ema import ModelEMA
from .losses import deep_supervision_loss, get_loss_function, get_optimizer, get_scheduler
from .run import publish_best

logger = get_logger("training")

_MEMORY_FRACTION_SET = False


@dataclass
class TrainConfig:
    model_name: str = "MGD-Former"
    pretrained: str = "auto"
    deep_supervision: bool = True
    epochs: int = 20
    batch_size: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "AdamW"
    scheduler: str = "Cosine Annealing"
    loss_name: str = "MGD 复合损失"
    pos_weight: float = 1.0
    cldice_weight: float = 0.2
    cldice_warmup: float = 0.1
    boundary_weight: float = 0.0
    ema_decay: float = 0.999
    grad_accum_steps: int = 1
    channels_last: bool = True
    vram_tier: str = "auto"
    tile_mode: bool = True
    tile_height: int = 384
    augmentation: bool = True
    num_workers: int = 0
    amp: bool = False
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_mode: str = "holdout"
    n_folds: int = 5
    fold: int = 0
    dual_task: bool = True
    eyelid_weight: float = 0.25
    seed: int = 42
    resume_from: str | None = None
    use_wandb: bool = False
    wandb_project: str = "mgd"
    wandb_run_name: str | None = None
    save_every: int = 1
    keep_checkpoints: int = 3
    mem_limit_gb: float | None = None
    device: str | None = None
    run_name: str | None = None
    run_dir: str | None = None
    deterministic: bool = True
    publish_best: bool = True
    export_reference_report: bool = False
    eval_tta: str = "关闭"
    eval_threshold: float = 0.7
    eval_tile_size: int = 384
    eval_tile_overlap: float = 0.25

    def validate(self) -> None:
        if int(self.epochs) < 1:
            raise TrainingError("epochs 必须 ≥ 1")
        if int(self.batch_size) < 1:
            raise TrainingError("batch_size 必须 ≥ 1")
        if float(self.lr) <= 0:
            raise TrainingError("lr 必须 > 0")
        if not (0 <= float(self.val_ratio) < 1) or not (0 <= float(self.test_ratio) < 1):
            raise TrainingError("val_ratio / test_ratio 必须在 [0, 1) 区间")
        if float(self.val_ratio) + float(self.test_ratio) >= 1:
            raise TrainingError("val_ratio + test_ratio 必须 < 1")
        if str(self.split_mode).lower() not in ("holdout", "kfold"):
            raise TrainingError("split_mode 必须是 holdout 或 kfold")
        if int(self.grad_accum_steps) < 1:
            raise TrainingError("grad_accum_steps 必须 ≥ 1")
        if not (0.0 <= float(self.cldice_warmup) <= 0.5):
            raise TrainingError("cldice_warmup 必须在 [0, 0.5] 区间")
        if not (0.0 <= float(self.ema_decay) <= 1.0):
            raise TrainingError("ema_decay 必须在 [0, 1] 区间")


@dataclass
class TrainingResult:
    arch: str
    start_epoch: int
    epochs_run: int
    best_metric: float
    best_epoch: int | None
    eval_scope: str
    scope_label: str
    history: list[dict] = field(default_factory=list)
    best_weights: str | None = None
    last_checkpoint: str | None = None
    device: str = "CPU"
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    dual_task: bool = True
    warnings: list[str] = field(default_factory=list)
    run_dir: str | None = None
    peak_vram_gb: float | None = None
    samples_per_second: float = 0.0
    pretrained_status: dict = field(default_factory=dict)

    @property
    def metric_is_self_evaluated(self) -> bool:
        return self.eval_scope == "train_self"


def _dice(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred = (torch.sigmoid(logits) > threshold).float()
    intersection = float((pred * target).sum())
    denominator = float(pred.sum() + target.sum())
    if denominator == 0:
        return float("nan")
    return 2.0 * intersection / denominator


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError, ValueError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _apply_memory_limit(device: torch.device, mem_limit_gb: float | None) -> None:
    """每个进程只设置一次（该设置不可逆）。"""
    global _MEMORY_FRACTION_SET
    if _MEMORY_FRACTION_SET or mem_limit_gb is None or device.type != "cuda":
        return
    try:
        total = torch.cuda.get_device_properties(device).total_memory
        fraction = min(1.0, float(mem_limit_gb) * (1024 ** 3) / float(total))
        if fraction > 0:
            torch.cuda.set_per_process_memory_fraction(fraction, device.index or 0)
            logger.info("已设置显存上限：%.2f GB (fraction=%.3f)", mem_limit_gb, fraction)
    except Exception as exc:  # 驱动差异不应阻断训练
        logger.warning("设置显存上限失败：%s", exc)
    finally:
        _MEMORY_FRACTION_SET = True


def _cleanup_checkpoints(directory: Path, model_name: str, keep: int) -> None:
    if not directory.is_dir():
        return
    pattern = re.compile(rf"{model_name.lower()}_epoch_(\d+).mgd\.ckpt$")
    files = []
    for path in directory.iterdir():
        match = pattern.search(path.name)
        if match:
            files.append((int(match.group(1)), path))
    files.sort(key=lambda item: item[0])
    for _, path in files[: max(0, len(files) - int(keep))]:
        try:
            path.unlink()
        except OSError as exc:
            logger.warning("清理检查点失败 %s：%s", path, exc)


def train_iter(config: TrainConfig, settings: Settings | None = None) -> Iterator[dict]:
    """流式产出训练进度事件；最后一个事件为 {"event": "done", "result": TrainingResult}。"""
    config.validate()
    settings = settings or get_settings()
    arch = normalize_model_name(config.model_name)
    device = resolve_device(config.device or settings.device)
    _apply_memory_limit(device, config.mem_limit_gb)
    seed_everything(int(config.seed), bool(config.deterministic))

    run_dir = Path(config.run_dir).expanduser().resolve() if config.run_dir else None
    checkpoint_dir = (run_dir / "checkpoints") if run_dir else (settings.model_dir / arch)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = (
        checkpoint_dir / f"{arch.lower()}_best.pth"
        if run_dir
        else model_best_path(arch, settings)
    )

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
    warnings = list(split.warnings)
    data_signature = stems_signature([pair.stem for pair in index.pairs])
    dataset_sizes = {
        "train": len(split.train),
        "val": len(split.val),
        "test": len(split.test),
        "total": len(index.pairs),
    }
    if not split.val:
        warnings.append("没有独立验证集，本轮所有指标为训练集自评（不代表泛化性能）")
    elif len(split.val) < 3:
        warnings.append(f"验证集仅 {len(split.val)} 例，指标波动较大")
    if not split.test:
        warnings.append("没有独立测试集；论文指标请另外用 test split 评估")

    train_transform = build_train_transforms(
        tile_mode=config.tile_mode,
        tile_height=config.tile_height,
        augmentation=config.augmentation,
        settings=settings,
    )
    eval_transform = build_eval_transforms(settings=settings)
    train_loader = build_dataloader(
        build_dataset(split.train, train_transform),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        device=device,
    )
    val_loader = None
    if split.val:
        val_loader = build_dataloader(
            build_dataset(split.val, eval_transform),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            device=device,
        )

    dual_task = bool(config.dual_task)
    model = create_model(
        arch,
        in_channels=1,
        out_channels=2 if dual_task else 1,
        deep_supervision=bool(config.deep_supervision),
    ).to(device)
    if bool(config.channels_last):
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception as exc:  # 个别自定义层可能不支持，回退即可
            warnings.append(f"channels_last 启用失败，已回退：{exc}")
    pretrained_status = load_pretrained(model, arch, config.pretrained)
    if arch == "MGD-Former" and not pretrained_status.get("loaded"):
        warnings.append(f"编码器未加载预训练权重：{pretrained_status.get('reason')}")
    criterion = get_loss_function(
        config.loss_name,
        pos_weight=config.pos_weight,
        dual_task=dual_task,
        eyelid_weight=config.eyelid_weight,
        cldice_weight=config.cldice_weight,
        boundary_weight=config.boundary_weight,
    )
    optimizer = get_optimizer(config.optimizer, model, config.lr, config.weight_decay)
    scheduler, batch_level = get_scheduler(
        config.scheduler, optimizer, config.epochs, max(1, len(train_loader))
    )

    use_amp = bool(config.amp) and device.type == "cuda"
    if config.amp and not use_amp:
        warnings.append("当前设备不支持 AMP（仅 CUDA 启用），已自动关闭")
    scaler = _make_scaler(use_amp)

    start_epoch = 1
    best_metric = float("nan")
    best_epoch: int | None = None
    history: list[dict] = []

    resumed_state = None
    if config.resume_from:
        state = load_checkpoint(config.resume_from, map_location=device)
        resumed_state = state
        if normalize_model_name(state.arch) != arch:
            raise TrainingError(f"检查点架构 {state.arch} 与当前 {arch} 不一致")
        start_epoch = apply_checkpoint(state, model, optimizer, scheduler, scaler)
        best_metric = state.best_metric
        best_epoch = state.best_epoch
        history = list(state.history)
        if start_epoch > config.epochs:
            warnings.append(f"检查点已完成 {state.epoch} 轮，已达到目标 {config.epochs} 轮，不再训练")
        logger.info("从 %s 续训：起始 epoch=%d", state.path, start_epoch)

    ema = ModelEMA(model, config.ema_decay)
    if resumed_state is not None and resumed_state.ema_state:
        try:
            ema.load_state_dict(resumed_state.ema_state)
        except Exception as exc:
            warnings.append(f"EMA 状态恢复失败（将重新累积）：{exc}")
    if not ema.enabled:
        warnings.append("ema_decay=1.0 或 <1 之外的取值，EMA 未启用；best 权重直接使用在线参数")
    if device.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass

    yield {
        "event": "start",
        "arch": arch,
        "device": describe_device(device),
        "train_samples": len(split.train),
        "val_samples": len(split.val),
        "warnings": warnings,
    }

    wandb_run = None
    if config.use_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=asdict(config),
                reinit=True,
            )
        except Exception as exc:  # W&B 不可用不应中断训练
            warnings.append(f"W&B 初始化失败，已继续本地训练：{exc}")
            logger.warning("W&B 初始化失败：%s", exc)
            wandb_run = None

    epochs_run = 0
    best_weights: Path | None = None
    last_checkpoint: Path | None = None
    eval_scope = "holdout" if split.val else "train_self"
    scope_label = "独立验证集" if eval_scope == "holdout" else "训练集自评（不代表泛化性能）"

    try:
        for epoch in range(start_epoch, int(config.epochs) + 1):
            model.train()
            if hasattr(criterion, "cldice_weight"):
                warmup = max(1, int(round(float(config.epochs) * float(config.cldice_warmup))))
                progress = min(1.0, (epoch - 1) / warmup)
                criterion.cldice_weight = float(config.cldice_weight) * progress
            loss_total = 0.0
            dice_values: list[float] = []
            batches = 0
            epoch_started = time.perf_counter()
            accum = max(1, int(config.grad_accum_steps))
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(train_loader):
                images = batch["image"].float().to(device)
                if bool(config.channels_last):
                    images = images.contiguous(memory_format=torch.channels_last)
                labels = batch["label"].float().to(device)
                eyelids = batch["eyelid"].float().to(device) if dual_task else None

                with torch.autocast(device_type="cuda", enabled=use_amp):
                    outputs = model(images)
                    logits, aux = unwrap_outputs(outputs)
                    loss = criterion(logits, labels, eyelids) if dual_task else criterion(logits, labels)
                    aux_loss = deep_supervision_loss(aux, labels) if aux else None
                    if aux_loss is not None:
                        loss = loss + aux_loss
                    loss = loss / accum

                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % accum == 0 or (step + 1) == len(train_loader):
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update(model)

                if batch_level:
                    scheduler.step()

                loss_total += float(loss.detach().cpu()) * accum
                dice_value = _dice(logits[:, 0:1].detach(), labels)
                if not math.isnan(dice_value):
                    dice_values.append(dice_value)
                batches += 1

                current_lr = float(optimizer.param_groups[0]["lr"])

            train_loss = loss_total / max(1, batches)
            train_dice = float(sum(dice_values) / len(dice_values)) if dice_values else float("nan")

            val_loss = float("nan")
            val_dice = float("nan")
            val_eyelid_dice = float("nan")
            if val_loader is not None:
                with ema.activated(model):
                    model.eval()
                    val_loss_total = 0.0
                    val_dices: list[float] = []
                    val_eyelid_dices: list[float] = []
                    val_batches = 0
                    def _val_predict(x: torch.Tensor) -> torch.Tensor:
                        return unwrap_outputs(model(x))[0]

                    tile_h = int(max(64, min(int(config.tile_height), settings.target_height)))
                    tile_w = int(max(128, min(tile_h * 2, settings.target_width)))
                    with torch.no_grad():
                        for batch in val_loader:
                            images = batch["image"].float().to(device)
                            labels = batch["label"].float().to(device)
                            eyelids = batch["eyelid"].float().to(device) if dual_task else None
                            if bool(config.tile_mode):
                                # 验证必须与训练同尺度：整图 640×1280 直接前向会让
                                # 在 384×768 分块上训练的模型出现严重尺度失配。
                                logits = sliding_window_inference(
                                    images, (tile_h, tile_w), sw_batch_size=1,
                                    predictor=_val_predict, overlap=0.25,
                                )
                            else:
                                logits = _val_predict(images)
                            val_loss_total += float(
                                criterion(logits, labels, eyelids) if dual_task else criterion(logits, labels)
                            )
                            value = _dice(logits[:, 0:1], labels)
                            if not math.isnan(value):
                                val_dices.append(value)
                            if dual_task and eyelids is not None:
                                eyelid_value = _dice(logits[:, 1:2], eyelids)
                                if not math.isnan(eyelid_value):
                                    val_eyelid_dices.append(eyelid_value)
                            val_batches += 1
                    val_loss = val_loss_total / max(1, val_batches)
                    val_dice = float(sum(val_dices) / len(val_dices)) if val_dices else float("nan")
                    if val_eyelid_dices:
                        val_eyelid_dice = float(sum(val_eyelid_dices) / len(val_eyelid_dices))
                model.train()

            epoch_seconds = max(1e-6, time.perf_counter() - epoch_started)
            epoch_samples = len(split.train)
            samples_per_second = epoch_samples / epoch_seconds

            monitor = val_dice if not math.isnan(val_dice) else train_dice
            if not math.isnan(monitor) and (math.isnan(best_metric) or monitor > best_metric):
                best_metric = monitor
                best_epoch = epoch
                best_path.parent.mkdir(parents=True, exist_ok=True)
                with ema.activated(model):
                    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                torch.save(state_dict, best_path)
                manifest = build_manifest(
                    best_path,
                    arch=arch,
                    model_version=__version__,
                    # 训练产物一律是草稿：必须再用 verify --evaluate 才能转正
                    verified=False,
                    source="mgd-training",
                    settings=settings,
                    trained_on={
                        "scope": eval_scope,
                        "metric": monitor,
                        "epoch": epoch,
                        "train_samples": len(split.train),
                        "val_samples": len(split.val),
                        "loss": config.loss_name,
                        "data_signature": data_signature,
                    },
                    extra={
                        "out_channels": 2 if dual_task else 1,
                        "dual_task": dual_task,
                        "draft": True,
                        "run_dir": str(run_dir) if run_dir else None,
                        "seed": int(config.seed),
                        "dataset_sizes": dataset_sizes,
                    },
                )
                write_manifest(manifest, best_path)
                best_weights = best_path

            if not batch_level:
                plateau_value = val_loss if not math.isnan(val_loss) else train_loss
                scheduler.step(plateau_value)

            record = {
                "epoch": epoch,
                "loss": train_loss,
                "dice": train_dice,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_eyelid_dice": val_eyelid_dice,
                "lr": current_lr,
                "monitor": monitor,
                "best": best_metric,
                "epoch_seconds": round(epoch_seconds, 3),
                "samples_per_second": round(samples_per_second, 3),
                "cldice_weight": float(getattr(criterion, "cldice_weight", 0.0)),
            }
            history.append(record)
            epochs_run += 1

            if epoch % max(1, int(config.save_every)) == 0:
                checkpoint_target = (
                    checkpoint_dir / f"{arch.lower()}_epoch_{epoch:03d}.mgd.ckpt"
                    if run_dir
                    else checkpoint_path(arch, epoch, settings)
                )
                last_checkpoint = save_checkpoint(
                    checkpoint_target,
                    model,
                    optimizer,
                    epoch,
                    arch=arch,
                    scheduler=scheduler,
                    scaler=scaler if use_amp else None,
                    best_metric=best_metric,
                    best_epoch=best_epoch,
                    history=history,
                    config=asdict(config),
                    rng_state=rng_state(),
                    data_signature=data_signature,
                    dataset_sizes=dataset_sizes,
                    steps=epoch * max(1, len(train_loader)),
                    ema_state=ema.state_dict(),
                )
                _cleanup_checkpoints(checkpoint_dir, arch, config.keep_checkpoints)

            if wandb_run is not None:
                try:
                    wandb_run.log(
                        {
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "train_dice": train_dice,
                            "val_loss": val_loss,
                            "val_dice": val_dice,
                            "learning_rate": current_lr,
                        },
                        step=epoch,
                    )
                except Exception as exc:  # 网络/服务端异常不应中断训练
                    warnings.append(f"W&B 记录失败（第 {epoch} 轮）：{exc}")

            yield {
                "event": "epoch",
                **record,
                "total_epochs": int(config.epochs),
                "scope": scope_label,
            }

    except torch.cuda.OutOfMemoryError as exc:
        empty_cache(device)
        raise TrainingError(
            "训练显存不足",
            hint="降低 batch_size / 开启分块模式并调小 tile_height / 关闭 AMP",
        ) from exc
    except MGDError:
        raise
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            empty_cache(device)
            raise TrainingError("训练显存不足", hint="降低 batch_size 或调小分块尺寸") from exc
        raise TrainingError(f"训练失败：{exc}") from exc
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass

    if config.publish_best and best_weights is not None and run_dir is not None:
        try:
            publish_best(best_weights, model_best_path(arch, settings))
        except OSError as exc:
            warnings.append(f"发布 best 权重到 model_dir 失败：{exc}")
            logger.warning("发布 best 权重失败：%s", exc)

    peak_vram_gb: float | None = None
    if device.type == "cuda":
        try:
            peak_vram_gb = float(torch.cuda.max_memory_allocated(device)) / (1024 ** 3)
        except Exception as exc:
            warnings.append(f"显存峰值统计失败：{exc}")
            logger.warning("显存峰值统计失败：%s", exc)

    result = TrainingResult(
        arch=arch,
        start_epoch=start_epoch,
        epochs_run=epochs_run,
        best_metric=best_metric,
        best_epoch=best_epoch,
        eval_scope=eval_scope,
        scope_label=scope_label,
        history=history,
        best_weights=str(best_weights) if best_weights else None,
        last_checkpoint=str(last_checkpoint) if last_checkpoint else None,
        device=describe_device(device),
        train_samples=len(split.train),
        val_samples=len(split.val),
        test_samples=len(split.test),
        dual_task=dual_task,
        warnings=warnings,
        run_dir=str(run_dir) if run_dir else None,
        peak_vram_gb=round(peak_vram_gb, 3) if peak_vram_gb is not None else None,
        samples_per_second=round(float(history[-1].get("samples_per_second", 0.0)), 3) if history else 0.0,
        pretrained_status=pretrained_status,
    )
    yield {"event": "done", "result": result}


def train(config: TrainConfig, settings: Settings | None = None, on_event=None) -> TrainingResult:
    """阻塞式训练，返回最终结果。"""
    result: TrainingResult | None = None
    for event in train_iter(config, settings):
        if on_event is not None:
            on_event(event)
        if event.get("event") == "done":
            result = event["result"]
    if result is None:
        raise TrainingError("训练未产生结果")
    return result

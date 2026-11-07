"""零泄漏数据集划分引擎（Phase 2）。

- 固定 seed=42 的确定性划分；
- holdout：默认 Train 70% / Val 15% / Test 15%；
- k-fold：5 折交叉验证（fold_idx=0..4），每折 test=1 折、val=下一折、其余 train；
- 结果持久化到 `<data_dir>/dataset_splits.json`，文件列表未变时后续训练/评估
  严格复用同一划分，保证多模型对比落在同一测试集上；
- 极小样本（< 10）自动降级为安全划分，`<= 2` 时默认拒绝（空载防火墙）。
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..config import Settings, get_settings
from ..runtime.errors import DataIntegrityError
from ..runtime.logging import get_logger
from .pairing import Pair, SplitResult

logger = get_logger("data.splits")

SPLITS_FILENAME = "dataset_splits.json"
SPLITS_FORMAT_VERSION = 1
DEFAULT_SEED = 42
SMALL_DATASET_THRESHOLD = 10
MIN_DATASET_SIZE = 3


@dataclass(frozen=True)
class SplitSpec:
    """划分参数。mode=holdout 用 val_ratio/test_ratio；mode=kfold 用 n_folds/fold。"""

    mode: str = "holdout"
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = DEFAULT_SEED
    n_folds: int = 5
    fold: int = 0

    def validate(self) -> None:
        if self.mode not in ("holdout", "kfold"):
            raise DataIntegrityError(f"未知划分模式：{self.mode}", hint="可选：holdout / kfold")
        if not (0 <= float(self.val_ratio) < 1) or not (0 <= float(self.test_ratio) < 1):
            raise DataIntegrityError("val_ratio / test_ratio 必须在 [0, 1) 区间")
        if float(self.val_ratio) + float(self.test_ratio) >= 1:
            raise DataIntegrityError("val_ratio + test_ratio 必须 < 1")
        if int(self.n_folds) < 2:
            raise DataIntegrityError("n_folds 必须 ≥ 2")
        if not 0 <= int(self.fold) < int(self.n_folds):
            raise DataIntegrityError(f"fold 必须在 [0, {int(self.n_folds) - 1}]")

    @property
    def key(self) -> str:
        if self.mode == "kfold":
            return f"kfold:{int(self.fold)}:of{int(self.n_folds)}"
        return "holdout"

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "val_ratio": float(self.val_ratio),
            "test_ratio": float(self.test_ratio),
            "seed": int(self.seed),
            "n_folds": int(self.n_folds),
            "fold": int(self.fold),
        }


@dataclass
class SplitRecord:
    spec: SplitSpec
    signature: str
    train: list[str]
    val: list[str]
    test: list[str]
    warnings: list[str] = field(default_factory=list)
    source: str = "computed"

    def as_dict(self) -> dict:
        return {
            "spec": self.spec.as_dict(),
            "signature": self.signature,
            "train": self.train,
            "val": self.val,
            "test": self.test,
            "warnings": self.warnings,
            "created_at": utc_now(),
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stems_signature(stems: list[str]) -> str:
    """只用文件列表（basename）计算签名；列表未变则复用已有划分。"""
    payload = "\n".join(sorted(str(item) for item in stems))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def splits_path(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.data_dir / SPLITS_FILENAME


def _split_counts(total: int, spec: SplitSpec) -> tuple[int, int]:
    val_count = int(round(total * float(spec.val_ratio))) if spec.val_ratio else 0
    test_count = int(round(total * float(spec.test_ratio))) if spec.test_ratio else 0
    val_count = min(val_count, max(0, total - 1))
    test_count = min(test_count, max(0, total - 1 - val_count))
    return val_count, test_count


def compute_holdout(stems: list[str], spec: SplitSpec) -> SplitRecord:
    items = sorted(stems)
    rng = random.Random(int(spec.seed))
    rng.shuffle(items)
    total = len(items)
    warnings: list[str] = []

    if total < SMALL_DATASET_THRESHOLD:
        warnings.append(
            f"样本仅 {total} 例（<{SMALL_DATASET_THRESHOLD}），已降级为安全划分；指标波动大，不可作为论文结论"
        )

    val_count, test_count = _split_counts(total, spec)

    # 极小样本：保证 train/val/test 至少各有 1 例（当总数 >= 3 且请求了对应划分时）
    if 3 <= total < SMALL_DATASET_THRESHOLD:
        val_count = max(1, val_count) if spec.val_ratio else 0
        test_count = max(1, test_count) if spec.test_ratio else 0
        while total - val_count - test_count < 1:
            if test_count >= val_count and test_count > 0:
                test_count -= 1
            elif val_count > 0:
                val_count -= 1
            else:
                break

    test_start = total - test_count if test_count else total
    val_start = test_start - val_count
    train, val, test = items[:val_start], items[val_start:test_start], items[test_start:]

    if not train:
        raise DataIntegrityError("划分后训练集为空，请降低 val_ratio/test_ratio")
    if spec.val_ratio and not val:
        warnings.append("验证集为空，本轮评估口径只能是训练集自评")
    if spec.test_ratio and not test:
        warnings.append("测试集为空，无法给出独立测试集指标")

    return SplitRecord(
        spec=spec,
        signature=stems_signature(stems),
        train=train,
        val=val,
        test=test,
        warnings=warnings,
    )


def compute_kfold(stems: list[str], spec: SplitSpec) -> SplitRecord:
    items = sorted(stems)
    rng = random.Random(int(spec.seed))
    rng.shuffle(items)
    total = len(items)
    n_folds = int(spec.n_folds)
    if total < n_folds:
        raise DataIntegrityError(
            f"样本 {total} 例不足以做 {n_folds} 折交叉验证",
            hint="请增加数据或改用 holdout 划分",
        )

    warnings: list[str] = []
    if total < SMALL_DATASET_THRESHOLD:
        warnings.append(
            f"样本仅 {total} 例（<{SMALL_DATASET_THRESHOLD}），K 折结果仅用于调试，不可作为论文结论"
        )

    folds: list[list[str]] = []
    base, extra = divmod(total, n_folds)
    cursor = 0
    for index in range(n_folds):
        size = base + (1 if index < extra else 0)
        folds.append(items[cursor:cursor + size])
        cursor += size

    fold = int(spec.fold)
    test = list(folds[fold])
    val_fold = (fold + 1) % n_folds
    val = list(folds[val_fold]) if n_folds > 2 else []
    train = [stem for i, group in enumerate(folds) if i not in (fold, val_fold) for stem in group]

    if not train:
        raise DataIntegrityError("K 折划分后训练集为空，请减少折数")
    return SplitRecord(
        spec=spec,
        signature=stems_signature(stems),
        train=train,
        val=val,
        test=test,
        warnings=warnings,
    )


def compute_split(stems: list[str], spec: SplitSpec) -> SplitRecord:
    spec.validate()
    if not stems:
        raise DataIntegrityError("没有可划分的样本")
    if len(stems) < MIN_DATASET_SIZE:
        raise DataIntegrityError(
            f"数据集仅 {len(stems)} 例（<= {MIN_DATASET_SIZE - 1}），已触发空载防火墙",
            hint="请至少准备 3 例影像/腺体/眼睑三元组，避免产出无意义的指标",
        )
    return compute_kfold(stems, spec) if spec.mode == "kfold" else compute_holdout(stems, spec)


def _read_store(path: Path) -> dict:
    if not path.is_file():
        return {"format_version": SPLITS_FORMAT_VERSION, "entries": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("dataset_splits.json 损坏，将重新生成：%s", exc)
        return {"format_version": SPLITS_FORMAT_VERSION, "entries": {}}
    if not isinstance(payload, dict) or not isinstance(payload.get("entries", {}), dict):
        return {"format_version": SPLITS_FORMAT_VERSION, "entries": {}}
    return payload


def _write_store(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_or_create_splits(
    pairs: list[Pair],
    spec: SplitSpec | None = None,
    settings: Settings | None = None,
) -> SplitRecord:
    """读取或生成持久化划分。文件列表不变时严格复用。"""
    spec = spec or SplitSpec()
    spec.validate()
    settings = settings or get_settings()
    stems = [pair.stem for pair in pairs]
    signature = stems_signature(stems)
    path = splits_path(settings)
    store = _read_store(path)

    stored_signature = store.get("signature")
    if stored_signature != signature:
        if stored_signature is not None:
            logger.warning("数据集文件列表已变化，重新生成划分：%s", path)
        store = {
            "format_version": SPLITS_FORMAT_VERSION,
            "signature": signature,
            "dataset_size": len(stems),
            "created_at": utc_now(),
            "entries": {},
        }

    entry = store.get("entries", {}).get(spec.key)
    record: SplitRecord | None = None
    if isinstance(entry, dict):
        try:
            train = [str(item) for item in entry["train"]]
            val = [str(item) for item in entry["val"]]
            test = [str(item) for item in entry["test"]]
        except (KeyError, TypeError):
            train = val = test = []
        known = set(stems)
        if train and set(train) | set(val) | set(test) == known and (
            len(train) + len(val) + len(test) == len(known)
        ):
            record = SplitRecord(
                spec=spec, signature=signature, train=train, val=val, test=test,
                warnings=[str(w) for w in entry.get("warnings", [])], source="persisted",
            )

    if record is None:
        record = compute_split(stems, spec)
        store.setdefault("entries", {})[spec.key] = record.as_dict()
        store.setdefault("format_version", SPLITS_FORMAT_VERSION)
        _write_store(path, store)

    # 若持久化文件里没有该 key，但已存在同签名文件，补写；已存在则不重复写
    elif spec.key not in store.get("entries", {}):
        store["entries"][spec.key] = record.as_dict()
        _write_store(path, store)

    return record


def to_pairs(record: SplitRecord, pairs: list[Pair]) -> SplitResult:
    lookup = {pair.stem: pair for pair in pairs}
    missing = [stem for stem in record.train + record.val + record.test if stem not in lookup]
    if missing:
        raise DataIntegrityError(f"划分记录中的样本在当前数据集中缺失：{missing[:5]}")
    return SplitResult(
        train=[lookup[stem] for stem in record.train],
        val=[lookup[stem] for stem in record.val],
        test=[lookup[stem] for stem in record.test],
        warnings=list(record.warnings),
    )


def get_splits(
    pairs: list[Pair],
    spec: SplitSpec | None = None,
    settings: Settings | None = None,
) -> SplitResult:
    """对外主入口：返回 Pair 级别的 train/val/test（含持久化复用）。"""
    return to_pairs(load_or_create_splits(pairs, spec, settings), pairs)


def resolve_spec(split_mode: str = "holdout", *, val_ratio: float = 0.15,
                 test_ratio: float = 0.15, seed: int = DEFAULT_SEED,
                 n_folds: int = 5, fold: int = 0) -> SplitSpec:
    return SplitSpec(
        mode=str(split_mode or "holdout").lower(),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
        seed=int(seed),
        n_folds=int(n_folds),
        fold=int(fold),
    )

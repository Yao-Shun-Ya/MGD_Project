"""训练参数的唯一 schema（YAML / CLI / WebUI 三层共用）。

规则：
- 任何参数只在这里定义一次；YAML 键、CLI `--set` 键、UI 控件都从这里派生，
  因此不会出现「UI 能设、CLI 不认」的分叉；
- 这里只登记**已经实现**的参数；没有落到引擎里的参数不允许出现在 schema；
- 所有数值在入口处统一走 `coerce_value`，类型/范围错误在进入训练前就被拒绝。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..runtime.errors import ConfigError

#: 分节顺序同时决定 UI 中「高级参数」的展示顺序
SECTIONS: tuple[str, ...] = ("run", "data", "model", "train", "eval", "artifacts")

TTA_OPTIONS: tuple[str, ...] = ("关闭", "2倍增强 (水平翻转)", "8倍增强 (多尺度+翻转)")
OPTIMIZER_OPTIONS: tuple[str, ...] = ("AdamW", "Adam", "SGD")
SCHEDULER_OPTIONS: tuple[str, ...] = (
    "ReduceLROnPlateau (默认)",
    "Cosine Annealing",
    "Step Decay",
    "CosineAnnealingWarmRestarts",
)
VRAM_TIERS: tuple[str, ...] = ("auto", "6", "8", "12", "16", "24", "40")
PRETRAINED_OPTIONS: tuple[str, ...] = ("auto", "none", "ade20k", "imagenet")


@dataclass(frozen=True)
class ParamSpec:
    section: str
    key: str
    field: str
    kind: str                      # bool | int | float | str | enum
    default: Any
    label: str
    help: str
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    advanced: bool = False
    #: 写入 YAML 的注释（train_default.yaml 由本表生成）
    unit: str = ""

    @property
    def name(self) -> str:
        return f"{self.section}.{self.key}"


SPECS: tuple[ParamSpec, ...] = (
    # ---------------- run ----------------
    ParamSpec("run", "name", "run_name", "str", None, "run 名称前缀",
              "留空则自动生成「时间戳_架构_代码版本」；目录名中的非法字符会被替换", advanced=True),
    ParamSpec("run", "output_dir", "_output_dir", "str", None, "输出根目录",
              "留空 → $MGD_OUTPUT_DIR/runs；每个 run 一个独立子目录", advanced=True),
    ParamSpec("run", "seed", "seed", "int", 42, "随机种子", "划分、数据增强、训练全部使用该种子", minimum=0),
    ParamSpec("run", "deterministic", "deterministic", "bool", True, "确定性模式",
              "固定 cudnn / 关闭 TF32 的非确定性算子；跨机器完全复现仍需相同驱动与 GPU", advanced=True),
    ParamSpec("run", "amp", "amp", "bool", True, "AMP 混合精度",
              "仅 CUDA 生效；CPU 会自动关闭并在日志中说明"),
    ParamSpec("run", "device", "device", "str", None, "设备",
              "留空 → 自动 cuda > cpu；也可写 cuda / cpu / cuda:0", advanced=True),
    ParamSpec("run", "num_workers", "num_workers", "int", 0, "DataLoader 线程数",
              "Windows 建议 0；Linux 可设 4–8", minimum=0, maximum=32, advanced=True),
    ParamSpec("run", "mem_limit_gb", "mem_limit_gb", "float", None, "显存上限 (GB)",
              "给 CUDA 进程设置 per-process 上限，调试 OOM 时使用；留空不限制", minimum=0.5, advanced=True),
    ParamSpec("run", "vram_tier", "vram_tier", "enum", "auto", "显存档位",
              "auto = 启动时探测显存并推荐；也可显式指定 6/8/12/16/24/40", choices=VRAM_TIERS),
    ParamSpec("run", "grad_accum_steps", "grad_accum_steps", "int", 1, "梯度累积步数",
              "等效放大 batch；显存不足时优先用它而不是减小分块", minimum=1, maximum=64,
              advanced=True),
    ParamSpec("run", "channels_last", "channels_last", "bool", True, "channels_last 内存格式",
              "卷积在 NHWC 下通常更快，显存占用略降；个别算子可能回退", advanced=True),

    # ---------------- data ----------------
    ParamSpec("data", "split_mode", "split_mode", "enum", "holdout", "划分模式",
              "holdout = 持久化 70/15/15；kfold = 5 折交叉验证", choices=("holdout", "kfold")),
    ParamSpec("data", "val_ratio", "val_ratio", "float", 0.15, "验证集比例",
              "holdout 生效；与测试集比例之和必须 < 1", minimum=0.0, maximum=0.9),
    ParamSpec("data", "test_ratio", "test_ratio", "float", 0.15, "测试集比例",
              "holdout 生效；持久化在 dataset_splits.json", minimum=0.0, maximum=0.9),
    ParamSpec("data", "n_folds", "n_folds", "int", 5, "折数", "kfold 生效", minimum=2, maximum=10,
              advanced=True),
    ParamSpec("data", "fold", "fold", "int", 0, "当前折", "kfold 生效，范围 0..n_folds-1", minimum=0,
              advanced=True),
    ParamSpec("data", "tile_mode", "tile_mode", "bool", True, "分块训练",
              "640×1280 整图在 8 GB 卡上会 OOM；默认分块"),
    ParamSpec("data", "tile_height", "tile_height", "int", 384, "分块高度",
              "宽度固定为高度的 2 倍（保持 2:1 宽高比）；384 → 384×768", minimum=64, maximum=640),
    ParamSpec("data", "augmentation", "augmentation", "bool", True, "数据增强",
              "水平翻转 + 亮度/对比度扰动；不启用垂直翻转（腺体走向不对称）"),

    # ---------------- model ----------------
    ParamSpec("model", "name", "model_name", "str", "MGD-Former", "模型",
              "MGD-Former 为默认；UNet/AttentionUNet/WRCANet/MGD-PriorNet 为基线对照"),
    ParamSpec("model", "pretrained", "pretrained", "enum", "auto", "编码器预训练",
              "auto = 本地有缓存则加载 ADE20K 编码器，否则随机初始化并记录降级；none = 强制不加载",
              choices=PRETRAINED_OPTIONS, advanced=True),
    ParamSpec("model", "dual_task", "dual_task", "bool", True, "双任务输出",
              "腺体 + 眼睑两个通道；眼睑通道同时提供 R_loss 分母与 ROI"),
    ParamSpec("model", "eyelid_weight", "eyelid_weight", "float", 0.25, "眼睑辅助权重 λ1",
              "只影响训练损失，不影响推理输出通道数", minimum=0.0, maximum=1.0),
    ParamSpec("model", "pos_weight", "pos_weight", "float", 1.0, "正样本权重",
              "仅 BCE 类损失使用；腺体占比低时可上调", minimum=0.1, maximum=20.0, advanced=True),
    ParamSpec("model", "deep_supervision", "deep_supervision", "bool", True, "深监督",
              "MGD-Former 在 1/4、1/8、1/16 三个尺度各接辅助头（仅腺体）；旧模型自动忽略"),
    ParamSpec("model", "cldice_weight", "cldice_weight", "float", 0.2, "clDice 权重",
              "骨架连续性约束；0 = 关闭", minimum=0.0, maximum=1.0, advanced=True),
    ParamSpec("model", "cldice_warmup", "cldice_warmup", "float", 0.1, "clDice warmup 比例",
              "前 x% 轮内由 0 线性升到目标权重，避免早期骨架噪声", minimum=0.0, maximum=0.5,
              advanced=True),
    ParamSpec("model", "boundary_weight", "boundary_weight", "float", 0.0, "边界损失权重",
              "默认关闭；对 HD95 敏感的场景可设 0.1", minimum=0.0, maximum=0.5, advanced=True),
    ParamSpec("model", "ema_decay", "ema_decay", "float", 0.999, "EMA 衰减",
              "1.0 = 关闭；评估与保存使用 EMA 权重", minimum=0.9, maximum=1.0, advanced=True),

    # ---------------- train ----------------
    ParamSpec("train", "epochs", "epochs", "int", 20, "总轮数", "冒烟请用 1–2", minimum=1, maximum=1000),
    ParamSpec("train", "batch_size", "batch_size", "int", 1, "batch_size",
              "分块模式下 8 GB 卡建议 1–2", minimum=1, maximum=64),
    ParamSpec("train", "optimizer", "optimizer", "enum", "AdamW", "优化器", "", choices=OPTIMIZER_OPTIONS),
    ParamSpec("train", "lr", "lr", "float", 1e-4, "学习率",
              "预训练编码器建议 1e-4；从零训练基线可用 1e-3", minimum=1e-7, maximum=1.0),
    ParamSpec("train", "weight_decay", "weight_decay", "float", 1e-4, "权重衰减", "", minimum=0.0,
              maximum=1.0, advanced=True),
    ParamSpec("train", "scheduler", "scheduler", "enum", "Cosine Annealing", "学习率调度",
              "Cosine 为长训练默认；ReduceLROnPlateau 适合短程调试", choices=SCHEDULER_OPTIONS,
              advanced=True),
    ParamSpec("train", "loss", "loss_name", "str", "MGD 复合损失", "损失函数",
              "默认 0.4 Dice + 0.4 CE + clDice + 眼睑辅助；其余旧损失保留可选"),
    ParamSpec("train", "save_every", "save_every", "int", 1, "检查点间隔",
              "每 N 轮保存一次完整状态", minimum=1, maximum=100, advanced=True),
    ParamSpec("train", "keep_checkpoints", "keep_checkpoints", "int", 3, "保留检查点数",
              "滚动删除更早的完整检查点，best 权重不受影响", minimum=1, maximum=50, advanced=True),
    ParamSpec("train", "resume_from", "resume_from", "str", None, "断点续训",
              "run 目录 / checkpoints 目录 / 具体 .mgd.ckpt；目录自动取最新", advanced=True),

    # ---------------- eval ----------------
    ParamSpec("eval", "tta", "eval_tta", "enum", "关闭", "TTA",
              "训练期监控始终关闭 TTA；论文指标建议 8 倍增强", choices=TTA_OPTIONS),
    ParamSpec("eval", "threshold", "eval_threshold", "float", 0.7, "分割阈值",
              "实测 0.7 在 384×768 分块下最优（见 KNOWN_LIMITATIONS）", minimum=0.05, maximum=0.95),
    ParamSpec("eval", "tile_size", "eval_tile_size", "int", 384, "推理分块尺寸",
              "384 为实测最优；整图推理会显著掉 Dice", minimum=64, maximum=640),
    ParamSpec("eval", "tile_overlap", "eval_tile_overlap", "float", 0.25, "推理分块重叠",
              "", minimum=0.0, maximum=0.9),

    # ---------------- artifacts ----------------
    ParamSpec("artifacts", "publish_best", "publish_best", "bool", False, "发布 best 权重",
              "true → 额外复制到 $MGD_MODEL_DIR/<arch>/；会覆盖同名文件，默认关闭", advanced=True),
    ParamSpec("artifacts", "export_reference_report", "export_reference_report", "bool", True,
              "生成 reference_report.json",
              "训练结束后在 run 目录生成带 split_signature 的验证口径报告"),
)

SPEC_BY_NAME: dict[str, ParamSpec] = {spec.name: spec for spec in SPECS}
SPEC_BY_FIELD: dict[str, ParamSpec] = {spec.field: spec for spec in SPECS}
_BY_SECTION: dict[str, dict[str, ParamSpec]] = {section: {} for section in SECTIONS}
for _spec in SPECS:
    _BY_SECTION[_spec.section][_spec.key] = _spec

#: 兼容旧接口：分节 → {键: TrainConfig 字段}
SCHEMA: dict[str, dict[str, str]] = {
    section: {spec.key: spec.field for spec in SPECS if spec.section == section}
    for section in SECTIONS
}


def specs_for_section(section: str) -> tuple[ParamSpec, ...]:
    if section not in _BY_SECTION:
        raise ConfigError(f"未知配置分节：{section}", hint=f"允许的分节：{', '.join(SECTIONS)}")
    return tuple(_BY_SECTION[section].values())


def basic_specs() -> tuple[ParamSpec, ...]:
    return tuple(spec for spec in SPECS if not spec.advanced)


def advanced_specs() -> tuple[ParamSpec, ...]:
    return tuple(spec for spec in SPECS if spec.advanced)


def coerce_value(spec: ParamSpec, value: Any, *, where: str | None = None) -> Any:
    """按 schema 强制类型并校验范围；错误信息直接指出参数全名。"""
    place = where or spec.name
    if value is None:
        return None
    try:
        if spec.kind == "bool":
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on", "是")
            return bool(value)
        if spec.kind == "int":
            result: Any = int(value)
        elif spec.kind == "float":
            result = float(value)
        else:
            result = str(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{place} 的值 {value!r} 无法转换为 {spec.kind}") from exc

    if spec.kind == "enum" and spec.choices and result not in spec.choices:
        raise ConfigError(
            f"{place} 的取值 {result!r} 不在允许列表",
            hint=f"可选：{', '.join(spec.choices)}",
        )
    if isinstance(result, (int, float)) and spec.kind in ("int", "float"):
        if spec.minimum is not None and result < spec.minimum:
            raise ConfigError(f"{place} 必须 ≥ {spec.minimum}（当前 {result}）")
        if spec.maximum is not None and result > spec.maximum:
            raise ConfigError(f"{place} 必须 ≤ {spec.maximum}（当前 {result}）")
    return result


def parse_set_expression(expression: str) -> tuple[str, str, str]:
    """解析 `a.b=c`，返回 (分节, 键, 原始字符串值)。"""
    text = str(expression or "").strip()
    if "=" not in text:
        raise ConfigError(f"--set 需要 a.b=c 形式（收到 {text!r}）")
    left, right = text.split("=", 1)
    if "." not in left:
        raise ConfigError(f"--set 的键必须带分节，如 model.name=MGD-Former（收到 {left!r}）")
    section, key = left.strip().split(".", 1)
    if section not in SCHEMA or key not in SCHEMA[section]:
        raise ConfigError(
            f"--set 未知参数 {left.strip()}",
            hint="可用参数见 mgd/core/training/specs.py 或 `mgd-train --print-schema`",
        )
    return section, key, right.strip()


def apply_set_expression(payload: dict, expression: str) -> dict:
    section, key, raw = parse_set_expression(expression)
    spec = _BY_SECTION[section][key]
    payload.setdefault(section, {})[key] = coerce_value(spec, raw, where=spec.name)
    return payload


def validate_payload(payload: dict) -> None:
    """校验整份配置：未知分节/键、类型、范围。"""
    unknown_sections = [key for key in payload if key not in SCHEMA]
    if unknown_sections:
        raise ConfigError(
            f"未知配置分节：{', '.join(sorted(unknown_sections))}",
            hint=f"允许的分节：{', '.join(SECTIONS)}",
        )
    for section, body in payload.items():
        if body is None:
            continue
        if not isinstance(body, dict):
            raise ConfigError(f"分节 {section} 必须是映射")
        unknown = [key for key in body if key not in SCHEMA[section]]
        if unknown:
            raise ConfigError(
                f"分节 {section} 含未知键：{', '.join(sorted(unknown))}",
                hint=f"允许的键：{', '.join(sorted(SCHEMA[section]))}",
            )
        for key, value in body.items():
            if key == "output_dir":
                continue
            coerce_value(_BY_SECTION[section][key], value)

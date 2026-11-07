# 换机 Runbook：从零到跑通重训

> 目标：不看代码就能在一台新机器上完成「安装 → 数据放置 → 自检 → 划分 → 训练 → 评估 → 导出」。
> 所有命令都使用 `mgd-*` 入口；`pip install -e .` 之后无需设置 PYTHONPATH。

## 0. 前置

- Python 3.10 或 3.11（64 位）
- 训练：NVIDIA GPU + 驱动（CUDA 12.1 对应 torch 2.5.1）
- 纯推理/调试：CPU 也可，但训练极慢
- 磁盘：数据 + 权重 + run 产物至少预留 20 GB

## 1. 安装（干净环境）

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux
source .venv/bin/activate

# CUDA 12.1
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# 或者 CPU
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

pip install -e .
pip install -e ".[ui]"      # 需要 WebUI 时
```

离线安装：在联网机器 `pip download -r requirements.txt -d wheels/`，目标机
`pip install --no-index --find-links wheels/ ...`。torch 需按平台单独下载 official wheel。

## 2. 数据放置

```
MGD_Project/
├── data/
│   ├── images/        640×1280 红外影像（.png/.jpg/.bmp/.tif）
│   ├── gland_masks/   同名腺体掩膜
│   └── eyelid_masks/  同名眼睑掩膜
├── models_weights/    预训练/历史权重
└── outputs/           run 产物（可改 MGD_OUTPUT_DIR）
```

路径可用环境变量覆盖：`MGD_HOME` / `MGD_DATA_DIR` / `MGD_MODEL_DIR` / `MGD_OUTPUT_DIR` / `MGD_DEVICE`。
所有路径都来自 `Settings`，不依赖当前工作目录。

## 3. 自检

```bash
mgd-doctor            # 人类可读报告，失败项返回非零
mgd-doctor --json     # CI 用
```

检查项：Python、torch/设备、依赖、数据目录、权重 manifest、磁盘余量、写权限。

## 4. 划分

```bash
mgd-data splits       # 打印 train/val/test 计数与 split_signature
mgd-data check        # 完整性：缺图/缺掩膜/孤儿/重名/命名异常
```

持久化划分写在 `data/dataset_splits.json`（seed=42）。**论文报告必须带该 signature。**

## 5. 训练

```bash
# 快速冒烟（1–2 epoch，只验证可训练性，不出精度结论）
mgd-train --profile quick --vram auto

# 学术档（5 折、Cosine、8 倍 TTA、自动 reference_report）
mgd-train --profile academic --vram auto --set data.fold=0

# 低显存档
mgd-train --profile lowvram --vram 6

# 任意参数覆盖（与 UI 完全同一份 schema）
mgd-train --set model.name=UNet --set train.epochs=10 --set run.seed=2026

# 断点续训
mgd-train --resume outputs/runs/<run_id>
```

查看全部参数：`mgd-train --print-schema`；查看显存档：`mgd-train --print-tiers`。

每个 run 目录结构见 `FREEZE_BASELINE.md §5`。

## 6. 评估

```bash
mgd-eval --model MGD-Former --scope test --tta "8倍增强 (多尺度+翻转)"
mgd-eval --scope val --out outputs/eval_manual
```

产物：`reference_report.json`（带 split_signature、bootstrap CI）、`per_sample_*.csv/.tex`。
训练集自评会标注 `scope=train_self`，不能进论文表格。

## 7. 流水线

```bash
mgd-pipeline --stages train,eval,report --folds 0-4 --profile academic --vram auto
mgd-pipeline --stages train --configs a.yaml,b.yaml --retries 2
```

阶段：`pretrain`（离线不可用，会标 skipped）/`train`/`finetune`/`eval`/`ablation`/`report`。
状态写入 `outputs/pipeline_state.json`；中断后重跑会自动跳过已完成条目。

## 8. 产物导出

- 单 run：`<run>/reference_report.json`、`metrics.jsonl`、`checkpoints/`
- 多 run 对比：WebUI「实验管理」→ 导出 CSV / LaTeX（也可用 `mgd.core.training.experiments`）
- 临床报告：WebUI「诊断」→ 生成 HTML/CSV（PDF 需安装 weasyprint）

## 9. 显存与时间估算

| 显存 | 分块 | batch | 状态 | 说明 |
|---|---|---|---|---|
| 6 GB | 256×512 | 1 | 实测 | 峰值 0.89 GB、3.54 样本/秒（本机 RTX 4060） |
| 8 GB | 384×768 | 1 | 实测 | 峰值 1.23 GB、3.10 样本/秒、225.6 s/epoch |
| 16 GB | 512×1024 | 2 | 估算 | 约 9–11 GB，长训练推荐 |
| 24 GB | 640×1280 | 2 | 估算 | 整图训练，约 14–18 GB |
| 32 GB | 640×1280 | 3 | 估算 | 整图训练，约 21–26 GB |

时间估算依赖数据量与设备，**本基线不提供虚假的绝对时长**；请以冒烟输出的
`samples_per_second` 自行外推：`每轮秒数 ≈ 训练样本数 / 吞吐`。

## 10. CUDA / CPU 两条路径

| 能力 | CUDA | CPU |
|---|---|---|
| 训练 | 推荐 | 仅冒烟 |
| AMP | 支持 | 自动关闭 |
| channels_last | 有效 | 无害 |
| 分块推理 | 支持 | 支持 |
| 线程数建议 | 0–4 | 0 |

> macOS/MPS 自 v2.3.0 起不再作为交付目标；`device.resolve_device()` 只接受 `cuda` 与 `cpu`。

## 11. 同步到运行目录 / 容器 / CI

```bash
python tools/sync_runtime.py --plan     # 新增/覆盖/保留清单
python tools/sync_runtime.py --apply    # 备份覆盖项后写入，并逐文件哈希核对
```

```bash
docker build -f docker/Dockerfile.cpu -t mgd-seg:cpu .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" \
  -v "$PWD/outputs:/app/outputs" -p 7860:7860 mgd-seg:cpu
```

CI：`.github/workflows/ci.yml` 的 `python-quality` job 执行 compileall + pytest + doctor（无 GPU）。

## 12. 常见故障

| 现象 | 原因 | 处理 |
|---|---|---|
| `CUDA out of memory` | 分块/batch 过大 | 降档位（`--vram 6`）或加大梯度累积 |
| doctor 数据目录为空 | 路径不对 | 设置 `MGD_DATA_DIR` 或放入 `data/` |
| 权重 `verified=false` | manifest 未转正 | 用 `mgd-verify` 在独立测试集评估后转正 |
| 评估 Dice 很低 | 分块过小/阈值不对 | 用 384×768、阈值 0.7 |
| UI 打不开 | 端口占用 | `mgd-ui --port 7861` |
| 训练中断 | 断电/抢占 | `mgd-train --resume <run目录>` |
| 中文乱码（Linux） | 缺中文字体 | 安装 Noto Sans CJK |

# 工程冻结基线 v2.3.0

> 冻结日期：2026-11-07　｜　包名 `mgd-seg`　｜　版本 `2.3.0`
> 论文、文档与后续二次开发均以本文件为准。任何偏离都需要在这里显式修订并升版本号。

本轮定位：在既定架构决议（`docs/engineering/ARCHITECTURE_DECISION.md`）之上完成
**可用性、自由度与工程完备性**。不重新选型；MGD-Former 为默认主干，旧四族为基线对照。

---

## 1. 模块划分

```
mgd/
├── core/                       不含 UI 依赖的内核
│   ├── config.py               Settings（路径/设备/manifest/凭据）、模型发现
│   ├── runtime/                device / errors / logging / doctor
│   ├── data/                   pairing / splits / dataset / transforms / versioning
│   ├── models/                 mgd_former / architectures / mgd_prior_net /
│   │                           registry / loader / manifest / pretrained / verify
│   ├── training/               specs / config_file / presets / engine / runner /
│   │                           run / checkpoint / determinism / ema / losses /
│   │                           experiments
│   ├── inference/              service（分块/TTA/阈值/置信度）、manifest 解析
│   ├── evaluation/             evaluator（Dice/IoU/HD95/ASSD/clDice）、stats（bootstrap/Wilcoxon）
│   ├── imaging/                preprocess / metrics / visualize（含 True Grad-CAM）
│   ├── reporting/              HTML/CSV 报告
│   └── workflow/               pipeline（阶段编排/批量/重试/断点续跑）
├── adapters/
│   ├── cli/                    train_cli / eval_cli / pipeline_cli / data_cli
│   └── gradio_app/app.py       WebUI（六分区 / 深色金属）
├── configs/                    train_default.yaml / inference_default.yaml
└── tests/                      契约与回归测试
```

前端（Next.js 的 `app/`、`components/`、`lib/`、`styles/`、`public/`（Paper 除外）、
`package.json`）不在本基线内，本轮未改动。

## 2. 公共接口

### 2.1 命令（`[project.scripts]`）

| 命令 | 入口 | 用途 |
|---|---|---|
| `mgd-doctor` | `mgd.core.runtime.doctor:main` | 换机第一件事：环境自检，失败返回非零 |
| `mgd-ui` | `mgd.adapters.gradio_app.app:main` | WebUI（默认 127.0.0.1:7860） |
| `mgd-train` | `mgd.adapters.cli.train_cli:main` | YAML + 预设 + `--set` + 显存档 |
| `mgd-pipeline` | `mgd.adapters.cli.pipeline_cli:main` | 学术流水线（阶段/多配置/多折/重试/续跑） |
| `mgd-eval` | `mgd.adapters.cli.eval_cli:main` | 独立测试集评估、CI、CSV/LaTeX |
| `mgd-data` | `mgd.adapters.cli.data_cli:main` | `check` / `version` / `backup` / `splits` |
| `mgd-verify` | `mgd.core.models.verify:main` | 权重验证与转正（manifest `verified`） |

### 2.2 核心 Python API（稳定契约）

```python
from mgd.core.training.config_file import build_train_config
from mgd.core.training.presets import apply_tier
from mgd.core.training.runner import run_training
from mgd.core.models.registry import create_model, unwrap_outputs
from mgd.core.evaluation.evaluator import evaluate_pairs
from mgd.core.evaluation.stats import reference_report_payload
from mgd.core.data.versioning import integrity_report, record_version, backup_dataset
from mgd.core.workflow.pipeline import run_pipeline
```

- `create_model(name, in_channels=1, out_channels=1, deep_supervision=True)`
- 输出契约：旧模型返回 `Tensor`；MGD-Former 返回
  `{"logits": Tensor, "aux": [Tensor, Tensor, Tensor]}`，统一用 `unwrap_outputs()` 解包。
- `run_training(config, meta, settings, on_event) -> (TrainingResult, RunContext)`；
  任何入口（CLI/UI/流水线）都必须经过它，保证产物一致。

## 3. 参数 schema（唯一来源）

`mgd/core/training/specs.py` 是唯一来源；YAML 键、CLI `--set` 键、WebUI 控件都由它派生，
未知键直接报错，**不存在「UI 能设、CLI 不认」**。

分节：`run` / `data` / `model` / `train` / `eval` / `artifacts`

三层入口与优先级：

```
默认 YAML  <  用户 YAML  <  命令行预设  <  --set / 快捷参数  <  显存档（安全边界）
```

- 底层：YAML（`mgd/configs/train_default.yaml`）+ `mgd-train --set a.b=c`
- 中层：`--profile academic|quick|lowvram`
- 上层：WebUI 表单（基础参数默认可用，高级参数折叠可改）

预设定义见 `mgd/core/training/presets.py::TRAIN_PROFILES`。
参数全表可在终端查看：`mgd-train --print-schema`。

## 4. 显存档位表

| 档位 | 分块 | batch | 梯度累积 | AMP | channels_last | 深监督 | TTA | 状态 | 依据 |
|---|---|---|---|---|---|---|---|---|---|
| 6 GB | 256×512 | 1 | 4 | 是 | 是 | 否 | 关闭 | **实测** | 256×512 冒烟：峰值 0.89 GB、3.54 样本/秒 |
| 8 GB | 384×768 | 1 | 4 | 是 | 是 | 是 | 关闭 | **实测** | RTX 4060 8 GB 冒烟（数值见 §9） |
| 12 GB | 384×768 | 2 | 2 | 是 | 是 | 是 | 关闭 | 估算 | 8 GB 档 batch 翻倍外推 |
| 16 GB | 512×1024 | 2 | 1 | 是 | 是 | 是 | 关闭 | 估算 | 分块 + batch 外推 |
| 24 GB | 640×1280 | 2 | 1 | 是 | 是 | 是 | 8 倍 | 估算 | 整图 + batch 外推 |
| 32 GB | 640×1280 | 3 | 1 | 是 | 是 | 是 | 8 倍 | 估算 | 整图 + batch 3 外推 |
| 40 GB+ | 640×1280 | 4 | 1 | 是 | 是 | 是 | 8 倍 | 估算 | 整图 + batch 4 外推 |

规则：
- `--vram auto` 启动时探测显存，取「标称容量 ≤ 探测显存」的最大档位；首轮若 OOM 由用户显式降一档；
- 探测失败（CPU）→ 6 GB 安全档；
- 每档在 `TIERS[...].status/basis/peak_gb/samples_per_second` 中标注实测或估算及依据；
- 档位是安全边界：它会覆盖分块/batch/AMP 等字段，避免「能跑但必 OOM」的组合。

## 5. run 目录约定

```
<output_dir>/runs/<run_id>/
  config.yaml            配置快照（resolved_yaml + train_config）
  env.json               代码版本 / git commit / 依赖版本 / 设备 / 路径
  train.log              日志
  metrics.jsonl          每轮一行（loss/dice/val_dice/lr/吞吐）
  checkpoints/           <arch>_epoch_N.mgd.ckpt（完整状态）、<arch>_best.pth（EMA 权重）
  reference_report.json  带 split_signature 的验证口径报告
  run_result.json        结果摘要（含 peak_vram_gb / samples_per_second / wall_seconds）
```

`run_id = 时间戳_名称_代码版本_配置指纹前 6 位`；重名自动追加序号，绝不覆盖已有 run。

断点续训保存：模型、优化器、调度器、AMP scaler、epoch、全局步数、数据规模、
rng state、EMA shadow、数据签名；`--resume <run目录|checkpoints|.ckpt>` 自动取最新。

## 6. 评测协议

1. 划分：**患者级 holdout**（694/153/152；239/50/49 名患者；seed=42；持久化 `data/dataset_splits.json`）；5 折作为扩展，复用同一患者分组；
2. 指标：Dice、IoU、HD95、ASSD、Precision、Recall、clDice、眼睑 Dice、R_loss、Meiboscore；
3. 统计：bootstrap 1000 次 95% CI；多模型对比用配对 Wilcoxon + Holm；
4. **任何报告必须同时带文件清单签名与实际派分签名**，缺失即无效；
5. 训练集自评（`scope=train_self`）会显式标注，不能进入论文表格；
6. 推理默认：分块 384×768、overlap 0.25、阈值 0.7；TTA 训练期关闭、论文评估建议 8 倍；MTE 为 256/384/512 概率平均；
7. 泄漏审计：官方 200 例 Meiboscore 测试清单含 166 名患者，其余 800 例中 326 例来自同一批患者；clean-official 子集为 60 图/50 患者；
8. 不确定性：多分块方差、预测熵与 8 倍 TTA 误差对失败病例（Dice<0.7）的 AUROC；
9. 权重状态：`verified=false` 的权重只能 `allow_unverified` 加载，界面/日志必须标注原因。

## 7. 数据集管理

- 版本记录：`data/dataset_version.json`（计数、哈希清单、划分签名、来源/采集设备）；
- 写前校验：完整性不通过则拒绝写版本记录；
- 备份：`backups/dataset_backup_<时间戳>/`，只复制、不移动、不覆盖；
- 完整性命令：`mgd-data check` —— 缺图、缺腺体/眼睑掩膜、孤儿掩膜、重名、命名异常；
- **绝不覆盖** `MGD_Project` 中用户已有的影像、掩膜与权重。

## 8. 版本与依赖矩阵

| 项 | 版本 / 约束（下限 = 实测已安装版本） |
|---|---|
| Python | ≥ 3.10（实测 3.10.6） |
| torch | 2.5.1+cu121（实测；CPU wheel 同为 2.5.1） |
| torchvision（可选） | 0.20.1+cu121（实测） |
| numpy | 1.26.4 |
| opencv-python | 4.8.0.74（实测） |
| scikit-image | 0.25.2 |
| monai | 1.5.2 |
| scipy | 1.15.3 |
| pandas | 2.3.3 |
| matplotlib | 3.10.8 |
| Pillow | 12.2.0（实测） |
| PyYAML | 6.0.3（实测） |
| pytest（dev） | 9.2.0（实测） |
| gradio（可选） | 6.16.0 |
| pydicom / highdicom（可选） | 3.0.2 / 0.28.2（实测，DICOM 往返已验证） |
| wandb（可选） | 0.27.0（实测） |
| weasyprint（可选） | 未安装、未实测（PDF 报告路径） |

平台：
- Windows：CUDA 或 CPU；transforms 已可 pickle，`num_workers=0–8` 均可用（默认仍 0）；交付形态为 Gradio WebUI + CLI；
- Linux：CUDA 或 CPU；`num_workers=4–8`；交付形态为训练/评估 CLI（不安装 gradio）；
- macOS/MPS：v2.3.0 起不再作为交付目标，相关脚本与分支已删除。

## 9. 验收快照（本机实测，2026-11-07）

运行环境：Windows 11、RTX 4060 Laptop 8 GB、MGD_Project venv（Python 3.11 + torch 2.5.1 CUDA 12.1）。

| 项目 | 命令 | 结果 |
|---|---|---|
| 单测 | `python -m pytest mgd/tests -q` | **146 passed**（含 TTA 视图数、流水线状态容错、档位实测回填回归） |
| 编译 | `python -m compileall -q mgd` | 通过 |
| doctor | `mgd-doctor --data-dir data` | 通过（Python/torch/CUDA/依赖/数据/权重/磁盘/写权限） |
| 数据集 | `mgd-data check` | 999 配对、0 孤儿掩膜、0 重复、0 命名异常；患者级划分 694/153/152（239/50/49 患者），文件清单签名 `6bdacfb0…` / 派分签名 `7a6fd581…`（1 组内容重复已移入 `data/_duplicates/`） |
| 数据版本 | `mgd-data version` | 写入 `data/dataset_version.json`，摘要以实际输出为准，999 配对 |
| UI | `python -m mgd.adapters.gradio_app` | HTTP 200；1280/1440/1920 三档截图实测，全宽深色、无控件溢出/换行错位 |
| 诊断 e2e | `ModelService.predict(UNet)` 单例 | 1.85 s、峰值 0.21 GB、分块 384×768；`verified=false` 原因与 ROI 回退提示均显式返回 |
| 评估 | `mgd-eval --scope test --bootstrap 1000` | 产物 3 件（report/per_sample CSV/LaTeX）；见下方指标 |
| 训练冒烟 | `mgd-train --profile quick --epochs 1` | 见下方档位实测；只报吞吐/显存，不报精度结论 |
| 同步 | `tools/sync_runtime.py --apply`（本地脚本，不随仓库分发） | 新增/覆盖清单后执行，逐文件 sha256 → `ALL_SYNCED` |

评估实测（旧 UNet 权重，`verified=false`，仅供工程回归，不作论文指标）：

- Dice 0.7021（95% CI 0.6841–0.7195）、IoU 0.5512、clDice 0.7204、
  Precision 0.6157、Recall 0.8484、HD95 312.7 px、ASSD 47.3 px、Meiboscore 一致率 80.7%；
- 报告内含 `split_signature`、`weight_verified=false` 与原因，评估入口默认允许未验证
  权重但强制标注（`--strict-weights` 可拒绝）。

训练冒烟实测（MGD-Former，1 epoch，700 训练 / 150 验证）：

- 6 GB 档（256×512，batch 1，累积 4，AMP）：197.5 s/epoch，3.54 样本/秒，峰值显存 0.89 GB；
- 8 GB 档（384×768，batch 1，累积 4，AMP）：225.6 s/epoch，3.10 样本/秒，峰值显存 1.227 GB
  （run `20261106T203908Z_MGD-Former_nogit_7c6ef4`）；
- 两次冒烟均明确记录「预训练编码器未加载，随机初始化」的降级原因。

流水线报告实测（`mgd-pipeline --stages report`，仅汇总产物、不回训练）：

- 报告自动扫描 `outputs/runs/*`，跳过中断目录，收录 2 个已完成冒烟 run；
- 两个 run 均无 `reference_report.json`（quick 档不导出），因此 `split_signature` 为空，
  被列入 `missing_split_signature` —— 这是**预期行为**：缺签名的 run 不能进论文表格。

> 冒烟训练不产生精度结论；未实测的档位一律标「估算」，不得用估算值冒充论文指标。

## 10. 已知限制

见 `docs/engineering/KNOWN_LIMITATIONS.md`。要点：`verified=false` 权重状态、
旧权重数据泄漏、分块敏感性、预训练编码器离线降级、Windows/Linux 脚本验证程度；macOS 支持已移除。

---

## 11. v2.4.0 增量：诊断中心级能力（2026-11-10 冻结）

本节是 v2.3.0 之上的**纯代码增量**（不跑数据、不产精度结论）。新增 11 个子包 +
`provenance.py`，全部落在 `mgd/core/`；UI 与 CLI 只做薄壳。版本单一来源仍是
`mgd/__init__.py` 的 `__version__`（现为 2.5.0）。

### 11.1 新增子包与公开接口

| 子包 | 公开接口（稳定契约） | 说明 |
|---|---|---|
| `quantification/` | `METRIC_REGISTRY`、`MetricSpec`、`register/get_spec/all_specs/describe`；`gland_area_ratio / width_profile / diameter_deformation_index / tortuosity_* / fractal_dimension / signal_index / local_contrast / gland_eyelid_distance`；`endpoints_and_branches / break_count / split_instances(method="cc"\|"watershed")`；`zone_edges / zonal_density / symmetry_index`；`resolve_pixel_scale / to_mm` | 可追溯指标层（ADR-003）：16 个指标全部带定义/单位/来源；`split_instances` 的 watershed 分割解决 1px 黏连实例的二义性 |
| `grading/` | `GradeSpec`、`ARITA_MEIBOSCORE`、`PULT_MEIBOSCALE`、`grade_from_loss_percent / grade_all / grade_table / incompatibility_note`；`GradeResult / build_grade_report / grade_rows` | 双分级体系（ADR-004）；Arita ≤5% 归 0 为本项目操作化 |
| `explain/` | `LayerCAM / ScoreCAM`（全分辨率、hook 自清理、绝不返回全零/NaN 图）、`OcclusionConfig / occlusion_map`、`instance_attribution`、`deletion_auc / insertion_auc / road_score / faithfulness_report` | 真归因 + 忠实度证据（ADR-002）；`road_score` 为 engine-defined 简化代理 |
| `qc/` | `QualityThresholds / QualityReport / assess_quality / default_thresholds`；`QCPolicy / decide` | 8 类采集缺陷启发式（模糊/欠曝/过曝/反光/翻转不全/过翻/离焦/睫毛）；阈值全部标注「工程默认，待临床标定」 |
| `conformal/` | `ConformalCalibrator(fit/quantile/interval/to_dict/from_dict)`、`conformal_upper_quantile`、`select_threshold_for_fnr / select_shrink_for_fp`、`save_calibration / load_calibration` | split conformal（有限样本修正 `ceil((n+1)(1-α))`）+ FNR/FP 风险控制阈值；n<1/α 拒绝出区间 |
| `atlas/` | `TYPE_RULES`（8 类）、`TypeEvidence`、`classify_instance / classify_case`、`FEATURE_NAMES / case_vector`、`AtlasIndex(add/search/to_json/load)` | 形态学分型规则全部返回证据值；相似病例余弦检索 |
| `longitudinal/` | `ClinicStore`（sqlite3，patient/visit/measurement/review + schema_version 幂等迁移）、`register_pair`（cv2 ECC）、`match_instances`（匈牙利算法） | 随访三件套；warp 约定为 follow→base |
| `interop/` | `export_seg / export_sr`（pydicom+highdicom 可选，缺→`InteropDependencyError` 含安装命令）、`to_fhir_observation / to_fhir_diagnostic_report` | DICOM SEG/SR（TID 1500 风格，实例表截断 20 条）+ FHIR R4 JSON（编码为工程占位，待临床信息学审核） |
| `governance/` | `AuditLog(append/verify/tail)`（哈希链）、`DEID_TAGS / scrub_image_exif / scrub_dicom_tags / pseudonymize`、`PERMISSIONS / can / require`（RBAC）、`ReviewLog(record/needs_arbitration/arbitrate/history)` | 审计/脱敏/权限/双人复核 |
| `feedback/` | `CorrectionRecord / record_correction / summarize / export_candidates` | 医生修正捕获与训练候选导出（只记录不训练） |
| `provenance.py` | `Provenance / collect_provenance` | 单次推理/报告的血缘（版本/权重 sha256/参数/校准 ID） |

### 11.2 配置与报告扩展（向后兼容）

- `Settings` 新增：`db_path / calibration_dir / atlas_index_path / qc_policy / physical_scale_mm_per_px`
  （全部带默认值；`calibration_dir` 默认跟随 `output_dir`；env：`MGD_DB_PATH / MGD_CALIBRATION_DIR / MGD_ATLAS_INDEX / MGD_SCALE_MM_PER_PX`）。
- `specs.py` 新增段 `infer`：`infer.qc_policy / infer.explain_method`；另加 `data.physical_scale_mm_per_px`、
  `eval.conformal_alpha`；`TrainConfig` 同步 4 个新字段（旧键名与默认值零改动）。
- `build_report()` 新增可选参数 `quality / grading / typology / provenance / explain`：
  HTML 区块顺序 Provenance → 质控 → 分级 → 定量 → 分型 → 解释性 → 图像；None 参数完全不渲染；
  CSV 追加行 r_loss/ga/di/ti/si/grade_arita/grade_pult/qc_score/qc_flags。
- 新 CLI：`mgd-qc / mgd-atlas / mgd-followup / mgd-report`（均支持 `--json` 与 `--help`）。
- 可选依赖组：`pip install "mgd-seg[dicom]"`（pydicom + highdicom）。

### 11.3 质量守卫（独立目录 `quality/`）

堆料式门禁体系：18 道门禁、432 项检查（语法/TODO 扫描/注册表完整性/NaN 纪律/分层/
API 表面冻结/测试速度/导入图/specs 对齐/文档同步/禁径/可选依赖纪律/CLI 表面/
Settings 契约/报告区块/分级交叉验证/INSTANCE_COLUMNS 冻结/严格 JSON）。
运行：`./venv/Scripts/python.exe quality/run_gates.py`。任一 FAIL 退出码 1。


---

## 12. v2.5.0 接线增量（2026-11-10 冻结）：从能力到生效路径

| 接线 | 生效点 | 事实 |
|---|---|---|
| conformal | `mgd-eval`（自动产出+持久化+reference_report.conformal）、`ModelService._predict`（诊断单例 Dice 覆盖下界） | 双口径（dice_coverage / rloss_interval）；`calibration_dir` 生命周期由 store 管 |
| governance/审计 | service.predict、runner、verify（strict）、versioning.backup、eval_cli、RBAC 拒绝 | `home/logs/audit.jsonl` 哈希链；事件表见 KNOWN_LIMITATIONS §11 |
| governance/RBAC | `mgd-eval`/`mgd-report`/`mgd-atlas` 经 `require_or_audit`（Settings.role，env MGD_ROLE） | 拒绝 = 先审计（strict）后 ConfigError |
| governance/脱敏 | `build_report`（MGD_DEID_SALT → 病例号伪名，原始 ID 不落盘）；corrections 落盘同步伪名化 | `pseudonymize` sha256(salt+id)[:16] |
| 日志 v2 | 三路输出（控制台/mgd_events.jsonl/errors.jsonl）+ 20MB×10 轮转 + contextvars 关联 ID | `mgd-logs tail/search --case/--run/summary` |
| feedback | 诊断页 ImageEditor 修正 + `mgd-feedback` CLI + `db/corrections.jsonl`（掩膜 sha256，不嵌像素） | export 产出训练候选清单 |
| DICOM | 真实 SEG/SR 写读往返（pydicom 3.0.2 + highdicom 0.28.2，21 断言） | Comprehensive SR = 1.2.840.10008.5.1.4.1.1.88.33 |
| 新 CLI | `mgd-logs`、`mgd-feedback`（console scripts 已注册） | 合计 11 个 mgd-* 入口 |

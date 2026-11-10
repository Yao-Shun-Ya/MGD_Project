# Known Limitations（工程冻结基线 v2.3.0）

以下限制是**已知且经过实测**的，后续接手的人必须先读这一页，不要重复踩坑。

## 0. 2026-11-09 深度审计补充（接手前必读）

- **MGD-Former 解码器的真实行为**：四级特征在 stride 32 分辨率自顶向下融合 + 32×
  上采样回原图，与 ADR 中「1/4 与 1/8 融合、stride=4 输出」的描述不一致；
  2026-11 患者级基准的全部数字都在该实现上测得。真正的 FPN 式 stride-4 融合属于
  v2.4.0 架构演进项，实施后必须重训并重新对比，不得拿旧权重冒充新架构的指标。
- **深监督的多样性有限**：3 个辅助头当前共享 stride-32 融合特征（插值到三个尺度），
  不是真正的多尺度监督；与上一条同属 v2.4 架构项。
- **batch=1 + 梯度累积下 BatchNorm 的统计噪声**：MGD-Former 融合块/头、AttentionUNet、
  MGD-PriorNet 含 BatchNorm2d；梯度累积不增大 BN 的统计 batch，running stats 来自
  单张图，训练/推理（eval 用 running stats）行为差距被放大。不崩溃（BN 只在
  N×H×W=1 时报错，空间 384×768 不触发），但该组合的收敛性弱于 batch≥2；MONAI UNet
  用 InstanceNorm 不受影响。若追求极致指标，把 BN 换 GroupNorm 属 v2.4 候选项。
- **引擎自建划分已患者级化**：`splits.py` 现按文件名 ≥6 位患者编号整组分配；
  无该编号的数据集退化为图像级并在 warnings 显式说明。MGD-1K 的持久化划分
  （signature `6bdacfb0…` / `7a6fd581…`）不受影响，照常复用。
- **官方 200 例清单仍是审计对象**：泄漏审计结论（326/800、116/166、clean 60/50）不变。


## 1. 权重 `verified=false`

- `models_weights/MGD_UNet_v1.0.0.pth` 的 manifest 仍为 `verified: false`；
- 该权重与旧训练集存在数据泄漏嫌疑（见第 2 条），在拿到独立测试集结果前**不得转正**；
- 任何入口加载未验证权重都必须显式 `allow_unverified=True`，并在界面/日志写明原因；
- 转正流程：`mgd-verify` 在独立划分上评估 → 生成带 `split_signature` 的
  `reference_report.json` → 人工确认后再改 manifest。

## 2. 旧权重存在数据泄漏（不能作为独立测试指标）

- 旧训练脚本 `MGD_Project/data/loader.py` 对全部样本（当时 1000 例）做 shuffle，没有 holdout；
- 因此旧权重很可能见过全部数据，其 Test 指标（Dice 0.7021 等）只能作为
  **工程回归基线**，不能作为论文的独立测试性能；
- 论文必须使用本基线的持久化划分（`dataset_splits.json`，signature 可追溯）
  或 5 折交叉验证重新训练。

## 3. 分块敏感性（实测）

在 40 例原始影像、阈值 0.7 下：

| 分块 | Dice |
|---|---|
| 128 | 0.37 |
| 192 | 0.41 |
| 256 | 0.48 |
| 384 | 0.644 |
| 448 | 0.643 |
| 512 | 0.655 |
| 整图 640×1280 | 0.593 |

结论：分块尺寸对结果影响显著（128 → 384 提升约 27 个百分点），
默认锁定 384×768 / overlap 0.25；跨数据集复现时必须报告分块参数。

## 4. 预训练编码器离线降级

- MGD-Former 使用 MiT-B2 结构，但默认不联网；
- 若本地 `MGD_PRETRAINED_DIR` 或 HuggingFace 缓存中没有 SegFormer-B2 权重，
  编码器为随机初始化，run 的 `pretrained_status` 与训练日志会记录降级原因；
- 论文中必须写明「编码器是否加载预训练、来源与覆盖率」；
- 冻结基线中的 8 GB 实测档为**随机初始化**下的吞吐/显存，不代表有预训练时的差异。

## 5. 显存档位的实测覆盖

- 6 GB 与 8 GB 档在本机（RTX 4060 Laptop 8 GB）实测：6 GB 档 256×512 峰值 0.89 GB、
  3.54 样本/秒；8 GB 档 384×768 峰值 1.23 GB、3.10 样本/秒（详见 `TIERS["8"]` 与冻结基线 §9）；
- 12 / 16 / 24 / 32 / 40 GB 档为外推估算，`TIERS[...].status` 标了「估算」并给出依据；
- 换到新机器后，首次训练请观察 `peak_vram_gb`，实测后回填档位表并改状态为「实测」。

## 6. 跨平台脚本验证程度

- Windows：实际运行验证；
- Linux：仅做语法与逻辑校验（本机无 Linux 环境）；`mgd/packaging/*.sh` 未在真实
  Linux 上执行，视为「逻辑可用、环境未实测」；
- macOS/MPS：v2.3.0 起不再交付，相关 requirements 与 packaging 脚本已删除；
- Docker：提供 `docker/Dockerfile.cuda` 与 `docker/Dockerfile.cpu`，两版均未在本机
  完成镜像构建验证（无 Docker 环境），仅完成语法与 COPY 路径核对。

## 7. 训练精度结论缺失

- 本轮只做 1–2 epoch 冒烟，验证可训练性、吞吐与显存；
- **没有任何新的精度结论**；所有 Dice/IoU/HD95 数字只能来自完整训练 + 独立测试集；
- 冒烟 run 的权重虽然产物齐全，但**不得**当作可部署模型。

## 8. 临床数值口径

- R_loss、Meiboscore 基于像素比例，缺少物理标定（mm/pixel）；
- 界面与报告已标注「比例参考值」，不能直接作为临床诊断依据。

## 9. 诊断链路的两个已知行为（本轮修复后）

- **ROI 冲突回退**：单通道模型用 Otsu 估计眼睑 ROI；当 ROI 内保留的预测像素不足 5%
  时，服务会自动回退全图并在结果（含 HTML 报告）写明「ROI 冲突回退」。这是
  **显式降级**，不是静默修正；看到该提示时必须以全图结果 + 人工复核为准。
- **骨架化终止**：`visualize._skeletonize` 与 `clinical._skeleton` 曾因「孤立像素无法被
  腐蚀」而死循环；已改为「腐蚀不再改变图像即终止」，并加了回归测试。若后续改动
  骨架算法，必须保留该终止条件。

## 10. v2.4.0 诊断中心级增量的诚实限制（2026-11-10）

1. **概率伪彩不等于归因**：旧 UI 的「概率热力图」实为分割概率的伪彩显示，不反映模型
   依据；现已改称「概率伪彩（非归因）」。真归因走 `mgd.core.explain`（LayerCAM /
   Score-CAM / 遮挡敏感度），且必须附忠实度 verdict；`road_score` 是 engine-defined
   简化代理（完整 ROAD 含噪声估计，未实现）。
2. **QC 阈值全部为工程默认，未经临床标定**：`qc/quality.py` 的 blur/曝光/反光/离焦/
   翻转/睫毛启发式阈值（含扣分权重）均为工程经验值；「睫毛遮挡」是未经临床验证的
   启发式。标定前 QC 分数只能用于初筛与复核建议，不能作为拒收的最终依据。
3. **形态分型中标注为 engine-defined 的规则尚无专家一致性验证**：`atlas/typology.py`
   8 类分型中除 tortuosity 引用 Pult & Nichols 2016 的 τ 口径外，其余阈值均为工程定义；
   在完成专家一致性（如 Krippendorff α）评估前，分型结果仅供结构化描述参考。
4. **物理标定缺失时所有长度/距离指标单位为 px，不可跨设备比较**：`quantification.scale`
   在无 DICOM PixelSpacing 且无 `MGD_SCALE_MM_PER_PX` 时返回未标定；此时 width/distance/
   HD95/ASSD 等一律保持像素单位，跨设备/跨分辨率比较无意义。
5. **无第二数据集，跨设备泛化未验证**：v2.4.0 全部能力只在 MGD-1K（LipiView II）语境下
   实现与自测；CGM/其他红外睑板腺设备的泛化、FHIR 编码（当前为工程占位码）均待后续阶段。
   DICOM SEG/SR 已在 v2.5.0 接线中完成**真实写读往返验证**（pydicom 3.0.2 +
   highdicom 0.28.2，21 项往返断言全过，见 tests/test_interop_roundtrip.py）；
   DICOM 一致性校验器（dciodvfy）级别的合规验证仍未做。

## 11. v2.5.0 接线增量的运行事实（2026-11-10）

- **conformal 已接入评估链路**：`mgd-eval` 每次评估自动产出双口径校准（Dice 覆盖下界
  「P(单例 Dice ≥ bound) ≥ 1-α」与 R_loss 预测区间「±q」），持久化到
  `calibration_dir` 并写入 reference_report 的 `conformal` 块；诊断单例从
  `load_latest_metric_calibration` 取最近校准并显示覆盖下界。保证只在可交换假设下成立
  （校准集=持久化评估划分）；`train_self` 口径不作泛化解释；α 默认 0.10。
- **governance 已生效**：审计事件（inference/inference_failed/training_done/
  training_failed/weight_verify/dataset_backup/evaluation/access_denied）落在
  `home/logs/audit.jsonl`（哈希链，`AuditLog.verify()` 可验）；RBAC 经
  `require_or_audit` 覆盖 `mgd-eval`（run_inference）、`mgd-report`（export_report）、
  `mgd-atlas` classify/search（run_inference）；拒绝事件 strict 落审计。UI 路径由口令层
  + 推理审计覆盖；报告脱敏经 `MGD_DEID_SALT`（病例号落盘伪名）。
- **日志 v2**：`mgd_events.jsonl`（全量结构化）+ `errors.jsonl`（WARNING+）+
  控制台三路输出，20MB×10 轮转；`bind_context` 携带 run_id/case_id/stage/arch/scope；
  检索用 `mgd-logs tail/search/summary`。
- **feedback 已接入**：UI 诊断页 ImageEditor 笔刷修正 + `mgd-feedback record/list/
  summary/export`；持久化 `home/db/corrections.jsonl`（掩膜只存 sha256，不嵌像素；
  deid_salt 设置时 case_id 伪名化）。
- **依赖声明对齐实测（无历史用户）**：声明下限一律等于「本机实测已安装版本」，不承诺
  未验证的旧版本；`pyproject` 与 `mgd/requirements/base.txt` 逐条一致（g10 断言）。
  - PDF 报告（`weasyprint`）本机未安装、未实测，`pdf` 可选组仅作声明，不构成支持承诺；
  - `pytest-timeout` 已从未使用状态中移除（g07 用自带子进程超时）。

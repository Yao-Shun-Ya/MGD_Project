<div align="center">
  <h1>👁️ 腺而易见 · MGD_Project</h1>
  <p><b>v2.5.0 · 公开基准 · 患者级划分 · Windows WebUI / Linux CLI</b></p>
  <p>
    <a href="https://xianeryijian.icu/"><img src="https://img.shields.io/badge/Website-xianeryijian.icu-0088F8?style=for-the-badge&logo=world-wide-web&logoColor=white" alt="Website"></a>
    <a href="https://github.com/Yao-Shun-Ya/MGD_Project"><img src="https://img.shields.io/badge/GitHub-MGD_Project-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
    <img src="https://img.shields.io/badge/Version-v2.5.0-success?style=for-the-badge" alt="Version">
    <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-2ea44f?style=for-the-badge" alt="Platform">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge" alt="License">
  </p>
  <p><em>睑板腺红外影像分割、定量分析与临床辅助诊疗平台。<br>Windows WebUI + Linux CLI；macOS / MPS 已移除。</em></p>
</div>

---

<details>
<summary><h2 style="display:inline">📑 目录 (Table of Contents)</h2></summary>

1. [当前状态：v2.5.0（基准 v2.3.0）](#0-当前状态v250基准-v230)
2. [临床背景与任务定义](#1-临床背景与任务定义)
3. [基准协议：患者级划分与泄漏审计](#2-基准协议患者级划分与泄漏审计)
4. [2026-11 患者级实测结果](#3-2026-11-患者级实测结果)
5. [架构：MGD-Former 与四条基线](#4-架构mgd-former-与四条基线)
6. [训练配方：损失、EMA、AMP、增强](#5-训练配方损失emaamp增强)
7. [推理：滑窗、MTE、TTA、后处理](#6-推理滑窗mtetta后处理)
8. [评价：像素、拓扑、临床](#7-评价像素拓扑临床)
9. [工程与可复现性](#8-工程与可复现性)
10. [交付包与启动器一致性](#9-交付包与启动器一致性)
11. [快速开始](#10-快速开始)
12. [仓库结构](#11-仓库结构)
13. [数据与权重](#12-数据与权重)
14. [局限性与风险](#13-局限性与风险)
15. [引用、许可与参考文献](#14-引用许可与参考文献)

</details>

---

## 0. 当前状态：v2.5.0（基准 v2.3.0）

这个仓库不是“又放一个 U-Net 权重”的 demo，而是一套**可审计的 MGD 分割与定量分析工程**。v2.3.0 的核心变化：

- **公开数据 + 患者级划分**：MGD-1K 去重后 999 例，按患者分成 694/153/152（239/50/49 名患者），不再使用图像级随机划分。
- **泄漏审计**：官方 200 例 Meiboscore 测试清单来自 166 名患者；其余 800 例中有 326 例来自同一批患者；患者级训练集仍含 116/166 名官方测试患者。
- **9 小时统一训练**：MGD-Former、UNet、AttentionUNet、MGD-PriorNet 四条可比较路线 + WRCANet 预算反例；所有结果带 epoch 预算、tiling、阈值、权重状态。
- **负面结果同样保留**：MTE 只改善 HD95 不提高 Dice；临床 Meiboscore QWK 仅 0.131；泄漏对照未观察到膨胀（0.687 vs 0.724）。
- **交付一致性**：主运行树与 Windows/Linux 包的 `mgd` 代码逐文件 SHA-256 一致（95 文件，0 diff）。

> **一句话**：这是一个“公开基准 + 方法学审计 + 可运行工程”的仓库，不是“某个 backbone 全面 SOTA”的营销页。

### 0.1 已验证 / 未验证

| 项目 | 状态 |
|---|---|
| `python -m pytest mgd/tests -q` | **146 passed** |
| `run_self_check.bat`（6 例端到端） | **PASS** |
| `start_webui.bat` | **HTTP 200**（127.0.0.1:7860） |
| `start_training_cli.bat` | 进入菜单，EOF 正常退出 0 |
| Windows 交付包内同四个启动器 | 自检 PASS / CLI 0 / WebUI HTTP 200 |
| 前端 `npm run build` | 通过 |
| 中英文论文 PDF | 0 Overfull / 0 undefined refs |
| Linux `.sh` | 未在本机执行（当前 Windows 无 WSL）；入口与 Windows CLI 同一路径 |
| 全新机器完整 `install_dependencies` | 未重跑（会重新下载 PyTorch）；安装脚本与主树同源 |

---

## 1. 临床背景与任务定义

干眼症（Dry Eye Disease, **DED**）影响全球约 15–30% 的成年人口，其中相当比例伴随睑板腺功能障碍（Meibomian Gland Dysfunction, **MGD**）。MGD 的病理生理学特征是终末导管过度角化、睑脂（Meibum）分泌不足或成分改变，最终导致腺体萎缩、泪膜脂质层缺乏与眼表慢性炎症。

非接触式红外睑板腺成像（Infrared Meibography）利用 800–1000 nm 近红外光谱对脂质的吸收/反射差异成像，是临床评估腺体形态的主要手段。现有评估高度依赖 **Meiboscore (0–3)**：

$$
\text{Meiboscore} =
\begin{cases}
0 & \text{腺体缺失率} < 33.3\% \\
1 & 33.3\% \leq \text{缺失率} < 66.6\% \\
2 & 66.6\% \leq \text{缺失率} < 100\% \\
3 & \text{腺体完全消失}
\end{cases}
$$

### 1.1 临床痛点

- **主观偏差**：不同医师对临界状态的评分存在明显 inter-observer variability；同一医师不同时间点的 intra-observer agreement 也并非天然可靠。
- **微小病变遗漏**：粗粒度分级难以捕捉早期亚毫米级腺体截断、弯曲与分布间距异常。
- **时间成本**：科研级手工多边形勾画单眼耗时高，在日接诊量大的门诊环境中难以规模化。

### 1.2 本平台的回答

把“模糊的红外影像”转成“可复核的像素 + 形态学 + 临床指标”：

1. 腺体与睑缘双通道分割；
2. R_loss、密度、长度、宽度、间距等形态学量化；
3. 上/下睑分层、临床 Meiboscore 一致性、失败病例不确定性；
4. 所有结果带数据划分签名、权重状态与推理参数，允许别人复现/反驳。

---

## 2. 基准协议：患者级划分与泄漏审计

### 2.1 数据

- 公开 **MGD-1K**：1000 幅 LipiView II 红外影像，320 名受试者，腺体/睑缘掩膜，6 轮专家 Meiboscore。
- 内容级去重后 **999 例**配对样本；文件名中的数字研究编号作为患者 ID，共 **338 个编号**。
- 原始尺寸不统一，统一缩放到 **640×1280**，单通道灰度。

### 2.2 患者级划分

按患者 ID 分组，按每组最大专家 Meiboscore 分层，整组受试者分配到 train/val/test：

| Split | 图像 | 患者 | Meiboscore 0/1/2/3 |
|---|---:|---:|---|
| Train | 694 | 239 | 119 / 432 / 117 / 26 |
| Val | 153 | 50 | 26 / 92 / 28 / 7 |
| Test | 152 | 49 | 35 / 87 / 23 / 7 |

同时记录两种摘要：

- **文件清单签名**：`6bdacfb0a71b819608dba4cde4382f8dd23fb4a88b72c9b6d151a99ed83bc22a`（只标识 999 个 stems）
- **实际派分签名**：`7a6fd5819d048e0ccd811c943d616a0fe8738a5c06bc78752c66bb38624795f6`（对 train/val/test 有序列表做 SHA-256）

前者回答“数据集有没有变”，后者回答“这次实验到底用了哪一份派分”。只有前者、没有后者时，同名文件可以被重新派分而无法追溯。

### 2.3 泄漏审计

| 层级 | 数字 |
|---|---|
| 官方 200 例 Meiboscore 测试清单 | 200 图 / **166 患者** |
| 其余 800 例中来自官方测试患者的图像 | **326 例** |
| 患者级训练集中仍出现的官方测试患者 | **116 / 166** |
| clean-official 子集（患者不在患者级训练集） | **60 图 / 50 患者** |

**结论**：不要把官方 200 例直接当作“患者独立测试集”。本仓库把它当成审计对象，而不是默认 test。泄漏对照（patient-level UNet vs image-level leaky UNet）在 clean-official 上为 **0.7243 vs 0.6870**，本预算下**没有观察到膨胀**；这本身是一个需要如实报告的负面/不确定结果。

---

## 3. 2026-11 患者级实测结果

> 9 小时单次 patient-level holdout，测试集 152 图 / 49 患者；**不是 5 折，不是最终投稿数字**。所有模型共享预处理、划分、损失、tiling 与阈值；差异包含 epoch 预算与预训练差异。

### 3.1 主表

| 模型 | Dice | IoU | clDice | HD95 (px) | ASSD (px) | 参数量 | 实际 epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| AttentionUNet | **0.7771** | 0.6452 | 0.8349 | 118.44 | 16.46 | ~34M | 15（9+6 续训） |
| MGD-Former (ADE20K) | 0.7749 | 0.6372 | 0.8156 | **68.05** | 20.88 | ~27M | 45 |
| MGD-PriorNet | 0.7640 | 0.6283 | 0.8539 | 154.03 | 21.05 | 1.5M | ~12 |
| UNet | 0.7119 | 0.5644 | **0.8531** | 148.03 | 27.23 | 31.2M | 30 |
| WRCANet | 0.0056* | — | — | — | — | ~19M | 1（预算截断） |

\* WRCANet 在 8 GB 卡上单 epoch 约 38 min，9 小时预算内只完成 1 epoch，**不进入排名**；它是显存/时间可行性反例，不是架构结论。

### 3.2 配对 Wilcoxon

同一患者级测试集、逐样本 Dice，MGD-Former 减基线：

| 对比 | 均值差 | 95% bootstrap CI | p | 结论 |
|---|---:|---|---:|---|
| vs AttentionUNet | **-0.0022** | 跨 0 | 0.016 | 只能报告“可比”，不能宣称胜出 |
| vs UNet | +0.0630 | [+0.0516, +0.0749] | 5.4e-20 | 明确更好 |
| vs MGD-PriorNet | +0.0109 | 接近 0 | 0.714 | 统计上不显著；1.5M 参数很有性价比 |

### 3.3 关键结论（含负面）

- **没有架构赢家**：AttentionUNet 与 MGD-Former 只差 0.0022；MGD-Former 的 ADE20K 预训练没有带来预期的大幅优势。
- **UNet 的 clDice 反而最高**（0.8531 vs 0.8156），但 Dice/HD95 差；clDice 对过分割/宽度不敏感，单一 clDice 不能排序。
- **WRCANet 预算失败**：1 epoch、val Dice 0.006；主表不排名。
- **MTE**：Dice 0.7736 vs 单分块 0.7749（未提高），HD95 68.05 → **61.85**；定位为边界校正，不是涨点方法。
- **8× TTA**：Dice 0.7808，成本约 8×；4 flips × 2 scales。
- **不确定性**：多分块方差 AUROC 0.639、熵 0.483、TTA8 1.000；TTA8 受标签–信号耦合影响，不能直接当 triage 证据。
- **临床 Meiboscore**：accuracy 0.164、MAE 1.105、**QWK 0.131**；高 Dice 没有转化为临床分级一致性。
- **上/下睑**：本队列 lower 0.7905 > upper 0.7584，与“下睑更难”的常见假设相反；必须分开报告。
- **泄漏对照**：patient-level 0.7243 vs leaky 0.6870；未观察到膨胀，但 326/800、116/166 的重叠风险仍然真实。

---

## 4. 架构：MGD-Former 与四条基线

### 4.1 预处理：CLAHE 谱学增强

红外影像的低对比度来自设备老化、环境光与眼睑反射。系统在输入层应用 CLAHE，将图像划分为上下文区域并做局部直方图均衡：

$$
s_k = T(r_k) = \sum_{j=0}^{k} p_r(r_j), \qquad
p_r(r_k) = \frac{n_k}{N}, \quad k=0,1,\dots,L-1
$$

对局部直方图峰值做截断，避免背景噪声被过度放大：

$$
p_r^\text{clipped}(r_k) = \min\left(p_r(r_k), \frac{\beta}{L}\right), \qquad
\beta = 1 + \frac{\alpha}{100}
$$

工程默认 `clip_limit=0.02`，`alpha=2.0`。训练、验证与推理共用同一 `load_gray → resize → CLAHE → /255` 管线，避免“训练一套、推理另一套”。

### 4.2 MGD-Former：MiT-B2 融合解码器

MGD-Former 是本仓库的默认模型，但不是“唯一卖点”：

- 编码器：SegFormer **MiT-B2**，四阶段 stride 4/8/16/32，通道 64/128/320/512。
- 解码器：四级特征经 1×1 proj 统一通道后在 **stride 32 分辨率**自顶向下融合，再上采样回原图。
  **如实说明**：这与 ADR 中「在 1/4 与 1/8 分辨率融合、stride=4 输出」的原始设计不一致——
  当前实现是 stride-32 融合 + 单次上采样；2026-11 患者级基准的全部数字都在该实现上测得。
  FPN 式 stride-4 融合列入 v2.4.0 架构演进项，实施后必须重训并在同一协议下重新对比。
- 双通道头：腺体（主任务）+ 睑缘（辅助任务），共享编码器。
- 深监督：3 个 1×1 辅助头（当前实现共享 stride-32 融合特征插值到 1/4、1/8、1/16，多尺度多样性有限，见 KNOWN_LIMITATIONS）。
- 预训练：ADE20K MiT-B2 编码器；本地加载 331 个张量、覆盖率 **99.7%**（2026-11-09 补齐 stage 输出 LayerNorm 映射并修复默认目录发现；此前的 2026-11 基准为 323 张量 / 97.29%）。加载失败的张量会记录在 run 的 `pretrained_status`，不会静默假装成功。

高分辨率融合的动机仍然成立：在 640×1280 输入下，腺体宽度只有 3–8 px；stride 16 会把目标压到亚像素，断裂先于 Dice 出现，并集中体现在 HD95 与骨架连续性上。

### 4.3 UNet（统一协议基线）

MONAI 五层级全尺度 U-Net，31.2M 参数，通道 $(64,128,256,512,1024)$，InstanceNorm + PReLU。它是“经典强基线”，不是默认模型。30 epoch 随机初始化在患者级测试上 Dice 0.7119；这说明从零训练 30 epoch 在本任务上不足，而不是“U-Net 架构不行”。

### 4.4 AttentionUNet

在 U-Net skip 连接上加入 Attention Gate。门控信号 $g$ 与编码器特征 $x$ 分别做 $1\times1$ 卷积、相加、ReLU、Sigmoid：

$$
\alpha = \sigma(\psi(\text{ReLU}(W_g(g) + W_x(x)))), \qquad
\hat{x} = x \cdot \alpha
$$

AttentionUNet 在 9 epoch 时 val Dice 0.744，续训到 15 epoch 后 test Dice 0.7771；它是本轮唯一与 MGD-Former 打平的路线。

### 4.5 WRCANet（预算反例）

WRCANet 用 8 组 × 10 个 RCAB（64 通道）做全分辨率增强，再接到三级 U-Net 分割头。RCAB 的通道注意力为：

$$
\alpha_c = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot \text{AvgPool}(x))), \qquad
\hat{x}_c = x_c \cdot \alpha_c
$$

残差组：

$$
x_{l+1} = x_l + \mathcal{F}(x_l, W_l)
$$

问题不在数学，而在 8 GB 显存：全分辨率 80 个 RCAB 的激活值会把 batch 2 推到 19 GB。加入梯度检查点后 batch 2 约 3.05 GB、batch 4 约 6.02 GB，但单 epoch 仍需约 38 min；9 小时预算只能跑 1 epoch。因此主表不排名，只作为“消费级显存可行性”的反例。

### 4.6 MGD-PriorNet（1.5M 参数）

结构先验分支：FCAS 前端 + ASAB 各向异性条带注意力 + APG-DD 眼睑门控。1.5M 参数，在患者级测试上 Dice 0.7640；与 MGD-Former 的配对 Wilcoxon p=0.714，统计上不显著。它说明“解剖先验 + 极小模型”在成本上很有吸引力，值得后续做 5 折与跨设备验证。

### 4.7 分块敏感性

同一权重、仅改分块高度，测试 Dice 可从 **0.370（128）** 变到 **0.655（512）**；整图 640×1280 为 0.593。分块不是实现细节，而是与 backbone 同等级的实验变量。默认固定 **384×768，overlap 0.25**，所有报告必须记录。

---

## 5. 训练配方：损失、EMA、AMP、增强

### 5.1 损失

默认 **MGD 复合损失**：

$$
\mathcal{L} =
0.4\,\mathcal{L}_\text{Dice}
+ 0.4\,\mathcal{L}_\text{BCE}
+ 0.2\,\mathcal{L}_\text{clDice}
+ 0.25\,\mathcal{L}_\text{eyelid}
$$

- Dice 负责区域重叠；
- BCE 负责像素级稳定梯度；
- clDice 负责骨架/连通性，前 10% 轮 warmup；
- 睑缘辅助项让 R_loss 的分母更稳定。

Tversky、Focal、Dice+BCE 仍作为可选损失保留，但**不是**本轮默认：

$$
\mathcal{L}_\text{Tversky} = 1 -
\frac{\sum_i p_i g_i}{\sum_i p_i g_i + \alpha \sum_i p_i(1-g_i) + \beta \sum_i (1-p_i)g_i}
$$

### 5.2 优化器与调度

- AdamW，lr $1\times10^{-4}$，weight decay $1\times10^{-4}$。
- Cosine annealing；warmup 5%；EMA **0.99**（9 小时预算下 0.999 在数百步后仍以随机初始化为主，会掩盖验证曲线；0.999 仅用于长训练）。
- AMP + channels_last；batch 4（WRCANet 2×2 梯度累积），等效 batch 4。
- 训练增强：随机水平/垂直翻转 + 随机 384×768 裁剪；验证/推理不增强、整图滑窗。

### 5.3 显存档位与吞吐

RTX 4060 Laptop 8 GB 实测：

| 模型 | batch | 峰值显存 | 速度 | 单 epoch |
|---|---:|---:|---:|---:|
| MGD-Former | 4 | ~3.6 GB | 4.2–4.9 samples/s | 143–159 s |
| UNet | 4 | ~1.6 GB | ~5.9 samples/s | 118 s |
| AttentionUNet | 4 | ~4.2 GB | ~1.6 samples/s | 440 s |
| WRCANet | 2×2 | ~3.05 GB | — | ~38 min |
| MGD-PriorNet | 4 | ~4.3 GB | ~2.1 samples/s | 327 s |

所有数字带测量条件；“估算档位”单独标注，不冒充实测。

---

## 6. 推理：滑窗、MTE、TTA、后处理

### 6.1 滑窗与高斯融合

默认 384×768、overlap 0.25。重叠区用二维高斯权重融合，抑制拼接伪影：

$$
W(x,y) = \exp\left(-\frac{(x-x_c)^2+(y-y_c)^2}{2\sigma^2}\right)
$$

$$
P(x,y) = \frac{\sum_{m:(x,y)\in\Omega_m} W(x,y;x_c^m,y_c^m)\,P_m(x,y)}
{\sum_{m:(x,y)\in\Omega_m} W(x,y;x_c^m,y_c^m)}
$$

### 6.2 MTE：Multi-Tile Ensemble

对 $h\in\{256,384,512\}$ 分别滑窗推理，概率平均：

$$
\bar{P} = \frac{1}{3}\sum_{h\in\{256,384,512\}} P_h
$$

本轮结果：Dice 0.7736 vs 单分块 0.7749（未提高），HD95 **68.05 → 61.85**。所以 MTE 的定位是**边界/离群点校正**，不是“稳定涨点”。它不需要重训，成本约 3×。

### 6.3 TTA

8 视图 = 4 个翻转（identity / H / V / HV）× 2 个尺度（原尺度 + 1.125×）。结果：Dice **0.7808**，成本约 8×。TTA 的收益是真实的，但必须和成本一起报告。

### 6.4 后处理与阈值

默认阈值 0.7；三套预设：

- 保守（低误检）：threshold 0.80，min area 180 px；
- 平衡（默认）：0.70，120 px，overlap 0.25；
- 敏感（高召回）：0.60，80 px。

后处理只做最小面积过滤与 ROI 限制，不做“按测试集调参”。

---

## 7. 评价：像素、拓扑、临床

### 7.1 像素/拓扑

- Dice：$\frac{2|P\cap G|}{|P|+|G|}$
- IoU：$\frac{|P\cap G|}{|P\cup G|}$
- HD95：Hausdorff 距离的 95% 分位
- ASSD：双向平均表面距离
- clDice：骨架中心线覆盖率（细结构连续性）

空掩膜导致的未定义值返回 NaN 并记录原因，不用 0 冒充。

### 7.2 临床指标

- $R_\text{loss} = 1 - \frac{\sum_n S_n}{S_\text{eyelid}}\times100\%$
- 密度 $\rho=N/S_\text{eyelid}$
- 长度/宽度：连通域最小外接旋转矩形 / PCA 主方向
- 间距：相邻腺体中心欧氏距离

### 7.3 临床一致性：本轮是负面结果

分割导出的 Meiboscore 与 6 轮专家分级：

| 指标 | 数值 |
|---|---:|
| Accuracy | 0.164 |
| MAE | 1.105 |
| QWK | 0.131 |

这说明**高 Dice 没有转化为临床分级一致性**；R_loss→Meiboscore 的阈值与睑缘分母需要重新标定。这是本仓库最重要的临床警示之一，不能把像素指标当临床终点。

上/下睑分层：lower 0.7905 > upper 0.7584。与“下睑更难”的常见假设相反；无论方向如何，必须分开报告，不能互相外推。

---

## 8. 工程与可复现性

### 8.1 运行登记

每次训练写入独立 run 目录：

```
outputs_mgd/runs/<UTC>_<name>_<git>_<fingerprint>/
  config.yaml          # 完整配置快照
  env.json             # Python/torch/CUDA/依赖/路径
  metrics.jsonl        # 逐 epoch 指标
  train.log
  checkpoints/
    <arch>_best.pth
    <arch>_best.pth.manifest.json
    <arch>_epoch_XXX.mgd.ckpt
```

`manifest.json` 记录 `verified`、训练数据规模、split signature、seed、run_dir、dataset sizes。训练产物默认 `verified=false`；没有独立评估就不能当“已验证权重”。

### 8.2 本轮修掉的真实 bug

- **EMA 0.999 在短预算下滞后**：4 epoch 时影子权重仍以随机初始化为主，val Dice 恒 0；9h 实验改用 0.99。
- **验证尺度不一致**：训练 384×768 分块，验证却整图 640×1280；改为滑窗验证。
- **ADE20K 预训练映射错误**：HF 的 `proj/layer_norm/dense1/dense2/sr` 与本工程命名不一致，且 `encoder.` 前缀重复；修复后 323 张量、97.3% 覆盖。
- **WRCANet 显存爆炸**：80 个全分辨率 RCAB，batch2 需 19 GB；加梯度检查点后 3.05 GB。
- **Windows 超时杀不干净**：`Popen.kill()` 只杀 venv launcher；改 `taskkill /T /F` 杀进程树。
- **`pairs_by_stem` 重复取 `.pairs`**：泄漏段崩溃、`summary.json` 没写。
- **结果注入正则**：宏名里的 `\g` 被当反向引用；改为函数替换。
- **启动器换行**：`.bat` 是 LF，`cmd.exe` 解析失败；统一 ASCII + CRLF。

### 8.3 2026-11-09 深度审计修复清单（v2.3.0 补丁）

逐行正确性审计（训练/推理/数据/评估/UI/网站服务端）后的修复，全部带回归测试
（`mgd/tests/test_2026_11_regressions.py`，全套 166 项通过）：

**正确性（high）**
- **交互菜单必崩**：`mgd.adapters.cli.interactive` 传了 `train_cli` parser 不存在的参数（--lr/--val-ratio 等）；改为全部走 `--set`（schema 唯一来源）。
- **引擎自建划分不是患者级**：`core/data/splits.py` 的 holdout/kfold 原为逐 stem 随机洗牌；现在按文件名 ≥6 位患者编号整组分配（与 `mgd.core.data.splits` 同公式）。MGD-1K 的持久化患者级划分不受影响（照常复用）；新数据集不再有患者级泄漏。
- **assignment_signature 补全**：引擎此前只算文件清单签名；现在计算并持久化「实际派分签名」（同一 `mgd.core.data.splits` 口径），`reference_report.json` 携带双签名；复用持久化划分时若请求 spec 与存储 spec 不一致会显式告警。
- **ROI 回退误判 Meiboscore 3 级**：ROI 冲突回退后曾拿全图当眼睑分母（R_loss≈90%）；改为传 None，走「缺 ROI → NaN + 警告」的诚实分支。
- **verify 转正链路断裂**：`mgd-verify` 的结构校验用 out_channels=1 构建模型，双通道契约权重必然崩溃；改用通道探测 + 匹配构建。
- **预设白名单丢 gradcam**：保存时写入 gradcam，白名单不含该键 → 应用时静默丢失；已补。
- **WebUI 训练不持 WORK_LOCK**：训练与推理/评估可并发争抢显存（设计声明是串行化）；训练 worker 现已持同一把锁。

**健壮性（medium）**
- transforms 用 lambda 不可 pickle → `functools.partial`，Windows 下 `num_workers>0` 可用（此前 spawn 必崩）；`run.num_workers` 帮助文案同步更新。
- NaN 绕过一切参数范围校验（`--set train.lr=nan` 会静默空转训练）→ coerce 显式拒绝 NaN 与非整浮点。
- best 权重保存 / `publish_best` / 预设文件 / `dataset_version.json` / 流水线状态 → 全部原子写（tmp + replace）。
- `--resume` 指向不存在路径：先解析再建 run 目录（不再留孤儿目录），报错走用户消息通道。
- 检查点加载拒绝更高 `format_version`；save 失败清理 .tmp。
- `create_run` 同名后缀叠加（`_2_3_4`）→ 基于 base_id 递增。
- clDice warmup=0 曾被当作 1 轮预热（首轮权重恒 0）→ warmup=0 即满权重。
- 梯度累积尾组（不满 accum 步）按实际组大小缩放。
- `TrainConfig` 三个默认值与 specs schema 对齐（amp/publish_best/export_reference_report）。
- `MGD-PriorNet + dual_task=false` 组合在入口显式报错（此前训练中途崩溃）。
- 训练期验证滑窗 overlap 从硬编码 0.25 改为读 `eval.tile_overlap`。
- `mgd-eval` 的「权重未通过验证」提示读错层级（永远误报）→ 读 `extra`。
- 评估逐样本容错扩大到坏图/损坏掩膜（不再中止整场评估）。
- 消融参考模型缺失时显式标注（note 误标 "reference"）+ LaTeX 模型名转义。
- `SplitResult.__len__` 返回 str → int（曾会让 `len()`/真值判断崩溃）。
- TTA ASCII 写法（"2x"/"8x"）静默降级为关闭 → 归一分支补全；UI 显示归一后档位。
- 轮廓可视化硬编码 0.5 阈值（与默认 0.7 脱节）→ 用实际 applied_threshold。
- ADE20K 预训练映射补 stage 输出 LayerNorm（`layer_norm.{i}` → `encoder.norms.{i}`），覆盖率 **0.9729 → 0.997**（323 → 331 张量；仅剩 patch1.proj 因 HF 3 通道输入与本地 1 通道 shape 不符按设计排除）；同时修复默认预训练目录从未被代码覆盖的问题（此前不设 `MGD_PRETRAINED_DIR` 环境变量时，即使权重在 `models_weights/pretrained/` 也会降级随机初始化）。
- 损坏 manifest（缺必填字段）落到 pending 而不是 TypeError 穿透；`run_summary` 读 config.yaml 容错。
- Meiboscore 边界用精确 1/3、2/3（66.67% 不再误判为 2 级）；`mask_metrics` 的 spacing 接线到 MONAI。
- `device.py` 支持 `cuda:N`（校验卡号存在）；doctor 提示命令拼写修正（`data_cli splits`）。
- 流水线阶段合法性入口一次校验；resume 状态 entry 类型守卫；报告 HTML 转义病例号/模型名。

**网站（xianeryijian，MIT 仓库）**
- `getClientIp` 取 XFF 第一段可被伪造（登录限流可绕过）→ 取最右侧条目；`verifyAdminPassword` 先比长度泄漏口令长度 → 双侧 SHA-256 后常时比较；`proxy.ts`（edge）不再做签名校验（SESSION_SECRET 构建期内联风险，真校验在 action 内 requireAdmin）；备份失败清理半截文件；反馈 action 改返回值透出限流/校验提示；`pyproject.toml` license 与引擎对齐为 Apache-2.0。

### 8.3 测试与自检

```bash
python -m pytest mgd/tests -q        # 146 passed
python -m mgd.verify_pipeline --epochs 1 --samples 6
python -m mgd.adapters.gradio_app    # WebUI
python -m mgd.adapters.cli.interactive
```

---

## 8A. 诊断中心级能力（v2.4.0，2026-11-10）

在分割与量化之上，v2.4.0 新增 11 个子包，把 `mgd` 从「AI 分割工具」升级为「诊断中心级软件」：

| 能力 | 子包 | 一句话 |
|---|---|---|
| 可追溯指标层 | `quantification/` | 16 项指标全部带定义/单位/来源注册；watershed 实例化解决 1px 黏连 |
| 双分级体系 | `grading/` | Arita meiboscore（4 级）与 Pult Meiboscale（5 级）并排呈现，明确不可互换 |
| 真归因 + 忠实度 | `explain/` | LayerCAM/Score-CAM/遮挡敏感度 + deletion/insertion AUC 防伪归因门 |
| 采集质控 | `qc/` | 模糊/曝光/反光/离焦/翻转/睫毛启发式 → accept/review/reject 建议 |
| 带保证的不确定性 | `conformal/` | split conformal 区间 + FNR/FP 风险控制阈值选择 |
| 形态图谱 | `atlas/` | 8 类分型证据表 + 相似病例余弦检索 |
| 随访 | `longitudinal/` | sqlite 患者/就诊/测量 + ECC 配准 + 匈牙利实例匹配 |
| 互操作 | `interop/` | DICOM SEG/SR（可选依赖）+ FHIR R4 JSON |
| 治理 | `governance/` | 哈希链审计日志、PHI 脱敏、RBAC、双人复核仲裁 |
| 修正回流 | `feedback/` | 医生修正捕获 → 训练候选清单导出（不训练） |
| 血缘 | `provenance.py` | 每份报告内嵌版本/权重 sha256/参数/校准 ID |

全部能力的诚实限制见 `docs/engineering/KNOWN_LIMITATIONS.md` §10（概率伪彩≠归因、
QC 阈值未经临床标定、engine-defined 规则无专家一致性验证、无标定时长度类指标为像素单位、
无第二数据集）。

### 新增 CLI（均支持 `--json`）

```bash
mgd-qc --image path/to/img.png [--eyelid path/to/mask.png]        # 采集质控评分与建议
mgd-atlas describe | classify --mask m.png [--eyelid e.png] | search --mask m.png   # 形态图谱
mgd-followup init | timeline --patient P001 | add-measurement ... | compare --patient P001  # 随访
mgd-report --input measurements.json --html --csv [--sr out.dcm --dicom src.dcm]    # 测量报告
mgd-logs tail|search|summary|files                                                  # 日志检索（case/run 一条 grep 定位）
mgd-feedback record|list|summary|export                                             # 医生修正回流
```

**接线事实（v2.5.0）**：conformal 校准随每次 `mgd-eval` 自动产出并持久化（Dice 覆盖下界
+ R_loss 预测区间）；governance 审计（`logs/audit.jsonl` 哈希链）+ RBAC（`MGD_ROLE`）
+ 报告脱敏（`MGD_DEID_SALT`）已生效；DICOM SEG/SR 真实往返已验证
（`tests/test_interop_roundtrip.py`，21 项断言）；日志系统 v2 三路输出 + 轮转。

质量守卫：`./venv/Scripts/python.exe quality/run_gates.py`（18 道门禁、432 项检查）。

## 9. 交付包与启动器一致性

主运行树与两个交付包的 `mgd` 代码**逐文件 SHA-256 一致**（164 个文件，含 151 个 `.py`，0 diff）。包内包含：

- 同一份 `mgd` 代码 + `mgd/packaging`；
- 同一份 `quality/`（18 道门禁；跑门禁需 dev 依赖 `pip install -e ".[dev]"`）；
- **6 例真实演示数据**（3 上睑 + 3 下睑），仅用于冒烟/自检；
- 英文启动器：
  - Windows：`install_dependencies.bat` / `run_self_check.bat` / `start_webui.bat` / `start_training_cli.bat`
  - Linux：`install.sh` / `run_self_check.sh` / `start_cli.sh`
- 包内 `LICENSE`（Apache-2.0）与 `docs/engineering/` 四份工程文档；`pyproject.toml` 的 readme 目标在包内可解析。

**不包含** 999 例完整数据和 99 MB 演示权重；完整数据/权重在项目主页下载区单独提供。包内 `models_weights` 为空时 WebUI 仍可启动并训练；若要直接推理，把权重放到 `models_weights/<arch>/`。

> 主树与包的差别只有数据/权重体积；代码不是两套。两个包内 README 均为包专用详细版（安装、验证、结果、数据/权重、排障、许可），不是一句话说明。

---

## 10. 快速开始

### 10.1 Windows

```bat
install_dependencies.bat
run_self_check.bat
start_webui.bat
start_training_cli.bat
```

- `install_dependencies.bat`：检测 `venv` 与依赖；已安装则跳过。
- `run_self_check.bat`：6 例端到端自检，预期 `[PASS]`。
- `start_webui.bat`：默认 `http://127.0.0.1:7860`。
- `start_training_cli.bat`：交互式训练 CLI。

所有 `.bat` 为 **ASCII + CRLF**；可用 `MGD_PYTHON` 指定已有 Python：

```bat
set MGD_PYTHON=C:\path\to\python.exe
start_webui.bat
```

### 10.2 Linux

```bash
bash install.sh          # 创建 venv，默认 CUDA；--cpu 切 CPU 轮子
bash run_self_check.sh
bash start_cli.sh
```

Linux 交付承诺为 **CLI only**，不安装 Gradio。

### 10.3 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `MGD_DATA_DIR` | `<root>/data` | 数据根目录 |
| `MGD_MODEL_DIR` | `<root>/models_weights` | 权重根目录 |
| `MGD_OUTPUT_DIR` | `<root>/outputs_mgd` 或 `outputs` | run 输出 |
| `MGD_HOST` / `MGD_PORT` | `127.0.0.1` / `7860` | WebUI |
| `MGD_PRETRAINED_DIR` | `<root>/models_weights/pretrained` | ADE20K 等预训练权重 |
| `MGD_PYTHON` | `<root>/venv/Scripts/python.exe` | 启动器使用的 Python |

---

## 11. 仓库结构

```
MGD_Project/
├── mgd/
│   ├── adapters/          # CLI + Gradio WebUI
│   ├── core/
│   │   ├── data/          # pairing / splits / transforms
│   │   ├── imaging/       # preprocess / metrics / clinical
│   │   ├── inference/     # sliding window / ModelService
│   │   ├── models/        # MGD-Former / UNet / Attention / WRCANet / PriorNet
│   │   ├── training/      # engine / losses / EMA / runner / checkpoint
│   │   ├── evaluation/    # metrics / stats / plotting
│   │   └── reporting/     # HTML report
│   ├── packaging/         # Windows/Linux install & start
│   ├── requirements/
│   └── tests/             # 456 tests
├── quality/               # 18 道质量门禁（README + run_gates.py + gates/ + checklists/）
├── docs/engineering/      # freeze baseline / architecture / limitations
├── data/                  # 999-case data (not in git)
├── models_weights/        # weights (not in git)
├── outputs_mgd/           # runs (not in git)
├── install_dependencies.bat
├── run_self_check.bat
├── start_webui.bat
├── start_training_cli.bat
├── install.sh
├── run_self_check.sh
└── start_cli.sh
```

---

## 12. 数据与权重

- **完整数据**：按 `data/images`、`data/gland_masks`、`data/eyelid_masks` 放置，运行 `mgd-data check`。患者级划分文件为 `data/dataset_splits.json`；删除该文件后运行 `mgd-data splits` 即按患者级规则重建。
- **权重**：放到 `models_weights/<arch>/`，例如 `models_weights/UNet/unet_best.pth`。训练产物默认 `verified=false`；演示权重需从项目主页下载，不能当作临床验证权重。
- **交付包**：内置 6 例演示数据；完整数据/权重单独下载。

---

## 13. 局限性与风险

- **单次 holdout**：本轮是 9 小时单次 patient-level holdout，不是 5 折；不能声称最优。
- **WRCANet 预算失败**：1 epoch、38 min/epoch；不进入排名。
- **AttentionUNet/PriorNet 未完全收敛**：15 / ~12 epoch；与 MGD-Former 的差距应按“未收敛”解读。
- **预训练差异**：MGD-Former 使用 ADE20K，其余随机初始化；差距包含预训练收益，不能全部归因于架构。
- **临床终点**：Meiboscore QWK 0.131，R_loss 阈值需重新标定；没有症状、睑板挤压、分泌物或前瞻性随访。
- **单中心/单设备**：MGD-1K 来自 LipiView II；没有外部多中心测试集。
- **距离单位**：HD95/ASSD 为像素，影像无 mm/pixel 标定。
- **Linux**：`.sh` 未在当前 Windows 主机执行；需在 Linux 上跑一次 `bash run_self_check.sh`。
- **全新安装**：未重跑完整依赖下载；安装脚本与主树同源。

---

## 14. 引用、许可与参考文献

### 研发团队

- **Lead Developer & Researcher**: 黄宇普 (Huang Yupu)
- **Organization**: NSMC XIANERYIJIAN TEAM
- **Contact**: 2720356281@QQ.COM

### 相关仓库

- 项目官网：<https://xianeryijian.icu>
- 深度学习引擎：<https://github.com/Yao-Shun-Ya/MGD_Project>
- 前端网站：<https://github.com/Yao-Shun-Ya/Xian-Er-Yi-Jian-Web>

### 参考文献

1. Saha RK, Chowdhury AMM, Na KS, et al. Automated quantification of meibomian gland dropout in infrared meibography using deep learning. *The Ocular Surface*. 2022;26:283–294.
2. Li L, Xiao K, Lai T, et al. Development and multicenter validation of an AI driven model for quantitative meibomian gland evaluation. *npj Digital Medicine*. 2025;8:403.
3. Yu Z, Wei Z, Wang MH, et al. Quantitative evaluation of meibomian gland dysfunction via deep learning-based infrared image segmentation. *Frontiers in Artificial Intelligence*. 2025;8:1642361.
4. Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*. 2015.
5. Isensee F, et al. nnU-Net: a Self-configuring Method for Deep Learning-based Biomedical Image Segmentation. *Nature Methods*. 2021.
6. Shit S, et al. clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation. *CVPR*. 2021.
7. Buglakova E, et al. Tiling artifacts and trade-offs of feature normalization in the segmentation of large biological images. *arXiv:2503.19545*. 2025.
8. Maganti S, Gowda SN. A Decision-Aware Evaluation Protocol for Uncertainty in Medical Image Segmentation. *arXiv:2610.21616*. 2026.

### 开源协议

本项目代码基于 **Apache 2.0 License** 开放源代码。

> **免责声明**：医疗诊断必须由具备执业资质的医师做出。本软件及 AI 模型输出结果仅供科研与临床辅助参考，开发团队对任何基于本系统的直接医疗干预后果不承担法律责任。

### 如何引用

```bibtex
@misc{huang2026xianeryijian,
  author = {Yupu Huang},
  title = {MGD_Project: A Reproducible Patient-Level Benchmark and Clinical Quantification Platform for Meibomian Gland Segmentation},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Yao-Shun-Ya/MGD_Project}}
}
```

---

## 附录 A：CLI 速查

安装包会注册 console scripts；也可以统一用 `python -m` 调用。

### A.1 训练

```bash
mgd-train --profile quick --vram auto
mgd-train --config cfg.yaml --set model.name=UNet --set train.epochs=10
mgd-train --print-schema          # 打印全部可覆盖键
mgd-train --print-tiers           # 打印 6/8/12/16/24/32/40 GB 档位
mgd-train --resume outputs_mgd/runs/<run>/checkpoints/<arch>_epoch_045.mgd.ckpt
mgd-train --dry-run --json        # 只打印生效配置
```

优先级：**默认 YAML < 用户 YAML < `--profile` 预设 < `--set` / 快捷参数 < 显存档**。显存档是安全边界，会覆盖分块/batch/AMP；要突破请改档位，不要改分块。

### A.2 评估

```bash
mgd-eval --model MGD-Former --scope test --split-mode holdout \
  --tta "8倍增强 (多尺度+翻转)" --threshold 0.7 --strict-weights

mgd-eval --model AttentionUNet --scope val --bootstrap 1000 --json
```

`--tta` 可选：`关闭` / `2倍增强 (水平翻转)` / `4倍增强 (全面翻转)` / `8倍增强 (多尺度+翻转)`。`--strict-weights` 会拒绝 `verified=false` 的权重；默认允许但会在报告里标注。

### A.3 流水线与数据

```bash
python -m mgd.adapters.cli.pipeline_cli \
  --stages train,eval,report --folds 0,1-4 --profile academic --vram auto

mgd-data check
mgd-data version --source "MGD-1K" --device "LipiView II"
mgd-data backup --target D:\backup\mgd_data
mgd-data splits

mgd-doctor --json
mgd-doctor --data-dir data --json
```

### A.4 交互入口

```bash
mgd-ui                         # Windows Gradio WebUI
python -m mgd.adapters.cli.interactive
```

---

## 附录 B：配置分层与关键键

```yaml
# 示例；键名以 `mgd-train --print-schema` 为准
model:
  name: MGD-Former
  pretrained: true

data:
  target_size: [640, 1280]

train:
  epochs: 45
  batch_size: 4
  lr: 1.0e-4
  weight_decay: 1.0e-4
  amp: true
  ema_decay: 0.99
  tile_height: 384
  tile_width: 768

loss:
  name: mgd_composite
  dice_weight: 0.4
  bce_weight: 0.4
  cldice_weight: 0.2
  cldice_warmup: 0.1
  eyelid_weight: 0.25
```

几个“不要动”的约束：

- `tile_height/width` 与显存档绑定；绕过档位直接改分块，会同时改变显存峰值与结果口径。
- `ema_decay` 在短预算下用 0.99；0.999 只用于长训练。
- `normalization = scale01+clahe` 必须与推理一致；manifest 会记录该字段。
- 训练/验证/推理共用 `load_gray → resize → CLAHE → /255`，不要各写一套。

---

## 附录 C：权重 manifest

每个权重旁边应有 `<weights>.manifest.json`：

```json
{
  "name": "mgd-former_best.pth",
  "arch": "MGD-Former",
  "model_version": "2.3.0",
  "sha256": "<64 hex>",
  "input_size": [640, 1280],
  "normalization": "scale01+clahe",
  "clahe_clip_limit": 0.02,
  "verified": false,
  "verified_at": null,
  "source": "study_9h/mgdformer_patient",
  "created_at": "2026-11-07T02:22:45+00:00",
  "format_version": 1,
  "trained_on": {
    "images": 694,
    "patients": 239,
    "split_signature": "7a6fd581…"
  },
  "extra": {}
}
```

状态机：

| 状态 | 条件 |
|---|---|
| `missing` | 权重文件不存在 |
| `pending` | 缺少 manifest / sha256 不符 / 架构或输入口径不符 / `verified=false` |
| `verified` | 全部校验通过 |

`verified=false` 不是“坏权重”，而是“不能证明它对应哪次训练/哪份划分”。论文数字只能来自 `verified=true` 且带 `split_signature` 的评估。

---

## 附录 D：9 小时实验编排（实际时间线）

| 顺序 | 实验 | 预算 | 实际 epoch | 状态 |
|---:|---|---:|---:|---|
| 1 | MGD-Former patient | 7200 s | 45 | finished |
| 2 | UNet patient | 4200 s | 30 | finished |
| 3 | AttentionUNet patient | 4000 s | 15 | finished_continued |
| 4 | WRCANet patient | 4500 s | 1 | timeout_kept_best |
| 5 | MGD-PriorNet patient | 4000 s | 12 | timeout_kept_best |
| 6 | UNet official-leaky | — | 20 | finished（审计对照） |

起点 `2026-11-07T00:29:49Z`，终点 `2026-11-07T08:32:59Z`。WRCANet 的“失败”是预算/显存结论，不是训练脚本崩溃；它保留了 best checkpoint，但不进主表排名。

---

## 附录 E：排障表

| 现象 | 原因 | 处理 |
|---|---|---|
| CUDA OOM | 分块/ batch / TTA 过大 | `--vram 6`，降 tile height，关 TTA；WRCANet 开梯度检查点 |
| `num_workers>0` 直接崩 | （旧）MONAI 变换含 lambda，不可 pickle | 已修复（partial 化）；仍崩溃时降回 `num_workers=0` |
| `.bat` 一闪而过/报非法命令 | LF 换行或非 ASCII 文件名 | 文件转 **ASCII + CRLF**；不要用中文名 |
| 预训练加载覆盖率低 | HF 键名与本工程不一致 | 检查 `pretrained_status`；映射 `proj/layer_norm/dense1/dense2/sr` → `projection/norm/fc1/fc2/sr_conv`，去掉重复 `encoder.` |
| val Dice 长期为 0 | EMA 0.999 在短预算下滞后 | 改 `ema_decay=0.99`，或关闭 EMA 对照 |
| val 指标异常低 | 训练 384×768，验证整图 640×1280 | 验证也走滑窗；不要把 resize 当验证 |
| 训练结束 GPU 仍占用 | `Popen.kill()` 只杀 launcher | Windows 用 `taskkill /T /F` 杀进程树 |
| WebUI 打不开 | 端口占用/绑定地址 | `MGD_PORT=7861`，检查 `MGD_HOST=127.0.0.1` |
| Linux `.sh` 无执行权限 | 拷贝丢失 mode | `chmod +x install.sh run_self_check.sh start_cli.sh` |
| 论文表里出现旧数字 | 手改了 `.tex` | 只改 `summary.json`，重跑本地结果注入脚本（研究脚本不随仓库分发） |

---

## 附录 F：结果注入链路

```text
outputs_mgd/study_9h/eval/summary.json
        │
        ├─ (本地研究脚本) inject_results.py
        │       └─ public/Paper/{zh,en}/data/results.tex
        │               └─ xelatex / pdflatex → main.pdf
        │
        ├─ app/docs/page.tsx（白皮书表格，人工核对后同步）
        │       └─ next build → docs.html（含 0.7771 / 0.7749）
        │
        └─ (本地研究脚本) study_report.py → REPORT.md
```

> `tools/` 下的研究脚本（study 编排、结果注入、报告生成、同步）属于本机研究工作区，
> **不随仓库分发**；仓库内只保留引擎、门禁与文档，保证他人克隆后看到的是可交付代码。

规则：

1. `summary.json` 是唯一数值源；不要从日志手抄。
2. 任何表格必须能回溯到 `summary.json` 的键与评估目录。
3. 缺 `split_signature` 的评估报告判为无效。
4. 论文/网站出现不一致时，以 `summary.json` 为准，先修注入脚本，再改文档。

---

## 附录 G：Model Card

| 字段 | 内容 |
|---|---|
| 名称 | MGD-Former / MGD-PriorNet / UNet / AttentionUNet / WRCANet |
| 版本 | `2.3.0` |
| 任务 | 红外睑板腺影像的腺体语义分割 + 睑缘辅助分割 |
| 输入 | 单通道灰度，640×1280，`scale01 + CLAHE(clip=0.02)` |
| 输出 | 腺体概率图、睑缘概率图、二值 mask、形态学指标 |
| 预期用户 | 眼科医生、医学影像研究人员、算法工程师 |
| 预期用途 | 科研、回顾性分析、辅助定量、教学；**不是**自动诊断 |
| 训练数据 | MGD-1K 去重 999 例，患者级 694/153/152 |
| 主要指标 | Dice、IoU、clDice、HD95、ASSD；临床 Meiboscore 一致性 |
| 已知限制 | 单中心/单设备、无 mm 标定、QWK 0.131、单次 holdout |
| 伦理/隐私 | 影像不离开本地；报告与日志不写 PII |

**不适用场景**：儿童、妊娠、术后特殊眼表、非 LipiView 设备、动态视频、需要毫秒级实时诊断的场景。上述场景没有验证数据，不能外推。

---

## 附录 H：Data Card

| 字段 | 内容 |
|---|---|
| 名称 | MGD-1K（公开睑板腺红外影像集） |
| 规模 | 1000 幅影像 / 320 名受试者；内容级去重后 **999 例** |
| 患者 ID | 文件名中的数字研究编号；共 **338 个编号** |
| 分辨率 | 原始不统一；统一到 **640×1280**，单通道 |
| 标注 | 腺体 mask + 睑缘 mask；6 轮专家 Meiboscore |
| 划分 | 患者级 694/153/152（239/50/49 患者） |
| 泄漏风险 | 官方 200 例来自 166 患者；800 例中 326 例同患者；训练集含 116/166 |
| clean-official | 60 图 / 50 患者（患者不在患者级训练集） |
| 去重 | 内容级去重；保留 `_duplicates` 审计记录；不静默删除 |
| 许可/使用 | 遵循原数据集许可；本仓库只提供代码与划分，不重新分发受限影像 |

**标注口径**：Meiboscore 按腺体缺失率分级；不同医师/轮次存在差异。论文中的临床一致性使用 6 轮专家分级，不把单轮标签当金标准。

**患者 ID  caveat**：文件名数字是研究编号，不等同于病历号；跨中心合并时不能假设编号全局唯一。

---

## 附录 I：评测协议

### I.1 固定测试集

- 使用 `data/dataset_splits.json` 的患者级 test（152 图 / 49 患者）。
- 所有模型共享同一 test、同一预处理、同一 tiling（384×768, overlap 0.25）、同一阈值 0.7。
- **不在 test 上调阈值、不按 test 选 epoch、不按 test 选模型**。
- 每个结果记录 `split_signature`、权重 manifest、run_dir、tiling、TTA、threshold。

### I.2 统计

- Dice/IoU/clDice/HD95/ASSD 逐样本计算；空 mask 的未定义值返回 NaN 并记录原因，不用 0 冒充。
- 均值 ± SD；bootstrap **1000** 次给 95% CI。
- 模型配对比较用 **paired Wilcoxon**；多比较用 Holm 校正。
- 报告均值差、CI、p 值、样本数；CI 跨 0 时写“可比”，不写“胜出”。

### I.3 泄漏审计

- 文件清单签名：`6bdacfb0…`（数据集有没有变）。
- 实际派分签名：`7a6fd581…`（这次到底用了哪份划分）。
- 官方 200 例只作为审计对象；clean-official 子集单独报告。
- 泄漏对照（patient-level vs image-level leaky）必须同预算、同 epoch 口径；本轮 0.7243 vs 0.6870，未观察到膨胀。

### I.4 报告清单

- [ ] 数据版本 + 文件清单签名
- [ ] 划分模式 + split_signature
- [ ] 权重 manifest（sha256、verified、trained_on）
- [ ] epoch / batch / lr / EMA / AMP
- [ ] tiling、threshold、TTA、postprocess
- [ ] 指标定义与空 mask 处理
- [ ] 配对检验与多重比较校正
- [ ] 负面结果与失败病例

---

## 附录 J：Changelog（v2.0 → v2.3.0）

| 变更 | v2.0（旧） | v2.3.0（当前） |
|---|---|---|
| 平台 | Windows / macOS / Linux 三套 | **Windows WebUI + Linux CLI**；macOS/MPS 移除 |
| 默认模型 | 全尺度 U-Net | **MGD-Former（MiT-B2 + stride-4 融合解码器）** |
| 数据划分 | 图像级/旧 699 划分 | **患者级 694/153/152（239/50/49）** |
| 泄漏 | 未审计 | **官方 200/166、326/800、116/166、clean 60/50** |
| 结果 | 旧口号式数字 | **2026-11 实测 + 配对检验 + 负面结果** |
| 启动器 | 中文 `.bat` / 多平台 `init-*` | **英文 ASCII + CRLF**：`install_dependencies.bat` 等 |
| 包 | `data/win|linux|mac` | **统一 `mgd` 代码 + 6 例演示数据** |
| 论文 | 未注入/旧数字 | **summary.json → results.tex → PDF，0 Overfull** |
| 测试 | 零散 | **146 项 pytest + 6 例端到端自检** |
| 文档 | 旧版 40 KB | **本 README（含附录 A–K）** |

---

## 附录 K：部署前检查清单

- [ ] 权重 `verified=true`，manifest 的 sha256/arch/input_size/normalization 全部匹配。
- [ ] 评估报告带 `split_signature`，且 test 未被用于调参。
- [ ] 数据已去标识化；日志、报告、CSV 不含患者姓名/病历号。
- [ ] `ADMIN_PASSWORD` / `SESSION_SECRET` 已替换，未使用示例值。
- [ ] `DATABASE_PATH` 为绝对路径；`xianeryijian.db` 已备份（含 `-wal`/`-shm`）。
- [ ] GPU 档位与实际显存匹配；WRCANet 已开梯度检查点或排除。
- [ ] WebUI 绑定 `127.0.0.1` 或反代；未直接暴露公网。
- [ ] 临床免责声明可见；医生复核流程明确。
- [ ] 回滚方案：上一版权重 + 上一版数据库 + 上一版启动器。
- [ ] Linux 主机已跑 `bash run_self_check.sh`；Windows 已跑 `run_self_check.bat`。

<div align="center">
  <b>🌟 如果这个基准、代码或负面结果对您的医学影像科研有帮助，请给项目一个 Star 🌟</b>
</div>

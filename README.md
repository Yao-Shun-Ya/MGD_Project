<div align="center">
  <h1>👁️ 腺而易见 · MGD_Project</h1>
  <p><b>v2.3.0 · 公开基准 · 患者级划分 · Windows WebUI / Linux CLI</b></p>
  <p>
    <img src="https://img.shields.io/badge/version-2.3.0-blue" alt="version">
    <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux-2ea44f" alt="platform">
    <img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="license">
  </p>
  <p><a href="https://xianeryijian.icu/">项目主页</a> · <a href="https://github.com/Yao-Shun-Ya/Xian-Er-Yi-Jian-Web">网站仓库</a></p>
</div>

---

睑板腺红外影像分割、定量分析与临床辅助诊疗平台。当前版本 **v2.3.0**，交付两端：

- **Windows WebUI**：Gradio 桌面端（`start_webui.bat`）
- **Linux CLI**：服务器/命令行端（`start_cli.sh`，仅训练与评估）

macOS / MPS 已从代码、打包和文档中移除。

## 快速开始（Windows）

在仓库根目录双击或命令行运行：

```bat
install_dependencies.bat
run_self_check.bat
start_webui.bat
start_training_cli.bat
```

- `install_dependencies.bat`：检测 `venv` 与依赖；已安装则直接跳过，不会重复下载。
- `run_self_check.bat`：6 例端到端自检（数据→划分→训练→评估→HTML 报告），预期输出 `[PASS]`。
- `start_webui.bat`：启动 WebUI，默认 `http://127.0.0.1:7860`。
- `start_training_cli.bat`：进入交互式训练 CLI。

所有启动器均为 **ASCII + CRLF**，避免中文/换行导致的 `cmd.exe` 解析失败。可用 `MGD_PYTHON` 指定已有 Python：

```bat
set MGD_PYTHON=C:\path\to\python.exe
start_webui.bat
```

## 快速开始（Linux）

```bash
bash install.sh          # 创建 venv，安装 CUDA 依赖；--cpu 可切换 CPU 轮子
bash run_self_check.sh   # 6 例端到端自检
bash start_cli.sh        # 交互式训练 CLI
```

Linux 交付承诺为 **CLI only**，不安装 Gradio。

## 数据、划分与泄漏审计

公开 MGD-1K 经内容级去重后为 **999 例**配对样本，来自 **338 个患者编号**。按患者独立划分（`data/dataset_splits.json`）：

| Split | 图像 | 患者 | 文件清单签名 | 实际派分签名 |
|---|---:|---:|---|---|
| Train | 694 | 239 | `6bdacfb0…` | `7a6fd581…` |
| Val | 153 | 50 | 同上 | 同上 |
| Test | 152 | 49 | 同上 | 同上 |

泄漏审计（重要）：

- 官方 200 例 Meiboscore 测试清单来自 **166 名患者**；
- 其余 800 例中有 **326 例**来自同一批患者；
- 即使患者级训练集，仍含 **116/166** 名官方测试患者；
- clean-official 子集只有 **60 图 / 50 患者**。

因此 **不要直接把官方 200 例当作患者独立测试集**。本仓库把“文件清单签名”和“实际派分签名”分开记录，避免同名文件被重新派分而无法追溯。

## 2026-11 患者级测试结果

9 小时单次 patient-level holdout，测试集 152 图/49 患者；**不是 5 折，不是最终投稿数字**。

| 模型 | Dice | clDice | HD95 (px) |
|---|---:|---:|---:|
| AttentionUNet (15 ep) | **0.7771** | 0.8349 | 118.44 |
| MGD-Former (45 ep, ADE20K) | 0.7749 | 0.8156 | **68.05** |
| MGD-PriorNet (≈12 ep) | 0.7640 | 0.8539 | 154.03 |
| UNet (30 ep) | 0.7119 | **0.8531** | 148.03 |

配对 Wilcoxon（MGD-Former 减基线）：

- vs AttentionUNet：**-0.0022**，CI 跨 0，p=0.016 → 只能报告“可比”，不能宣称胜出；
- vs UNet：+0.063，p=5.4e-20 → 明确更好；
- vs MGD-PriorNet：+0.011，p=0.714 → 统计上不显著，1.5M 参数很有性价比。

其他关键结果与负面结论：

- **WRCANet 不进入排名**：8 GB 卡上单 epoch 约 38 min，预算内只完成 1 epoch（val Dice 0.006）。
- **MTE**：Dice 0.7736 vs 单分块 0.7749（未提高），但 HD95 从 68.05 降到 61.85；定位为边界校正，不是涨点方法。
- **8× TTA**：Dice 0.7808，成本约 8×。
- **不确定性**：多分块方差 AUROC 0.639、熵 0.483、TTA8 1.000（受标签–信号耦合影响，不能直接当 triage 证据）。
- **临床 Meiboscore**：accuracy 0.164、MAE 1.105、QWK 0.131。高 Dice 没有转化为临床分级一致性，R_loss/Meiboscore 链条需要重新标定。
- **上/下睑**：本队列 lower 0.7905 > upper 0.7584，与“下睑更难”的常见假设相反；必须分开报告。

## 交付包与代码一致性

`xianeryijian/public/file/Windows.腺而易见WebUI` 与 `Linux.腺而易见` 的 `mgd` 代码已与主运行树 **逐文件 SHA-256 一致**（95 个代码文件，0 diff）。两个包内含：

- 完整运行代码与 `mgd/packaging`；
- 6 例真实演示数据（3 上睑 + 3 下睑），用于自检/CLI/WebUI 冒烟；
- 英文启动器（Windows：`install_dependencies.bat` / `start_webui.bat` / `start_training_cli.bat` / `run_self_check.bat`；Linux：`install.sh` / `start_cli.sh` / `run_self_check.sh`）。

两个包**不包含** 999 例完整数据和 99 MB 演示权重；完整数据/权重在项目主页下载区单独提供。包内 `models_weights` 为空时，WebUI 仍可启动并训练；若要直接推理，请把权重放入 `models_weights/<arch>/` 并刷新。

## 已验证

- `python -m pytest mgd/tests -q`：**146 passed**
- `run_self_check.bat`：**PASS**（6 例端到端）
- `start_webui.bat`：**HTTP 200**（本机 127.0.0.1:7860）
- `start_training_cli.bat`：可进入菜单并正常退出
- 前端 `npm run build`：通过
- 中英文论文 PDF：0 Overfull / 0 undefined refs

Linux CLI 的 `.sh` 入口与 Windows CLI 共用同一 `mgd.verify_pipeline` / `mgd.adapters.cli.interactive` 路径；当前 Windows 主机未安装 WSL，因此 Linux 脚本未在本机执行，需在 Linux 主机上做一次 `bash run_self_check.sh` 验收。

## 目录

```
mgd/                核心引擎、模型、训练、推理、CLI、Gradio
mgd/packaging/      Windows/Linux 安装与启动脚本
tools/              9h 实验编排、评估、结果注入、报告
docs/engineering/   冻结基线、架构决策、已知限制
data/               999 例数据、患者级划分、元数据
models_weights/     权重目录（演示权重需单独下载）
outputs_mgd/        训练 run、评估、报告
```

## 说明

本仓库优先保证“可复现、可审计”，不把单一 Dice 或单一架构排名当卖点。所有结果都带数据划分、签名、epoch 预算与推理参数；负面结果同样保留。投稿前仍需补 5 折、epoch-matched 泄漏对照与临床分级校准。

# MGD 质量守卫（堆料式 Quality Gates）

> 定位：**独立于业务代码的守卫系统**，用"尽可能多的检查项、多重校验、多维度把关"换可靠性——
> 宁可冗余，不遗漏。核心纪律全部落成可执行断言：NaN 纪律、指标可追溯、分层（core 无 UI）、
> 向后兼容、可选依赖优雅降级。mgd/ 与 docs/ 一行不改，全部检查只读。

## 运行

```bash
cd <bundle>/MGD_Project   # quality/ 所在仓库根（示例为相对路径，搬移免疫）
./venv/Scripts/python.exe quality/run_gates.py            # 全量 18 道
./venv/Scripts/python.exe quality/run_gates.py --list     # 只列门禁清单
./venv/Scripts/python.exe quality/run_gates.py --only g03,g04   # 运行子集
./venv/Scripts/python.exe quality/run_gates.py --json     # 尾部追加机器可读 JSON
```

- 每道门禁在**独立子进程**运行（单门禁崩溃/挂起不影响其余），单门禁超时 90 秒；
- 汇总表格输出（门禁 / 检查数 / 耗时 / 结果），尾部 `PASS x/18`；
- **任一 FAIL → 退出码 1**；结果 JSON 始终写入 `quality/reports/last_run.json`。

## 18 道门禁一句话清单

| 门禁 | 一句话 |
|---|---|
| g01_syntax_compile | `compileall -q mgd` 全绿（可编译底线）。 |
| g02_todo_scan | mgd/（排除 tests）无 TODO/FIXME/XXX/HACK/pass#待/NotImplementedError。 |
| g03_metric_registry_integrity | 指标注册表 ≥16 项，definition/unit/source 齐全且 source 在三类词表内，key 小写蛇形，describe() 字段齐全。 |
| g04_nan_discipline | NaN 纪律静态面：核心子包无裸吞异常；morphometry/quality 有 NaN 证据；json 落盘点 allow_nan=False 或就近 sanitize。 |
| g05_core_no_ui | core 顶层无 gradio/pydicom/highdicom；matplotlib 仅 plotting/visualize；core 不反向 import adapters；v2.4.0 新子包顶层无 torch。 |
| g06_api_surface | 向后兼容面快照：config/inference/clinical/stats/report/training/registry/errors 逐符号存在；InferenceOptions 12 字段、INSTANCE_COLUMNS 9 列、meiboscore 六点行为锁定。 |
| g07_test_speed | v2.4.0 增量 13 个测试文件一条 pytest 命令全绿 + 墙钟 <60s。 |
| g08_import_graph | AST 导入图方向断言：core↛adapters、新子包顶层无 torch、interop 顶层无 dicom 库、adapters→core 正向边存在、reporting↛gradio。 |
| g09_specs_alignment | specs（唯一参数源）↔ TrainConfig 字段/默认值/默认 YAML/SCHEMA 四方一致。 |
| g10_docs_sync | 版本唯一源、pyproject 动态版本+dicom 组+4 新脚本、README/冻结基线/已知局限 5 关键词/ADR-002..004 与 v2.4.0 对齐（文档未就绪时 FAIL 属预期）。 |
| g11_forbidden_paths | v2.4.0 新模块字符串字面量无 models_weights/outputs_mgd/_transfer/data/images（路径必须走 Settings）。 |
| g12_optional_dep_discipline | interop 顶层无 dicom 库；缺依赖时 export_seg/export_sr 真实调用抛 InteropDependencyError 且含安装命令。 |
| g13_cli_surface | 6 个 CLI `--help` 退出码 0 且 <10s；未知 action 退出码非 0。 |
| g14_settings_fields | Settings 新字段存在、calibration_dir 跟随 output_dir、from_env 识别 4 个新环境变量、非法 scale 拒绝。 |
| g15_report_blocks | 报告旧调用无 5 个新区块且无 N/A 占位；全参数调用五区块（血缘/质控/分级/分型/解释性）齐全。 |
| g16_grading_crosscheck | 冗余交叉验证：Arita 与 clinical 逐点一致、Pult 单调且 [0,4]、30% → Arita 1 / Pult 2、grade_table 3 行。 |
| g17_instance_columns | INSTANCE_COLUMNS 精确 9 列冻结；atlas.typology 列常量一致；longitudinal.REQUIRED_COLUMNS ⊆ 冻结清单。 |
| g18_json_strictness | 运行时三连：审计日志/图谱索引产物过严格 JSON 解析（parse_constant 拒 NaN）；校准参数 NaN 在写端被 ValueError 拦截。 |

## 架构

```
quality/
├── run_gates.py          # 总入口：子进程隔离运行 + 表格汇总 + last_run.json
├── gates/
│   ├── _common.py        # 共享工具：PROJECT_ROOT / AST 扫描 / Gate 累积器 / 结果输出协议
│   └── gNN_*.py          # 18 道门禁，各暴露 run() -> dict
├── checklists/           # 交付清单（人工复核项）
└── reports/              # last_run.json（每次运行覆盖写）
```

门禁结果协议（`run() -> dict`）：

```python
{"ok": bool, "name": str, "details": str(≤2000字), "checks": int, "failed": [str, ...]}
```

门禁子进程向 stdout 打印一行 `GATE_RESULT_JSON: {...}`，由 run_gates.py 解析；
解析不到（崩溃）或超时（>90s）一律按 FAIL 归档，不会静默通过。

## 如何加新门禁（gNN 模板）

1. 复制任意 `gates/gNN_*.py` 为 `g19_your_gate.py`（编号递增、小写下划线命名）；
2. 实现 `run() -> dict`，用 `quality/gates/_common.py` 的 `Gate` 累积器逐条 `check(ok, label, fix)`——
   失败信息必须**可操作**（指出文件/行/修法）；
3. 文件尾部挂统一出口：

```python
if __name__ == "__main__":
    _common.run_as_main(run)
```

4. 保持单文件 ≤150 行、不 import pytest 之外的测试私有物、不改 mgd/ 与 docs/；
5. `./venv/Scripts/python.exe quality/run_gates.py --list` 确认被发现，再全量跑一遍。

## 与 CI 的关系

- 本体系定位**本地守卫**：交付前在本机（Windows Git Bash + venv）跑通 18/18 再出包；
- CI 可选接入：`quality/run_gates.py` 的退出码（0/1/2）与 `quality/reports/last_run.json`
  可直接作为流水线门禁信号；`--only` 支持按分支阶段挑子集（如 PR 只跑 g01–g06 快面）；
- 门禁与 pytest 互补：pytest 验"行为对"，本守卫验"结构纪律 + 兼容面 + 速度预算 +
  文档同步"这些 pytest 不方便覆盖的维度。两边都绿才算交付。

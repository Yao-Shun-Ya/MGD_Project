# v2.4.0 交付清单（任务书 §10 十条 + 质量守卫自有 5 条）

> 使用方式：交付前逐项勾选。前 10 条对应任务书 §10 交付清单；后 5 条是质量守卫体系
> 自有的复核项。机器可验证的部分由 `./venv/Scripts/python.exe quality/run_gates.py`
> 自动把关（括号内为对应门禁），人工判断项请逐条确认后再打勾。

## 任务书 §10 十条

- [ ] 1. `mgd.__version__` 为 X.Y.Z 形式且与全部文档声称的当前版本一致（`README.md` /
       `FREEZE_BASELINE.md` 均含该版本、无更高版本号、无"现为旧版本"陈述）。
       pyproject 动态读取，版本号唯一源未分叉。（g10 + `test_version_matches_docs`）
- [ ] 2. 诊断中心级增量子包（quantification/grading/atlas/qc/conformal/longitudinal/
       interop/governance/feedback/provenance）全部落地，且顶层无 torch（g05/g08）。
- [ ] 3. 指标注册表 ≥16 项，每项 definition/unit/source 可追溯，source 只落在
       Arita 2008 / Pult & Nichols 2016 / engine-defined 词表内。（g03）
- [ ] 4. NaN 纪律全链成立：空输入返回 NaN 或显式拒绝、无裸吞异常、JSON 落盘
       allow_nan=False / sanitize 豁免、运行时三连（审计/图谱/校准）通过。（g04/g18）
- [ ] 5. 向后兼容面无破坏：config/inference/clinical/stats/report/training/registry/
       errors 全部旧符号存在，InferenceOptions 12 字段与 INSTANCE_COLUMNS 9 列冻结，
       meiboscore 六点行为快照锁定。（g06/g17）
- [ ] 6. 分层纪律成立：core 无 UI、无反向依赖 adapters、matplotlib 只在
       plotting/visualize、可选依赖（pydicom/highdicom）延迟导入并抛
       InteropDependencyError。（g05/g08/g12）
- [ ] 7. specs（唯一参数源）↔ TrainConfig ↔ 默认 YAML ↔ SCHEMA 四方一致，
       v2.4.0 四个新键（data.physical_scale_mm_per_px / infer.qc_policy /
       infer.explain_method / eval.conformal_alpha）全部登记。（g09）
- [ ] 8. v2.4.0 增量测试套件全绿且墙钟 <60s；6 个 CLI `--help` 秒级响应、
       未知 action 显式拒绝。（g07/g13）
- [ ] 9. 文档同步：README/冻结基线/已知局限（≥5 条诚实条目：概率伪彩、工程默认、
       engine-defined、像素单位、跨设备泛化）/ADR-002..004 与 v2.4.0 对齐；
       pyproject 含 dicom optional group 与 mgd-qc/mgd-atlas/mgd-followup/mgd-report
       四个新脚本。（g10）
- [ ] 10. 新模块零硬编码路径（models_weights/outputs_mgd/_transfer/data/images 不出现
       在字符串字面量），路径一律经 Settings 属性访问，可用 MGD_HOME 重定向测试。（g11/g14）

## 质量守卫自有五条

- [ ] 11. `quality/run_gates.py` 全量 18/18 PASS（或 g10 因文档排期未就绪并已记录
       待办），`quality/reports/last_run.json` 已随交付归档。（g01–g18）
- [ ] 12. 每道门禁的失败信息可操作（文件/行/修法），抽查 3 条失败路径确认
       不是"只说 FAIL 不说怎么修"。（gates/*.py）
- [ ] 13. 冗余校验未放松：g16 分级双通道逐点对照（Arita 101 点 + Pult 401 点）、
       g06 行为快照六点、g17 三消费端逐列核对全部仍在执行——禁止为了绿灯删断言。
- [ ] 14. 门禁自身健康：单门禁 ≤150 行、不 import pytest 之外的测试私有物、
       `--list` / `--only` / `--json` 三个入口行为正确；本次运行的门禁总检查数
       与上一轮相比没有无故缩水。
- [ ] 15. 报告产物抽查：`build_report` 全参数 HTML 五区块顺序正确（Provenance →
       质控 → 分级 → 定量指标 → 形态分型 → 解释性 → 图像），未传参区块无空表头、
       无 N/A 占位。（g15）

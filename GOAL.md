# 当前唯一目标

## 交付约定
- 目标：**让 MathModeler 在模型形式化阶段实现可靠的符号闭包修复**——使 `unknown variable/parameter` 失败能够产生**精确的缺失符号反馈**，并在下一次尝试中**针对性修正**，而不是重新盲生成整个模型。
- 交付物与位置：
  - `src/mathmodel/domain/math_model.py`：结构化符号闭包检查（`SymbolClosureIssue` / `closure_issues` / `describe_closure_issue` / `parse_closure_issues`）与校验器
  - `src/mathmodel/agents/math_modeler.py`：把缺失符号清单转成针对性修复提示并回灌下一次尝试
  - `tests/test_autopilot.py`：`TestSymbolClosureRepair`、`TestR05FalsePassRegression`
  - `tests/fixtures/r05_false_pass.json`：r05 永久 negative fixture
- 范围：只修符号闭包这条路径。**不**改 `MAX_REPAIR_ATTEMPTS`、**不**换模型、**不**重构 pipeline、**不**再升级 verifier、**不**处理 solve 沙箱的其他失败、**不**处理论文质量与性能。
- 明确不做：不新增 `SymbolAgent`、新 subsystem 或复杂 AST 架构。原方案（现有 schema 校验已能发现未知符号）保留；最小修改是把已有 validator 的缺失符号清单**结构化地传回**下一轮 MathModeler。
- 必须保留与约束：保留 `AGENTS.md` 全部既有要求；保留 r05 作为永久 negative fixture（旧门禁 PASS → 独立校验 WRONG → 新门禁 FAIL）。

## 验收
| 标准 | 应达到的结果 | 检查方法 | 实际证据与状态 |
| --- | --- | --- | --- |
| S1 | 已知含未声明符号的 model_spec 必须 FAIL | 构造含未声明符号的 payload 调 `MathematicalModel.model_validate` | **通过**。`test_undeclared_symbols_fail_validation`：抛错且消息含 `unknown variable v_y_missing` / `unknown parameter p_M_missing` |
| S2 | FAIL 返回具体 unknown variable / parameter 列表 | 解析失败消息为结构化 issue | **通过**。`test_failure_message_parses_back_into_the_exact_symbols`、`test_describe_and_parse_round_trip`：`parse_closure_issues` 精确还原 variable/parameter/dependency 三类符号与所在方程；`test_duplicate_and_colliding_symbols_are_reported` 覆盖重复符号与变量/参数同名冲突 |
| S3 | repair prompt 明确包含这些列表 | 用假 router 捕获第二次尝试的 prompt | **通过**。`test_retry_prompt_contains_the_exact_missing_symbols`：第二次 prompt 含 `v_y_missing`、`p_M_missing` 与 `SYMBOL CLOSURE FAILURE` |
| S4 | 下一轮不是无反馈重生成 | 检查 prompt 的针对性指令 | **通过**。同一测试断言 prompt 含 “Keep every other part of the model as it is” 与 “do not rebuild the model from scratch”；`test_first_attempt_has_no_corrective_block` 确认首次尝试不带该块；`test_a_non_closure_error_still_gets_generic_feedback` 确认非闭包错误仍走通用反馈 |
| S5 | 修复后的 model_spec 满足 symbol closure | 补齐声明后重新校验 | **通过**。`test_declaring_the_missing_symbols_clears_the_failure`：补齐后 `closure_issues(model) == []` |
| S6 | 原有 799 tests 不回归 | 全量 `pytest` | **通过**。**812 passed, 1 skipped**（799 + 9 闭包 + 4 r05 fixture，无回归） |
| S7 | 用真实 D 题重跑时，model 阶段不再因同类符号闭包错误耗尽 3 次预算 | 真实 D 题全新运行并观察 model 阶段 | **通过（附限制）**。3 次全新运行（`runs/reliability/c01–c03`）**model 阶段 3/3 completed，0 次 `MathModeler failed`**；修复前 12 次运行中有 1 次 model 阶段彻底失败（p02）。每次运行仍会在 attempt 1 出现闭包错误（3/3），靠回灌清单修好，其中 c03 用满 3 次尝试才通过——**反馈能修好，但尚未做到首次即正确**。日志：`.tmp/closure_verify.log` |

## 附加：r05 永久 negative fixture
`tests/fixtures/r05_false_pass.json` 保存了真实 r05 运行的验证报告（12 项检查、`input_records=150`、`overall=PASS`）与独立复算的 ground truth（result2 残余冲突 297、撤销 0、result3 0 行）。`TestR05FalsePassRegression` 断言：
1. fixture 记录的 ground truth 是 `WRONG`；
2. **去掉 objectives 契约时该报告仍会 PASS**（锁定回归）；
3. 当前门禁判 FAIL；
4. fixture 里每一项检查都是输入数据属性，且唯一跑过的 `constraint_satisfaction` 只比对了 result1（检测），从不判定答案——记录它当初为何漏判。

## 上一目标的完成记录
上一目标（把本项目实现成可运行的 CUMCM Autopilot，A1–A15）**已完成**，验收与证据见 `DONE.md`「2026-09-24 — 把 MathModel AI 实现成可运行的 CUMCM 数学建模 Autopilot」。该目标下的 24 项已修缺陷、可靠性测量（12 次运行正确收敛 0/12）与 #25 的定位过程保留在下方历史小节中。

## 历史：上一目标的交付约定（已完成）
- 目标：把本项目（MathModel AI，`D:\deepseek工作区`）实现成真正可运行的 CUMCM 数学建模 Autopilot。用户上传赛题与附件后，系统沿一条连续工作流完成赛题理解、建模、代码求解、结果验证、图表生成、论文写作与最终打包，产出可检查的论文 PDF + 支撑材料包。
- 交付物与位置：
  - 运行入口：`src/mathmodel/autopilot/`（`CUMCMAutopilot`）与 `src/mathmodel/cli.py`（`mathmodel run / resume / status`）
  - 单次运行产物：`runs/<run_id>/output/paper.pdf`、`runs/<run_id>/output/support/{source_code,data_or_processed_data,figures,tables,necessary_supporting_files}`、`runs/<run_id>/output/manifest.json`
  - 运行过程与状态：`runs/<run_id>/pipeline_state.json` 与 `runs/<run_id>/artifacts/`
- 范围：
  - 上传 → 动态澄清 → 问题理解 → 歧义/假设/数据模式登记 → 候选模型 → 模型选择审计 → 数学建模 → 求解代码生成 → 真实沙箱执行 → 确定性验证 + 独立复算 → 验证门禁与回修 → 证据/图表/表格 → 论文大纲与分节写作 → 提交审查 → PDF → 支撑材料包 → 最终一致性检查。
  - 验证未通过不得进入论文；验证失败回到建模/求解修复。
  - 真实赛题、真实附件、真实模型、真实代码执行、真实验证、完整论文、可打开 PDF、完整支撑材料包。
- 明确不做：不新增路线图/阶段/子项目/长期 TODO；不引入新的 agent 体系、数据库或服务层；不做 UI 控制台化改造；不以 mock/demo/截图/单元测试/“理论可行”替代真实产物。
- 必须保留与约束：
  - 保留 `AGENTS.md` 全部既有项目要求与执行规则。
  - 保留既有 `ProblemState`/domain schema/`EvidenceStore`/`PaperIR`/`SubmissionCheckAgent`/沙箱后端等既有实现与其测试（基线 660 passed, 1 skipped）。
  - 仅实现参考实现中【原项目明确做法】的部分：Autopilot 连续性、pipeline 持久化与暂停/恢复、求解必须有验证、验证失败阻断论文、回修循环；`ambiguity_register`/`assumption_ledger`/`data_schema`/`model_spec`/`model_selection_audit`；图→源数据→结果→结论可追溯；轻量 skill/harness 而非新 agent 体系；chat-first 动态澄清、不重复询问已知信息、不在信息缺失时推进、渐进式论文生成、仅在关键节点暂停、始终显示当前阶段/进度/阻塞；verified results → Paper Context Package → 大纲 → 分节写作 → 组装；数据状态区分 `formal`/`demo`/`blocked`；运行/暂存/输出/manifest/包分离。

## 历史：上一目标的验收（A1–A15，全部通过）
| 标准 | 应达到的结果 | 检查方法 | 实际证据与状态 |
| --- | --- | --- | --- |
| A1 | 上传赛题与附件后能启动一条连续工作流，无需手工分步调用各 agent | 用真实赛题文件调用 `mathmodel run`，观察各阶段自动依次执行 | **通过**。`CUMCMAutopilot.run()` 单次调用连续跑完 18 个阶段（intake→final_check），无任何手工分步调用各 agent；`src/mathmodel/cli.py` 提供 `run`/`resume`/`status`。最终交付件由 `resume()` 续跑产出（设计内的暂停/恢复路径，非手工分步）。证据：`runs/cumcm-2026-dti-e2e/pipeline_state.json` 全部 18 阶段 `completed`；`.tmp/final_run.log` |
| A2 | 系统真实读懂赛题与附件（提取正文、真实读取附件数据表） | 检查 `artifacts/intake/problem_context.json` 的正文长度、附件行列与解析状态 | **通过**。`problem_text` 2176 字符（真实取自 `D题.pdf`）；6 个附件角色识别正确：`D题.pdf`=problem_statement、`附件1.xlsx`=data（150 行 × 5 列，表头 `用频装备编号/频段区间/时间区间/间隔时长/使用次数`）、`result1–4.xlsx`=template（各自表头被真实解析）；导出 1 个处理后数据文件。证据：`artifacts/intake/problem_context.json`、`artifacts/intake/files.json`、`artifacts/data/data_schema.json` |
| A3 | 信息不足时提出动态澄清问题；信息足够时不打扰用户 | 检查 `clarify` 阶段产出的问题列表与是否阻塞 | **通过**。`clarify` 阶段产出 1 个澄清问题（赛题未明确处），答毕后 `all resolved` 继续推进，未产生多余追问。证据：`pipeline_state.json` 中 `clarify: completed 1 question(s), all resolved` |
| A4 | 产出结构化问题理解（子问题、事实、歧义、隐式条件） | 检查 `artifacts/analysis/analysis.json` | **通过**。4 个子问题、25 条 evidence、5 条 ambiguity，均来自真实赛题正文。证据：`artifacts/analysis/analysis.json` |
| A5 | 产出 `ambiguity_register`、`assumption_ledger`、`data_schema`、`model_selection_audit` | 检查对应 JSON 产物内容非空且来自真实分析 | **通过**。`ambiguity_register.json` 2027B、`assumption_ledger.json` 1538B、`data_schema.json` 2612B、`model_selection_audit.json` 9032B，均由真实分析与真实数据生成。证据：`artifacts/analysis/`、`artifacts/data/`、`artifacts/models/` |
| A6 | 自动生成候选数学模型并完成可审计的选择 | 检查 `artifacts/models/candidates.json` 与 `model_selection_audit.json` | **通过**。5 个候选模型，选择过程可审计（含评分与理由）。证据：`artifacts/models/candidates.json`、`artifacts/models/model_selection_audit.json` |
| A7 | 产出形式化数学模型（变量/参数/目标/约束/方程） | 检查 `artifacts/model/math_model.json` | **通过**。`时频冲突检测与消解模型`：23 变量、22 约束、4 方程，含目标函数与参数表。证据：`artifacts/model/math_model.json`（另有 `math_model.repaired.json` 记录一次由验证反馈触发的模型修复） |
| A8 | 自动生成求解代码并在真实沙箱中真实执行 | 检查 `artifacts/code/*/solver.py` 与 `artifacts/solve/*/outcome*.json` 的执行状态/退出码/耗时 | **通过**。`artifacts/code/attempt1/solver.py` 17504 字节；`artifacts/solve/outcome1.json`：`status=success`、`exit_code=0`、`runtime_seconds=4.881`、镜像 `mathmodel-ai-autopilot:latest`，产出 4 个结果文件。证据：`artifacts/code/attempt1/`、`artifacts/solve/` |
| A9 | 产出赛题要求的全部结果文件，且结构符合模板 | 检查 `artifacts/solve/*/` 下 result 文件与 `verify` 报告中的结构检查 | **通过**。result1–4.xlsx 全部产出，**表头与附件2模板逐列完全一致**（`output_structure_matches_template` PASS）；按计划编号建表的 result2/result4 均为 150 行，覆盖每个计划（`per_plan_table_covers_every_plan` PASS）。行数：result1 297、result2 150、result3 32809、result4 150。证据：`output/support/data_or_processed_data/`、`artifacts/verify/report1.json` |
| A10 | 求解结果经过独立验证：确定性检查 + 不看求解代码的独立复算 | 检查 `artifacts/verify/report*.json` 中 `independent` 类检查与 `artifacts/verify/*/verifier.py` | **通过**。`report1.json` overall=PASS，共 **49 项检查**：execution 1、outputs 4、data 1、correctness 1、independent 42（含 `independent_read_the_input`、`independent_independent_recomputation`、`independent_result1_pair_match`、`independent_constraint_satisfaction`、`independent_constraint_1..34`、`independent_feasibility`、`independent_small_case_check`、`independent_baseline_comparison`、`independent_statistics_agree`）。独立验证程序在沙箱内生成并运行，**从不接触求解代码**。证据：`artifacts/verify/report1.json`、`artifacts/verify/attempt1/verifier1.py`、`verifier2.py` |
| A11 | 验证失败时自动回到求解修复；验证未通过时不得进入论文 | 检查回修循环日志与验证门禁分支 | **通过**。`report2.json`/`report3.json` 为 FAIL 并触发回修；当独立复算指出**模型本身**错误时，回修循环回到 `MathModeler` 重新形式化（产出 `math_model.repaired.json`），而不只是重生成代码——这是“回到建模/求解修复”的实现。验证未通过的运行一律 `blocked`，不产出新论文（实测多次拦下：237 vs 297、39 处残余冲突、残余冲突 16/121、空转验证器 `input_records=0`）。证据：`artifacts/verify/report2.json`、`report3.json`、`artifacts/model/math_model.repaired.json`、`.tmp/resume_case*.log` |
| A12 | 论文由 verified results 生成，数值与验证结果一致，无占位符 | 检查 `artifacts/paper/paper_ir.json`、`final_check.json` 与 PDF 正文 | **通过**。`artifacts/audit/final_check.json` = `{"issues": [], "warnings": []}`；摘要与正文数值全部可追溯到已验证统计量（297 对冲突、0 残余冲突、32809 个新增 C 类计划等）。“不可追溯数字”检查扫描全部整数与小数，命中即**只重写含该数字的章节/摘要**（最多 2 轮），实测拦下并修正了编造的“A、B、C 类各 50 个”（真实 20/40/90）。证据：`artifacts/paper/paper_ir.json`、`artifacts/audit/final_check.json`、`artifacts/paper/paper.tex` |
| A13 | 图表由真实结果生成且可追溯到来源数据/执行 | 检查 `FigureRecord.source_execution_ids`、`TableCell.source_id` 与 `artifacts/figures/` | **通过**。5 张图，每张均带 `source_execution_ids=['RUN-1d3eaad0']`（真实求解执行 id）；5 张表由已验证统计量构建。证据：`artifacts/figures/figures.json`、`artifacts/tables/`、`output/support/figures/fig1..5.png`、`output/support/tables/TAB-001..005.csv` |
| A14 | 产出可打开的论文 PDF（中文正常渲染） | 打开 `output/paper.pdf` 并检查页数与正文 | **通过**。`output/paper.pdf` 507310 字节，文件头 `%PDF-1.5`，**25 页**，A4（595.28×841.89 pt），内嵌 CJK 字体 `FandolSong-Regular/Bold`、`FandolKai-Regular`；8 个 CUMCM 标准章节（问题重述/问题分析/模型假设/符号说明/模型建立与求解/结果分析/模型检验/模型评价与推广）。注：中文字形无 ToUnicode 映射，`pypdf` 抽取文本为空属字体特性而非渲染失败——pypdfium2 渲染 25 页成功且字体表可证。证据：`runs/cumcm-2026-dti-e2e/output/paper.pdf` |
| A15 | 产出完整支撑材料包与 manifest，且 PDF/结果/图表/支撑材料相互一致 | 检查 `output/support/**` 与 `output/manifest.json`，并核对最终一致性检查结果 | **通过**。`output/manifest.json`：`verification.overall=PASS`（49 项全 PASS）、`data_status=formal`、`counts={source_code:3, data:5, figures:5, tables:5, extra:7}`、`result_files` 四项齐全、`submission_check=ready_to_submit`；`final_check` 无 issue 无 warning。另有**手写独立校验器**（`.tmp/verify_deliverable.py`，不 import 任何 autopilot 代码）复核交付结果文件：**ALL INDEPENDENT CHECKS PASSED**。证据：`runs/cumcm-2026-dti-e2e/output/`、`.tmp/verify_deliverable.json`、`.tmp/independent_conflicts.json` |

### 独立复算（不依赖任何 autopilot 代码，A10/A15 的人级证据）
- `.tmp/independent_check.py`：按赛题原文与附录语义独立实现，得 150 个计划（A=20、B=40、C=90）、**冲突对 297 对**（A-B 21、A-C 66、B-C 181、B-B 10、C-C 19），并复现附录 A001 示例区间 35/100/165。
- `.tmp/verify_deliverable.py` 独立校验**交付的结果文件本身**：result1 冲突对集合与独立复算**完全一致**（297 对，无缺无多）；result2、result4 重建后**残余冲突 0**、每计划只改一个参数、平移未越界；result3 新增计划与 result2 方案及彼此冲突 **0**；四个文件表头与附件2模板一致。
- 该 297 与运行中求解程序自报值一致；此前本地网关某次运行自报 237，经比对为**错误答案**并被验证门禁拦下——证明独立验证层不可省略。

## 当前状态
- 交付状态：**已完成**（S1–S7 全部通过，证据见上表）
- 当前工作目录：`D:\deepseek工作区`
- 已完成与证据：
  - `math_model.py` 新增结构化符号闭包检查：`SymbolClosureIssue` / `closure_issues` / `describe_closure_issue` / `parse_closure_issues`；校验器改为复用同一份 issue 列表，错误消息格式保持不变。
  - `math_modeler.py` 在 schema 失败时先尝试把消息解析为闭包 issue；成功则生成**精确修复清单**（缺失 variable / parameter / dependency、重复符号、符号冲突 + 两条可选修法 + “其余部分保持不变”），失败才回退到通用反馈。
  - 测试 **812 passed, 1 skipped**（新增 9 项闭包测试 + 4 项 r05 fixture 测试，无回归）。
  - 真实 D 题 3 次全新运行：model 阶段 **3/3 completed**，0 次 `MathModeler failed`（修复前 12 次运行中有 1 次该阶段彻底失败）。其中 `c03` 全流程 `completed` + `VERIFICATION: PASS`，并经手写独立校验器复核 **ALL INDEPENDENT CHECKS PASSED**（result1 297 对冲突与独立复算一致、result2/result4 残余冲突 0、result3 冲突 0）。
- 唯一下一步：无
- 关键决定与依据：
  - **不新增 `SymbolAgent`/subsystem/AST**：现有 schema 校验已经能发现未知符号，缺的只是把清单结构化传回，因此只补这一条链路。
  - pydantic 会把 `model_validator` 抛出的 `ValueError` 包装成 `ValidationError` 并丢弃异常对象，消息是唯一存活的通道；因此 `describe_closure_issue` 与 `parse_closure_issues` 成对定义在同一模块并做往返测试，而不是靠散落的字符串匹配。
  - 解析时必须切掉 pydantic 追加的 ` [type=value_error, input_value=...]` 尾巴，否则最后一个符号会被粘上元数据。
- 阻塞与解除条件：无
- 已知限制：
  - 反馈能**修好**闭包错误，但尚未做到首次即正确：3 次真实运行每次 attempt 1 仍出现闭包错误（3/3），其中 1 次用满 3 次尝试才通过。
  - 闭包检查只覆盖方程显式声明的 `variable_ids` / `parameter_ids` / `dependencies`，不解析 `expression` 文本里的自由符号——超出本次范围。
  - 非闭包的 schema 错误（例如 `variable_type` 枚举写错）仍走通用反馈，不在本次范围内。
- 审查结果：自审完成；S1–S7 逐条有测试或真实运行证据

## 历史：上一目标的当前状态（已完成）
- 交付状态：**已完成**（A1–A15 全部通过，证据见上表）
- 当前工作目录：`D:\deepseek工作区`
- 已完成与证据：
  - 真实赛题 **2026 高教社杯 D 题（时频冲突检测与消解）** 全流程跑通：`runs/cumcm-2026-dti-e2e`，`STATUS: completed`、`VERIFICATION: PASS`、`final_check: all consistency checks passed`。
  - 交付物：`runs/cumcm-2026-dti-e2e/output/paper.pdf`（25 页 A4，中文正常）、`output/support/{source_code,data_or_processed_data,figures,tables,necessary_supporting_files}`、`output/manifest.json`。
  - 结果：150 个计划中检出 **297 对时频冲突**（与独立复算完全一致），四个结果文件无残余冲突、表头与模板一致、每计划只改一个参数且平移未越界。
  - 测试：`pytest` **799 passed, 1 skipped**。
- 唯一下一步：无
- 关键决定与依据：
  - 求解代码生成为**单个自包含 Python 文件**：`DockerSandboxBackend` 只执行一个 `code.py`。
  - 附件在宿主机转成 CSV 后作为文本输入传入沙箱：后端只接受文本输入。
  - 图表在宿主机用 matplotlib 从已验证统计量渲染：可追溯且规避沙箱字体/缓存问题。
  - 验证分两层：宿主机确定性检查 + 沙箱内运行的对抗式独立验证程序（不看求解代码）。
  - `SimpleLPCompiler`/`solve_lp_scipy` 无法处理该组合优化赛题，真实路径是 LLM 代码生成 + 沙箱执行。
  - DeepSeek 推理模型会把整个输出预算耗在 reasoning 上并返回空 JSON（实测 16384/32768 completion tokens 全部计入 reasoning，`finish_reason=length`，content 为空），因此结构化生成固定使用标准模型，推理模型仅用于自由文本生成。
  - DeepSeek 账户余额耗尽（402 Insufficient Balance）后，改用环境内既有 OpenAI 兼容网关继续跑真实赛题。**运行时配置（已实测确认）**：`OPENAI_BASE_URL=http://127.0.0.1:7863/v1`、密钥取 Windows 用户环境变量 `MATHMODEL_LOCAL_7863_API_KEY`、模型 `global:deepseek-v4.1-flash`（`_run_case.py` / `_resume_case.py` 已以此为默认，可用 `MATHMODEL_LOCAL_MODEL` 覆盖）。该网关同时提供 `cn:deepseek-v4.1-flash` 等 70 个模型；两者均可正常响应，`global:` 变体在 `thinking` 关闭时输出更完整。
  - 结构化生成（非自由文本）在网关侧会把 reasoning 计入 completion 预算，长 JSON 因此可能被截断（`finish_reason=length`）。已在 `OpenAIProvider.structured_generate` 增加自适应重试：一旦检测到截断，立即以 `thinking: {"type": "disabled"}` 重发一次，避免整个阶段因一份永远无法解析的 JSON 而失败。
  - 验证门禁新增“求解程序自报残余冲突”确定性检查：求解程序若自报 `residual_conflicts>0`，说明它自己承认没解出可行解，此时必须判 FAIL 而不是放行。
- 阻塞与解除条件：无（DeepSeek 官方 API 余额耗尽已由环境内网关替代；若要回到官方接口，充值后设 `MATHMODEL_PROVIDER=deepseek`）
- 已知限制：见 `DONE.md`
- 审查结果：自审完成，共发现并修复 24 个真实缺陷（另定位 1 个未修复，见第 25 条）（见“实施进展”）；另有手写独立校验器复核交付件，全部通过

## 实施进展
- `src/mathmodel/autopilot/state.py`：阶段状态与持久化（`pipeline_state.json`），支持 resume。
- `src/mathmodel/autopilot/intake.py`：真实解析 PDF/Excel，识别赛题/数据/结果模板角色，导出处理后数据，按真实缺口生成澄清问题。
- `src/mathmodel/autopilot/codegen.py`：求解程序生成 + 沙箱执行（含产物回收）。
- `src/mathmodel/autopilot/verify.py`：确定性验证 + 独立复算验证 + 报告与回修反馈。
- `src/mathmodel/autopilot/deliver.py`：Paper Context Package、图表构建、CUMCM LaTeX 渲染、xelatex 编译、支撑材料包与 manifest。
- `src/mathmodel/autopilot/pipeline.py`：连续编排、持久化、回修循环、最终一致性检查。
- `src/mathmodel/cli.py`：`run` / `resume` / `status`。
- 修复：`agents/model_explorer.py` 语法错误；`files/parsers.py` 真实读取 PDF/Excel；`sandbox/docker_backend.py` 产物回收；`providers/deepseek.py` JSON 解析健壮性与推理模型结构化输出问题。
- `providers/json_repair.py`（新增）：把 fence/散文/枚举大小写容错抽成 provider 共用模块，`DeepSeekProvider` 与 `OpenAIProvider` 共用。
- `tests/test_autopilot.py`（新增 59 项）：覆盖 RunState 持久化与 resume、intake 真实解析与澄清问题、残余冲突检测、确定性验证门禁、代码提取、LaTeX 注入守卫、图表与 LaTeX 渲染。

### 本轮自审发现并修复的真实缺陷（均由测试或真实运行暴露）
1. **LaTeX 注入守卫误杀 `\includegraphics`**：`paper/renderer.py` 用子串匹配禁用命令，`\include` 命中 `\includegraphics`，导致**任何带图的论文都无法编译**。改为按完整控制序列匹配（`(?![A-Za-z])`）。
2. **验证门禁形同虚设**：求解程序自报 `residual_conflicts: 291`（自己承认没解出可行解）时，原有 5 项确定性检查全部 PASS。新增 `no_self_reported_residual_violations` 检查。
3. **代码提取被截断回复击穿**：`_CODE_FENCE_RE` 要求闭合围栏，回复被截断时 ``` 泄漏进程序，导致独立验证程序 `SyntaxError`（实测 `/workspace/code/code.py` line 13）。改为容忍未闭合围栏 + 丢弃围栏行 + 前导散文 + 取最长可编译前缀。
4. **验证永远无法通过**：验证程序输出契约未要求 `statistics`，而 `compare_statistics` 返回 `NOT_RUN` 会让 `build_report` 判 FAIL，形成死循环。已在契约中强制要求 `statistics`，并把该反馈跨 attempt 传给验证程序而非错误地归咎于求解程序。
5. **图已就位时 PDF 构建崩溃**：`shutil.copy2` 把文件复制到自身在 Windows 上抛 `PermissionError`。
6. **短赛题正文被静默丢弃**：`intake` 只把 >200 字符的文档当赛题，短正文会被判为“未提取到赛题”并产生阻塞性追问；改为无长文档时回退取最长候选。
7. **PDF 构建崩溃：`'NoneType' object is not subscriptable`**：`subprocess.run(..., text=True)` 未指定 `encoding`，宿主机 GBK 区域设置去解码 xelatex 的 UTF-8 中文日志时在读线程抛 `UnicodeDecodeError`，导致 `completed.stdout is None`，随后 `stdout[-4000:]` 崩溃。该缺陷使**任何中文日志含非 ASCII 的论文都无法出 PDF**。已为 `deliver.py`、`renderer.py`、`sandbox/docker_backend.py` 的全部 subprocess 调用补 `encoding="utf-8", errors="replace"` 与 `or ""` 兜底，并新增回归测试（用 `\typeout{中文}` 强制日志含中文）。
8. **论文编造未验证数字**：摘要写“A、B、C 类各 50 个”，而附件1真实分布为 A=20、B=40、C=90。原一致性检查只用 `re.findall(r"\d+\.\d+", ...)` 扫小数，**整数根本不检查**，且只取 `statistics` 顶层数值（嵌套的 `by_class.*` 全部漏掉）。已改为递归收集嵌套统计量 + 表格单元格 + 赛题原文数字 + 验证报告数字 + 产出文件行数作为“可追溯集合”，并对全部整数与小数扫描；发现不可追溯数字时**只重写含该数字的章节与摘要**（最多 2 轮），而不是静默通过。
9. **模板表头从未被真正核对**：`DeterministicVerifier` 计算了 `expected_norm` 与 `actual_norm` 却**只比较列数、从不比较文字**，导致 result1–result4 全部表头与附件2模板不一致（如模板 `冲突装备1/冲突设备2`，产出 `用频装备编号1/用频装备编号2`）仍判 PASS。已改为逐列比较（忽略大小写与首尾空白）并报出差异列。
10. **按计划编号建表的模板只填了改动过的计划**：result2/result4 只写 113 行（39 调整 + 74 撤销），未列出 76 个保留未动的计划。已新增 `per_plan_table_covers_every_plan` 确定性检查（模板首列为 `用频装备编号` 时要求每计划一行）并在代码生成提示中明确该要求。
11. **resume 会重跑已完成且已验证的求解**：`resume()` 只缓存到 `model`，codegen/solve/verify 每次都重跑，既烧模型预算又可能用未验证的新解替换已验证的解。已新增 `_load_verified_attempt`：存在 PASS 的 `reportN.json` + `outcomeN.json` + 全部必需产出文件时直接复用。
12. **单次瞬时网络错误摧毁整条运行**：网关偶发 `APIConnectionError`/`ReadError` 会从 `_drive` 抛出并终止运行（实测在论文阶段崩溃）。已新增 `_retry_call`，对论文大纲/分节/摘要等调用做有界重试。
13. **resume 报告里残留上一轮的告警**：`state.notes` 跨 resume 累积，最终一致性检查已通过却仍报告旧告警。已改为每次 `_drive` 开始时清空 notes（notes 属于本次尝试的诊断）。
14. **行内公式被转义成字面文本**：`_latex_escape` 把 `$\Delta t$` 转成 `\$\textbackslash\{\}Delta t\$`，PDF 里显示为乱码而非数学符号。已改为保留 `$...$` 数学片段、只转义片段外的文本（注入守卫仍扫描全文，已加测试确保 `$...$` 内不能夹带危险命令）。
15. **回修循环无法收敛到正确模型**：求解程序按 `t0 + k*gap` 展开占用（应为 `t0 + k*(len+gap)`），报 237 对冲突。根因更深——**缓存的 `math_model.json` 本身就错**（`eq_time_overlap` 用 `(l-1)*(g_i + d_i^g)`，而 `g_i` 是间隔），而原回修循环只根据验证反馈**重新生成代码**、从不回头修模型，因此永远无法收敛。已新增 `_needs_model_repair` + `_repair_model`：验证指出**输入推导量**不一致时，带 `external_feedback` 回到 `MathModeler` 重新形式化。修复后模型被正确重写，求解程序报 297，与独立复算一致。
16. **回修循环被“解的质量问题”误触发**：`_needs_model_repair` 原本对残余冲突、基线比较失败也触发模型重写，而这类问题属于求解质量、模型本身没错，重写模型会丢弃已正确的建模成果（实测把模型从正确的 20 变量版本改坏成 14 变量、冲突数变成 71）。已收紧为只在 `recomputation` 类检查或 `recomputed_total`/`recomputed`+`claimed` 证据出现时触发。
17. **独立验证程序读不到输入却全判 PASS**：该验证程序把每行过滤空单元格后要求 `len(vals) >= 6`，而附件1每行只有 5 列，于是 0 条记录被解析，18 项约束检查却全部 PASS（空集合上的空转循环）。已要求验证程序在 JSON 中上报 `input_records`，并新增确定性检查 `independent_read_the_input`：缺失或 ≤0 一律 FAIL。**验证程序自己从不看输入数据这一失败模式，从此无法通过门禁。**
18. **验证程序不知道输入数据的真实列结构**：求解代码生成提示带了 `data_schema`，独立验证程序却只拿到赛题正文与模型，于是自行假设列数并解析失败。已把真实 `data_schema`（真实表头、类型、示例行、行数）加入验证程序提示，并声明其列为权威。
19. **“无证据的 PASS”被当成已覆盖**：逐约束检查 `constraint_<n>` 即使什么都没查也会返回 PASS（`detail` 形如 `Check of: c6`、`evidence` 为空），于是模型已声明的 `c6: f_i + t_i + g_i <= 1`（每个计划最多调整一个参数）从未被真正检查，而交付件确实违反了它。已规定：`constraint*` 检查若报 PASS 却无 `evidence`，一律降级为 `NOT_RUN`（`build_report` 遇 `NOT_RUN` 即不通过）。
20. **运行结果报告了过期的验证结论**：`_result` 取 `sorted(report*.json)[-1]`，即编号最大的报告；上一轮遗留的 `report3.json`（FAIL）会让**本轮实际 PASS 的运行**仍显示 `VERIFICATION: FAIL`。已改为按修改时间取最新报告，即真正把关本次运行的那一份。
21. **结构化生成的 JSON 被 reasoning 截断**：网关把 reasoning 计入 completion 预算，`structured_generate` 的 32768 token 上限会被 reasoning 吃掉大半，JSON 在对象中途被切断（`finish_reason=length`），整段阶段随之失败。已在 `OpenAIProvider.structured_generate` 加入自适应重试：检测到 `finish_reason == "length"` 时立刻以 `thinking: {"type": "disabled"}` 重发一次，并容忍网关不支持该参数（失败则保留首次响应）。
22. **验证程序与求解程序意见不一致时只责怪求解程序**：`compare_statistics` 判 FAIL 时，反馈只送给 `MathModeler`（“你的模型错了”），从不送回验证程序。但**验证程序本身也会算错**——实测一次运行中求解程序报 297 对冲突（经手写独立复算确认正确），验证程序却算出 237 并因此把一个**正确答案**判为失败。已改为：不一致时同时把该分歧作为 `carried_verifier_feedback` 送给下一轮的验证程序，要求它先从赛题原文重新推导展开规则、并与赛题中的例题/附录数值核对后再下结论。

### E2E 收敛可靠性测量（固定模型 / 固定赛题 / 固定配置）
**协议**：模型 `global:deepseek-v4.1-flash`，赛题 2026 CUMCM D 题，配置固定（网关 `http://127.0.0.1:7863/v1`、`MAX_REPAIR_ATTEMPTS=3`、`MAX_AGENT_ATTEMPTS=3`、`MAX_VERIFIER_ATTEMPTS=3`），每次均为**全新完整运行**（独立 run_id，intake→final_check），并行度 2，共 6 次。测量脚本 `_measure_reliability.py`，逐次记录 `runs/reliability/*`、`.tmp/reliability_records.jsonl`。

**结果（修复前，n=6）**

| 运行 | 门禁判定 | status | 独立复核 | 墙钟 | 主要失败模式 |
| --- | --- | --- | --- | --- | --- |
| r01 | FAIL | blocked | — | 2734s | 残余冲突 |
| r02 | FAIL | blocked | — | 2464s | 约束违反 + 复算分歧 |
| r03 | FAIL | blocked | — | 990s | 约束违反 + 复算分歧 |
| r04 | FAIL | `running`（异常） | — | 2222s | 约束违反 + 复算分歧 + 残余冲突 |
| r05 | **PASS** | completed | **错误答案** | 2749s | **假通过** |
| r06 | FAIL | blocked | — | 1744s | 约束违反 + 复算分歧 + 残余冲突 |

- 门禁通过率 **1/6 = 16.7%**；平均墙钟 2152s（36 min），范围 990–2749s。
- **经手写独立校验器复核，唯一“通过”的 r05 是错误答案**：result2 的 150 行只改参数、0 撤销、**297 对冲突全部残留**；result3 **0 行**；result4 同样 297 对残留。即**真正正确收敛 0/6 = 0%**。
- 样本量小：0/6 的 95% 单侧上界约 39%，真实正确收敛率可能落在 0%–40%。

**r05 假通过的根因**：该轮验证程序的 12 项检查几乎全是**输入数据属性**（band 范围、使用次数为正、区间长度为正、gap 取值），`constraint_satisfaction` 只比对了 **result1（冲突检测）**，从未判定 result2/result4 是否真的消解了冲突；`no_self_reported_residual_violations` 因求解程序未上报残余计数而返回 `NOT_APPLICABLE`（不阻断）；result3 为 **0 行**仍通过 `required_outputs_present`。

**结果（修复后，n=6，同一协议同一模型同一赛题）**

| 运行 | 门禁判定 | status | 墙钟 | 主要失败模式 |
| --- | --- | --- | --- | --- |
| p01 | FAIL | blocked | 1439s | 约束违反 + 复算分歧 + 残余冲突 |
| p02 | FAIL（未产生报告） | blocked | 2291s | MathModeler 三次均未通过 schema 校验 |
| p03 | FAIL | blocked | 1294s | 残余冲突 |
| p04 | FAIL | blocked | 1280s | 复算分歧 |
| p05 | FAIL | blocked | 2657s | 约束违反 + 复算分歧 + 残余冲突 |
| p06 | FAIL | blocked | 3827s | 约束违反 + 复算分歧 + 残余冲突 |

**结论（12 次运行汇总）**

| 指标 | 修复前 (n=6) | 修复后 (n=6) | 合计 (n=12) |
| --- | --- | --- | --- |
| 门禁判定通过 | 1/6 = 16.7% | **0/6 = 0%** | 1/12 = 8.3% |
| **经独立复核真正正确** | **0/6 = 0%** | **0/6 = 0%** | **0/12 = 0%** |
| 平均墙钟 | 2152s | 2132s | 2142s（36 min） |

- 真正正确收敛率 **0/12**；n=12 时 95% 单侧上界约 **22%**，即该配置下的真实 E2E 正确收敛率不高于约 22%。
- **失败位置分布**：verify 门禁 10 次、solve 沙箱执行 3 次、model 形式化 1 次。即 10/12 的运行跑到了验证，但 3 次回修内没解出合格解。
- **上游主因**：`MathModeler` 反复产出**引用未声明符号的方程**而 schema 校验失败（每次运行日志里都出现多次；p02 因此三次全败、运行在 model 阶段终止）。这是当前最影响收敛率的单点，**尚未修复**。
- 修复后的门禁更严：r05 那类假通过被消除，代价是通过率由 1/6 降到 0/6——**原先那个“通过”本来就是错的，所以正确率没有变化（两次都是 0%）**。

**由本次测量发现并修复的缺陷**
23. **验证程序可以把检查全花在输入数据属性上，从不判定答案是否达成目标**，于是一个什么都没解决的答案被判 PASS。已新增强制契约 `objectives`：验证程序必须对**每个必需输出文件**给出一条 `{"output", "achieved", "evidence"}` 判定，缺项或 `achieved != true` 一律 FAIL。用 r05 的**真实报告**回放：旧门禁 PASS → 新门禁 FAIL（`.tmp/replay_r05_gate.py`）。修复后在 p01 的运行日志中可见该检查按预期工作：`outputs whose purpose was not achieved: {'result2.xlsx': 'residual_conflicts=274, adjusted/cancelled=150', ...}`。
24. **运行可能以 `status=running` 退出**（r04：solve failed + verify failed，但 `state.fail()` 只标记阶段、不设置运行状态，调用方无法区分“已结束但受阻”与“仍在运行”）。已在 `_result` 统一收敛：带 blockers 返回时若状态仍为 running 则落为 `blocked`。
25. **（已定位，未修复）`MathModeler` 符号闭包不满足**：模型反复在方程中引用未声明的变量/参数（`unknown variable/parameter`），schema 校验连续失败并耗尽 3 次尝试，直接导致 p02 整条运行在 model 阶段终止，并在其余各次运行中消耗大量尝试。符号闭包目前只在提示中要求，**没有在失败后给出可操作的反馈**（例如把缺失符号清单回灌给下一次尝试）。

**关于 `global:deepseek-v4.1-flash` 的补充实测**
- 用上述配置全新跑了一条运行（`runs/cumcm-2026-dti-global`）：18 个阶段的编排、真实沙箱求解、确定性检查与独立复算全部正常执行，**验证门禁按设计拦下了不达标解**（实测拦下：band 越界 56/65 处、result3 可行域 8001 处、残余冲突 20 对、以及一次“求解程序 297 vs 验证程序 237”的分歧）。
- 该运行在 3 次回修预算内未收敛到可通过的解，因此**未产出交付件**；已交付件仍为 `runs/cumcm-2026-dti-e2e`（独立复算全部通过）。这说明编排与验证层工作正常，剩余变量是生成式求解程序在该组合规模下的收敛率。

### 独立复算证据（不依赖任何 autopilot 代码）
- `.tmp/independent_check.py` 按赛题原文与附录语义独立实现：100 个等宽频段、首次时间区间 `[t0,t0+len)`、第 k 次占用起点 `t0 + k*(len+gap)`、时间与频段同时交叠才算冲突。
- 结果：150 个计划（A=20, B=40, C=90），**冲突对 297 对**（A-B 21、A-C 66、B-C 181、B-B 10、C-C 19），并复现附录 A001 示例区间 35/100/165。
- `.tmp/verify_deliverable.py` 独立校验**交付的结果文件本身**（同样不 import 任何 autopilot 模块），按赛题语义重建占用区间后逐条核对，最终交付件的结果：
  - result1 的 297 对冲突与独立复算**完全一致**（无缺无多）；
  - result2 的 150 行方案重建后**残余冲突 0**，46 个计划被调整（撤销 104），每个计划只改一个参数、频段平移 ≤10、时间平移 ≤5；
  - result3 新增的 32809 个 C 类计划与 result2 方案及彼此之间**冲突 0**，且都落在 100 个频段内；
  - result4 的 150 行方案重建后**残余冲突 0**，79 个活跃计划中 33 个调整了间隔时长且均 ≤10Δt；
  - 四个文件表头与附件2模板逐列一致。
- 该 297 与运行中求解程序自报值完全一致；而此前本地网关某次运行自报 237，经比对为**错误答案**——证明独立验证层是必要的，不能只靠内部一致性检查。

## dsh 运行状态快照
- goal 工具：可用
- 原生 goal 标识与阶段：未创建（本目标为单会话连续实施，未使用原生续跑）
- 续跑激活情况与轮数限制：无
- 状态同步结果：未操作

本节只记录观察快照，不能替代更新前的 get_goal。
GOAL.md 的交付状态与 dsh 原生阶段分别记录，不强行一一映射。

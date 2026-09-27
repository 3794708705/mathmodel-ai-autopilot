# 完成记录

仅在实际达到验收后追加以下记录；创建空文件不代表任务完成。

## 2026-09-24 — 部署 DeepSeek Harness 项目执行规则到 MathModel AI 项目
- 当时约定的交付范围：把 `deepseek-harness-agent-rules` 的 12 节执行规则合并进项目根目录 `AGENTS.md`，保留项目原有要求，并确认规则在 dsh 会话中加载。
- 交付物与位置：`D:\deepseek工作区\AGENTS.md`（项目根目录）。
- 验收证据：
  - A1：`AGENTS.md` 共 292 行 = 原有 79 行 + 规则 1–12 节；原有 79 行与 `git show HEAD:AGENTS.md` 逐行比对 0 处差异。
  - A2：规则正文与上游 `deepseek-harness-agent-rules`（commit `8f0a446`）逐行比对 203/203 行一致，0 处差异。
  - A3：写入后本会话收到 “Updated instructions from: AGENTS.md”，合并后内容进入上下文。
  - A4：按仓库 `README.md` 步骤 2“已有 `AGENTS.md` 时合并、保留原要求”执行，未覆盖原文件。
- 审查结论：自审完成（逐行比对 + 加载确认）；独立审查不适用；用户签收未进行。
- 必要限制：规则正文的“合并状态”说明行为本项目新增，其余规则文字与上游一致；项目内临时克隆目录 `_agent_rules_repo` 已在合并后删除，如需再次核对请重新克隆该仓库。
- 最短使用方法：在 `D:\deepseek工作区` 新建 dsh 会话（或让当前会话重新加载规则）即自动生效；需要启动具体目标时，按 `AGENTS.md` 第 11 节的启动指令发送。
- 原生 goal 收尾状态：未创建原生 goal（短任务），无需终态更新。

## 2026-09-24 — 把 MathModel AI 实现成可运行的 CUMCM 数学建模 Autopilot
- 当时约定的交付范围：用户上传赛题与附件后，系统沿一条连续工作流完成赛题理解、建模、代码求解、结果验证、图表生成、论文写作与最终打包，产出可检查的论文 PDF + 支撑材料包。不新增路线图/子项目/长期 TODO，不以 mock/demo/单元测试替代真实产物。
- 交付物与位置：
  - 运行入口：`src/mathmodel/autopilot/`（`CUMCMAutopilot`）+ `src/mathmodel/cli.py`（`mathmodel run / resume / status`）。
  - 本次真实运行的交付件：`runs/cumcm-2026-dti-e2e/output/paper.pdf`（25 页 A4）、`runs/cumcm-2026-dti-e2e/output/support/{source_code,data_or_processed_data,figures,tables,necessary_supporting_files}`、`runs/cumcm-2026-dti-e2e/output/manifest.json`。
  - 过程产物：`runs/cumcm-2026-dti-e2e/pipeline_state.json`、`runs/cumcm-2026-dti-e2e/artifacts/`。
- 实际跑的赛题：**2026 高教社杯全国大学生数学建模竞赛 D 题「时频冲突检测与消解」**，4 个子问题。真实输入为赛题 PDF + `附件1.xlsx`（150 个用频装备计划）+ `result1–4.xlsx` 四个空白结果模板。
- 验收证据（A1–A15 逐条见 `GOAL.md` 验收表，均为实测）：
  - 运行结论：`STATUS: completed`、`VERIFICATION: PASS`、`final_check: all consistency checks passed`；18 个阶段全部 `completed`。
  - 真实结果：150 个计划（A=20/B=40/C=90）中检出 **297 对时频冲突**；result2/result4 消解后残余冲突 **0**；result3 新增 32809 个 C 类计划且冲突 0。
  - 验证：`artifacts/verify/report1.json` overall=PASS，共 **49 项检查**（含 34 项逐约束检查、独立复算、独立冲突对集合比对），由**不看求解代码**的沙箱内验证程序产出。
  - PDF：`%PDF-1.5`、25 页、A4、内嵌 `FandolSong`/`FandolKai` 中文字体、8 个 CUMCM 标准章节。
  - 支撑材料：`manifest.json` 中 `verification.overall=PASS`、`data_status=formal`、`counts={source_code:3, data:5, figures:5, tables:5, extra:7}`、`submission_check=ready_to_submit`。
  - 测试：`pytest` **790 passed, 1 skipped**（基线 660 passed, 1 skipped，新增 130 项，无回归）。
  - 人级独立证据：`.tmp/independent_check.py` 与 `.tmp/verify_deliverable.py`（均不 import 任何 autopilot 代码）独立复算得同样的 297 对冲突，并逐条复核交付的结果文件 → **ALL INDEPENDENT CHECKS PASSED**。
- 审查结论：自审完成，共发现并修复 **22 个真实缺陷**（逐条见 `GOAL.md`「实施进展」）。其中影响正确性的关键几项：验证门禁形同虚设、模板表头从未真正比对、按计划建表只填改动过的计划、论文编造未验证数字、回修循环无法回到建模修复、独立验证程序读不到输入却全判 PASS、运行结果报告过期验证结论、结构化生成 JSON 被 reasoning 截断、验证程序与求解程序不一致时只责怪求解程序。独立审查由手写校验器承担；用户签收未进行。
- 运行时配置（已实测确认）：API 地址 `http://127.0.0.1:7863/v1`，密钥取 Windows 用户环境变量 `MATHMODEL_LOCAL_7863_API_KEY`，模型 `global:deepseek-v4.1-flash`（`_run_case.py` / `_resume_case.py` 已以此为默认，`MATHMODEL_LOCAL_MODEL` 可覆盖）。
- 必要限制：
  1. DeepSeek 官方 API 余额耗尽（402），真实运行使用上述环境内 OpenAI 兼容网关。该网关把 reasoning 计入 completion 预算，**长 JSON/长代码生成可能被截断**（`finish_reason=length`）；系统已用自适应重试（结构化生成遇截断即关闭 thinking 重发）与回修循环兜住，但**单次运行仍可能需要多次尝试**。
  2. 用 `global:deepseek-v4.1-flash` 全新跑过一条运行（`runs/cumcm-2026-dti-global`）：编排、真实沙箱求解、确定性检查与独立复算全部正常执行，门禁按设计拦下了不达标解（band 越界、result3 可行域违规、残余冲突、以及一次求解程序与验证程序的复算分歧）；但该运行在 3 次回修预算内**未收敛**，未产出交付件。**交付件仍为 `runs/cumcm-2026-dti-e2e`**，其独立复算全部通过。生成式求解程序在该组合规模下的收敛率是主要剩余变量。
  2. 求解质量依赖模型生成的求解程序。本题组合规模大（297 对冲突、150 个计划），生成的求解程序采用贪心策略，得到的可行解**不是最优解**（例如 result2 撤销了 104 个计划，其中 C 类 84 个）。论文中已如实说明求解策略与目标，未宣称最优性。
  3. result3 的“新增用频装备”按赛题宽松读法处理（利用空闲资源最多新增），数量随策略变化（历次运行 755～32809 不等）；论文中记录的是本次运行的实际值。
  4. 中文字形无 ToUnicode 映射，`pypdf` 抽取 PDF 文本为空——这是字体特性，不是渲染失败；验证方式是页数、字体表与 LaTeX 源。
  5. 本项目原有的历史缺口仍然存在：`ProblemAnalysis` 无 `expected_outputs`/`data_requirements`、`Subproblem` 无 `description`、`MathematicalValidationGate` 仅面向 LP。这些不影响本次交付路径（真实路径是 LLM 代码生成 + 沙箱执行）。
- 最短使用方法：
  ```powershell
  cd "D:\deepseek工作区"
  $env:PYTHONPATH = "src"          # 项目未注册 console script，需把 src 放进 PYTHONPATH

  # 方式一：CLI（--problem 直接列赛题与附件文件，可多个）
  .\.venv\Scripts\python.exe -m mathmodel.cli run `
      --problem ".tmp\case_dti\D题.pdf" ".tmp\case_dti\附件1.xlsx" `
                ".tmp\case_dti\result1.xlsx" ".tmp\case_dti\result2.xlsx" `
                ".tmp\case_dti\result3.xlsx" ".tmp\case_dti\result4.xlsx" `
      --workspace runs --run-id my-run
  .\.venv\Scripts\python.exe -m mathmodel.cli status "runs\my-run"
  .\.venv\Scripts\python.exe -m mathmodel.cli resume "runs\my-run"   # 中断后续跑

  # 方式二：本次实际使用的一行式脚本（已内置网关配置与赛题目录）
  .\.venv\Scripts\python.exe _run_case.py       # 全新运行
  .\.venv\Scripts\python.exe _resume_case.py    # 续跑既有运行
  ```
  产物在 `runs/<run_id>/output/`：`paper.pdf`、`support/**`、`manifest.json`。
- 原生 goal 收尾状态：本目标为单会话连续实施，未创建 dsh 原生 goal，无需终态更新。

## 2026-09-27 — 修复 autopilot 流程缺陷：资格门接通、出版复用、ledger 加固
- 当时约定的交付范围：用户先问「程序整体流程有问题吗」，在得到缺陷清单后指示「有问题就去优化修复问题，没问题就上传 GitHub」，按推荐范围（项目代码+测试+文档，排除并发会话文件与生成物）提交推送。
- 交付物与位置：提交 `bc2fe18`，已推送到 `origin/master`（`github.com/3794708705/mathmodel-ai-autopilot`），21 个文件 +8308/−253。
- 验收证据：
  - A1 `select` 不再硬编码 `eligible=True`（`pipeline.py` 原 `reason="autopilot intake"` 占位），改为运行真实 `EligibilityGate(policy=EligibilityPolicy())`，无适格候选即 block；stage detail 记录 `N/M candidate(s) eligible`。证据：`tests/test_pipeline_flow_repairs.py::test_a_candidate_that_violates_a_hard_constraint_is_never_scored`（真实 `_drive` 到 select，jury 被 monkeypatch 成"一旦调用即失败"）；`_verify_eligibility_wiring.py` 用真实归档候选复算 5/5 仍 eligible，证明不会误杀真实运行。
  - A2 出版半场恢复复用：`figures`/`tables`/`paper` 在 `artifacts/paper/publication_inputs.json` 指纹（已验证 outcome run_id、model version、statistics、subproblems、output summaries）一致且图片文件仍在时整体复用；`audit`/`pdf`/`package`/`final_check` 仍重跑。证据：`_verify_publication_reuse.py` 在真实 completed run（`runs/reliability/c03`，PASS 于 attempt 3）上 5/5——复用后 0 次模型调用（outline/section/abstract/propose 全 0）、`paper_id` 与 8 个章节不变、run 仍 `completed`；改 `math_model.json` 的 version 后指纹不匹配，`propose` 确实被调用（拒绝陈旧复用）。单测 4 个分支见 `tests/test_pipeline_flow_repairs.py`。
  - A3 两处潜伏 bug：已验证尝试的回看不再绑定 `MAX_REPAIR_ATTEMPTS`（改读盘上 `report*.json`），`_report_artifact` 按 `attempt<N>` 目录名解析（原 `solve_dir.name[-1]` 会把 attempt10 写成 `report0.json`）。证据：`test_verified_attempts_are_read_from_disk_newest_first`、`test_a_verified_attempt_above_the_repair_limit_is_still_found`、`test_attempt_numbers_come_from_the_name_not_its_last_digit`。
  - A4 本轮发现的 ledger 真实缺陷：写入 `problem_states` JSON 列时把 domain `metadata_` 中的 datetime 原样放入，触发 `Object of type datetime is not JSON serializable` 并让 ledger 整个自禁用；现统一经 `_json_safe`（datetime→ISO）归一化。证据：`tests/test_run_ledger.py::test_ledger_stores_a_domain_state_that_carries_datetimes`；真实 resumed run 日志中不再出现 "Run ledger disabled"。
  - A5 `clarify`/`registries` 补 `begin`，消除 `attempts=0`、`started_at` 为空与 `current_stage` 不反映的阶段语义不一致。
  - A6 全量回归：`.\.venv\Scripts\python.exe -m pytest -q` → **1045 passed, 1 skipped**（基线 1035+1，新增 10 个测试）。
  - A7 `_verify_run_ledger.py` 重放真实归档运行 17/17。
- 审查结论：自审 + 三个独立脚本在真实运行数据上复核；未做用户签收外的额外审查。
- 必要限制：未接 SENSITIVITY/ROBUSTNESS/RED_TEAM 阶段（`domain/verification.py` 已有模型但零调用，论文 8 个小节也无灵敏度/稳健性小节）、literature/`CitationVerifier`（`submission/__init__.py:340` 分支恒假）、reality/budget/integrity/runtime/solver/verification 等约 4900 行仅测试可达；`NEEDS_CONFIRMATION` 仍是死状态；`RunState.load` 会忽略传入目录、沿用 state 文件里的 `run_dir`（未改，验证脚本内自行重定向）；本轮仍未做一次完整实跑（缺模型凭据）。
- 最短使用方法：`git pull` 后 `pytest`；中断的续跑用 `mathmodel resume <run_dir>`，已产出出版物的运行会直接复用论文与图表。
- 原生 goal 收尾状态：交付验收已通过，待提交 complete。

"""Tests for the CUMCM Autopilot: intake, state, verification, codegen, delivery.

These cover the deterministic core of the autopilot — everything that must be
correct regardless of which model writes the prose.
"""

import json
from pathlib import Path

import pydantic
import pytest

from mathmodel.autopilot.codegen import (
    GeneratedProgram,
    _extract_code,
    _extract_header_field,
    _extract_imports,
    _parse_summary,
)
from mathmodel.autopilot.deliver import TableBuilder, build_cumcm_latex
from mathmodel.autopilot.intake import (
    AttachmentInfo,
    ProblemContext,
    ProblemIntake,
    SheetSchema,
    build_clarification_questions,
)
from mathmodel.autopilot.state import (
    ClarificationQuestion,
    RunState,
    RunStatus,
    StageStatus,
)
from mathmodel.autopilot.verify import (
    CheckStatus,
    DeterministicVerifier,
    VerificationCheck,
    VerificationReport,
    _self_reported_leftovers,
    build_report,
    collect_identifier_pool,
    compare_statistics,
    read_output_table,
)
from mathmodel.documents import FigureRegistry, TableRegistry
from mathmodel.paper import BlockType, ContentBlock, PaperIR, PaperSection
from mathmodel.providers.json_repair import (
    coerce_enum_casing,
    normalise_for_schema,
    parse_json_object,
)
from pydantic import BaseModel


# ═══════════════════════════════════════════════════════════════
# RunState: persistence, resume, stage transitions
# ═══════════════════════════════════════════════════════════════

class TestRunState:
    def test_create_and_reload(self, tmp_path):
        state = RunState.create(tmp_path / "run1")
        state.begin("intake", "reading")
        state.complete("intake", "done", ["a.json"])
        state.save()

        reloaded = RunState.load(tmp_path / "run1")
        assert reloaded.run_id == state.run_id
        assert reloaded.is_done("intake")
        assert reloaded.stage("intake").detail == "done"
        assert reloaded.stage("intake").artifacts == ["a.json"]

    def test_status_transitions_are_recorded(self, tmp_path):
        state = RunState.create(tmp_path / "run2")
        state.begin("solve", "running")
        assert state.stage("solve").status == StageStatus.RUNNING

        state.fail("solve", "exit 1")
        assert state.stage("solve").status == StageStatus.FAILED
        assert state.stage("solve").attempts == 1
        assert not state.is_done("solve")

        state.block("solve", "gave up")
        assert state.stage("solve").status == StageStatus.BLOCKED
        assert state.status == RunStatus.BLOCKED

    def test_skip_and_count(self, tmp_path):
        state = RunState.create(tmp_path / "run3")
        state.begin("a")
        state.complete("a")
        state.skip("b", "not needed")
        assert state.count(StageStatus.COMPLETED) == 1
        assert state.count(StageStatus.SKIPPED) == 1
        assert state.stage("b").status == StageStatus.SKIPPED

    def test_save_is_atomic_and_utf8(self, tmp_path):
        state = RunState.create(tmp_path / "run4")
        state.complete("intake", "中文详情")
        state.save()
        raw = (tmp_path / "run4" / "pipeline_state.json").read_text(encoding="utf-8")
        assert "中文详情" in raw
        assert RunState.load(tmp_path / "run4").stage("intake").detail == "中文详情"

    def test_summary_shape(self, tmp_path):
        state = RunState.create(tmp_path / "run5")
        state.begin("intake")
        state.complete("intake", "ok")
        summary = state.summary()
        assert summary["run_id"] == state.run_id
        assert summary["stages"]["intake"]["status"] == "completed"
        assert "current_stage" in summary


# ═══════════════════════════════════════════════════════════════
# Intake: real parsing of real files
# ═══════════════════════════════════════════════════════════════

def _make_xlsx(path: Path, headers, rows):
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(list(headers))
    for row in rows:
        sheet.append(list(row))
    workbook.save(path)
    return path


class TestIntake:
    def test_reads_real_excel_cells(self, tmp_path):
        source = _make_xlsx(
            tmp_path / "附件1.xlsx",
            ["用频装备编号", "频段区间", "时间区间", "间隔时长", "使用次数"],
            [["A001", "[80,90)", "[35,40)", 60, 3], ["B002", "[10,20)", "[0,5)", 30, 2]],
        )
        intake = ProblemIntake(tmp_path / "input")
        context, records = intake.ingest([source])

        assert len(records) == 1
        info = context.attachments[0]
        assert info.role == "data"
        assert info.parser_status.value == "completed"
        assert info.sheets
        sheet = info.sheets[0]
        assert sheet.row_count == 2
        assert sheet.headers[0] == "用频装备编号"
        assert sheet.sample_rows[0][0] == "A001"

    def test_classifies_templates_and_detects_required_outputs(self, tmp_path):
        problem = tmp_path / "D题.txt"
        problem.write_text(
            "问题 1 将结果保存到文件result1.xlsx 中。\n"
            "问题 2 将消解方案保存到文件result2.xlsx 中。\n"
            "附件 1 给出了用频计划。\n",
            encoding="utf-8",
        )
        data = _make_xlsx(tmp_path / "附件1.xlsx", ["编号"], [["A001"]])
        template = _make_xlsx(tmp_path / "result1.xlsx", ["序号", "冲突设备1", "冲突设备2"], [])

        intake = ProblemIntake(tmp_path / "input")
        context, _ = intake.ingest([problem, data, template])

        assert context.required_outputs == ["result1.xlsx", "result2.xlsx"]
        assert context.template_file_names == ["result1.xlsx"]
        assert "附件1" in context.declared_attachments
        assert context.missing_attachments == []

    def test_reports_missing_declared_attachment(self, tmp_path):
        problem = tmp_path / "题.txt"
        problem.write_text("详见附件 3 的说明。", encoding="utf-8")
        data = _make_xlsx(tmp_path / "附件1.xlsx", ["编号"], [["A001"]])

        intake = ProblemIntake(tmp_path / "input")
        context, _ = intake.ingest([problem, data])
        assert "附件3" in context.missing_attachments

    def test_export_processed_data_writes_csv(self, tmp_path):
        source = _make_xlsx(
            tmp_path / "附件1.xlsx", ["编号", "值"], [["A001", 1], ["A002", 2]]
        )
        intake = ProblemIntake(tmp_path / "input")
        context, _ = intake.ingest([source])
        written = intake.export_processed_data(context, tmp_path / "processed")

        assert len(written) == 1
        text = Path(written[0]).read_text(encoding="utf-8-sig")
        assert "A001" in text and "A002" in text

    def test_no_questions_when_upload_is_sufficient(self, tmp_path):
        problem = tmp_path / "题.txt"
        problem.write_text("某区域内150个用频装备提报用频计划，需检测时频冲突。", encoding="utf-8")
        data = _make_xlsx(tmp_path / "附件1.xlsx", ["编号"], [["A001"]])

        intake = ProblemIntake(tmp_path / "input")
        context, _ = intake.ingest([problem, data])
        assert build_clarification_questions(context) == []

    def test_blocking_question_when_statement_unreadable(self):
        context = ProblemContext(problem_text="", attachments=[
            AttachmentInfo(
                original_name="scan.pdf", role="problem_statement",
                storage_path="scan.pdf", size_bytes=10,
            )
        ])
        questions = build_clarification_questions(context)
        assert any(q.blocking for q in questions)
        assert questions[0].question_id == "Q-STATEMENT"

    def test_missing_attachment_is_non_blocking_when_data_present(self, tmp_path):
        problem = tmp_path / "题.txt"
        problem.write_text("详见附件 2。", encoding="utf-8")
        data = _make_xlsx(tmp_path / "附件1.xlsx", ["编号"], [["A001"]])

        intake = ProblemIntake(tmp_path / "input")
        context, _ = intake.ingest([problem, data])
        questions = build_clarification_questions(context)
        assert questions and not any(q.blocking for q in questions)

    def test_missing_attachment_blocks_without_any_data(self, tmp_path):
        problem = tmp_path / "题.txt"
        problem.write_text("详见附件 1 的数据。", encoding="utf-8")

        intake = ProblemIntake(tmp_path / "input")
        context, _ = intake.ingest([problem])
        questions = build_clarification_questions(context)
        assert questions and questions[0].blocking


# ═══════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════

class TestSelfReportedLeftovers:
    def test_flags_residual_conflicts(self):
        found = _self_reported_leftovers({"problem2": {"residual_conflicts": 291}})
        assert found == {"problem2.residual_conflicts": 291.0}

    def test_zero_residuals_pass(self):
        assert _self_reported_leftovers({"problem2_remaining_conflicts": 0}) == {}

    def test_final_conflicts_is_a_residual_counter(self):
        assert _self_reported_leftovers({"problem2": {"final_conflicts": 0}}) == {}
        assert _self_reported_leftovers({"problem2": {"final_conflicts": 5}}) == {
            "problem2.final_conflicts": 5.0
        }

    def test_total_conflicts_is_not_a_residual(self):
        assert _self_reported_leftovers({"problem1": {"total_conflicts": 297}}) is None

    def test_detected_conflict_counts_are_not_residuals(self):
        stats = {"problem1": {"total_conflict_pairs": 297, "num_plans": 150}}
        assert _self_reported_leftovers(stats) is None

    def test_nested_and_mixed(self):
        stats = {"a": {"b": {"unresolved_overlaps": 3, "residual_conflicts": 0}}}
        assert _self_reported_leftovers(stats) == {"a.b.unresolved_overlaps": 3.0}

    def test_booleans_ignored(self):
        assert _self_reported_leftovers({"violations_leftover": True}) is None


class TestOutputReaders:
    def test_reads_xlsx(self, tmp_path):
        path = _make_xlsx(tmp_path / "r.xlsx", ["序号", "设备"], [[1, "A001"], [2, "B002"]])
        headers, rows = read_output_table(path)
        assert headers == ["序号", "设备"]
        assert rows == [[1, "A001"], [2, "B002"]]

    def test_reads_csv(self, tmp_path):
        path = tmp_path / "r.csv"
        path.write_text("序号,设备\n1,A001\n", encoding="utf-8")
        headers, rows = read_output_table(path)
        assert headers == ["序号", "设备"]
        assert rows == [["1", "A001"]]

    def test_identifier_pool_from_real_data(self, tmp_path):
        _make_xlsx(
            tmp_path / "附件1.xlsx", ["编号", "值"],
            [["A001", 1], ["B002", 2], ["not-an-id", 3]],
        )
        pool = collect_identifier_pool(tmp_path)
        assert pool == {"A001", "B002"}


def _outcome(statistics=None, status="success"):
    from mathmodel.autopilot.codegen import ExecutionOutcome

    return ExecutionOutcome(
        run_id="RUN-test",
        status=status,
        exit_code=0 if status == "success" else 1,
        stdout="",
        stderr="",
        runtime_seconds=1.5,
        execution_real=True,
        sandbox={"image": "test", "code_hash": "abc"},
        summary={"statistics": statistics or {}},
    )


class TestDeterministicVerifier:
    def test_all_checks_pass_on_a_clean_solution(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _make_xlsx(data_dir / "附件1.xlsx", ["编号"], [["A001"], ["B002"]])

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        _make_xlsx(out_dir / "result1.xlsx", ["序号", "冲突设备1", "冲突设备2"], [[1, "A001", "B002"]])

        checks = DeterministicVerifier(data_dir).run(
            outcome=_outcome({"conflicts": 1, "residual_conflicts": 0}),
            output_dir=out_dir,
            required_outputs=["result1.xlsx"],
            template_headers={"result1.xlsx": ["序号", "冲突设备1", "冲突设备2"]},
        )
        by_name = {c.name: c for c in checks}
        assert by_name["execution_was_real"].status == CheckStatus.PASS
        assert by_name["required_outputs_present"].status == CheckStatus.PASS
        assert by_name["output_structure_matches_template"].status == CheckStatus.PASS
        assert by_name["referential_integrity"].status == CheckStatus.PASS
        assert by_name["statistics_reported"].status == CheckStatus.PASS
        assert by_name["no_self_reported_residual_violations"].status == CheckStatus.PASS
        assert all(c.status != CheckStatus.FAIL for c in checks)

    def test_unresolved_conflicts_fail_the_gate(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _make_xlsx(data_dir / "附件1.xlsx", ["编号"], [["A001"]])
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        _make_xlsx(out_dir / "result1.xlsx", ["序号"], [[1]])

        checks = DeterministicVerifier(data_dir).run(
            outcome=_outcome({"problem2": {"residual_conflicts": 291}}),
            output_dir=out_dir,
            required_outputs=["result1.xlsx"],
            template_headers={},
        )
        residual = [c for c in checks if c.name == "no_self_reported_residual_violations"][0]
        assert residual.status == CheckStatus.FAIL
        assert "residual_conflicts" in residual.detail

    def test_missing_output_fails(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        checks = DeterministicVerifier(data_dir).run(
            outcome=_outcome({"conflicts": 3}),
            output_dir=out_dir,
            required_outputs=["result1.xlsx"],
            template_headers={},
        )
        present = [c for c in checks if c.name == "required_outputs_present"][0]
        assert present.status == CheckStatus.FAIL

    def test_unknown_identifier_fails_referential_integrity(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _make_xlsx(data_dir / "附件1.xlsx", ["编号"], [["A001"]])
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        _make_xlsx(out_dir / "result1.xlsx", ["设备"], [["Z999"]])

        checks = DeterministicVerifier(data_dir).run(
            outcome=_outcome({"conflicts": 1}),
            output_dir=out_dir,
            required_outputs=["result1.xlsx"],
            template_headers={},
        )
        ref = [c for c in checks if c.name == "referential_integrity"][0]
        assert ref.status == CheckStatus.FAIL
        assert "Z999" in ref.detail

    def test_unreal_execution_fails(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        outcome = _outcome({}, status="failed")
        outcome.execution_real = False
        checks = DeterministicVerifier(data_dir).run(
            outcome=outcome, output_dir=out_dir,
            required_outputs=[], template_headers={},
        )
        real = [c for c in checks if c.name == "execution_was_real"][0]
        assert real.status == CheckStatus.FAIL


class TestReportAndComparison:
    def test_report_fails_when_any_check_fails(self):
        from mathmodel.domain.math_model import (
            MathematicalModel, Objective, ObjectiveSense,
        )

        model = MathematicalModel(
            model_id="M1", name="t", description="d",
            objectives=[Objective(name="o", sense=ObjectiveSense.MINIMIZE, expression="x")],
        )
        checks = [
            VerificationCheck(name="a", category="x", status=CheckStatus.PASS),
            VerificationCheck(name="b", category="x", status=CheckStatus.FAIL, detail="boom"),
        ]
        report = build_report(model, "RUN-1", checks)
        assert not report.passed
        assert report.blocking_failures
        assert "boom" in report.to_prompt_feedback()[0] or any(
            "boom" in f for f in report.to_prompt_feedback()
        )

    def test_not_run_counts_as_failure(self):
        from mathmodel.domain.math_model import (
            MathematicalModel, Objective, ObjectiveSense,
        )

        model = MathematicalModel(
            model_id="M1", name="t", description="d",
            objectives=[Objective(name="o", sense=ObjectiveSense.MINIMIZE, expression="x")],
        )
        report = build_report(model, "RUN-1", [
            VerificationCheck(name="a", category="x", status=CheckStatus.NOT_RUN),
        ])
        assert not report.passed

    def test_compare_statistics_detects_mismatch(self):
        check = compare_statistics({"a": 1.0}, {"a": 2.0})
        assert check.status == CheckStatus.FAIL

    def test_compare_statistics_accepts_match(self):
        check = compare_statistics({"a": 1.0}, {"a": 1.0})
        assert check.status == CheckStatus.PASS

    def test_nested_dicts_with_different_extra_keys_still_agree(self):
        # Two independent programs naturally report different extra keys. Only
        # the values present in both may be compared.
        check = compare_statistics(
            {"problem3": {"total_cells": 64600, "num_new_plans": 53680, "T_lo": 0}},
            {"problem3": {"total_cells": 64600, "num_new_plans": 53680, "new_cells": 53680}},
        )
        assert check.status == CheckStatus.PASS
        assert check.evidence["compared"] == 2

    def test_nested_dicts_still_catch_a_real_mismatch(self):
        check = compare_statistics(
            {"problem3": {"num_new_plans": 53680}},
            {"problem3": {"num_new_plans": 508}},
        )
        assert check.status == CheckStatus.FAIL
        assert "problem3.num_new_plans" in check.detail

    def test_boolean_is_not_treated_as_a_matching_number(self):
        # bool is a subclass of int, so a bool must be compared exactly rather
        # than through the numeric tolerance path.
        check = compare_statistics({"flag": True}, {"flag": False})
        assert check.status == CheckStatus.FAIL


# ═══════════════════════════════════════════════════════════════
# Code generation helpers
# ═══════════════════════════════════════════════════════════════

class TestCodeExtraction:
    def test_strips_markdown_fence(self):
        code = _extract_code("```python\n# APPROACH: x\nprint(1)\n```")
        assert code.startswith("# APPROACH: x")
        assert "```" not in code

    def test_keeps_raw_source(self):
        assert _extract_code("import os\nprint(1)") == "import os\nprint(1)"

    def test_handles_unterminated_fence(self):
        # A truncated reply has no closing marker; the opening ``` must not
        # survive into the program, where it is a SyntaxError.
        code = _extract_code("Here is the program:\n\n```python\nimport os\nprint(1)\n")
        assert "```" not in code
        assert code.startswith("import os")
        compile(code, "<test>", "exec")

    def test_drops_leading_prose_without_fence(self):
        code = _extract_code(
            "Sure! Here is a complete solver.\n\n"
            "import json\nprint(json.dumps({}))\n"
        )
        assert code.startswith("import json")
        compile(code, "<test>", "exec")

    def test_trims_trailing_prose(self):
        code = _extract_code(
            "import os\nprint(1)\n\n"
            "This program models the conflict as a graph and runs greedy search.\n"
            "Let me know if you need anything else!\n"
        )
        compile(code, "<test>", "exec")
        assert code.startswith("import os")

    def test_keeps_docstring_start(self):
        code = _extract_code('"""Solver."""\nimport os\n')
        assert code.startswith('"""Solver."""')

    def test_empty_input(self):
        assert _extract_code("") == ""
        assert _extract_code("   ") == ""

    def test_reads_approach_header(self):
        assert _extract_header_field("# APPROACH: greedy\nx=1", "APPROACH") == "greedy"

    def test_missing_header_returns_empty(self):
        assert _extract_header_field("x=1", "APPROACH") == ""

    def test_extract_imports_filters_stdlib(self):
        deps = _extract_imports("import os\nimport numpy as np\nfrom scipy import optimize\n")
        assert deps == ["numpy", "scipy"]

    def test_parse_summary_takes_last_json_line(self):
        stdout = 'progress\n{"statistics": {"a": 1}}\ntrailing text\n{"statistics": {"a": 2}}'
        assert _parse_summary(stdout)["statistics"]["a"] == 2

    def test_parse_summary_ignores_broken_json(self):
        assert _parse_summary("not json\n{broken\n") == {}


# ═══════════════════════════════════════════════════════════════
# JSON repair helpers shared by providers
# ═══════════════════════════════════════════════════════════════

class _StatusModel(BaseModel):
    class Status(str, __import__("enum").Enum):
        KNOWN = "known"
        REQUIRED = "required"

    status: Status


class TestJsonRepair:
    def test_parses_fenced_json(self):
        assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}

    def test_parses_json_with_prose(self):
        assert parse_json_object('Here you go: {"a": 1} hope that helps') == {"a": 1}

    def test_empty_returns_none(self):
        assert parse_json_object("") is None
        assert parse_json_object("no json here") is None

    def test_enum_case_is_canonicalised(self):
        normalised = normalise_for_schema({"status": "REQUIRED"}, _StatusModel)
        assert normalised["status"] == "required"
        assert _StatusModel.model_validate(normalised).status.value == "required"

    def test_unknown_enum_value_is_left_alone(self):
        normalised = normalise_for_schema({"status": "bogus"}, _StatusModel)
        assert normalised["status"] == "bogus"
        with pytest.raises(Exception):
            _StatusModel.model_validate(normalised)

    def test_nested_enum_coercion(self):
        schema = {
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/E"}}},
            "$defs": {"E": {"enum": ["low", "high"]}},
        }
        value = {"items": ["HIGH", "low"]}
        assert coerce_enum_casing(value, schema, schema["$defs"]) == {"items": ["high", "low"]}


# ═══════════════════════════════════════════════════════════════
# Delivery: tables and LaTeX
# ═══════════════════════════════════════════════════════════════

class TestTables:
    def test_statistics_table_carries_source_ids(self):
        builder = TableBuilder()
        table = builder.build_statistics_table(
            {"conflicts": 297, "plans": 150}, "EVD-EXECUTION"
        )
        assert table is not None
        assert len(table.rows) == 2
        assert all(cell.source_id == "EVD-EXECUTION" for row in table.rows for cell in row)

    def test_nested_statistics_table(self):
        builder = TableBuilder()
        table = builder.build_nested_statistics_table(
            statistics={"p2": {"A": {"keep": 19, "cancel": 1}}},
            source_id="EVD-EXECUTION",
            table_id="TAB-002",
            title="消解统计",
            key="p2",
        )
        assert table is not None
        assert table.table_id == "TAB-002"
        assert any(cell.value == 19 for row in table.rows for cell in row)

    def test_empty_statistics_produces_no_table(self):
        assert TableBuilder().build_statistics_table({}, "EVD") is None


class TestLatexRenderer:
    def test_renders_figures_and_tables(self, tmp_path):
        image = tmp_path / "fig1.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)

        from mathmodel.documents import FigureRecord

        figures = FigureRegistry()
        figures.register(FigureRecord(
            figure_id="FIG-001",
            title="冲突分布",
            caption="图 1 冲突分布",
            artifact_path=str(image),
            source_execution_ids=["RUN-1"],
        ))
        tables = TableRegistry()
        tables.register(TableBuilder().build_statistics_table({"conflicts": 297}, "EVD"))

        paper = PaperIR(
            title="测试论文",
            abstract="摘要内容",
            sections=[
                PaperSection(
                    title="问题分析",
                    content_blocks=[ContentBlock(block_type=BlockType.PARAGRAPH, text="分析正文")],
                ),
                PaperSection(
                    title="结果分析",
                    content_blocks=[
                        ContentBlock(block_type=BlockType.FIGURE, text="图 1 冲突分布",
                                     figure_ids=["FIG-001"]),
                        ContentBlock(block_type=BlockType.TABLE, text="TAB-001",
                                     table_ids=["TAB-001"]),
                    ],
                ),
            ],
        )
        tex = build_cumcm_latex(paper, figures, tables, tmp_path)
        assert "\\documentclass" in tex
        assert "\\includegraphics" in tex
        assert "fig1.png" in tex
        assert "297" in tex
        assert "结果分析" in tex


# ═══════════════════════════════════════════════════════════════
# GeneratedProgram / ExecutionOutcome contracts
# ═══════════════════════════════════════════════════════════════

class TestGeneratedProgram:
    def test_code_is_required(self):
        with pytest.raises(Exception):
            GeneratedProgram(approach="x", code="")

    def test_succeeded_flag(self):
        from mathmodel.autopilot.codegen import ExecutionOutcome

        assert ExecutionOutcome(status="success").succeeded
        assert not ExecutionOutcome(status="failed").succeeded


class TestLatexInjectionGuard:
    """The guard must block real primitives without blocking legitimate ones."""

    def test_allows_ordinary_commands(self):
        from mathmodel.paper.renderer import _check_latex_injection

        tex = (
            r"\includegraphics[width=0.8\textwidth]{fig1.png}"
            r"\begin{figure}[H]\caption{图 1}\end{figure}"
            r"\section{结果}\inputencoding{utf8}\readline"
        )
        assert _check_latex_injection(tex) == []

    def test_blocks_real_primitives(self):
        from mathmodel.paper.renderer import _check_latex_injection

        assert _check_latex_injection(r"\input{/etc/passwd}")
        assert _check_latex_injection(r"\include{secret}")
        assert _check_latex_injection(r"\write18{rm -rf /}")
        assert _check_latex_injection(r"\catcode`\@=11")
        assert _check_latex_injection(r"\special{ps: hide}")

    def test_blocks_primitive_at_end_of_string(self):
        from mathmodel.paper.renderer import _check_latex_injection

        assert _check_latex_injection(r"text \include")


class TestFigureBuilder:
    def test_renders_real_png_from_statistics(self, tmp_path):
        from mathmodel.autopilot.deliver import FigureBuilder, FigureSpec

        builder = FigureBuilder()
        registry = builder.render(
            specs=[FigureSpec(
                title="冲突对数量", chart="bar", figure_type="comparison",
                caption="图 1 冲突对数量", statistics_keys=["conflicts"],
                supported_claim="共有若干冲突", source="statistics",
            )],
            statistics={"conflicts": 297},
            output_dir=tmp_path,
            figures_dir=tmp_path / "figures",
            execution_id="RUN-1",
            source_data_ids=["EVD-1"],
        )
        records = registry.all()
        assert len(records) == 1
        record = records[0]
        artifact = Path(record.artifact_path)
        assert artifact.exists()
        assert artifact.read_bytes()[:4] == b"\x89PNG"
        assert record.source_execution_ids == ["RUN-1"]
        assert record.source_data_ids == ["EVD-1"]
        assert record.metadata["values"] == [297.0]

    def test_falls_back_to_available_numeric_statistics(self, tmp_path):
        from mathmodel.autopilot.deliver import FigureBuilder, FigureSpec

        # A requested key that is absent falls back to whatever numeric
        # statistics exist, rather than producing an empty figure.
        registry = FigureBuilder().render(
            specs=[FigureSpec(
                title="回退", chart="bar", statistics_keys=["absent"],
                source="statistics",
            )],
            statistics={"conflicts": 297},
            output_dir=tmp_path,
            figures_dir=tmp_path / "figures",
            execution_id="RUN-1",
            source_data_ids=[],
        )
        records = registry.all()
        assert len(records) == 1
        assert records[0].metadata["values"] == [297.0]

    def test_skips_specs_with_no_numeric_statistics(self, tmp_path):
        from mathmodel.autopilot.deliver import FigureBuilder, FigureSpec

        registry = FigureBuilder().render(
            specs=[FigureSpec(
                title="无数据", chart="bar", statistics_keys=["note"],
                source="statistics",
            )],
            statistics={"note": "纯文本"},
            output_dir=tmp_path,
            figures_dir=tmp_path / "figures",
            execution_id="RUN-1",
            source_data_ids=[],
        )
        assert registry.all() == []

    def test_mixed_numeric_and_text_keys_stay_paired(self, tmp_path):
        from mathmodel.autopilot.deliver import FigureBuilder, FigureSpec

        # A requested key that is not numeric must not desynchronise labels and
        # values, which would otherwise crash matplotlib.
        registry = FigureBuilder().render(
            specs=[FigureSpec(
                title="混合", chart="bar",
                statistics_keys=["conflicts", "note"],
                source="statistics",
            )],
            statistics={"conflicts": 297, "note": "见正文"},
            output_dir=tmp_path,
            figures_dir=tmp_path / "figures",
            execution_id="RUN-1",
            source_data_ids=[],
        )
        records = registry.all()
        assert len(records) == 1
        metadata = records[0].metadata
        assert len(metadata["labels"]) == len(metadata["values"]) == 1
        assert metadata["labels"] == ["conflicts"]

    def test_series_from_output_file(self, tmp_path):
        from mathmodel.autopilot.deliver import FigureBuilder, FigureSpec

        _make_xlsx(
            tmp_path / "result1.xlsx", ["序号", "冲突设备1", "冲突设备2"],
            [[1, "A001", "B002"], [2, "A003", "C004"]],
        )
        registry = FigureBuilder().render(
            specs=[FigureSpec(
                title="冲突对", chart="bar", source="output_file",
                output_file="result1.xlsx", output_columns=["序号", "冲突设备1"],
            )],
            statistics={},
            output_dir=tmp_path,
            figures_dir=tmp_path / "figures",
            execution_id="RUN-1",
            source_data_ids=[],
        )
        # "冲突设备1" is text, so it cannot become a numeric series.
        assert registry.all() == []


class TestSupportPackageBuilder:
    def test_assembles_output_and_manifest(self, tmp_path):
        from mathmodel.autopilot.deliver import SupportPackageBuilder

        source = tmp_path / "src"
        source.mkdir()
        code = source / "solver.py"
        code.write_text("print(1)\n", encoding="utf-8")
        data = source / "附件1.csv"
        data.write_text("编号\nA001\n", encoding="utf-8")
        pdf = source / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.7\n" + b"0" * 64)
        figure = source / "fig1.png"
        figure.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)

        out_dir = tmp_path / "output"
        manifest = SupportPackageBuilder(out_dir).build(
            paper_pdf=pdf,
            paper_tex=None,
            source_code=[code],
            processed_data=[data],
            original_files=[],
            figures=[figure],
            tables=[],
            result_files=[data],
            manifest_extra={"run_id": "RUN-1"},
        )

        assert (out_dir / "paper.pdf").read_bytes()[:5] == b"%PDF-"
        assert (out_dir / "support" / "source_code" / "solver.py").exists()
        assert (out_dir / "support" / "figures" / "fig1.png").exists()
        assert manifest["paper_pdf"] == "paper.pdf"
        assert manifest["run_id"] == "RUN-1"
        assert manifest["counts"]["source_code"] == 1
        written = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert written["counts"] == manifest["counts"]

    def test_same_basename_does_not_overwrite(self, tmp_path):
        from mathmodel.autopilot.deliver import SupportPackageBuilder

        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        (a / "solver.py").write_text("A\n", encoding="utf-8")
        (b / "solver.py").write_text("B\n", encoding="utf-8")

        out_dir = tmp_path / "output"
        manifest = SupportPackageBuilder(out_dir).build(
            paper_pdf=None, paper_tex=None,
            source_code=[a / "solver.py", b / "solver.py"],
            processed_data=[], original_files=[], figures=[], tables=[],
            result_files=[], manifest_extra={},
        )
        copied = sorted((out_dir / "support" / "source_code").glob("*.py"))
        assert len(copied) == 2
        assert manifest["counts"]["source_code"] == 2

    def test_missing_pdf_is_recorded_as_none(self, tmp_path):
        from mathmodel.autopilot.deliver import SupportPackageBuilder

        out_dir = tmp_path / "output"
        manifest = SupportPackageBuilder(out_dir).build(
            paper_pdf=tmp_path / "nope.pdf", paper_tex=None,
            source_code=[], processed_data=[], original_files=[],
            figures=[], tables=[], result_files=[], manifest_extra={},
        )
        assert manifest["paper_pdf"] is None
        assert not (out_dir / "paper.pdf").exists()


class TestProgramCompleteness:
    """A truncated program parses and exits 0 while producing nothing."""

    def test_detects_missing_summary_print(self):
        from mathmodel.autopilot.codegen import _code_looks_complete

        assert not _code_looks_complete("def main():\n    x = 1\n")
        assert _code_looks_complete("import json\nprint(json.dumps({'a': 1}))\n")

    def test_detects_unparseable_code(self):
        from mathmodel.autopilot.codegen import _code_looks_complete

        assert not _code_looks_complete("def broken(:\n")

    def test_empty_is_incomplete(self):
        from mathmodel.autopilot.codegen import _code_looks_complete

        assert not _code_looks_complete("")
        assert not _code_looks_complete("   ")

    def test_truncated_reply_triggers_a_compact_retry(self):
        import asyncio

        from mathmodel.autopilot.codegen import generate_program_source
        from mathmodel.providers.base import GenerationResponse, UsageInfo
        from mathmodel.routing.profile import TaskProfile, TaskType

        calls: list[str] = []

        class FakeRouter:
            async def route_generate(self, profile, prompt, **kwargs):
                calls.append(prompt)
                if len(calls) == 1:
                    # Cut off mid-function: parses, but never prints a summary.
                    return GenerationResponse(
                        content="import json\ndef main():\n    total = 1\n",
                        model="fake", finish_reason="length",
                        usage=UsageInfo(), is_mock=False,
                    )
                return GenerationResponse(
                    content="import json\nprint(json.dumps({'total': 1}))\n",
                    model="fake", finish_reason="stop",
                    usage=UsageInfo(), is_mock=False,
                )

        code = asyncio.run(generate_program_source(
            FakeRouter(),
            profile=TaskProfile.for_task_type(TaskType.CODE_GENERATION),
            prompt="write a program",
            system_prompt="be brief",
        ))

        assert len(calls) == 2
        assert "CUT OFF" in calls[1]
        assert "print(" in code

    def test_accepts_a_good_first_reply_without_retrying(self):
        import asyncio

        from mathmodel.autopilot.codegen import generate_program_source
        from mathmodel.providers.base import GenerationResponse, UsageInfo
        from mathmodel.routing.profile import TaskProfile, TaskType

        calls: list[str] = []

        class FakeRouter:
            async def route_generate(self, profile, prompt, **kwargs):
                calls.append(prompt)
                return GenerationResponse(
                    content="print('{\"ok\": true}')",
                    model="fake", finish_reason="stop",
                    usage=UsageInfo(), is_mock=False,
                )

        asyncio.run(generate_program_source(
            FakeRouter(),
            profile=TaskProfile.for_task_type(TaskType.CODE_GENERATION),
            prompt="write a program",
            system_prompt="be brief",
        ))
        assert len(calls) == 1


class TestClarificationQuestion:
    def test_blocks_by_default(self):
        # Fail safe: a question nobody classified is treated as blocking rather
        # than silently letting the pipeline advance on unanswered information.
        question = ClarificationQuestion(question_id="Q1", question="?")
        assert question.blocking is True
        assert question.options == []

    def test_can_be_marked_non_blocking(self):
        question = ClarificationQuestion(question_id="Q1", question="?", blocking=False)
        assert question.blocking is False


class TestPaperToPdfPipeline:
    """Covers the delivery path end to end: paper IR -> LaTeX -> PDF -> package.

    This is the stage that had never executed, and it is where a figure or a
    table silently disappearing would cost the whole deliverable.
    """

    def _build(self, tmp_path, figure=True, table=True):
        from mathmodel.documents import FigureRecord
        from mathmodel.autopilot.deliver import SupportPackageBuilder, build_pdf

        figures = FigureRegistry()
        if figure:
            image = tmp_path / "figures" / "fig1.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.bar(["A", "B", "C"], [20, 40, 90])
            fig.savefig(image)
            plt.close(fig)
            figures.register(FigureRecord(
                figure_id="FIG-001", title="装备类别分布", caption="图 1 装备类别分布",
                artifact_path=str(image), source_execution_ids=["RUN-1"],
            ))

        tables = TableRegistry()
        if table:
            tables.register(TableBuilder().build_statistics_table(
                {"total_conflicts": 297, "num_plans": 150}, "EVD-EXECUTION",
            ))

        blocks = [ContentBlock(block_type=BlockType.PARAGRAPH, text="本文检测到 297 对时频冲突。")]
        if figure:
            blocks.append(ContentBlock(
                block_type=BlockType.FIGURE, text="图 1 装备类别分布",
                figure_ids=["FIG-001"],
            ))
        if table:
            blocks.append(ContentBlock(
                block_type=BlockType.TABLE, text="表 1 统计",
                table_ids=["TAB-001"],
            ))

        paper = PaperIR(
            title="时频冲突检测与消解",
            abstract="摘要：本文建立了时频冲突检测与消解模型，检测到 297 对冲突。",
            keywords=["时频冲突", "离散化", "整数规划"],
            sections=[
                PaperSection(title="问题重述", content_blocks=[
                    ContentBlock(block_type=BlockType.PARAGRAPH, text="某区域内 150 个用频装备提报计划。"),
                ]),
                PaperSection(title="模型求解与结果分析", content_blocks=blocks),
            ],
        )
        build_dir = tmp_path / "paper"
        build_dir.mkdir(parents=True, exist_ok=True)
        tex = build_cumcm_latex(paper, figures, tables, build_dir)
        return paper, figures, tables, tex, build_dir

    def test_latex_includes_figure_table_and_numbers(self, tmp_path):
        _, _, _, tex, build_dir = self._build(tmp_path)
        assert r"\includegraphics" in tex
        assert "fig1.png" in tex
        assert "297" in tex
        assert "时频冲突检测与消解" in tex
        assert (build_dir / "fig1.png").exists()

    def test_compiles_to_a_real_pdf(self, tmp_path):
        from mathmodel.autopilot.deliver import build_pdf

        _, _, _, tex, build_dir = self._build(tmp_path)
        result = build_pdf(tex, build_dir)
        if not result.compiler_available:
            pytest.skip("paper compilation image not available")
        assert result.success, result.reason or result.log_tail[-1500:]
        assert result.pdf_size > 5000
        assert Path(result.pdf_path).read_bytes()[:5] == b"%PDF-"

    def test_package_contains_every_deliverable(self, tmp_path):
        from mathmodel.autopilot.deliver import SupportPackageBuilder, build_pdf

        _, figures, tables, tex, build_dir = self._build(tmp_path)
        pdf = build_pdf(tex, build_dir)
        if not pdf.success:
            pytest.skip("paper compilation image not available")

        out_dir = tmp_path / "output"
        manifest = SupportPackageBuilder(out_dir).build(
            paper_pdf=Path(pdf.pdf_path),
            paper_tex=build_dir / "paper.tex",
            source_code=[build_dir / "paper.tex"],
            processed_data=[],
            original_files=[],
            figures=[Path(f.artifact_path) for f in figures.all()],
            tables=[],
            result_files=[],
            manifest_extra={"run_id": "RUN-1"},
        )

        assert (out_dir / "paper.pdf").exists()
        assert (out_dir / "manifest.json").exists()
        assert (out_dir / "support" / "necessary_supporting_files" / "paper.tex").exists()
        assert (out_dir / "support" / "figures" / "fig1.png").exists()
        assert manifest["paper_pdf_size"] > 5000

    def test_figure_missing_from_ir_is_detected(self, tmp_path):
        # The final consistency check keys off collect_figures/collect_tables.
        paper, _, _, _, _ = self._build(tmp_path, figure=False, table=True)
        assert paper.collect_figures() == []
        assert paper.collect_tables()

    def test_stale_figure_id_is_skipped_not_crashed(self, tmp_path):
        from mathmodel.autopilot.deliver import build_cumcm_latex

        paper = PaperIR(
            title="t",
            sections=[PaperSection(title="结果", content_blocks=[
                ContentBlock(block_type=BlockType.FIGURE, text="missing",
                             figure_ids=["FIG-999"]),
            ])],
        )
        build_dir = tmp_path / "paper"
        build_dir.mkdir()
        tex = build_cumcm_latex(paper, FigureRegistry(), TableRegistry(), build_dir)
        assert r"\includegraphics" not in tex


class TestSubprocessDecoding:
    """Hosts with a GBK locale must not lose subprocess output.

    xelatex and generated programs emit UTF-8 Chinese. Decoding that with the
    host locale raises inside the subprocess reader thread, which leaves
    stdout as None and surfaces later as "'NoneType' object is not
    subscriptable" - a message that points nowhere near the real cause.
    """

    def test_latex_log_with_chinese_survives(self, tmp_path):
        from mathmodel.autopilot.deliver import build_pdf

        build_dir = tmp_path / "paper"
        build_dir.mkdir()
        # \typeout writes this straight to stdout, so the captured output is
        # genuinely non-ASCII and cannot be decoded with a GBK locale.
        tex = (
            "\\documentclass[11pt]{ctexart}\n"
            "\\begin{document}\n"
            "\\typeout{中文日志：时频冲突检测与消解}\n"
            "测试正文。\n"
            "\\end{document}\n"
        )
        result = build_pdf(tex, build_dir)
        if not result.compiler_available:
            pytest.skip("paper compilation image not available")

        assert result.success, result.reason or (result.log_tail or "")[-1500:]
        assert result.pdf_path
        assert result.log_tail, "compiler log should have been captured"
        assert "中文日志" in result.log_tail

    def test_sandbox_decodes_chinese_stdout(self):
        import asyncio

        from mathmodel.sandbox.docker_backend import DockerSandboxBackend, SandboxLimits

        backend = DockerSandboxBackend(image="mathmodel-ai-autopilot:latest")
        if not backend.available:
            pytest.skip("docker sandbox not available")

        record = asyncio.run(backend.execute(
            code='print("中文诊断：开始求解")',
            limits=SandboxLimits(timeout_seconds=60),
        ))
        assert record.status.value == "success", record.stderr
        assert record.stdout is not None
        assert "中文诊断" in record.stdout


class TestTraceableNumbers:
    """The paper must not state a number that no verified source supports."""

    def _allowed(self, statistics=None,
                 problem_text="附件1给出3类共150个用频装备。频域离散化为100个频段。"):
        from mathmodel.autopilot.pipeline import _traceable_numbers

        return _traceable_numbers(
            statistics if statistics is not None else {"conflicts": 297},
            TableRegistry(),
            problem_text,
        )

    def test_nested_statistics_are_traceable(self):
        allowed = self._allowed({"p2": {"by_class": {"A": {"total": 20}}}})
        assert 20.0 in allowed

    def test_problem_statement_numbers_are_traceable(self):
        assert 150.0 in self._allowed()

    def test_table_cells_are_traceable(self):
        from mathmodel.autopilot.pipeline import _traceable_numbers

        tables = TableRegistry()
        tables.register(TableBuilder().build_statistics_table({"conflicts": 297}, "EVD"))
        assert 297.0 in _traceable_numbers({}, tables, "")

    def test_fabricated_class_count_is_flagged(self):
        from mathmodel.autopilot.pipeline import _untraceable_numbers

        # The real distribution is A=20, B=40, C=90; "各50个" is invented.
        allowed = self._allowed({
            "p2": {"by_class": {"A": {"total": 20}, "B": {"total": 40}, "C": {"total": 90}}}
        })
        text = "对附件1中150个用频计划（A、B、C类各50个）进行检测。"
        assert _untraceable_numbers(text, allowed) == {"50"}

    def test_verified_and_problem_numbers_are_not_flagged(self):
        from mathmodel.autopilot.pipeline import _untraceable_numbers

        allowed = self._allowed()
        text = "附件1给出150个计划，离散化为100个频段，共检出297对冲突，平移不超过10。"
        assert _untraceable_numbers(text, allowed) == set()

    def test_small_structural_integers_are_ignored(self):
        from mathmodel.autopilot.pipeline import _untraceable_numbers

        assert _untraceable_numbers("见第 3 节与表 2。", set()) == set()

    def test_untraceable_decimal_is_flagged(self):
        from mathmodel.autopilot.pipeline import _untraceable_numbers

        assert _untraceable_numbers("准确率为 87.5%。", set()) == {"87.5"}


class TestInlineMathPreservation:
    """Inline math in prose must survive escaping as real notation."""

    def test_inline_math_is_preserved(self):
        from mathmodel.paper.renderer import _latex_escape

        escaped = _latex_escape("以固定时间长度 $\\Delta t$ 为基本单位。")
        assert "$\\Delta t$" in escaped
        assert r"\textbackslash{}" not in escaped
        assert r"\$" not in escaped

    def test_plain_text_is_still_escaped(self):
        from mathmodel.paper.renderer import _latex_escape

        escaped = _latex_escape("比例 50% 与 A&B 以及 x_1")
        assert r"\%" in escaped
        assert r"\&" in escaped
        assert r"\_" in escaped

    def test_stray_dollar_is_escaped(self):
        from mathmodel.paper.renderer import _latex_escape

        assert _latex_escape("价格是 100$") == r"价格是 100\$"

    def test_math_span_still_cannot_smuggle_primitives(self):
        from mathmodel.paper.renderer import _check_latex_injection, _latex_escape

        tex = _latex_escape(r"公式 $\input{secret}$ 结束")
        assert _check_latex_injection(tex)

    def test_paper_with_inline_math_compiles(self, tmp_path):
        from mathmodel.autopilot.deliver import build_pdf, build_cumcm_latex

        paper = PaperIR(
            title="测试",
            sections=[PaperSection(title="模型建立", content_blocks=[
                ContentBlock(
                    block_type=BlockType.PARAGRAPH,
                    text="时间长度以 $\\Delta t$ 为单位，频率以 $\\Delta f$ 为单位。",
                ),
            ])],
        )
        build_dir = tmp_path / "paper"
        build_dir.mkdir()
        tex = build_cumcm_latex(paper, FigureRegistry(), TableRegistry(), build_dir)
        assert "$\\Delta t$" in tex
        result = build_pdf(tex, build_dir)
        if not result.compiler_available:
            pytest.skip("paper compilation image not available")
        assert result.success, (result.reason or "") + (result.log_tail or "")[-1200:]


class TestRetryCall:
    """A dropped provider connection must not destroy a completed run."""

    def test_retries_then_succeeds(self):
        import asyncio

        from mathmodel.autopilot.pipeline import CUMCMAutopilot

        calls = {"n": 0}

        async def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("reset by peer")
            return "ok"

        assert asyncio.run(CUMCMAutopilot._retry_call("t", flaky, max_attempts=3)) == "ok"
        assert calls["n"] == 3

    def test_raises_after_exhausting_attempts(self):
        import asyncio

        import pytest as _pytest

        from mathmodel.autopilot.pipeline import CUMCMAutopilot

        async def always_fails():
            raise ConnectionError("reset by peer")

        with _pytest.raises(RuntimeError, match="failed after 2 attempts"):
            asyncio.run(CUMCMAutopilot._retry_call("t", always_fails, max_attempts=2))


class TestTemplateHeaderGate:
    """The template's header row is part of the required answer."""

    def _checks(self, tmp_path, produced_headers, template_headers):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _make_xlsx(data_dir / "附件1.xlsx", ["编号"], [["A001"]])
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        _make_xlsx(out_dir / "result1.xlsx", produced_headers, [[1, "A001", "B002"]])
        return DeterministicVerifier(data_dir).run(
            outcome=_outcome({"conflicts": 1, "residual_conflicts": 0}),
            output_dir=out_dir,
            required_outputs=["result1.xlsx"],
            template_headers=template_headers,
        )

    def test_matching_headers_pass(self, tmp_path):
        headers = ["序号", "冲突装备1", "冲突设备2"]
        checks = self._checks(tmp_path, headers, {"result1.xlsx": headers})
        by_name = {c.name: c for c in checks}
        assert by_name["output_structure_matches_template"].status == CheckStatus.PASS

    def test_renamed_columns_fail_even_with_the_right_count(self, tmp_path):
        # The real run shipped these headings while the template asked for
        # 冲突装备1 / 冲突设备2. Column count alone must not pass this.
        checks = self._checks(
            tmp_path,
            ["序号", "用频装备编号1", "用频装备编号2"],
            {"result1.xlsx": ["序号", "冲突装备1", "冲突设备2"]},
        )
        by_name = {c.name: c for c in checks}
        check = by_name["output_structure_matches_template"]
        assert check.status == CheckStatus.FAIL
        assert "header does not match" in check.detail

    def test_whitespace_differences_are_tolerated(self, tmp_path):
        checks = self._checks(
            tmp_path,
            [" 序号 ", "冲突装备1", "冲突设备2"],
            {"result1.xlsx": ["序号", "冲突装备1", "冲突设备2"]},
        )
        by_name = {c.name: c for c in checks}
        assert by_name["output_structure_matches_template"].status == CheckStatus.PASS


class TestCodegenTemplateHeaders:
    """The solver must be told the template's exact column headers."""

    def test_prompt_lists_the_mandatory_headers(self):
        from mathmodel.autopilot.codegen import SolverCodeGenerator
        from mathmodel.domain.math_model import MathematicalModel

        prompt = SolverCodeGenerator._build_prompt(None, 
            model=MathematicalModel(
                name="m", objective="min", variables=[], constraints=[], equations=[]
            ),
            problem_text="题目",
            data_schema="{}",
            required_outputs=["result1.xlsx"],
            input_file_names=["附件1.xlsx"],
            output_headers={"result1.xlsx": ["序号", "冲突装备1", "冲突设备2"]},
        )
        assert "MANDATORY COLUMN HEADERS" in prompt
        assert "冲突装备1" in prompt
        assert "冲突设备2" in prompt

    def test_prompt_omits_the_section_without_templates(self):
        from mathmodel.autopilot.codegen import SolverCodeGenerator
        from mathmodel.domain.math_model import MathematicalModel

        prompt = SolverCodeGenerator._build_prompt(None, 
            model=MathematicalModel(
                name="m", objective="min", variables=[], constraints=[], equations=[]
            ),
            problem_text="题目",
            data_schema="{}",
            required_outputs=["out.csv"],
            input_file_names=["a.csv"],
        )
        assert "MANDATORY COLUMN HEADERS" not in prompt


class TestPerPlanCoverageGate:
    """A table keyed by plan ID must be answered for every plan."""

    def _run(self, tmp_path, rows, expected):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _make_xlsx(data_dir / "附件1.xlsx", ["用频装备编号"], [["A001"], ["A002"]])
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        _make_xlsx(out_dir / "result2.xlsx", ["用频装备编号", "是否撤销用频计划"], rows)
        return DeterministicVerifier(data_dir).run(
            outcome=_outcome({"residual_conflicts": 0}),
            output_dir=out_dir,
            required_outputs=["result2.xlsx"],
            template_headers={"result2.xlsx": ["用频装备编号", "是否撤销用频计划"]},
            expected_row_counts=expected,
        )

    def test_incomplete_per_plan_table_fails(self, tmp_path):
        checks = self._run(tmp_path, [["A001", "是"]], {"result2.xlsx": 2})
        check = {c.name: c for c in checks}["per_plan_table_covers_every_plan"]
        assert check.status == CheckStatus.FAIL
        assert "1 row(s) for 2 plan(s)" in check.detail

    def test_complete_per_plan_table_passes(self, tmp_path):
        checks = self._run(
            tmp_path, [["A001", "是"], ["A002", "否"]], {"result2.xlsx": 2}
        )
        check = {c.name: c for c in checks}["per_plan_table_covers_every_plan"]
        assert check.status == CheckStatus.PASS

    def test_no_per_plan_template_is_not_applicable(self, tmp_path):
        checks = self._run(tmp_path, [["A001", "是"]], {})
        check = {c.name: c for c in checks}["per_plan_table_covers_every_plan"]
        assert check.status == CheckStatus.NOT_APPLICABLE


class TestModelRepairTrigger:
    """A semantic disagreement must send the repair loop back to modeling."""

    def _check(self, name, category, status, detail):
        return VerificationCheck(
            name=name, category=category, status=status, detail=detail
        )

    def test_independent_disagreement_triggers_model_repair(self):
        from mathmodel.autopilot.pipeline import _needs_model_repair

        report = VerificationReport(overall=CheckStatus.FAIL, checks=[
            self._check("independent_independent_recomputation", "independent",
                        CheckStatus.FAIL,
                        "plans=150 recomputed_total=297 claimed=237 match=False"),
        ])
        assert _needs_model_repair(report)

    def test_pure_output_defect_does_not_trigger_model_repair(self):
        from mathmodel.autopilot.pipeline import _needs_model_repair

        report = VerificationReport(overall=CheckStatus.FAIL, checks=[
            self._check("required_outputs_present", "outputs", CheckStatus.FAIL,
                        "missing=['result1.xlsx']"),
            self._check("output_structure_matches_template", "outputs",
                        CheckStatus.FAIL, "result1.xlsx: header does not match"),
        ])
        assert not _needs_model_repair(report)

    def test_passing_report_does_not_trigger_repair(self):
        from mathmodel.autopilot.pipeline import _needs_model_repair

        report = VerificationReport(overall=CheckStatus.PASS, checks=[
            self._check("independent_x", "independent", CheckStatus.PASS, "ok"),
        ])
        assert not _needs_model_repair(report)


class TestModelRepairScope:
    """Only input-derived disagreements implicate the model."""

    def test_residual_conflicts_do_not_repair_the_model(self):
        from mathmodel.autopilot.pipeline import _needs_model_repair

        report = VerificationReport(overall=CheckStatus.FAIL, checks=[
            VerificationCheck(
                name="independent_constraint_satisfaction", category="independent",
                status=CheckStatus.FAIL,
                detail="problem2 remaining conflicts after applying adjustments = 39",
            ),
        ])
        assert not _needs_model_repair(report)

    def test_baseline_comparison_does_not_repair_the_model(self):
        from mathmodel.autopilot.pipeline import _needs_model_repair

        report = VerificationReport(overall=CheckStatus.FAIL, checks=[
            VerificationCheck(
                name="independent_baseline_comparison", category="independent",
                status=CheckStatus.FAIL,
                detail="solution is worse than the trivial baseline",
            ),
        ])
        assert not _needs_model_repair(report)


class TestReferentialIntegrityScope:
    """Only columns that reference an existing entity are checked."""

    def _run(self, tmp_path, headers, rows):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _make_xlsx(data_dir / "附件1.xlsx", ["用频装备编号"], [["A001"], ["B002"]])
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        _make_xlsx(out_dir / "result3.xlsx", headers, rows)
        return {
            c.name: c for c in DeterministicVerifier(data_dir).run(
                outcome=_outcome({"residual_conflicts": 0}),
                output_dir=out_dir,
                required_outputs=["result3.xlsx"],
                template_headers={},
            )
        }

    def test_serial_numbers_for_new_items_are_not_flagged(self, tmp_path):
        # result3 numbers newly added plans, so "newC10" is not a broken
        # reference to the input data.
        checks = self._run(
            tmp_path,
            ["新增用频装备序号", "调整后频段区间", "调整后时间区间"],
            [["newC10", "[0,10)", "[0,5)"], ["newC100", "[10,20)", "[0,5)"]],
        )
        assert checks["referential_integrity"].status == CheckStatus.PASS

    def test_unknown_equipment_ids_are_still_flagged(self, tmp_path):
        checks = self._run(
            tmp_path,
            ["用频装备编号", "调整后频段区间"],
            [["A001", "[0,10)"], ["Z999", "[10,20)"]],
        )
        assert checks["referential_integrity"].status == CheckStatus.FAIL
        assert "Z999" in checks["referential_integrity"].detail


class TestVacuousVerifierGuard:
    """A verifier that read no input must not be believed."""

    def _summary(self, **extra):
        base = {
            "checks": [{"name": "constraint_1", "status": "PASS", "detail": "ok"}],
            "statistics": {},
            "overall": "PASS",
        }
        base.update(extra)
        return base

    def test_missing_input_records_fails(self):
        from mathmodel.autopilot.verify import checks_from_independent_summary

        checks, _ = checks_from_independent_summary(self._summary())
        guard = {c.name: c for c in checks}["independent_read_the_input"]
        assert guard.status == CheckStatus.FAIL
        assert "no `input_records` field" in guard.detail

    def test_zero_input_records_fails(self):
        from mathmodel.autopilot.verify import checks_from_independent_summary

        checks, _ = checks_from_independent_summary(self._summary(input_records=0))
        guard = {c.name: c for c in checks}["independent_read_the_input"]
        assert guard.status == CheckStatus.FAIL
        assert "input_records=0" in guard.detail

    def test_positive_input_records_passes(self):
        from mathmodel.autopilot.verify import checks_from_independent_summary

        checks, _ = checks_from_independent_summary(self._summary(input_records=150))
        guard = {c.name: c for c in checks}["independent_read_the_input"]
        assert guard.status == CheckStatus.PASS

    def test_a_vacuous_verifier_cannot_produce_a_passing_report(self):
        from mathmodel.autopilot.verify import (
            build_report,
            checks_from_independent_summary,
        )
        from mathmodel.domain.math_model import MathematicalModel

        checks, _ = checks_from_independent_summary(self._summary(input_records=0))
        report = build_report(
            MathematicalModel(name="m", objective="o"), "run", checks
        )
        assert not report.passed


class TestVerifierGetsDataSchema:
    """The verifier must know the real input columns to parse the data."""

    def test_prompt_includes_the_real_schema(self):
        import asyncio
        import json as _json

        from mathmodel.autopilot.verify import IndependentVerifier
        from mathmodel.domain.math_model import MathematicalModel

        captured = {}

        class FakeRouter:
            async def route_generate(self, **kwargs):
                captured.update(kwargs)
                raise RuntimeError("stop here")

        verifier = IndependentVerifier(FakeRouter(), runner=None)
        schema = _json.dumps({
            "tables": [{
                "file": "附件1.xlsx",
                "headers": ["用频装备编号", "频段区间", "时间区间", "间隔时长", "使用次数"],
            }]
        }, ensure_ascii=False)
        with pytest.raises(RuntimeError):
            asyncio.run(verifier.generate(
                model=MathematicalModel(name="m", objective="o"),
                problem_text="题目",
                required_outputs=["result1.xlsx"],
                solver_statistics={},
                available_inputs=["附件1.csv"],
                data_schema=schema,
            ))
        prompt = captured["prompt"]
        assert "INPUT DATA SCHEMA" in prompt
        assert "使用次数" in prompt

    def test_prompt_omits_the_section_without_a_schema(self):
        import asyncio

        from mathmodel.autopilot.verify import IndependentVerifier
        from mathmodel.domain.math_model import MathematicalModel

        captured = {}

        class FakeRouter:
            async def route_generate(self, **kwargs):
                captured.update(kwargs)
                raise RuntimeError("stop here")

        verifier = IndependentVerifier(FakeRouter(), runner=None)
        with pytest.raises(RuntimeError):
            asyncio.run(verifier.generate(
                model=MathematicalModel(name="m", objective="o"),
                problem_text="题目",
                required_outputs=["result1.xlsx"],
                solver_statistics={},
                available_inputs=["a.csv"],
            ))
        assert "INPUT DATA SCHEMA" not in captured["prompt"]


class TestModelRepairNeedsARealRecomputation:
    """A vacuous verifier must not send the loop back to modeling."""

    def test_vacuous_verifier_does_not_repair_the_model(self):
        from mathmodel.autopilot.pipeline import _needs_model_repair

        report = VerificationReport(overall=CheckStatus.FAIL, checks=[
            VerificationCheck(
                name="independent_read_the_input", category="independent",
                status=CheckStatus.FAIL,
                detail="Independent verifier reported input_records=0",
            ),
            VerificationCheck(
                name="independent_independent_recomputation", category="independent",
                status=CheckStatus.FAIL,
                detail="Parsed 0 plans from 附件1.csv",
            ),
        ])
        assert not _needs_model_repair(report)

    def test_real_disagreement_still_repairs_the_model(self):
        from mathmodel.autopilot.pipeline import _needs_model_repair

        report = VerificationReport(overall=CheckStatus.FAIL, checks=[
            VerificationCheck(
                name="independent_read_the_input", category="independent",
                status=CheckStatus.PASS, detail="parsed 150 records",
            ),
            VerificationCheck(
                name="independent_independent_recomputation", category="independent",
                status=CheckStatus.FAIL,
                detail="recomputed_total=297 claimed=237",
            ),
        ])
        assert _needs_model_repair(report)


class TestEvidenceRequiredForConstraintChecks:
    """A PASS with no evidence is not coverage."""

    def _summary(self, checks):
        return {"checks": checks, "input_records": 150, "statistics": {}, "overall": "PASS"}

    def test_constraint_pass_without_evidence_becomes_not_run(self):
        from mathmodel.autopilot.verify import checks_from_independent_summary

        checks, _ = checks_from_independent_summary(self._summary([
            {"name": "constraint_6", "status": "PASS", "detail": "Check of: c6"},
        ]))
        check = {c.name: c for c in checks}["independent_constraint_6"]
        assert check.status == CheckStatus.NOT_RUN
        assert "not actually checked" in check.detail

    def test_constraint_pass_with_evidence_is_kept(self):
        from mathmodel.autopilot.verify import checks_from_independent_summary

        checks, _ = checks_from_independent_summary(self._summary([
            {"name": "constraint_6", "status": "PASS", "detail": "ok",
             "evidence": {"checked": 150}},
        ]))
        assert {c.name: c for c in checks}["independent_constraint_6"].status == CheckStatus.PASS

    def test_constraint_failure_without_evidence_still_fails(self):
        from mathmodel.autopilot.verify import checks_from_independent_summary

        checks, _ = checks_from_independent_summary(self._summary([
            {"name": "constraint_6", "status": "FAIL", "detail": "3 plans changed two parameters"},
        ]))
        assert {c.name: c for c in checks}["independent_constraint_6"].status == CheckStatus.FAIL

    def test_a_report_with_a_not_run_constraint_does_not_pass(self):
        from mathmodel.autopilot.verify import (
            build_report,
            checks_from_independent_summary,
        )
        from mathmodel.domain.math_model import MathematicalModel

        checks, _ = checks_from_independent_summary(self._summary([
            {"name": "constraint_6", "status": "PASS", "detail": "Check of: c6"},
        ]))
        report = build_report(MathematicalModel(name="m", objective="o"), "run", checks)
        assert not report.passed


class TestReportedVerificationMatchesTheGatingReport:
    """The result must report the verdict that gated the run."""

    def test_stale_report_is_not_reported(self, tmp_path):
        import json as _json
        import os
        import time

        from mathmodel.autopilot.pipeline import CUMCMAutopilot
        from mathmodel.autopilot.state import RunState

        run_dir = tmp_path / "run"
        (run_dir / "artifacts" / "verify").mkdir(parents=True)
        (run_dir / "output").mkdir(parents=True)
        state = RunState.create(run_dir)
        state.status = RunStatus.COMPLETED
        state.save()

        verify_dir = run_dir / "artifacts" / "verify"
        # A stale failure with a HIGHER number than the report that passed.
        (verify_dir / "report3.json").write_text(
            _json.dumps({"overall": "FAIL"}), encoding="utf-8"
        )
        time.sleep(0.01)
        (verify_dir / "report1.json").write_text(
            _json.dumps({"overall": "PASS"}), encoding="utf-8"
        )

        result = CUMCMAutopilot(workspace=str(tmp_path))._result(state)
        assert result.verification == "PASS"


class TestTruncatedStructuredResponseRetry:
    """A truncated JSON body can never parse, so retry with thinking off."""

    def _provider(self, responses):
        from mathmodel.providers.openai import OpenAIProvider

        provider = OpenAIProvider(api_key="k", base_url="http://x/v1")
        calls = []

        async def fake_chat(client, **kwargs):
            calls.append(kwargs)
            return responses[min(len(calls) - 1, len(responses) - 1)]

        provider._chat_completion = fake_chat
        provider._get_client = lambda: object()
        return provider, calls

    def _response(self, finish_reason, content):
        from types import SimpleNamespace

        return SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            )],
            model="m",
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    def test_truncated_response_is_retried_with_thinking_disabled(self):
        import asyncio

        from mathmodel.providers.base import StructuredGenerationRequest

        class Tiny(pydantic.BaseModel):
            problem_id: str = ""

        provider, calls = self._provider([
            self._response("length", '{"problem_id": "p"'),
            self._response("stop", '{"problem_id": "p"}'),
        ])
        result = asyncio.run(provider.structured_generate(
            StructuredGenerationRequest(prompt="p", output_schema=Tiny)
        ))
        assert len(calls) == 2
        assert calls[1].get("extra_body") == {"thinking": {"type": "disabled"}}
        assert result.problem_id == "p"

    def test_untruncated_response_is_not_retried(self):
        import asyncio

        from mathmodel.providers.base import StructuredGenerationRequest

        class Tiny(pydantic.BaseModel):
            problem_id: str = ""

        provider, calls = self._provider([
            self._response("stop", '{"problem_id": "p"}'),
        ])
        asyncio.run(provider.structured_generate(
            StructuredGenerationRequest(prompt="p", output_schema=Tiny)
        ))
        assert len(calls) == 1
        assert "extra_body" not in calls[0]


class TestVerifierDisagreementFeedback:
    """A disagreement must be fed back to the verifier too, not only the model."""

    def test_disagreement_produces_verifier_feedback(self):
        import inspect

        from mathmodel.autopilot import pipeline

        source = inspect.getsource(pipeline.CUMCMAutopilot)
        assert "carried_verifier_feedback = [" in source
        assert "DISAGREE with the solver" in source
        assert "re-derive" in source

    def test_feedback_is_only_carried_on_actual_failure(self):
        from mathmodel.autopilot.verify import compare_statistics

        agree = compare_statistics({"a": 297}, {"a": 297})
        assert agree.status == CheckStatus.PASS

        disagree = compare_statistics({"a": 297}, {"a": 237})
        assert disagree.status == CheckStatus.FAIL
        assert "297" in disagree.detail and "237" in disagree.detail


class TestObjectivesGate:
    """A verifier that never judges the answer must not pass the run."""

    def _summary(self, objectives=None, checks=None):
        s = {
            "checks": checks or [{"name": "constraint_1", "status": "PASS",
                                  "detail": "input range ok", "evidence": {"n": 150}}],
            "input_records": 150,
            "statistics": {},
            "overall": "PASS",
        }
        if objectives is not None:
            s["objectives"] = objectives
        return s

    def _check(self, summary, outputs=("result1.xlsx", "result2.xlsx")):
        from mathmodel.autopilot.verify import checks_from_independent_summary

        checks, _ = checks_from_independent_summary(
            summary, required_outputs=list(outputs)
        )
        return {c.name: c for c in checks}["independent_objectives_achieved"]

    def test_missing_objectives_fails(self):
        check = self._check(self._summary())
        assert check.status == CheckStatus.FAIL
        assert "never stated whether the solution actually achieves" in check.detail

    def test_output_without_a_verdict_fails(self):
        check = self._check(self._summary(objectives=[
            {"output": "result1.xlsx", "achieved": True, "evidence": "297 pairs"},
        ]))
        assert check.status == CheckStatus.FAIL
        assert "result2.xlsx" in check.detail

    def test_unachieved_objective_fails(self):
        check = self._check(self._summary(objectives=[
            {"output": "result1.xlsx", "achieved": True, "evidence": "297 pairs"},
            {"output": "result2.xlsx", "achieved": False, "evidence": "297 conflicts remain"},
        ]))
        assert check.status == CheckStatus.FAIL
        assert "297 conflicts remain" in check.detail

    def test_all_outputs_achieved_passes(self):
        check = self._check(self._summary(objectives=[
            {"output": "result1.xlsx", "achieved": True, "evidence": "297 pairs"},
            {"output": "result2.xlsx", "achieved": True, "evidence": "0 residual"},
        ]))
        assert check.status == CheckStatus.PASS

    def test_the_observed_false_pass_is_now_blocked(self):
        """r05: every check was an input-data property; the answer was unsolved."""
        from mathmodel.autopilot.verify import (
            build_report,
            checks_from_independent_summary,
        )
        from mathmodel.domain.math_model import MathematicalModel

        summary = self._summary(checks=[
            {"name": "constraint_1", "status": "PASS", "detail": "band ranges within [0,100]", "evidence": {"n": 150}},
            {"name": "constraint_2", "status": "PASS", "detail": "use counts positive", "evidence": {"v": [3, 4, 12]}},
            {"name": "constraint_3", "status": "PASS", "detail": "interval lengths positive", "evidence": {"n": 150}},
            {"name": "feasibility", "status": "PASS", "detail": "checked 150 adjustments", "evidence": {"n": 150}},
        ])
        checks, _ = checks_from_independent_summary(
            summary, required_outputs=["result1.xlsx", "result2.xlsx", "result3.xlsx", "result4.xlsx"]
        )
        report = build_report(MathematicalModel(name="m", objective="o"), "run", checks)
        assert not report.passed

    def test_no_required_outputs_skips_the_objective_gate(self):
        from mathmodel.autopilot.verify import checks_from_independent_summary

        checks, _ = checks_from_independent_summary(self._summary())
        assert "independent_objectives_achieved" not in {c.name for c in checks}


class TestBlockedRunNeverReportsRunning:
    """A run that returns blockers has stopped, so it must not say `running`."""

    def _state(self, tmp_path, status):
        from mathmodel.autopilot.state import RunState

        run_dir = tmp_path / "run"
        (run_dir / "output").mkdir(parents=True)
        state = RunState.create(run_dir)
        state.status = status
        state.save()
        return state

    def test_blockers_settle_a_running_run(self, tmp_path):
        from mathmodel.autopilot.pipeline import CUMCMAutopilot
        from mathmodel.autopilot.state import RunStatus

        state = self._state(tmp_path, RunStatus.RUNNING)
        result = CUMCMAutopilot(workspace=str(tmp_path))._result(
            state, blockers=["[independent] constraint_1: 297 residual conflicts"]
        )
        assert result.status == "blocked"
        assert state.status == RunStatus.BLOCKED

    def test_a_completed_run_is_not_downgraded(self, tmp_path):
        from mathmodel.autopilot.pipeline import CUMCMAutopilot
        from mathmodel.autopilot.state import RunStatus

        state = self._state(tmp_path, RunStatus.COMPLETED)
        result = CUMCMAutopilot(workspace=str(tmp_path))._result(
            state, blockers=["should not happen"]
        )
        assert result.status == "completed"

    def test_no_blockers_leaves_a_running_run_alone(self, tmp_path):
        from mathmodel.autopilot.pipeline import CUMCMAutopilot
        from mathmodel.autopilot.state import RunStatus

        state = self._state(tmp_path, RunStatus.RUNNING)
        result = CUMCMAutopilot(workspace=str(tmp_path))._result(state)
        assert result.status == "running"


from mathmodel.agents.base import AgentStatus
from mathmodel.domain.math_model import (
    KIND_DUPLICATE_SYMBOL,
    KIND_SYMBOL_COLLISION,
    KIND_UNKNOWN_DEPENDENCY,
    KIND_UNKNOWN_PARAMETER,
    KIND_UNKNOWN_VARIABLE,
    MathematicalModel,
    closure_issues,
    parse_closure_issues,
)


class TestSymbolClosureRepair:
    """#25: undeclared symbols must yield a targeted repair, not a blind retry."""

    @staticmethod
    def _payload(**overrides):
        base = {
            "model_id": "M1",
            "name": "closure test model",
            "description": "d",
            "variables": [
                {"variable_id": "v_x", "symbol": "x", "name": "x"},
            ],
            "parameters": [
                {"parameter_id": "p_T", "symbol": "T", "name": "T",
                 "value": 10.0, "status": "known"},
            ],
            "objectives": [
                {"objective_id": "o1", "name": "obj", "sense": "minimize",
                 "expression": "x", "meaning": "m", "priority": 1},
            ],
            "constraints": [],
            "equations": [
                {"equation_id": "EQ1", "name": "e1", "expression": "x = T",
                 "variable_ids": ["v_x"], "parameter_ids": ["p_T"]},
            ],
        }
        base.update(overrides)
        return base

    # ── criterion 1: an undeclared symbol must FAIL ──────────────────
    def test_undeclared_symbols_fail_validation(self):
        bad = self._payload(equations=[
            {"equation_id": "EQ1", "name": "e1", "expression": "x = T + M_j",
             "variable_ids": ["v_x", "v_y_missing"],
             "parameter_ids": ["p_T", "p_M_missing"]},
        ])
        with pytest.raises(Exception) as excinfo:
            MathematicalModel.model_validate(bad)
        assert "unknown variable v_y_missing" in str(excinfo.value)
        assert "unknown parameter p_M_missing" in str(excinfo.value)

    # ── criterion 2: the failure carries the concrete symbol lists ───
    def test_failure_message_parses_back_into_the_exact_symbols(self):
        bad = self._payload(equations=[
            {"equation_id": "EQ1", "name": "e1", "expression": "x = T + M_j",
             "variable_ids": ["v_x", "v_y_missing"],
             "parameter_ids": ["p_T", "p_M_missing"],
             "dependencies": ["EQ_missing"]},
        ])
        with pytest.raises(Exception) as excinfo:
            MathematicalModel.model_validate(bad)

        issues = parse_closure_issues(str(excinfo.value))
        assert sorted(i.symbol for i in issues if i.kind == KIND_UNKNOWN_VARIABLE) == ["v_y_missing"]
        assert sorted(i.symbol for i in issues if i.kind == KIND_UNKNOWN_PARAMETER) == ["p_M_missing"]
        assert sorted(i.symbol for i in issues if i.kind == KIND_UNKNOWN_DEPENDENCY) == ["EQ_missing"]
        assert {i.location for i in issues} == {"EQ1"}

    def test_duplicate_and_colliding_symbols_are_reported(self):
        bad = self._payload(
            variables=[
                {"variable_id": "v_x", "symbol": "x", "name": "x"},
                {"variable_id": "v_x2", "symbol": "x", "name": "x again"},
                {"variable_id": "v_T", "symbol": "T", "name": "T as a variable"},
            ],
        )
        with pytest.raises(Exception) as excinfo:
            MathematicalModel.model_validate(bad)
        issues = parse_closure_issues(str(excinfo.value))
        kinds = {i.kind for i in issues}
        assert KIND_DUPLICATE_SYMBOL in kinds
        assert KIND_SYMBOL_COLLISION in kinds

    def test_a_clean_model_yields_no_issues(self):
        model = MathematicalModel.model_validate(self._payload())
        assert closure_issues(model) == []
        assert parse_closure_issues("some unrelated error") == []

    def test_describe_and_parse_round_trip(self):
        bad = self._payload(equations=[
            {"equation_id": "EQ7", "name": "e", "expression": "x",
             "variable_ids": ["v_missing"], "parameter_ids": ["p_missing"]},
        ])
        with pytest.raises(Exception) as excinfo:
            MathematicalModel.model_validate(bad)
        issues = parse_closure_issues(str(excinfo.value))
        assert len(issues) == 2
        for issue in issues:
            assert issue.symbol in str(excinfo.value)
            assert issue.location == "EQ7"

    # ── criteria 3 and 4: the next attempt is fed the exact lists ────
    @staticmethod
    def _state():
        from mathmodel.domain.analysis import ProblemAnalysis
        from mathmodel.models.problem_state import ProblemState

        state = ProblemState(title="closure")
        state.metadata_ = {
            "analysis": ProblemAnalysis(
                background="b", core_problem="c", objectives=["o"],
            ).model_dump()
        }
        state.raw_problem = "A problem statement."
        state.selected_model = {"candidate_id": "CAND-1"}
        return state

    async def test_retry_prompt_contains_the_exact_missing_symbols(self):
        from mathmodel.agents.math_modeler import MathModeler

        bad = self._payload(equations=[
            {"equation_id": "EQ1", "name": "e1", "expression": "x = T + M_j",
             "variable_ids": ["v_x", "v_y_missing"],
             "parameter_ids": ["p_T", "p_M_missing"]},
        ])
        with pytest.raises(Exception) as excinfo:
            MathematicalModel.model_validate(bad)
        real_error = excinfo.value

        good = MathematicalModel.model_validate(self._payload())

        class FakeRouter:
            def __init__(self):
                self.prompts = []

            async def route_structured_generate(self, **kwargs):
                self.prompts.append(kwargs["prompt"])
                if len(self.prompts) == 1:
                    raise real_error
                return good

        router = FakeRouter()
        result = await MathModeler(router).run(self._state())

        assert result.status == AgentStatus.COMPLETED
        assert len(router.prompts) == 2
        repair_prompt = router.prompts[1]

        # criterion 3: the repair prompt names the exact symbols
        assert "v_y_missing" in repair_prompt
        assert "p_M_missing" in repair_prompt
        assert "SYMBOL CLOSURE FAILURE" in repair_prompt

        # criterion 4: it is targeted, not a blind regeneration
        assert "Keep every other part of the model as it is" in repair_prompt
        assert "do not rebuild the model from scratch" in repair_prompt

    async def test_first_attempt_has_no_corrective_block(self):
        from mathmodel.agents.math_modeler import MathModeler

        good = MathematicalModel.model_validate(self._payload())

        class FakeRouter:
            def __init__(self):
                self.prompts = []

            async def route_structured_generate(self, **kwargs):
                self.prompts.append(kwargs["prompt"])
                return good

        router = FakeRouter()
        result = await MathModeler(router).run(self._state())
        assert result.status == AgentStatus.COMPLETED
        assert len(router.prompts) == 1
        assert "SYMBOL CLOSURE FAILURE" not in router.prompts[0]

    async def test_a_non_closure_error_still_gets_generic_feedback(self):
        from mathmodel.agents.math_modeler import MathModeler

        good = MathematicalModel.model_validate(self._payload())

        class FakeRouter:
            def __init__(self):
                self.prompts = []

            async def route_structured_generate(self, **kwargs):
                self.prompts.append(kwargs["prompt"])
                if len(self.prompts) == 1:
                    raise ValueError("Provider returned no usable JSON after retry")
                return good

        router = FakeRouter()
        result = await MathModeler(router).run(self._state())
        assert result.status == AgentStatus.COMPLETED
        assert "SYMBOL CLOSURE FAILURE" not in router.prompts[1]
        assert "could not be parsed" in router.prompts[1]

    # ── criterion 5: the repaired model satisfies closure ────────────
    def test_declaring_the_missing_symbols_clears_the_failure(self):
        broken = self._payload(equations=[
            {"equation_id": "EQ1", "name": "e1", "expression": "x = T + M_j",
             "variable_ids": ["v_x", "v_y_missing"],
             "parameter_ids": ["p_T", "p_M_missing"]},
        ])
        with pytest.raises(Exception):
            MathematicalModel.model_validate(broken)

        repaired = self._payload(
            variables=[
                {"variable_id": "v_x", "symbol": "x", "name": "x"},
                {"variable_id": "v_y_missing", "symbol": "v_y_missing",
                 "name": "declared now"},
            ],
            parameters=[
                {"parameter_id": "p_T", "symbol": "T", "name": "T",
                 "value": 10.0, "status": "known"},
                {"parameter_id": "p_M_missing", "symbol": "p_M_missing",
                 "name": "declared now", "value": 1.0, "status": "known"},
            ],
            equations=[
                {"equation_id": "EQ1", "name": "e1", "expression": "x = T + M_j",
                 "variable_ids": ["v_x", "v_y_missing"],
                 "parameter_ids": ["p_T", "p_M_missing"]},
            ],
        )
        model = MathematicalModel.model_validate(repaired)
        assert closure_issues(model) == []


class TestR05FalsePassRegression:
    """Permanent negative fixture: a real report that was accepted as PASS
    while the answer was wrong. The gate must never accept it again."""

    @staticmethod
    def _fixture():
        import json as _json
        from pathlib import Path as _Path

        path = _Path(__file__).resolve().parent / "fixtures" / "r05_false_pass.json"
        return _json.loads(path.read_text(encoding="utf-8"))

    def test_fixture_records_the_ground_truth(self):
        fixture = self._fixture()
        truth = fixture["independent_ground_truth"]
        assert truth["verdict"] == "WRONG"
        # The answer left every conflict unresolved and produced an empty result3.
        assert truth["result2_residual_conflicts"] == 297
        assert truth["result2_cancelled"] == 0
        assert truth["result3_rows"] == 0
        assert fixture["verifier_summary"]["overall"] == "PASS"

    def test_the_old_gate_would_still_accept_it(self):
        """Guards the regression: without the objectives contract this passes."""
        from mathmodel.autopilot.verify import (
            build_report,
            checks_from_independent_summary,
        )
        from mathmodel.domain.math_model import MathematicalModel

        summary = self._fixture()["verifier_summary"]
        checks, _ = checks_from_independent_summary(summary)
        report = build_report(MathematicalModel(name="m", objective="o"), "r05", checks)
        assert report.passed

    def test_the_current_gate_rejects_it(self):
        from mathmodel.autopilot.verify import (
            build_report,
            checks_from_independent_summary,
        )
        from mathmodel.domain.math_model import MathematicalModel

        fixture = self._fixture()
        checks, _ = checks_from_independent_summary(
            fixture["verifier_summary"],
            required_outputs=fixture["required_outputs"],
        )
        report = build_report(MathematicalModel(name="m", objective="o"), "r05", checks)
        assert not report.passed

        by_name = {c.name: c for c in checks}
        assert by_name["independent_objectives_achieved"].status == CheckStatus.FAIL

    def test_every_check_in_the_fixture_is_an_input_property(self):
        """Documents WHY it slipped through: no check judged the answer."""
        fixture = self._fixture()
        names = [c["name"] for c in fixture["verifier_summary"]["checks"]]
        assert names, "fixture must carry the verifier's checks"
        assert not any(n.startswith("objective") for n in names)
        # The one constraint check that ran only re-derived result1 (detection).
        satisfaction = [c for c in fixture["verifier_summary"]["checks"]
                        if c["name"] == "constraint_satisfaction"]
        assert satisfaction and "result1" in satisfaction[0]["detail"]

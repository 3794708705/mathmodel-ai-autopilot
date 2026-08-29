"""Phase 4-6 Integration Acceptance tests.

Runs the full production chain:
MathematicalModel → ValidationGate → Compiler → Solver → Validation →
Sensitivity → Robustness → RedTeam → QualityGate → Evidence → Claims →
Figures/Tables → PaperIR → Numerical/Equation consistency → SubmissionCheck

Plus the 8 required cross-phase adversarial tests.
"""

import hashlib
import os
import tempfile

import pytest

from mathmodel.domain.math_model import (
    MathematicalModel, Variable, Parameter, Objective, Constraint,
    ConstraintRelation, ObjectiveSense, VariableType,
)
from mathmodel.domain.verification import (
    VerificationStatus, GateStatus, IssueSeverity, RedTeamReport, RedTeamIssue,
    RepairPlan, RepairAction, RepairType,
)
from mathmodel.solver import SimpleLPCompiler, solve_lp_scipy, SolverStatus
from mathmodel.verification import MathematicalValidationGate, build_validation_report
from mathmodel.agents.verification_agents import (
    SensitivityAgent, RobustnessAgent, RedTeamAgent, ModelRepairAgent,
    VerificationQualityGate,
)
from mathmodel.agents.code_agent import CodeAgent, check_model_code_drift
from mathmodel.evidence import (
    EvidenceStore, EvidenceRef, EvidenceSourceType, Claim, ClaimType,
    ClaimStatus,
)
from mathmodel.documents import (
    FigureRecord, FigureType, FigureRegistry, verify_figure,
    TableRecord, TableType, TableRegistry, TableCell, verify_table,
)
from mathmodel.paper import PaperIR, PaperSection, ContentBlock, BlockType
from mathmodel.paper.renderer import LaTeXRenderer
from mathmodel.submission import (
    CompetitionProfile, SubmissionCheckAgent, SubmissionStatus,
    NumericalConsistencyChecker,
)


def make_model(version: int = 1) -> MathematicalModel:
    return MathematicalModel(
        model_id="MODEL-INTEG",
        name="Integration LP",
        version=version,
        variables=[
            Variable(variable_id="VAR-x1", symbol="x1", name="P1", variable_type=VariableType.CONTINUOUS, lower_bound=0.0),
            Variable(variable_id="VAR-x2", symbol="x2", name="P2", variable_type=VariableType.CONTINUOUS, lower_bound=0.0),
            Variable(variable_id="VAR-x3", symbol="x3", name="P3", variable_type=VariableType.CONTINUOUS, lower_bound=0.0),
        ],
        parameters=[
            Parameter(parameter_id="PAR-p1", symbol="p1", name="Profit P1", value=30.0, source="Problem"),
            Parameter(parameter_id="PAR-p2", symbol="p2", name="Profit P2", value=25.0, source="Problem"),
            Parameter(parameter_id="PAR-p3", symbol="p3", name="Profit P3", value=20.0, source="Problem"),
        ],
        objectives=[
            Objective(objective_id="OBJ-1", name="Max Profit", sense=ObjectiveSense.MAXIMIZE, expression="30*x1 + 25*x2 + 20*x3"),
        ],
        constraints=[
            Constraint(constraint_id="CON-1", name="M1", expression="2*x1 + 1.5*x2 + 1*x3", relation=ConstraintRelation.LE, rhs=40.0),
            Constraint(constraint_id="CON-2", name="M2", expression="1.5*x1 + 2*x2 + 1*x3", relation=ConstraintRelation.LE, rhs=35.0),
            Constraint(constraint_id="CON-3", name="P1 demand", expression="x1", relation=ConstraintRelation.LE, rhs=15.0),
            Constraint(constraint_id="CON-4", name="P3 min", expression="x3", relation=ConstraintRelation.GE, rhs=5.0),
        ],
    )


def solve_model(model: MathematicalModel):
    compiled = SimpleLPCompiler.compile(model)
    compiled["model_id"] = model.model_id
    return solve_lp_scipy(compiled)


# ═══════════════════════════════════════════════════════════════
# 1. Full production-path E2E
# ═══════════════════════════════════════════════════════════════

class TestFullChain:
    def test_end_to_end_production_path(self):
        # Phase 4: model + gate + compile + solve
        model = make_model(version=1)
        gate = MathematicalValidationGate()
        gate_result = gate.validate(model)
        assert gate_result.status == GateStatus.PASS

        code_agent = CodeAgent()
        code_result = code_agent.generate(model)
        assert code_result.status == "completed"
        assert check_model_code_drift(model, code_result.mappings) == []

        solver_result = solve_model(model)
        assert solver_result.status == SolverStatus.OPTIMAL
        assert solver_result.objective_value == 700.0
        assert solver_result.model_version == 1

        # Phase 5: validation, sensitivity, robustness, red team, quality gate
        validation = build_validation_report(model, solver_result)
        assert validation.overall_status == GateStatus.PASS
        assert validation.model_version == 1

        sensitivity = SensitivityAgent().run(model, parameter_ids=["PAR-p1"])
        robustness = RobustnessAgent().run(model)
        red_team = RedTeamAgent().review(model, solver_result, validation, sensitivity, robustness)
        assert red_team.critical_count == 0

        quality = VerificationQualityGate.evaluate(validation, red_team, sensitivity, robustness)
        assert quality in (VerificationStatus.VERIFIED.value, VerificationStatus.VERIFIED_WITH_WARNINGS.value)

        # Phase 6: evidence → claims → figures/tables → paper → submission
        evidence = EvidenceStore()
        evidence.register_evidence(EvidenceRef(
            evidence_id="EVD-SOLVER",
            source_type=EvidenceSourceType.SOLVER_RESULT,
            source_id=solver_result.solver_run_id,
            metadata={"model_version": 1},
        ))
        evidence.register_claim(Claim(
            claim_id="CLAIM-1",
            text="The maximum total profit is 700.",
            claim_type=ClaimType.NUMERICAL,
            importance="high",
            evidence_ids=["EVD-SOLVER"],
            verification_status=ClaimStatus.SUPPORTED,
        ))

        # Figure with real artifact
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"real-figure-bytes")
            fig_path = f.name
        fig_hash = hashlib.sha256(b"real-figure-bytes").hexdigest()[:16]
        figures = FigureRegistry()
        figures.register(FigureRecord(
            figure_id="FIG-001", title="Solution",
            figure_type=FigureType.SOLUTION,
            source_execution_ids=[solver_result.solver_run_id],
            artifact_path=fig_path, hash=fig_hash,
            verification_status="verified",
        ))

        # Table with sourced cells
        tables = TableRegistry()
        tables.register(TableRecord(
            table_id="TAB-001", title="Optimal values",
            table_type=TableType.OPTIMIZATION_RESULT,
            headers=["Variable", "Value"],
            rows=[
                [TableCell(value="x1", source_id="MODEL"), TableCell(value=10.0, source_id="SRC-X1")],
                [TableCell(value="x2", source_id="MODEL"), TableCell(value=0.0, source_id="SRC-X2")],
                [TableCell(value="x3", source_id="MODEL"), TableCell(value=20.0, source_id="SRC-X3")],
            ],
            source_ids=[solver_result.solver_run_id],
        ))

        # Paper
        paper = PaperIR(
            title="Production Planning Optimization",
            abstract="We optimize production achieving a maximum profit of 700.",
            sections=[
                PaperSection(section_id="SEC-001", title="Results", content_blocks=[
                    ContentBlock(
                        block_type=BlockType.PARAGRAPH,
                        text="The maximum total profit is 700.",
                        claim_ids=["CLAIM-1"],
                    ),
                    ContentBlock(
                        block_type=BlockType.EQUATION,
                        text=r"P = 30x_1 + 25x_2 + 20x_3",
                    ),
                    ContentBlock(block_type=BlockType.TABLE, text="Optimal values", table_ids=["TAB-001"]),
                    ContentBlock(block_type=BlockType.FIGURE, text="Solution", figure_ids=["FIG-001"]),
                ]),
            ],
            metadata={"model_version": 1},
        )

        # Numerical consistency
        checker = NumericalConsistencyChecker()
        assert len(checker.check_all(paper, {"profit": 700.0})) == 0

        # LaTeX render (no injection)
        tex = LaTeXRenderer().render(paper)
        assert r"\begin{document}" in tex

        # Submission
        profile = CompetitionProfile()
        submission = SubmissionCheckAgent(
            profile,
            evidence=evidence,
            figures=figures,
            tables=tables,
            current_model_version=1,
            source_values={"SRC-X1": 10.0, "SRC-X2": 0.0, "SRC-X3": 20.0},
            red_team_report=red_team,
        )
        result = submission.check(paper)
        assert result.status == SubmissionStatus.READY_TO_SUBMIT

        os.unlink(fig_path)


# ═══════════════════════════════════════════════════════════════
# 2. Required adversarial tests
# ═══════════════════════════════════════════════════════════════

class TestAdversarial1StalePaper:
    """Repaired model invalidates old paper."""

    def test_repaired_model_invalidates_old_paper(self):
        v1_model = make_model(version=1)
        v1_result = solve_model(v1_model)

        # Repair produces model v2
        broken = v1_model.model_copy(deep=True)
        broken.parameters[0].value = 300.0
        broken.objectives[0].expression = "300*x1 + 25*x2 + 20*x3"
        repair_agent = ModelRepairAgent()
        plan = RepairPlan(actions=[RepairAction(
            kind="parameter_set", parameter_id="PAR-p1",
            new_value=30.0, source_reference="EVD-1: problem profit table",
        )])
        repaired = repair_agent.apply_repair(broken, plan)
        assert repaired.version == 2

        # Paper built from v1
        paper_v1 = PaperIR(title="T", abstract="A", metadata={"model_version": 1})
        profile = CompetitionProfile()
        agent = SubmissionCheckAgent(profile, current_model_version=2)
        result = agent.check(paper_v1)
        assert result.status == SubmissionStatus.BLOCKED
        assert any("stale" in f for f in result.failures)


class TestAdversarial2OldSolverResult:
    """Old solver result cannot support new paper."""

    def test_old_result_blocked(self):
        v1_model = make_model(version=1)
        v1_result = solve_model(v1_model)
        assert v1_result.model_version == 1

        # Paper claims current model is v2 but evidence is v1
        evidence = EvidenceStore()
        evidence.register_evidence(EvidenceRef(
            evidence_id="EVD-OLD",
            source_type=EvidenceSourceType.SOLVER_RESULT,
            source_id=v1_result.solver_run_id,
            metadata={"model_version": 1},
        ))
        evidence.register_claim(Claim(
            claim_id="C1", text="profit=700", claim_type=ClaimType.NUMERICAL,
            importance="high", evidence_ids=["EVD-OLD"],
            verification_status=ClaimStatus.SUPPORTED,
        ))
        paper = PaperIR(title="T", abstract="A", metadata={"model_version": 2})
        agent = SubmissionCheckAgent(
            CompetitionProfile(), evidence=evidence, current_model_version=2,
        )
        result = agent.check(paper)
        assert result.status == SubmissionStatus.BLOCKED
        assert any("stale evidence" in f for f in result.failures)


class TestAdversarial3ValidationFailBlocks:
    """Validation FAIL blocks numerical claim."""

    def test_validation_fail_blocks_claim(self):
        model = make_model(version=1)
        result = solve_model(model)
        # Tamper objective → validation FAIL
        result.objective_value = 9999.0
        validation = build_validation_report(model, result)
        assert validation.overall_status == GateStatus.FAIL

        # Even with "pretty" number, no supported claim
        quality = VerificationQualityGate.evaluate(
            validation, RedTeamReport(model_id=model.model_id, issues=[]),
        )
        assert quality == VerificationStatus.FAILED.value


class TestAdversarial4StaleSensitivity:
    """Stale sensitivity evidence blocks sensitivity claim."""

    def test_stale_sensitivity_blocked(self):
        v1_model = make_model(version=1)
        sensitivity = SensitivityAgent().run(v1_model, parameter_ids=["PAR-p1"])
        assert sensitivity.model_version == 1

        # v2 current model: v1 sensitivity claim must not be READY
        evidence = EvidenceStore()
        evidence.register_evidence(EvidenceRef(
            evidence_id="EVD-SENS",
            source_type=EvidenceSourceType.SENSITIVITY,
            source_id=sensitivity.report_id,
            metadata={"model_version": 1},
        ))
        evidence.register_claim(Claim(
            claim_id="C2", text="p1 is most sensitive",
            claim_type=ClaimType.INTERPRETIVE, importance="high",
            evidence_ids=["EVD-SENS"], verification_status=ClaimStatus.SUPPORTED,
        ))
        paper = PaperIR(title="T", abstract="A", metadata={"model_version": 2})
        agent = SubmissionCheckAgent(
            CompetitionProfile(), evidence=evidence, current_model_version=2,
        )
        result = agent.check(paper)
        assert result.status == SubmissionStatus.BLOCKED


class TestAdversarial5FigureTampering:
    """Figure hash tampering propagates to SubmissionCheck."""

    def test_figure_tampering_blocks(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"original-bytes")
            path = f.name
        figures = FigureRegistry()
        figures.register(FigureRecord(
            figure_id="FIG-1", title="x", artifact_path=path,
            hash="recorded-hash", source_execution_ids=["E1"],
        ))
        paper = PaperIR(title="T", abstract="A", sections=[
            PaperSection(section_id="S", title="R", content_blocks=[
                ContentBlock(block_type=BlockType.FIGURE, figure_ids=["FIG-1"]),
            ]),
        ])
        agent = SubmissionCheckAgent(CompetitionProfile(), figures=figures)
        result = agent.check(paper)
        assert result.status == SubmissionStatus.BLOCKED
        assert any("hash mismatch" in f for f in result.failures)
        os.unlink(path)


class TestAdversarial6TableTampering:
    """Table numerical tampering propagates to SubmissionCheck."""

    def test_table_tampering_blocks(self):
        tables = TableRegistry()
        tables.register(TableRecord(
            table_id="TAB-1", title="R", headers=["V"],
            rows=[[TableCell(value=701.0, source_id="SRC-1")]],
        ))
        paper = PaperIR(title="T", abstract="A", sections=[
            PaperSection(section_id="S", title="R", content_blocks=[
                ContentBlock(block_type=BlockType.TABLE, table_ids=["TAB-1"]),
            ]),
        ])
        agent = SubmissionCheckAgent(
            CompetitionProfile(), tables=tables,
            source_values={"SRC-1": 700.0},
        )
        result = agent.check(paper)
        assert result.status == SubmissionStatus.BLOCKED
        assert any("does not match" in f for f in result.failures)


class TestAdversarial7RedTeamCritical:
    """Unresolved RedTeam CRITICAL blocks READY_TO_SUBMIT."""

    def test_redteam_critical_blocks(self):
        red_team = RedTeamReport(
            model_id="M", issues=[
                RedTeamIssue(
                    severity=IssueSeverity.CRITICAL, category="constraints",
                    title="Missing constraint", description="Capacity missing",
                    evidence=[{"constraint_id": "CON-X"}],
                ),
            ],
        )
        paper = PaperIR(title="T", abstract="A")
        agent = SubmissionCheckAgent(
            CompetitionProfile(), red_team_report=red_team,
        )
        result = agent.check(paper)
        assert result.status == SubmissionStatus.BLOCKED
        assert any("RedTeam" in f for f in result.failures)


class TestAdversarial8ProvenanceRoundtrip:
    """Full provenance chain survives serialization roundtrip."""

    def test_roundtrip_preserves_chain(self):
        model = make_model(version=1)
        result = solve_model(model)
        validation = build_validation_report(model, result)

        # Serialize everything
        model_data = model.model_dump()
        validation_data = validation.model_dump()

        # Roundtrip
        restored_model = MathematicalModel.model_validate(model_data)
        from mathmodel.domain.verification import ValidationReport
        restored_validation = ValidationReport.model_validate(validation_data)

        assert restored_model.model_id == model.model_id
        assert restored_model.version == 1
        assert restored_validation.model_id == model.model_id
        assert restored_validation.model_version == 1
        assert restored_validation.solver_run_id == result.solver_run_id
        assert restored_validation.overall_status == GateStatus.PASS
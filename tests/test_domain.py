"""Tests for Phase 2 typed domain schemas (TD-2).

Covers: ProblemAnalysis, EvidenceItem, Subproblem, Ambiguity,
ModelCandidate, ModelComponent, EligibilityResult, JuryScore,
ModelJuryResult, LiteratureQueryPlan.
"""

import pytest
from pydantic import ValidationError

from mathmodel.domain.evidence import EvidenceItem, EvidenceType, EvidenceStatus
from mathmodel.domain.analysis import (
    ProblemAnalysis,
    Subproblem,
    Ambiguity,
    AmbiguityImpact,
    ModelingTaskType,
)
from mathmodel.domain.candidates import (
    ModelCandidate,
    ModelComponent,
    ModelFamily,
    ComponentRole,
)
from mathmodel.domain.eligibility import (
    EligibilityPolicy,
    EligibilityResult,
    EligibilityFailure,
    EligibilityCheck,
)
from mathmodel.domain.jury import (
    JuryDimension,
    JuryScore,
    ModelJuryResult,
    SelectionOverride,
    DEFAULT_JURY_WEIGHTS,
    compute_jury_scores,
)
from mathmodel.domain.literature import (
    LiteratureQuery,
    LiteratureQueryPlan,
    LiteratureQueryStatus,
)


# ═══════════════════════════════════════════════════════════════
# EvidenceItem
# ═══════════════════════════════════════════════════════════════

class TestEvidenceItem:
    def test_create_fact(self):
        ev = EvidenceItem(
            type=EvidenceType.FACT,
            content="The sky is blue",
            source="Observation",
        )
        assert ev.type == EvidenceType.FACT
        assert ev.evidence_id.startswith("EVD-")
        assert ev.confidence == 1.0

    def test_create_proposed_assumption(self):
        ev = EvidenceItem(
            type=EvidenceType.PROPOSED_ASSUMPTION,
            content="Demand is normally distributed",
            source="Modeling assumption",
            confidence=0.7,
        )
        assert ev.is_assumption
        assert not ev.is_accepted

    def test_accept_assumption(self):
        ev = EvidenceItem(
            type=EvidenceType.PROPOSED_ASSUMPTION,
            content="Demand is normal",
            source="Modeling assumption",
        )
        accepted = ev.accept("ProblemAgent", "Standard assumption in inventory models")
        assert accepted.type == EvidenceType.ACCEPTED_ASSUMPTION
        assert accepted.accepted_by == "ProblemAgent"
        assert accepted.is_accepted

    def test_cannot_accept_fact(self):
        ev = EvidenceItem(
            type=EvidenceType.FACT,
            content="A fact",
            source="Problem",
        )
        with pytest.raises(ValueError, match="Cannot accept"):
            ev.accept("Someone", "reason")

    def test_derivation_must_reference_sources(self):
        ev = EvidenceItem(
            type=EvidenceType.DERIVATION,
            content="Derived conclusion",
            source="Derivation",
        )
        violations = ev.validate_invariants()
        assert len(violations) > 0
        assert "must reference" in violations[0]

    def test_derivation_with_sources(self):
        ev = EvidenceItem(
            type=EvidenceType.DERIVATION,
            content="Derived from EVD-1",
            source="Derivation",
            derived_from=["EVD-1"],
        )
        violations = ev.validate_invariants()
        assert len(violations) == 0

    def test_accepted_assumption_must_have_acceptor(self):
        ev = EvidenceItem(
            type=EvidenceType.ACCEPTED_ASSUMPTION,
            content="An accepted assumption",
            source="Modeling",
        )
        violations = ev.validate_invariants()
        assert len(violations) > 0
        assert "accepted_by" in violations[0]


# ═══════════════════════════════════════════════════════════════
# ProblemAnalysis
# ═══════════════════════════════════════════════════════════════

class TestProblemAnalysis:
    def test_valid_analysis(self):
        analysis = ProblemAnalysis(
            background="Test background",
            core_problem="Test problem",
            objectives=["Objective 1"],
            subproblems=[
                Subproblem(
                    subproblem_id="SUB-1",
                    original_text="Subproblem text",
                    normalized_goal="Achieve X",
                    task_types=[ModelingTaskType.OPTIMIZATION],
                ),
            ],
            evidence=[
                EvidenceItem(
                    evidence_id="EVD-1",
                    type=EvidenceType.FACT,
                    content="A fact",
                    source="Problem",
                ),
            ],
            confidence=0.8,
        )
        issues = analysis.validate_domain()
        assert len(issues) == 0

    def test_duplicate_subproblem_ids(self):
        analysis = ProblemAnalysis(
            background="Test",
            core_problem="Test",
            subproblems=[
                Subproblem(
                    subproblem_id="SUB-1",
                    original_text="First",
                    normalized_goal="Goal 1",
                ),
                Subproblem(
                    subproblem_id="SUB-1",
                    original_text="Second",
                    normalized_goal="Goal 2",
                ),
            ],
        )
        issues = analysis.validate_domain()
        assert any("Duplicate subproblem" in i for i in issues)

    def test_duplicate_evidence_ids(self):
        analysis = ProblemAnalysis(
            background="Test",
            core_problem="Test",
            evidence=[
                EvidenceItem(
                    evidence_id="EVD-1",
                    type=EvidenceType.FACT,
                    content="Fact 1",
                    source="Source",
                ),
                EvidenceItem(
                    evidence_id="EVD-1",
                    type=EvidenceType.FACT,
                    content="Fact 2",
                    source="Source",
                ),
            ],
        )
        issues = analysis.validate_domain()
        assert any("Duplicate evidence" in i for i in issues)

    def test_invalid_subproblem_dependency(self):
        with pytest.raises(Exception):  # ValidationError from pydantic
            ProblemAnalysis(
                background="Test",
                core_problem="Test",
                subproblems=[
                    Subproblem(
                        subproblem_id="SUB-1",
                        original_text="First",
                        normalized_goal="Goal 1",
                        dependencies=["SUB-NONEXISTENT"],
                    ),
                ],
            )

    def test_invalid_task_type(self):
        analysis = ProblemAnalysis(
            background="Test",
            core_problem="Test",
            subproblems=[
                Subproblem(
                    subproblem_id="SUB-1",
                    original_text="Test",
                    normalized_goal="Goal",
                    task_types=[ModelingTaskType.OPTIMIZATION],
                ),
            ],
        )
        issues = analysis.validate_domain()
        # Valid task types should not produce issues
        assert not any("invalid task type" in i for i in issues)

    def test_serialization_roundtrip(self):
        analysis = ProblemAnalysis(
            background="Test",
            core_problem="Test",
            subproblems=[
                Subproblem(
                    subproblem_id="SUB-1",
                    original_text="Test",
                    normalized_goal="Goal",
                    task_types=[ModelingTaskType.OPTIMIZATION],
                ),
            ],
            evidence=[
                EvidenceItem(
                    evidence_id="EVD-1",
                    type=EvidenceType.FACT,
                    content="A fact",
                    source="Source",
                ),
            ],
        )
        data = analysis.model_dump()
        restored = ProblemAnalysis.model_validate(data)
        assert restored.analysis_id == analysis.analysis_id
        assert len(restored.subproblems) == 1
        assert restored.subproblems[0].subproblem_id == "SUB-1"

    def test_malformed_json_rejected(self):
        with pytest.raises(ValidationError):
            ProblemAnalysis.model_validate({"background": 123, "core_problem": None})


# ═══════════════════════════════════════════════════════════════
# ModelCandidate
# ═══════════════════════════════════════════════════════════════

class TestModelCandidate:
    def test_valid_candidate(self):
        c = ModelCandidate(
            name="Linear Programming",
            model_family=ModelFamily.LINEAR_PROGRAMMING,
            applicable_subproblems=["SUB-1"],
            summary="LP model",
            components=[
                ModelComponent(
                    name="Solver",
                    model_family=ModelFamily.LINEAR_PROGRAMMING,
                    role=ComponentRole.CORE_MODEL,
                ),
            ],
        )
        assert c.candidate_id.startswith("CAND-")
        assert not c.is_chain

    def test_candidate_with_chain(self):
        c = ModelCandidate(
            name="Forecast + Optimize",
            model_family=ModelFamily.HYBRID,
            applicable_subproblems=["SUB-1", "SUB-2"],
            summary="Two-stage model",
            components=[
                ModelComponent(
                    component_id="MC-1",
                    name="Forecaster",
                    model_family=ModelFamily.TIME_SERIES,
                    role=ComponentRole.CORE_MODEL,
                ),
                ModelComponent(
                    component_id="MC-2",
                    name="Optimizer",
                    model_family=ModelFamily.LINEAR_PROGRAMMING,
                    role=ComponentRole.CORE_MODEL,
                    dependencies=["MC-1"],
                ),
            ],
        )
        assert c.is_chain
        assert len(c.core_components) == 2

    def test_similarity_key(self):
        c1 = ModelCandidate(
            name="LP Model",
            model_family=ModelFamily.LINEAR_PROGRAMMING,
            applicable_subproblems=["SUB-1"],
            summary="LP",
        )
        c2 = ModelCandidate(
            name="Linear Programming",
            model_family=ModelFamily.LINEAR_PROGRAMMING,
            applicable_subproblems=["SUB-1"],
            summary="Linear Programming",
        )
        assert c1.similarity_key() == c2.similarity_key()

    def test_serialization_roundtrip(self):
        c = ModelCandidate(
            name="Test Model",
            model_family=ModelFamily.INTEGER_PROGRAMMING,
            applicable_subproblems=["SUB-1"],
            summary="Test",
            components=[
                ModelComponent(
                    name="Core",
                    model_family=ModelFamily.INTEGER_PROGRAMMING,
                    role=ComponentRole.CORE_MODEL,
                ),
            ],
        )
        data = c.model_dump()
        restored = ModelCandidate.model_validate(data)
        assert restored.candidate_id == c.candidate_id
        assert restored.model_family == ModelFamily.INTEGER_PROGRAMMING


# ═══════════════════════════════════════════════════════════════
# Eligibility
# ═══════════════════════════════════════════════════════════════

class TestEligibilityGate:
    def test_eligible_candidate(self):
        policy = EligibilityPolicy()
        result = policy.evaluate(
            candidate_id="CAND-1",
            has_required_data=True,
            math_plausible=True,
            has_implementation_path=True,
            can_validate=True,
            competition_feasible=True,
            hard_constraint_violations=[],
            missing_data=[],
            assumption_risks=[],
            implementation_risks=[],
            validation_risks=[],
        )
        assert result.eligible
        assert len(result.hard_failures) == 0

    def test_missing_data_hard_fail(self):
        policy = EligibilityPolicy(strict_data_requirements=True)
        result = policy.evaluate(
            candidate_id="CAND-1",
            has_required_data=False,
            math_plausible=True,
            has_implementation_path=True,
            can_validate=True,
            competition_feasible=True,
            hard_constraint_violations=[],
            missing_data=["Distance matrix"],
            assumption_risks=[],
            implementation_risks=[],
            validation_risks=[],
        )
        assert not result.eligible
        assert any(
            f.check == EligibilityCheck.REQUIRED_DATA
            for f in result.hard_failures
        )

    def test_math_implausible_hard_fail(self):
        policy = EligibilityPolicy(strict_math_plausibility=True)
        result = policy.evaluate(
            candidate_id="CAND-1",
            has_required_data=True,
            math_plausible=False,
            has_implementation_path=True,
            can_validate=True,
            competition_feasible=True,
            hard_constraint_violations=[],
            missing_data=[],
            assumption_risks=[],
            implementation_risks=[],
            validation_risks=[],
        )
        assert not result.eligible

    def test_hard_constraint_violation(self):
        policy = EligibilityPolicy()
        result = policy.evaluate(
            candidate_id="CAND-1",
            has_required_data=True,
            math_plausible=True,
            has_implementation_path=True,
            can_validate=True,
            competition_feasible=True,
            hard_constraint_violations=["Violates non-negativity"],
            missing_data=[],
            assumption_risks=[],
            implementation_risks=[],
            validation_risks=[],
        )
        assert not result.eligible

    def test_competition_infeasible(self):
        policy = EligibilityPolicy()
        result = policy.evaluate(
            candidate_id="CAND-1",
            has_required_data=True,
            math_plausible=True,
            has_implementation_path=True,
            can_validate=True,
            competition_feasible=False,
            hard_constraint_violations=[],
            missing_data=[],
            assumption_risks=[],
            implementation_risks=[],
            validation_risks=[],
        )
        assert not result.eligible

    def test_warnings_not_hard_fail(self):
        policy = EligibilityPolicy()
        result = policy.evaluate(
            candidate_id="CAND-1",
            has_required_data=True,
            math_plausible=True,
            has_implementation_path=False,
            can_validate=False,
            competition_feasible=True,
            hard_constraint_violations=[],
            missing_data=[],
            assumption_risks=["Risky assumption"],
            implementation_risks=["Complex implementation"],
            validation_risks=["Hard to validate"],
        )
        # Warnings should not make candidate ineligible
        assert result.eligible
        assert len(result.warnings) > 0


# ═══════════════════════════════════════════════════════════════
# Jury
# ═══════════════════════════════════════════════════════════════

class TestJuryScore:
    def test_weights_sum_to_100(self):
        total = sum(DEFAULT_JURY_WEIGHTS.values())
        assert abs(total - 100.0) < 0.01

    def test_deterministic_score_calculation(self):
        raw_scores = {
            JuryDimension.PROBLEM_FIT: 80.0,
            JuryDimension.DATA_FIT: 70.0,
            JuryDimension.MATHEMATICAL_VALIDITY: 90.0,
            JuryDimension.EXPLAINABILITY: 75.0,
            JuryDimension.VALIDATION_POTENTIAL: 85.0,
            JuryDimension.INNOVATION: 60.0,
            JuryDimension.COMPETITION_FEASIBILITY: 95.0,
            JuryDimension.COMPUTATIONAL_EFFICIENCY: 80.0,
        }
        reasoning = {dim: "Good" for dim in JuryDimension}

        score1 = compute_jury_scores("CAND-1", raw_scores, reasoning)
        score2 = compute_jury_scores("CAND-1", raw_scores, reasoning)

        # Same inputs -> same total
        assert score1.total_score == score2.total_score

        # Total should be in 0-100 range
        assert 0 <= score1.total_score <= 100

    def test_score_calculation_correct(self):
        # Test with known values
        raw_scores = {
            JuryDimension.PROBLEM_FIT: 100.0,
            JuryDimension.DATA_FIT: 0.0,
            JuryDimension.MATHEMATICAL_VALIDITY: 0.0,
            JuryDimension.EXPLAINABILITY: 0.0,
            JuryDimension.VALIDATION_POTENTIAL: 0.0,
            JuryDimension.INNOVATION: 0.0,
            JuryDimension.COMPETITION_FEASIBILITY: 0.0,
            JuryDimension.COMPUTATIONAL_EFFICIENCY: 0.0,
        }
        reasoning = {dim: "" for dim in JuryDimension}

        score = compute_jury_scores("CAND-1", raw_scores, reasoning)
        # problem_fit weight = 25, raw = 100 -> weighted = 25.0
        assert score.total_score == 25.0

    def test_invalid_score_range(self):
        with pytest.raises(ValueError, match="0-100"):
            JuryScore(
                candidate_id="CAND-1",
                raw_scores={JuryDimension.PROBLEM_FIT: 150.0},
            )

    def test_selected_and_backup_must_differ(self):
        with pytest.raises(ValueError, match="must differ"):
            ModelJuryResult(
                candidate_scores=[],
                ranking=["CAND-1", "CAND-2"],
                selected_model="CAND-1",
                backup_model="CAND-1",
                decision_reason="Test",
            )

    def test_selected_must_be_in_ranking(self):
        with pytest.raises(ValueError, match="not in ranking"):
            ModelJuryResult(
                candidate_scores=[],
                ranking=["CAND-1"],
                selected_model="CAND-NOT",
                decision_reason="Test",
            )

    def test_weights_configurable(self):
        custom_weights = {
            JuryDimension.PROBLEM_FIT: 50.0,
            JuryDimension.DATA_FIT: 50.0,
            JuryDimension.MATHEMATICAL_VALIDITY: 0.0,
            JuryDimension.EXPLAINABILITY: 0.0,
            JuryDimension.VALIDATION_POTENTIAL: 0.0,
            JuryDimension.INNOVATION: 0.0,
            JuryDimension.COMPETITION_FEASIBILITY: 0.0,
            JuryDimension.COMPUTATIONAL_EFFICIENCY: 0.0,
        }
        raw_scores = {
            JuryDimension.PROBLEM_FIT: 100.0,
            JuryDimension.DATA_FIT: 100.0,
        }
        reasoning = {dim: "" for dim in JuryDimension}
        score = compute_jury_scores("CAND-1", raw_scores, reasoning, custom_weights)
        assert score.total_score == 100.0

    def test_override_requires_reason(self):
        with pytest.raises(ValidationError):
            SelectionOverride(
                original_top_candidate="CAND-1",
                override_candidate="CAND-2",
                reason="",  # Empty reason
                impact="Test",
            )


# ═══════════════════════════════════════════════════════════════
# Literature
# ═══════════════════════════════════════════════════════════════

class TestLiteratureQueryPlan:
    def test_all_pending(self):
        plan = LiteratureQueryPlan(
            queries=[
                LiteratureQuery(
                    purpose="Find related work",
                    keywords=["test"],
                ),
            ],
        )
        assert plan.all_pending()

    def test_query_status_default(self):
        query = LiteratureQuery(
            purpose="Test query",
            keywords=["optimization"],
        )
        assert query.status == LiteratureQueryStatus.SEARCH_PENDING
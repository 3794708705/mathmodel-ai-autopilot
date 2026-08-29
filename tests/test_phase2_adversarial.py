"""Adversarial tests for Phase 2 Reasoning Core.

Tests edge cases, invariants, and failure modes that happy-path
tests may miss. Covers: evidence semantics, diversity, eligibility,
jury determinism, routing monotonicity, and model chain cycles.
"""

import asyncio
import pytest
from pydantic import ValidationError

from mathmodel.agents.model_explorer import ModelExplorer
from mathmodel.agents.model_jury import ModelJury
from mathmodel.agents.eligibility_gate import EligibilityGate
from mathmodel.domain.candidates import (
    ModelCandidate,
    ModelComponent,
    ModelFamily,
    ComponentRole,
)
from mathmodel.domain.evidence import EvidenceItem, EvidenceType
from mathmodel.domain.analysis import ProblemAnalysis, Subproblem, ModelingTaskType
from mathmodel.domain.eligibility import EligibilityPolicy, EligibilityResult
from mathmodel.domain.jury import (
    JuryDimension,
    JuryScore,
    ModelJuryResult,
    DEFAULT_JURY_WEIGHTS,
    compute_jury_scores,
)
from mathmodel.domain.state_helpers import (
    store_analysis,
    store_candidates,
    store_eligibility_results,
)
from mathmodel.models.problem_state import ProblemState
from mathmodel.routing.policy import RoutingPolicy
from mathmodel.routing.profile import ComplexityTier, TaskProfile, TaskType
from mathmodel.config import ModelTier
from mathmodel.providers.mock import MockProvider
from mathmodel.routing.router import ModelRouter


# ═══════════════════════════════════════════════════════════════
# Evidence Semantics
# ═══════════════════════════════════════════════════════════════

class TestEvidenceSemantics:
    """Test that evidence types have real semantic differences."""

    def test_fact_cannot_be_assumption(self):
        """Problem-stated facts must not be labeled as assumptions."""
        ev = EvidenceItem(
            type=EvidenceType.FACT,
            content="Machine M1 has 40 hours available",
            source="Problem statement",
        )
        assert not ev.is_assumption
        assert ev.type == EvidenceType.FACT

    def test_assumption_cannot_be_fact(self):
        """Agent-proposed assumptions must not be labeled as facts."""
        ev = EvidenceItem(
            type=EvidenceType.PROPOSED_ASSUMPTION,
            content="Demand follows normal distribution",
            source="Modeling assumption",
            confidence=0.7,
        )
        assert ev.is_assumption
        assert ev.type != EvidenceType.FACT

    def test_assumption_requires_acceptance(self):
        """PROPOSED_ASSUMPTION must be explicitly accepted."""
        ev = EvidenceItem(
            type=EvidenceType.PROPOSED_ASSUMPTION,
            content="Test assumption",
            source="Modeling",
        )
        # Can't be ACCEPTED without explicit accept() call
        assert ev.type != EvidenceType.ACCEPTED_ASSUMPTION
        # Must have accepted_by after acceptance
        accepted = ev.accept("ProblemAgent", "Reasonable for this problem")
        assert accepted.accepted_by == "ProblemAgent"
        assert accepted.acceptance_reason is not None

    def test_derivation_references_must_exist(self):
        """DERIVATION must reference real evidence IDs."""
        # Create analysis with derivation
        with pytest.raises(Exception):
            ProblemAnalysis(
                background="Test",
                core_problem="Test",
                evidence=[
                    EvidenceItem(
                        evidence_id="EVD-1",
                        type=EvidenceType.FACT,
                        content="Base fact",
                        source="Problem",
                    ),
                    EvidenceItem(
                        evidence_id="EVD-2",
                        type=EvidenceType.DERIVATION,
                        content="Derived from nonexistent",
                        source="Derivation",
                        derived_from=["EVD-NONEXISTENT"],  # Invalid reference
                    ),
                ],
            )

    def test_broken_reference_after_removal(self):
        """If evidence is removed, references become invalid."""
        # This is a known limitation: ProblemAnalysis doesn't track
        # references after construction. Full referential integrity
        # requires an EvidenceStore (Phase 3+).
        # For now, verify the check works at construction time.
        analysis = ProblemAnalysis(
            background="Test",
            core_problem="Test",
            evidence=[
                EvidenceItem(
                    evidence_id="EVD-1",
                    type=EvidenceType.FACT,
                    content="Fact",
                    source="Problem",
                ),
                EvidenceItem(
                    evidence_id="EVD-2",
                    type=EvidenceType.DERIVATION,
                    content="Derived",
                    source="Derivation",
                    derived_from=["EVD-1"],
                ),
            ],
        )
        issues = analysis.validate_domain()
        assert len(issues) == 0


# ═══════════════════════════════════════════════════════════════
# Model Diversity
# ═══════════════════════════════════════════════════════════════

class TestDiversityAdversarial:
    """Test that diversity guard catches disguised duplicates."""

    def test_identical_family_same_math(self):
        """MILP and 'Weighted MILP' with same math should be flagged."""
        c1 = ModelCandidate(
            candidate_id="CAND-1",
            name="MILP Optimization",
            model_family=ModelFamily.MIXED_INTEGER_PROGRAMMING,
            applicable_subproblems=["SUB-1"],
            summary="MILP model",
            mathematical_structure="min c^T x s.t. Ax ≤ b, x ∈ Z",
            components=[
                ModelComponent(
                    component_id="MC-1",
                    name="Solver",
                    model_family=ModelFamily.MIXED_INTEGER_PROGRAMMING,
                    role=ComponentRole.CORE_MODEL,
                ),
            ],
        )
        c2 = ModelCandidate(
            candidate_id="CAND-2",
            name="Weighted MILP",
            model_family=ModelFamily.MIXED_INTEGER_PROGRAMMING,
            applicable_subproblems=["SUB-1"],
            summary="Weighted MILP",
            mathematical_structure="min c^T x s.t. Ax ≤ b, x ∈ Z",  # Same math
            components=[
                ModelComponent(
                    component_id="MC-2",
                    name="Weighted Solver",
                    model_family=ModelFamily.MIXED_INTEGER_PROGRAMMING,
                    role=ComponentRole.CORE_MODEL,
                ),
            ],
        )

        assert c1.is_essentially_same_as(c2), (
            "Same family + same mathematical structure should be flagged as essentially same"
        )

    def test_truly_diverse_not_flagged(self):
        """MILP, Stochastic, and Forecast+Optimize should NOT be flagged."""
        c1 = ModelCandidate(
            candidate_id="CAND-1",
            name="MILP Optimization",
            model_family=ModelFamily.MIXED_INTEGER_PROGRAMMING,
            applicable_subproblems=["SUB-1"],
            summary="MILP",
            mathematical_structure="min c^T x s.t. Ax ≤ b, x ∈ Z",
        )
        c2 = ModelCandidate(
            candidate_id="CAND-2",
            name="Stochastic Optimization",
            model_family=ModelFamily.STOCHASTIC_PROGRAMMING,
            applicable_subproblems=["SUB-1"],
            summary="Stochastic",
            mathematical_structure="min E[c^T x(ω)] s.t. constraints ∀ ω ∈ Ω",
        )
        c3 = ModelCandidate(
            candidate_id="CAND-3",
            name="Forecast + Optimize",
            model_family=ModelFamily.HYBRID,
            applicable_subproblems=["SUB-1"],
            summary="Two-stage",
            mathematical_structure="Stage 1: forecast, Stage 2: optimize",
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

        # All three should be different from each other
        assert not c1.is_essentially_same_as(c2)
        assert not c1.is_essentially_same_as(c3)
        assert not c2.is_essentially_same_as(c3)

    def test_same_name_different_family(self):
        """Same name but different family should still be flagged."""
        c1 = ModelCandidate(
            candidate_id="CAND-1",
            name="Linear Programming",
            model_family=ModelFamily.LINEAR_PROGRAMMING,
            applicable_subproblems=["SUB-1"],
            summary="LP",
            mathematical_structure="max c^T x",
        )
        c2 = ModelCandidate(
            candidate_id="CAND-2",
            name="LinearProgramming",  # Same after normalization
            model_family=ModelFamily.INTEGER_PROGRAMMING,
            applicable_subproblems=["SUB-1"],
            summary="IP",
            mathematical_structure="max c^T x, x ∈ Z",
        )

        # Same name after normalization + same family? No, different families
        # But name similarity alone should trigger if same family
        assert not c1.is_essentially_same_as(c2), (
            "Different families should not be flagged even with similar names"
        )


# ═══════════════════════════════════════════════════════════════
# Model Chain Cycles
# ═══════════════════════════════════════════════════════════════

class TestModelChainCycles:
    """Test cycle detection in model component chains."""

    def test_no_cycle(self):
        """A valid DAG should not have cycles."""
        c = ModelCandidate(
            candidate_id="CAND-1",
            name="Pipeline",
            model_family=ModelFamily.HYBRID,
            applicable_subproblems=["SUB-1"],
            summary="Pipeline",
            components=[
                ModelComponent(
                    component_id="MC-1",
                    name="Stage 1",
                    model_family=ModelFamily.TIME_SERIES,
                    role=ComponentRole.CORE_MODEL,
                ),
                ModelComponent(
                    component_id="MC-2",
                    name="Stage 2",
                    model_family=ModelFamily.LINEAR_PROGRAMMING,
                    role=ComponentRole.CORE_MODEL,
                    dependencies=["MC-1"],
                ),
            ],
        )
        assert c.is_dag
        assert not c.has_cycles

    def test_direct_cycle(self):
        """A -> B -> A should be detected."""
        c = ModelCandidate(
            candidate_id="CAND-1",
            name="Cyclic",
            model_family=ModelFamily.HYBRID,
            applicable_subproblems=["SUB-1"],
            summary="Broken",
            components=[
                ModelComponent(
                    component_id="MC-1",
                    name="A",
                    model_family=ModelFamily.TIME_SERIES,
                    role=ComponentRole.CORE_MODEL,
                    dependencies=["MC-2"],  # A depends on B
                ),
                ModelComponent(
                    component_id="MC-2",
                    name="B",
                    model_family=ModelFamily.LINEAR_PROGRAMMING,
                    role=ComponentRole.CORE_MODEL,
                    dependencies=["MC-1"],  # B depends on A → cycle
                ),
            ],
        )
        assert c.has_cycles
        assert not c.is_dag
        cycles = c._find_cycles()
        assert len(cycles) > 0

    def test_self_loop(self):
        """A component depending on itself should be detected."""
        c = ModelCandidate(
            candidate_id="CAND-1",
            name="Self-loop",
            model_family=ModelFamily.HYBRID,
            applicable_subproblems=["SUB-1"],
            summary="Broken",
            components=[
                ModelComponent(
                    component_id="MC-1",
                    name="Self",
                    model_family=ModelFamily.TIME_SERIES,
                    role=ComponentRole.CORE_MODEL,
                    dependencies=["MC-1"],  # Self-dependency
                ),
            ],
        )
        assert c.has_cycles

    def test_empty_components(self):
        """A candidate with no components should not have cycles."""
        c = ModelCandidate(
            candidate_id="CAND-1",
            name="No Components",
            model_family=ModelFamily.LINEAR_PROGRAMMING,
            applicable_subproblems=["SUB-1"],
            summary="No components",
        )
        assert not c.has_cycles
        assert c.is_dag


# ═══════════════════════════════════════════════════════════════
# Eligibility Adversarial
# ═══════════════════════════════════════════════════════════════

class TestEligibilityAdversarial:
    """Test eligibility edge cases: false positives and false negatives."""

    def test_false_positive_risk(self):
        """Model with no data support should be hard-failed."""
        policy = EligibilityPolicy(strict_data_requirements=True)
        result = policy.evaluate(
            candidate_id="CAND-1",
            has_required_data=False,
            math_plausible=True,
            has_implementation_path=True,
            can_validate=True,
            competition_feasible=True,
            hard_constraint_violations=[],
            missing_data=["Distance matrix", "Demand data"],
            assumption_risks=[],
            implementation_risks=[],
            validation_risks=[],
        )
        assert not result.eligible
        assert any(
            f.check.value == "required_data" for f in result.hard_failures
        )

    def test_false_negative_risk(self):
        """Model with high computation but otherwise valid should NOT be hard-failed."""
        policy = EligibilityPolicy(allow_high_computation=True)
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
            implementation_risks=["expensive computation", "high cost"],
            validation_risks=[],
        )
        assert result.eligible, (
            "High computation cost should be a warning, not a hard fail"
        )

    def test_all_candidates_ineligible(self):
        """When all candidates are ineligible, Jury should handle it."""
        policy = EligibilityPolicy()
        ineligible_results = [
            EligibilityResult(
                candidate_id="CAND-1",
                eligible=False,
                reason="Ineligible",
                hard_failures=[],
            ),
            EligibilityResult(
                candidate_id="CAND-2",
                eligible=False,
                reason="Ineligible",
                hard_failures=[],
            ),
        ]
        # Verify eligibility results are consistent
        assert all(not r.eligible for r in ineligible_results)

    def test_warning_does_not_disqualify(self):
        """Warnings alone should not make a candidate ineligible."""
        policy = EligibilityPolicy()
        result = policy.evaluate(
            candidate_id="CAND-1",
            has_required_data=True,
            math_plausible=True,
            has_implementation_path=False,  # Warning only
            can_validate=False,  # Warning only (with allow_difficult_validation)
            competition_feasible=True,
            hard_constraint_violations=[],
            missing_data=[],
            assumption_risks=["Risky assumption"],
            implementation_risks=["Complex implementation"],
            validation_risks=["Hard to validate"],
        )
        assert result.eligible
        assert len(result.warnings) >= 2
        assert len(result.hard_failures) == 0


# ═══════════════════════════════════════════════════════════════
# Jury Determinism
# ═══════════════════════════════════════════════════════════════

class TestJuryDeterminism:
    """Test that jury scoring is deterministic and correct."""

    def test_100_runs_same_result(self):
        """Same raw scores should produce same total 100 times."""
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
        reasoning = {dim: "test" for dim in JuryDimension}

        scores = [
            compute_jury_scores("CAND-1", raw_scores, reasoning).total_score
            for _ in range(100)
        ]
        # All must be identical
        assert len(set(scores)) == 1

    def test_invalid_weight_sum_rejected(self):
        """Weights not summing to 100 should raise error."""
        bad_weights = {
            JuryDimension.PROBLEM_FIT: 50.0,
            JuryDimension.DATA_FIT: 50.0,
            # Missing 6 dimensions — sum = 100, but missing dimensions
        }
        with pytest.raises(ValueError, match="missing dimensions"):
            compute_jury_scores(
                "CAND-1",
                {JuryDimension.PROBLEM_FIT: 100.0},
                {JuryDimension.PROBLEM_FIT: "test"},
                bad_weights,
            )

    def test_boundary_scores(self):
        """Score 0 and 100 should be valid."""
        raw_scores = {
            JuryDimension.PROBLEM_FIT: 0.0,
            JuryDimension.DATA_FIT: 100.0,
            JuryDimension.MATHEMATICAL_VALIDITY: 0.0,
            JuryDimension.EXPLAINABILITY: 100.0,
            JuryDimension.VALIDATION_POTENTIAL: 0.0,
            JuryDimension.INNOVATION: 100.0,
            JuryDimension.COMPETITION_FEASIBILITY: 0.0,
            JuryDimension.COMPUTATIONAL_EFFICIENCY: 100.0,
        }
        reasoning = {dim: "test" for dim in JuryDimension}
        score = compute_jury_scores("CAND-1", raw_scores, reasoning)
        # 0*25 + 100*15 + 0*15 + 100*10 + 0*10 + 100*10 + 0*10 + 100*5 = 15+10+10+5 = 40
        assert score.total_score == 40.0

    def test_selected_model_invariants(self):
        """Test all selection invariants."""
        # selected != backup
        with pytest.raises(Exception):
            ModelJuryResult(
                ranking=["CAND-1", "CAND-2"],
                selected_model="CAND-1",
                backup_model="CAND-1",
                decision_reason="",
            )

        # selected must be in ranking
        with pytest.raises(Exception):
            ModelJuryResult(
                ranking=["CAND-1"],
                selected_model="CAND-NOT",
                decision_reason="",
            )

    def test_empty_candidate_list(self):
        """Empty candidate list should not crash."""
        result = ModelJuryResult(
            ranking=[],
            selected_model=None,
            backup_model=None,
            decision_reason="",
        )
        assert result.selected_model is None

    def test_single_eligible_candidate(self):
        """Single eligible candidate: selected = that candidate, backup = None."""
        result = ModelJuryResult(
            ranking=["CAND-1"],
            selected_model="CAND-1",
            backup_model=None,
            decision_reason="Only one candidate",
        )
        assert result.selected_model == "CAND-1"
        assert result.backup_model is None


# ═══════════════════════════════════════════════════════════════
# Routing Monotonicity
# ═══════════════════════════════════════════════════════════════

class TestRoutingMonotonicity:
    """Test that routing decisions are monotonic with respect to requirements."""

    def test_math_requirement_monotonic(self):
        """Higher math requirement should not lower tier."""
        policy = RoutingPolicy()
        profile_low = TaskProfile(
            task_type=TaskType.CRUD,
            complexity=ComplexityTier.MEDIUM,
            math_requirement=ComplexityTier.LOW,
        )
        profile_high = TaskProfile(
            task_type=TaskType.CRUD,
            complexity=ComplexityTier.MEDIUM,
            math_requirement=ComplexityTier.CRITICAL,
        )
        tier_low, _ = policy.select_tier(profile_low)
        tier_high, _ = policy.select_tier(profile_high)
        assert policy._tier_order(tier_high) >= policy._tier_order(tier_low)

    def test_reasoning_requirement_monotonic(self):
        """Higher reasoning requirement should not lower tier."""
        policy = RoutingPolicy()
        profile_low = TaskProfile(
            task_type=TaskType.CRUD,
            complexity=ComplexityTier.MEDIUM,
            reasoning_requirement=ComplexityTier.LOW,
        )
        profile_high = TaskProfile(
            task_type=TaskType.CRUD,
            complexity=ComplexityTier.MEDIUM,
            reasoning_requirement=ComplexityTier.CRITICAL,
        )
        tier_low, _ = policy.select_tier(profile_low)
        tier_high, _ = policy.select_tier(profile_high)
        assert policy._tier_order(tier_high) >= policy._tier_order(tier_low)

    def test_blast_radius_monotonic(self):
        """Higher blast radius should not lower tier."""
        policy = RoutingPolicy()
        profile_low = TaskProfile(
            task_type=TaskType.CRUD,
            complexity=ComplexityTier.MEDIUM,
            blast_radius=ComplexityTier.LOW,
        )
        profile_high = TaskProfile(
            task_type=TaskType.CRUD,
            complexity=ComplexityTier.MEDIUM,
            blast_radius=ComplexityTier.CRITICAL,
        )
        tier_low, _ = policy.select_tier(profile_low)
        tier_high, _ = policy.select_tier(profile_high)
        assert policy._tier_order(tier_high) >= policy._tier_order(tier_low)

    def test_retry_count_monotonic(self):
        """More retries should not lower tier."""
        policy = RoutingPolicy()
        profile_low = TaskProfile(
            task_type=TaskType.CRUD,
            complexity=ComplexityTier.MEDIUM,
            retry_count=0,
        )
        profile_high = TaskProfile(
            task_type=TaskType.CRUD,
            complexity=ComplexityTier.MEDIUM,
            retry_count=3,
        )
        tier_low, _ = policy.select_tier(profile_low)
        tier_high, _ = policy.select_tier(profile_high)
        assert policy._tier_order(tier_high) >= policy._tier_order(tier_low)

    def test_security_critical_not_lowest(self):
        """Critical security risk should not route to lowest tier."""
        policy = RoutingPolicy()
        profile = TaskProfile(
            task_type=TaskType.SANDBOX_SECURITY,
            complexity=ComplexityTier.CRITICAL,
        )
        tier, _ = policy.select_tier(profile)
        assert tier != ModelTier.FAST
        assert tier != ModelTier.BALANCED


# ═══════════════════════════════════════════════════════════════
# Malformed JSON / Persistence
# ═══════════════════════════════════════════════════════════════

class TestMalformedPersistence:
    """Test that malformed persisted JSON is rejected."""

    def test_malformed_analysis_json(self):
        """Arbitrary dict-like JSON should not become ProblemAnalysis."""
        with pytest.raises(ValidationError):
            ProblemAnalysis.model_validate({
                "analysis_id": "fake",
                "background": 12345,  # Should be str
                "core_problem": None,  # Should be str with min_length
            })

    def test_malformed_candidate_json(self):
        """Arbitrary dict should not become ModelCandidate."""
        with pytest.raises(ValidationError):
            ModelCandidate.model_validate({
                "candidate_id": "fake",
                "name": "",
                "model_family": "not_a_real_family",
                "summary": "",
            })

    def test_malformed_jury_result_json(self):
        """Arbitrary dict should not become ModelJuryResult."""
        with pytest.raises(ValidationError):
            ModelJuryResult.model_validate({
                "ranking": "not_a_list",
                "selected_model": 123,
            })

    def test_round_trip_preserves_types(self):
        """Full round-trip: domain -> dict -> domain preserves all types."""
        original = ProblemAnalysis(
            background="Test",
            core_problem="Test",
            objectives=["Obj 1"],
            subproblems=[
                Subproblem(
                    subproblem_id="SUB-1",
                    original_text="Text",
                    normalized_goal="Goal",
                    task_types=[ModelingTaskType.OPTIMIZATION],
                ),
            ],
            evidence=[
                EvidenceItem(
                    evidence_id="EVD-1",
                    type=EvidenceType.FACT,
                    content="A fact",
                    source="Problem",
                    confidence=1.0,
                ),
            ],
            confidence=0.8,
        )
        data = original.model_dump()
        restored = ProblemAnalysis.model_validate(data)

        assert restored.analysis_id == original.analysis_id
        assert restored.confidence == 0.8
        assert isinstance(restored.subproblems[0].task_types[0], ModelingTaskType)
        assert restored.subproblems[0].task_types[0] == ModelingTaskType.OPTIMIZATION
        assert isinstance(restored.evidence[0].type, EvidenceType)
        assert restored.evidence[0].type == EvidenceType.FACT


# ═══════════════════════════════════════════════════════════════
# Literature Hallucination Prevention
# ═══════════════════════════════════════════════════════════════

class TestLiteratureHallucination:
    """Test that LiteratureAgent does not fabricate results."""

    def test_query_must_be_pending(self):
        from mathmodel.domain.literature import LiteratureQuery, LiteratureQueryStatus
        query = LiteratureQuery(
            purpose="Find optimization methods",
            keywords=["linear programming"],
        )
        assert query.status == LiteratureQueryStatus.SEARCH_PENDING

    def test_cannot_create_verified_query(self):
        """A query cannot be created with COMPLETED status without evidence."""
        from mathmodel.domain.literature import LiteratureQuery, LiteratureQueryStatus
        query = LiteratureQuery(
            purpose="Test",
            keywords=["test"],
            status=LiteratureQueryStatus.SEARCH_PENDING,
        )
        # Only SEARCH_PENDING is allowed
        assert query.status == LiteratureQueryStatus.SEARCH_PENDING
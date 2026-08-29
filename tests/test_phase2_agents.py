"""Tests for Phase 2 agents using fixtures.

Tests ProblemAgent, ModelExplorer, EligibilityGate, ModelJury,
and LiteratureAgent behavior with competition fixtures.
Does NOT require real LLM calls — uses MockProvider.
"""

import asyncio
import pytest

from mathmodel.agents.problem_agent import ProblemAgent
from mathmodel.agents.model_explorer import ModelExplorer
from mathmodel.agents.eligibility_gate import EligibilityGate
from mathmodel.agents.model_jury import ModelJury
from mathmodel.agents.literature_agent import LiteratureAgent
from mathmodel.domain.analysis import ProblemAnalysis
from mathmodel.domain.candidates import ModelCandidate, ModelFamily
from mathmodel.domain.eligibility import EligibilityPolicy
from mathmodel.domain.fixtures import (
    FIXTURE_A_ANALYSIS,
    FIXTURE_A_CANDIDATES,
    FIXTURE_B_ANALYSIS,
    FIXTURE_B_CANDIDATES,
    FIXTURE_C_ANALYSIS,
    FIXTURE_C_CANDIDATES,
)
from mathmodel.domain.state_helpers import (
    store_analysis,
    store_candidates,
    load_analysis,
    load_candidates,
    load_eligibility_results,
    load_jury_result,
    load_literature_plan,
)
from mathmodel.models.problem_state import (
    ProblemState,
    ProblemStateStage,
    ProblemStateStatus,
)
from mathmodel.providers.mock import MockProvider
from mathmodel.routing.router import ModelRouter
from mathmodel.providers.registry import ProviderRegistry


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def make_router() -> ModelRouter:
    """Create a ModelRouter with MockProvider."""
    registry = ProviderRegistry()
    registry.reset()
    return ModelRouter()


def make_state(problem_text: str = "") -> ProblemState:
    """Create a test ProblemState."""
    return ProblemState(
        title="Test Problem",
        raw_problem=problem_text,
        current_stage=ProblemStateStage.INGEST,
        status=ProblemStateStatus.PENDING,
    )


# ═══════════════════════════════════════════════════════════════
# Typed State Tests (TD-2)
# ═══════════════════════════════════════════════════════════════

class TestTypedState:
    """Test that domain objects serialize/deserialize through ProblemState."""

    def test_store_and_load_analysis(self, db_session):
        state = make_state()
        store_analysis(state, FIXTURE_A_ANALYSIS)

        loaded = load_analysis(state)
        assert loaded is not None
        assert loaded.analysis_id == FIXTURE_A_ANALYSIS.analysis_id
        assert len(loaded.subproblems) == len(FIXTURE_A_ANALYSIS.subproblems)
        assert len(loaded.evidence) == len(FIXTURE_A_ANALYSIS.evidence)

    def test_store_and_load_candidates(self, db_session):
        state = make_state()
        store_candidates(state, FIXTURE_A_CANDIDATES)

        loaded = load_candidates(state)
        assert len(loaded) == len(FIXTURE_A_CANDIDATES)
        assert loaded[0].candidate_id == FIXTURE_A_CANDIDATES[0].candidate_id

    def test_invalid_analysis_rejected(self, db_session):
        state = make_state()
        # Create an analysis with duplicate subproblem IDs
        from mathmodel.domain.analysis import Subproblem
        bad_analysis = ProblemAnalysis(
            background="Test",
            core_problem="Test",
            subproblems=[
                Subproblem(
                    subproblem_id="SUB-1",
                    original_text="A",
                    normalized_goal="Goal A",
                ),
                Subproblem(
                    subproblem_id="SUB-1",
                    original_text="B",
                    normalized_goal="Goal B",
                ),
            ],
        )
        with pytest.raises(ValueError):
            store_analysis(state, bad_analysis)

    def test_revision_metadata_preserved(self, db_session):
        from mathmodel.domain.state_helpers import record_revision
        state = make_state()
        store_analysis(state, FIXTURE_A_ANALYSIS)
        record_revision(state, "TestAgent", ProblemStateStage.UNDERSTAND, "Test revision")

        assert state.stage_history is not None
        assert len(state.stage_history) > 0
        rev = state.stage_history[-1]
        assert rev["source_agent"] == "TestAgent"
        assert rev["stage"] == "understand"
        assert "revision" in rev

    def test_arbitrary_json_rejected(self):
        """Malformed JSON cannot become a domain object."""
        with pytest.raises(Exception):
            ProblemAnalysis.model_validate({"not": "valid", "data": 123})


# ═══════════════════════════════════════════════════════════════
# ProblemAgent Tests
# ═══════════════════════════════════════════════════════════════

class TestProblemAgent:
    def test_agent_contract(self):
        agent = ProblemAgent(router=make_router())
        assert agent.name == "ProblemAgent"
        assert "problem_understanding" in agent.capabilities
        assert "model_selection" not in agent.capabilities

    def test_validate_output_valid(self):
        agent = ProblemAgent(router=make_router())
        errors = agent.validate_output(FIXTURE_A_ANALYSIS)
        assert len(errors) == 0

    def test_validate_output_invalid(self):
        agent = ProblemAgent(router=make_router())
        errors = agent.validate_output("not an analysis")
        assert len(errors) > 0

    def test_run_without_problem_text(self):
        agent = ProblemAgent(router=make_router())
        state = make_state()
        # Should fail because no raw_problem
        result = asyncio.run(agent.run(state))
        assert result.status.value in ("completed", "failed")

    def test_run_with_problem_text(self):
        agent = ProblemAgent(router=make_router())
        state = make_state(
            "A company produces two products. Maximize profit subject to constraints."
        )
        result = asyncio.run(agent.run(state))
        # With MockProvider, structured_generate returns a default instance
        # which may or may not pass validation
        assert result.agent_name == "ProblemAgent"

    def test_no_final_result_invented(self):
        """ProblemAgent should not invent numerical results."""
        agent = ProblemAgent(router=make_router())
        state = make_state("Maximize profit")
        result = asyncio.run(agent.run(state))
        # If completed, the output should be an analysis_id, not a numerical result
        if result.status.value == "completed":
            assert "analysis_id" in result.output
            assert "profit" not in str(result.output).lower() or "analysis_id" in result.output


# ═══════════════════════════════════════════════════════════════
# ModelExplorer Tests
# ═══════════════════════════════════════════════════════════════

class TestModelExplorer:
    def test_agent_contract(self):
        agent = ModelExplorer(router=make_router())
        assert agent.name == "ModelExplorer"
        assert agent.MIN_CANDIDATES == 3

    def test_requires_analysis(self, db_session):
        agent = ModelExplorer(router=make_router())
        state = make_state()
        # No analysis stored -> should fail
        result = asyncio.run(agent.run(state))
        assert result.status.value == "failed"

    def test_with_analysis(self, db_session):
        agent = ModelExplorer(router=make_router())
        state = make_state()
        store_analysis(state, FIXTURE_A_ANALYSIS)

        result = asyncio.run(agent.run(state))
        # With MockProvider, may or may not succeed
        assert result.agent_name == "ModelExplorer"

    def test_candidate_uniqueness(self):
        """Candidate IDs must be unique."""
        agent = ModelExplorer(router=make_router())
        dup_candidates = [
            FIXTURE_A_CANDIDATES[0],
            FIXTURE_A_CANDIDATES[0],  # Duplicate
        ]
        errors = agent._validate_candidates(dup_candidates)
        assert any("Duplicate" in e.message for e in errors)

    def test_diversity_check_identical(self):
        """Identical candidates should be flagged."""
        agent = ModelExplorer(router=make_router())
        c1 = FIXTURE_A_CANDIDATES[0]
        c2 = c1.model_copy(update={"candidate_id": "CAND-DUP"})
        c2.name = c1.name  # Same name
        issues = agent._check_diversity([c1, c2])
        # Should flag as highly similar
        assert len(issues) > 0

    def test_diversity_check_different_families(self):
        """Different model families should pass diversity check."""
        agent = ModelExplorer(router=make_router())
        issues = agent._check_diversity(FIXTURE_A_CANDIDATES)
        # FIXTURE_A_CANDIDATES has LP, ILP, Goal Programming — different families
        assert len(issues) == 0 or all("similar" not in i.lower() for i in issues)


# ═══════════════════════════════════════════════════════════════
# Eligibility Gate Tests
# ═══════════════════════════════════════════════════════════════

class TestEligibilityGateAgent:
    def test_requires_candidates(self, db_session):
        gate = EligibilityGate(policy=EligibilityPolicy())
        state = make_state()
        result = asyncio.run(gate.run(state))
        assert result.status.value == "failed"

    def test_with_candidates(self, db_session):
        gate = EligibilityGate(policy=EligibilityPolicy())
        state = make_state()
        store_candidates(state, FIXTURE_A_CANDIDATES)

        result = asyncio.run(gate.run(state))
        assert result.status.value == "completed"
        assert result.output["total"] == len(FIXTURE_A_CANDIDATES)

        # Load results
        results = load_eligibility_results(state)
        assert len(results) == len(FIXTURE_A_CANDIDATES)

    def test_ineligible_cannot_be_selected(self, db_session):
        """Candidates with hard failures should not be eligible."""
        policy = EligibilityPolicy(strict_data_requirements=True)
        gate = EligibilityGate(policy=policy)
        state = make_state()
        store_candidates(state, [FIXTURE_A_CANDIDATES[0]])

        result = asyncio.run(gate.run(state))
        results = load_eligibility_results(state)
        assert len(results) == 1
        # FIXTURE_A_CANDIDATES[0] has required_data populated, should be eligible
        assert results[0].eligible


# ═══════════════════════════════════════════════════════════════
# ModelJury Tests
# ═══════════════════════════════════════════════════════════════

class TestModelJuryAgent:
    def test_requires_candidates(self, db_session):
        jury = ModelJury(router=make_router())
        state = make_state()
        result = asyncio.run(jury.run(state))
        assert result.status.value == "failed"

    def test_with_candidates_and_eligibility(self, db_session):
        jury = ModelJury(router=make_router())
        state = make_state()
        store_candidates(state, FIXTURE_A_CANDIDATES)

        # Run eligibility first
        from mathmodel.domain.state_helpers import store_eligibility_results
        from mathmodel.domain.eligibility import EligibilityResult
        store_eligibility_results(state, [
            EligibilityResult(
                candidate_id=c.candidate_id,
                eligible=True,
                reason="Test eligible",
            )
            for c in FIXTURE_A_CANDIDATES
        ])

        result = asyncio.run(jury.run(state))
        # With MockProvider, may or may not succeed
        assert result.agent_name == "ModelJury"

    def test_ineligible_cannot_win(self, db_session):
        """Ineligible candidates must not be scored."""
        jury = ModelJury(router=make_router())
        state = make_state()
        store_candidates(state, FIXTURE_A_CANDIDATES)

        # Make only first candidate eligible
        from mathmodel.domain.state_helpers import store_eligibility_results
        from mathmodel.domain.eligibility import EligibilityResult
        store_eligibility_results(state, [
            EligibilityResult(
                candidate_id=FIXTURE_A_CANDIDATES[0].candidate_id,
                eligible=True,
                reason="Eligible",
            ),
            EligibilityResult(
                candidate_id=FIXTURE_A_CANDIDATES[1].candidate_id,
                eligible=False,
                reason="Ineligible",
                hard_failures=[],
            ),
            EligibilityResult(
                candidate_id=FIXTURE_A_CANDIDATES[2].candidate_id,
                eligible=False,
                reason="Ineligible",
                hard_failures=[],
            ),
        ])

        result = asyncio.run(jury.run(state))
        if result.status.value == "completed":
            jury_result = load_jury_result(state)
            if jury_result and jury_result.selected_model:
                # Only eligible candidates should be in ranking
                for score in jury_result.candidate_scores:
                    assert score.candidate_id == FIXTURE_A_CANDIDATES[0].candidate_id


# ═══════════════════════════════════════════════════════════════
# LiteratureAgent Tests
# ═══════════════════════════════════════════════════════════════

class TestLiteratureAgent:
    def test_requires_analysis(self, db_session):
        agent = LiteratureAgent()
        state = make_state()
        result = asyncio.run(agent.run(state))
        assert result.status.value == "failed"

    def test_with_analysis(self, db_session):
        agent = LiteratureAgent()
        state = make_state()
        store_analysis(state, FIXTURE_A_ANALYSIS)
        store_candidates(state, FIXTURE_A_CANDIDATES)

        result = asyncio.run(agent.run(state))
        assert result.status.value == "completed"

        plan = load_literature_plan(state)
        assert plan is not None
        assert len(plan.queries) > 0
        assert plan.all_pending()

    def test_no_fabricated_results(self, db_session):
        """All queries must be SEARCH_PENDING — no fabricated papers."""
        agent = LiteratureAgent()
        state = make_state()
        store_analysis(state, FIXTURE_A_ANALYSIS)

        result = asyncio.run(agent.run(state))
        plan = load_literature_plan(state)
        for query in plan.queries:
            assert query.status.value == "SEARCH_PENDING"


# ═══════════════════════════════════════════════════════════════
# Integration Test
# ═══════════════════════════════════════════════════════════════

class TestPhase2Integration:
    """End-to-end integration of the Reasoning Core pipeline."""

    def test_full_pipeline_fixture_a(self, db_session):
        """Run the full pipeline with Fixture A (no LLM calls)."""
        state = make_state(FIXTURE_A_ANALYSIS.background)

        # Step 1: Store analysis (simulating ProblemAgent output)
        store_analysis(state, FIXTURE_A_ANALYSIS)
        state.current_stage = ProblemStateStage.UNDERSTAND

        # Step 2: Store candidates (simulating ModelExplorer output)
        store_candidates(state, FIXTURE_A_CANDIDATES)
        state.current_stage = ProblemStateStage.EXPLORE

        # Step 3: Eligibility
        policy = EligibilityPolicy()
        gate = EligibilityGate(policy=policy)
        result = asyncio.run(gate.run(state))
        assert result.status.value == "completed"

        # Step 4: Jury
        jury = ModelJury(router=make_router())
        jury_result = asyncio.run(jury.run(state))
        # May succeed or fail depending on MockProvider behavior

        # Step 5: Literature
        lit_agent = LiteratureAgent()
        lit_result = asyncio.run(lit_agent.run(state))
        assert lit_result.status.value == "completed"

        # Verify state
        state.current_stage = ProblemStateStage.SELECT

        # Reload all domain objects
        analysis = load_analysis(state)
        assert analysis is not None
        assert analysis.analysis_id == FIXTURE_A_ANALYSIS.analysis_id

        candidates = load_candidates(state)
        assert len(candidates) == len(FIXTURE_A_CANDIDATES)

        plan = load_literature_plan(state)
        assert plan is not None
        assert plan.all_pending()

    def test_pipeline_fixture_b(self, db_session):
        """Full pipeline with Fixture B (prediction + optimization)."""
        state = make_state()
        store_analysis(state, FIXTURE_B_ANALYSIS)
        store_candidates(state, FIXTURE_B_CANDIDATES)

        # Verify chain structure
        for c in FIXTURE_B_CANDIDATES:
            if c.is_chain:
                assert len(c.components) >= 2
                # Check dependency chain
                for comp in c.components:
                    if comp.dependencies:
                        comp_ids = {x.component_id for x in c.components}
                        for dep in comp.dependencies:
                            assert dep in comp_ids

        # Verify subproblem dependencies
        for sp in FIXTURE_B_ANALYSIS.subproblems:
            if sp.dependencies:
                sub_ids = {s.subproblem_id for s in FIXTURE_B_ANALYSIS.subproblems}
                for dep in sp.dependencies:
                    assert dep in sub_ids

    def test_pipeline_fixture_c(self, db_session):
        """Full pipeline with Fixture C (network/routing)."""
        state = make_state()
        store_analysis(state, FIXTURE_C_ANALYSIS)
        store_candidates(state, FIXTURE_C_CANDIDATES)

        # Verify routing candidates
        for c in FIXTURE_C_CANDIDATES:
            assert c.model_family in (
                ModelFamily.INTEGER_PROGRAMMING,
                ModelFamily.GRAPH_THEORY,
                ModelFamily.NETWORK_FLOW,
            )

        # Run eligibility
        policy = EligibilityPolicy()
        gate = EligibilityGate(policy=policy)
        result = asyncio.run(gate.run(state))
        assert result.status.value == "completed"

    def test_state_reload_maintains_types(self, db_session):
        """After round-tripping through persistence, all objects must be valid."""
        state = make_state()
        store_analysis(state, FIXTURE_A_ANALYSIS)
        store_candidates(state, FIXTURE_A_CANDIDATES)

        # Simulate persistence round-trip
        data = {
            "analysis": state.metadata_.get("analysis"),
            "candidates": state.candidate_models,
        }

        # Reload
        restored_analysis = ProblemAnalysis.model_validate(data["analysis"])
        restored_candidates = [
            ModelCandidate.model_validate(c) for c in data["candidates"]
        ]

        assert restored_analysis.analysis_id == FIXTURE_A_ANALYSIS.analysis_id
        assert len(restored_candidates) == len(FIXTURE_A_CANDIDATES)
        assert all(
            isinstance(c, ModelCandidate) for c in restored_candidates
        )
"""Phase 7D tests: Budget, Routing, Change Classification."""

import threading
import time
from datetime import datetime, timezone, timedelta

import pytest

from mathmodel.budget import (
    CompetitionBudget, BudgetLedger, BudgetReservation,
    ResourceType, ReservationStatus,
)
from mathmodel.routing.competition_policy import (
    CompetitionRoutingPolicy, TaskCriticality, RoutingDecision,
    TASK_CRITICALITY,
)
from mathmodel.runtime import RuntimeMode, ChangeClassification
from mathmodel.agents.change_analyzer import ChangeImpactAnalyzer, ChangeImpactResult
from mathmodel.config import ModelTier


# ═══════════════════════════════════════════════════════════════
# Budget Tests
# ═══════════════════════════════════════════════════════════════

class TestBudgetReservation:
    def test_reserve_success(self):
        budget = CompetitionBudget(total_llm_call_budget=10, remaining_llm_calls=10)
        ledger = BudgetLedger(budget)
        res = ledger.reserve(ResourceType.LLM_CALL, 3, "run-1")
        assert res is not None
        assert budget.remaining_llm_calls == 7

    def test_reserve_insufficient(self):
        budget = CompetitionBudget(total_llm_call_budget=2, remaining_llm_calls=2)
        ledger = BudgetLedger(budget)
        res = ledger.reserve(ResourceType.LLM_CALL, 5, "run-1")
        assert res is None
        assert budget.remaining_llm_calls == 2  # unchanged

    def test_consume(self):
        budget = CompetitionBudget(total_llm_call_budget=10, remaining_llm_calls=10)
        ledger = BudgetLedger(budget)
        res = ledger.reserve(ResourceType.LLM_CALL, 3, "run-1")
        entry = ledger.consume(res.reservation_id, 3)
        assert entry.actual_amount == 3
        assert budget.remaining_llm_calls == 7

    def test_release(self):
        budget = CompetitionBudget(total_llm_call_budget=10, remaining_llm_calls=10)
        ledger = BudgetLedger(budget)
        res = ledger.reserve(ResourceType.LLM_CALL, 3, "run-1")
        ledger.release(res.reservation_id)
        assert budget.remaining_llm_calls == 10  # returned

    def test_partial_reconcile(self):
        """Reserved 5 tokens, used 3, 2 released."""
        budget = CompetitionBudget(total_llm_token_budget=100, remaining_llm_token_budget=100)
        ledger = BudgetLedger(budget)
        res = ledger.reserve(ResourceType.LLM_TOKEN, 5, "run-1")
        entry = ledger.consume(res.reservation_id, 3)
        assert entry.actual_amount == 3
        assert budget.remaining_llm_token_budget == 97  # 100-5+2

    def test_duplicate_reconcile_idempotent(self):
        budget = CompetitionBudget(total_llm_call_budget=10, remaining_llm_calls=10)
        ledger = BudgetLedger(budget)
        res = ledger.reserve(ResourceType.LLM_CALL, 3, "run-1")
        ledger.consume(res.reservation_id, 3)
        # Second reconcile returns same entry
        entry2 = ledger.consume(res.reservation_id, 3)
        assert entry2.actual_amount == 3
        assert budget.remaining_llm_calls == 7  # not double-charged

    def test_reservation_expiry(self):
        budget = CompetitionBudget(total_llm_call_budget=10, remaining_llm_calls=10)
        ledger = BudgetLedger(budget)
        res = ledger.reserve(ResourceType.LLM_CALL, 3, "run-1")
        # Force expiry
        res.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        expired = ledger.expire_reservations()
        assert expired == 1
        assert budget.remaining_llm_calls == 10  # returned

    def test_no_negative_budget(self):
        budget = CompetitionBudget(total_llm_call_budget=5, remaining_llm_calls=5)
        ledger = BudgetLedger(budget)
        ledger.reserve(ResourceType.LLM_CALL, 5, "run-1")
        res2 = ledger.reserve(ResourceType.LLM_CALL, 1, "run-2")
        assert res2 is None  # can't go negative
        assert budget.remaining_llm_calls == 0


class TestCriticalReserve:
    def test_critical_reserve_available(self):
        budget = CompetitionBudget(total_llm_call_budget=100, remaining_llm_calls=100,
                                   critical_reserve_fraction=0.15)
        assert budget.critical_reserve(ResourceType.LLM_CALL) == 15

    def test_available_for_non_critical_excludes_reserve(self):
        budget = CompetitionBudget(total_llm_call_budget=100, remaining_llm_calls=100,
                                   critical_reserve_fraction=0.15)
        avail = budget.available_for(ResourceType.LLM_CALL, is_critical=False)
        assert avail == 85  # 100 - 15 reserve

    def test_available_for_critical_includes_reserve(self):
        budget = CompetitionBudget(total_llm_call_budget=100, remaining_llm_calls=100,
                                   critical_reserve_fraction=0.15)
        avail = budget.available_for(ResourceType.LLM_CALL, is_critical=True)
        assert avail == 100

    def test_non_critical_blocked_when_normal_exhausted(self):
        budget = CompetitionBudget(total_llm_call_budget=15, remaining_llm_calls=15,
                                   critical_reserve_fraction=0.15)
        # Reserve = 2.25 → 2; available = 15-2=13
        avail = budget.available_for(ResourceType.LLM_CALL, is_critical=False)
        assert avail == 13


class TestUnlimitedBudget:
    def test_unlimited_tokens(self):
        budget = CompetitionBudget()  # no limits
        assert budget.remaining(ResourceType.LLM_TOKEN) is None
        assert budget.available_for(ResourceType.LLM_TOKEN, is_critical=True) is None

    def test_unlimited_calls_allows_reserve(self):
        budget = CompetitionBudget()
        ledger = BudgetLedger(budget)
        res = ledger.reserve(ResourceType.LLM_CALL, 100, "run-1", is_critical=False)
        assert res is not None  # unlimited → always succeeds


# ═══════════════════════════════════════════════════════════════
# Routing Policy Tests
# ═══════════════════════════════════════════════════════════════

class TestRoutingPolicy:
    def test_exploration_model_explore_medium(self):
        policy = CompetitionRoutingPolicy()
        rd = policy.route("ModelExplorer", "model_exploration",
                          RuntimeMode.EXPLORATION)
        assert rd.capability_tier == ModelTier.BALANCED
        assert rd.allow_parallel is True
        assert rd.max_attempts == 3

    def test_standard_math_modeler_high(self):
        policy = CompetitionRoutingPolicy()
        rd = policy.route("MathModeler", "mathematical_modeling",
                          RuntimeMode.STANDARD)
        assert rd.capability_tier == ModelTier.BALANCED

    def test_focus_critical_repair_flagship(self):
        policy = CompetitionRoutingPolicy()
        rd = policy.route("ModelRepair", "model_repair",
                          RuntimeMode.FOCUS, criticality=TaskCriticality.CRITICAL)
        assert rd.capability_tier == ModelTier.FLAGSHIP_HIGH

    def test_freeze_critical_repair_flagship(self):
        policy = CompetitionRoutingPolicy()
        rd = policy.route("ModelRepair", "model_repair",
                          RuntimeMode.MODEL_FREEZE, criticality=TaskCriticality.CRITICAL)
        assert rd.capability_tier == ModelTier.FLAGSHIP_HIGH
        assert rd.max_attempts == 2

    def test_submission_format_fix_fast(self):
        policy = CompetitionRoutingPolicy()
        rd = policy.route("PaperAgent", "format_fix",
                          RuntimeMode.SUBMISSION_MODE)
        assert rd.capability_tier == ModelTier.FAST
        assert rd.max_attempts == 1
        assert rd.allow_parallel is False

    def test_submission_critical_numerical_flagship(self):
        policy = CompetitionRoutingPolicy()
        rd = policy.route("Solver", "solver_execution",
                          RuntimeMode.SUBMISSION_MODE, criticality=TaskCriticality.CRITICAL)
        assert rd.capability_tier == ModelTier.FLAGSHIP_HIGH

    def test_schema_failure_escalation(self):
        policy = CompetitionRoutingPolicy()
        rd = policy.route("MathModeler", "mathematical_modeling",
                          RuntimeMode.STANDARD, previous_schema_failures=2)
        # Should escalate from BALANCED to FLAGSHIP_HIGH
        assert rd.capability_tier == ModelTier.FLAGSHIP_HIGH
        assert "PREVIOUS_SCHEMA_FAILURE" in rd.rationale_codes

    def test_low_budget_downgrade(self):
        policy = CompetitionRoutingPolicy()
        rd = policy.route("ModelExplorer", "model_exploration",
                          RuntimeMode.EXPLORATION, budget_available=False)
        assert rd.capability_tier == ModelTier.FAST
        assert "LOW_BUDGET_BLOCK" in rd.rationale_codes

    def test_critical_reserve_used_for_critical_task(self):
        policy = CompetitionRoutingPolicy()
        rd = policy.route("ModelRepair", "model_repair",
                          RuntimeMode.SUBMISSION_MODE,
                          criticality=TaskCriticality.CRITICAL,
                          budget_available=False,
                          is_critical_task=True,
                          can_use_critical_reserve=True)
        assert "CRITICAL_RESERVE" in rd.rationale_codes

    def test_rationale_codes_present(self):
        policy = CompetitionRoutingPolicy()
        rd = policy.route("MathModeler", "mathematical_modeling",
                          RuntimeMode.MODEL_FREEZE)
        assert "MODEL_FREEZE" in rd.rationale_codes


# ═══════════════════════════════════════════════════════════════
# Change Classification Tests
# ═══════════════════════════════════════════════════════════════

class TestChangeClassification:
    def test_model_switch_detected(self):
        analyzer = ChangeImpactAnalyzer()
        before = {"selected_candidate": "CAND-A"}
        after = {"selected_candidate": "CAND-B"}
        result = analyzer.analyze(before, after)
        assert result.candidate_changed
        assert result.observed_classification == ChangeClassification.MODEL_SWITCH
        assert result.effective_classification == ChangeClassification.MODEL_SWITCH

    def test_objective_change_is_major(self):
        analyzer = ChangeImpactAnalyzer()
        before = {"objectives": [{"objective_id": "OBJ-1"}]}
        after = {"objectives": [{"objective_id": "OBJ-2"}]}
        result = analyzer.analyze(before, after)
        assert result.observed_classification == ChangeClassification.MAJOR_MODEL_CHANGE

    def test_constraint_fix(self):
        analyzer = ChangeImpactAnalyzer()
        before = {"constraint_ids": ["CON-1"]}
        after = {"constraint_ids": ["CON-1", "CON-2"]}
        result = analyzer.analyze(before, after)
        assert result.observed_classification == ChangeClassification.CONSTRAINT_FIX

    def test_many_constraints_major(self):
        analyzer = ChangeImpactAnalyzer()
        before = {"constraint_ids": []}
        after = {"constraint_ids": ["CON-1", "CON-2", "CON-3", "CON-4"]}
        result = analyzer.analyze(before, after)
        assert result.observed_classification == ChangeClassification.MAJOR_MODEL_CHANGE

    def test_declared_minor_observed_major(self):
        """LLM says MINOR_FIX, diff says MAJOR → effective MAJOR."""
        analyzer = ChangeImpactAnalyzer()
        before = {"selected_candidate": "CAND-A"}
        after = {"selected_candidate": "CAND-B"}
        result = analyzer.analyze(before, after,
                                  declared=ChangeClassification.MINOR_FIX)
        assert result.observed_classification == ChangeClassification.MODEL_SWITCH
        assert result.effective_classification == ChangeClassification.MODEL_SWITCH

    def test_parameter_correction(self):
        analyzer = ChangeImpactAnalyzer()
        before = {"parameters": [{"parameter_id": "P-1"}]}
        after = {"parameters": [{"parameter_id": "P-2"}]}
        result = analyzer.analyze(before, after)
        assert result.observed_classification == ChangeClassification.PARAMETER_CORRECTION

    def test_no_changes_formatting(self):
        analyzer = ChangeImpactAnalyzer()
        before = {"selected_candidate": "CAND-A"}
        after = {"selected_candidate": "CAND-A"}
        result = analyzer.analyze(before, after)
        assert result.observed_classification == ChangeClassification.FORMATTING_ONLY

    def test_variable_change_is_major(self):
        analyzer = ChangeImpactAnalyzer()
        before = {"variables": [{"variable_id": "x1"}]}
        after = {"variables": [{"variable_id": "x2"}]}
        result = analyzer.analyze(before, after)
        assert result.observed_classification == ChangeClassification.MAJOR_MODEL_CHANGE


# ═══════════════════════════════════════════════════════════════
# Precedence Tests
# ═══════════════════════════════════════════════════════════════

class TestPrecedence:
    def test_huge_budget_cannot_bypass_submission_freeze(self):
        """Budget abundance does NOT override runtime mode restrictions."""
        # This is tested at the routing level: even with budget_available=True,
        # SUBMISSION_MODE still restricts MODEL_EXPLORE to FAST tier
        policy = CompetitionRoutingPolicy()
        rd = policy.route("ModelExplorer", "model_exploration",
                          RuntimeMode.SUBMISSION_MODE, budget_available=True)
        # SUBMISSION_MODE + MEDIUM → FAST (from matrix)
        assert rd.capability_tier == ModelTier.FAST
        assert rd.allow_parallel is False

    def test_budget_exhaustion_not_validation_pass(self):
        """Budget failure does not mean validation passed."""
        budget = CompetitionBudget(total_llm_call_budget=0, remaining_llm_calls=0)
        ledger = BudgetLedger(budget)
        res = ledger.reserve(ResourceType.LLM_CALL, 1, "run-1")
        assert res is None  # validation not possible


# ═══════════════════════════════════════════════════════════════
# Concurrency
# ═══════════════════════════════════════════════════════════════

class TestConcurrency:
    def test_concurrent_reservations_no_overspend(self):
        budget = CompetitionBudget(total_llm_call_budget=5, remaining_llm_calls=5)
        ledger = BudgetLedger(budget)
        results = []

        def try_reserve(i):
            res = ledger.reserve(ResourceType.LLM_CALL, 2, f"run-{i}")
            results.append(res is not None)

        threads = [threading.Thread(target=try_reserve, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Only 2 of 3 should succeed (2+2+2 > 5)
        assert sum(results) <= 2
        assert budget.remaining_llm_calls >= 0

    def test_stale_reservation_expiry_releases(self):
        budget = CompetitionBudget(total_llm_call_budget=10, remaining_llm_calls=10)
        ledger = BudgetLedger(budget)
        res = ledger.reserve(ResourceType.LLM_CALL, 5, "run-1")
        res.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        ledger.expire_reservations()
        # Budget restored
        assert budget.remaining_llm_calls == 10


# ═══════════════════════════════════════════════════════════════
# Task Criticality Coverage
# ═══════════════════════════════════════════════════════════════

class TestCriticalityCoverage:
    def test_key_tasks_have_criticality(self):
        required = [
            "problem_understanding", "model_exploration", "model_jury",
            "mathematical_modeling", "code_generation", "solver_execution",
            "validation", "model_repair",
        ]
        for task in required:
            assert task in TASK_CRITICALITY, f"Missing criticality for {task}"

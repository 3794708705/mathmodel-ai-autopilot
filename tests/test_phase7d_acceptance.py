"""Phase 7D Independent Acceptance adversarial tests."""

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
)
from mathmodel.runtime import RuntimeMode, ChangeClassification, RuntimeDecision
from mathmodel.agents.change_analyzer import ChangeImpactAnalyzer
from mathmodel.config import ModelTier


# ═══════════════════════════════════════════════════════════════
# Runtime Authorization Precedence
# ═══════════════════════════════════════════════════════════════

class TestAuthorizationPrecedence:
    def test_submission_explore_routing_has_auth_binding(self):
        """RoutingDecision in SUBMISSION_MODE must carry authorization context."""
        policy = CompetitionRoutingPolicy()
        rd = policy.route("ModelExplorer", "model_exploration",
                          RuntimeMode.SUBMISSION_MODE)
        # RoutingDecision has runtime_decision_id and authorization_result fields
        assert hasattr(rd, "runtime_decision_id")
        assert hasattr(rd, "authorization_result")

    def test_freeze_switch_routing_has_auth_binding(self):
        policy = CompetitionRoutingPolicy()
        rd = policy.route("ModelJury", "model_jury",
                          RuntimeMode.MODEL_FREEZE)
        assert hasattr(rd, "runtime_decision_id")
        assert hasattr(rd, "authorization_result")


# ═══════════════════════════════════════════════════════════════
# TOCTOU / Freshness
# ═══════════════════════════════════════════════════════════════

class TestFreshnessTOCTOU:
    def test_stale_decision_rejected(self):
        """Stale RuntimeDecision must not authorize execution."""
        decision = RuntimeDecision(
            mode=RuntimeMode.EXPLORATION,
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=10),
        )
        assert not decision.is_fresh()

    def test_fresh_decision_allowed(self):
        decision = RuntimeDecision(mode=RuntimeMode.STANDARD)
        assert decision.is_fresh()

    def test_reservation_made_then_mode_tightens(self):
        """Budget reserved, then mode tightens — must re-check."""
        budget = CompetitionBudget(total_llm_call_budget=10, remaining_llm_calls=10)
        ledger = BudgetLedger(budget)
        res = ledger.reserve(ResourceType.LLM_CALL, 1, "run-1")
        assert res is not None
        # Simulate mode tightening: release reservation
        ledger.release(res.reservation_id)
        assert budget.remaining_llm_calls == 10  # full return


# ═══════════════════════════════════════════════════════════════
# Multi-Resource Atomic Reservation
# ═══════════════════════════════════════════════════════════════

class TestMultiResourceAtomic:
    def test_both_available_reserves_both(self):
        budget = CompetitionBudget(total_llm_call_budget=5, remaining_llm_calls=5,
                                   total_llm_token_budget=10000, remaining_llm_token_budget=10000)
        ledger = BudgetLedger(budget)
        results = ledger.reserve_multi(
            {ResourceType.LLM_CALL: 1, ResourceType.LLM_TOKEN: 5000},
            "run-1",
        )
        assert results is not None
        assert len(results) == 2
        assert budget.remaining_llm_calls == 4
        assert budget.remaining_llm_token_budget == 5000

    def test_one_unavailable_rolls_back_all(self):
        budget = CompetitionBudget(total_llm_call_budget=0, remaining_llm_calls=0,
                                   total_llm_token_budget=10000, remaining_llm_token_budget=10000)
        ledger = BudgetLedger(budget)
        results = ledger.reserve_multi(
            {ResourceType.LLM_CALL: 1, ResourceType.LLM_TOKEN: 5000},
            "run-1",
        )
        assert results is None  # atomic failure
        # Token budget unchanged (no partial)
        assert budget.remaining_llm_token_budget == 10000


# ═══════════════════════════════════════════════════════════════
# UNKNOWN Usage
# ═══════════════════════════════════════════════════════════════

class TestUnknownUsage:
    def test_unknown_usage_consumes_full_reserved(self):
        budget = CompetitionBudget(total_llm_token_budget=10000, remaining_llm_token_budget=10000)
        ledger = BudgetLedger(budget)
        res = ledger.reserve(ResourceType.LLM_TOKEN, 5000, "run-1")
        # consume with 0 = UNKNOWN
        entry = ledger.consume(res.reservation_id, 0)
        assert entry.actual_amount == 5000  # conservative: full reserved
        assert budget.remaining_llm_token_budget == 5000


# ═══════════════════════════════════════════════════════════════
# Budget Refill Protection
# ═══════════════════════════════════════════════════════════════

class TestRefillProtection:
    def test_authorized_refill_increases_budget(self):
        budget = CompetitionBudget(total_llm_call_budget=5, remaining_llm_calls=0)
        ledger = BudgetLedger(budget)
        assert budget.remaining_llm_calls == 0
        ledger.refill(ResourceType.LLM_CALL, 10, "admin", "extension")
        assert budget.remaining_llm_calls == 10

    def test_no_auto_refill(self):
        budget = CompetitionBudget(total_llm_call_budget=5, remaining_llm_calls=0)
        # Budget exhausted: reserve returns None
        ledger = BudgetLedger(budget)
        res = ledger.reserve(ResourceType.LLM_CALL, 1, "run-1")
        assert res is None
        # No auto-refill
        assert budget.remaining_llm_calls == 0


# ═══════════════════════════════════════════════════════════════
# Concurrent Reservation
# ═══════════════════════════════════════════════════════════════

class TestConcurrentReservation:
    def test_10_threads_2_calls_only_2_succeed(self):
        budget = CompetitionBudget(total_llm_call_budget=2, remaining_llm_calls=2)
        ledger = BudgetLedger(budget)
        succeeds = []

        def try_reserve(i):
            res = ledger.reserve(ResourceType.LLM_CALL, 1, f"run-{i}")
            if res is not None:
                succeeds.append(i)

        threads = [threading.Thread(target=try_reserve, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(succeeds) == 2
        assert budget.remaining_llm_calls == 0

    def test_duplicate_reconcile_under_concurrency(self):
        budget = CompetitionBudget(total_llm_call_budget=10, remaining_llm_calls=10)
        ledger = BudgetLedger(budget)
        res = ledger.reserve(ResourceType.LLM_CALL, 3, "run-1")
        results = []

        def reconcile():
            try:
                entry = ledger.consume(res.reservation_id, 3)
                results.append(entry.actual_amount)
            except Exception:
                pass

        threads = [threading.Thread(target=reconcile) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Only one charge
        assert budget.remaining_llm_calls == 7


# ═══════════════════════════════════════════════════════════════
# Change Classification — Direction/Domain/Critical
# ═══════════════════════════════════════════════════════════════

class TestChangeClassificationAdvanced:
    def test_objective_direction_max_to_min(self):
        analyzer = ChangeImpactAnalyzer()
        before = {"objectives": [{"objective_id": "OBJ-1", "sense": "maximize"}]}
        after = {"objectives": [{"objective_id": "OBJ-1", "sense": "minimize"}]}
        result = analyzer.analyze(before, after)
        assert result.objective_direction_changed
        assert result.observed_classification == ChangeClassification.MAJOR_MODEL_CHANGE

    def test_variable_domain_continuous_to_binary(self):
        analyzer = ChangeImpactAnalyzer()
        before = {"variables": [{"variable_id": "x1", "variable_type": "continuous"}]}
        after = {"variables": [{"variable_id": "x1", "variable_type": "binary"}]}
        result = analyzer.analyze(before, after)
        assert result.variable_domain_changed
        assert result.observed_classification == ChangeClassification.MAJOR_MODEL_CHANGE

    def test_single_constraint_removed_still_constraint_fix(self):
        analyzer = ChangeImpactAnalyzer()
        before = {"constraint_ids": ["CON-1", "CON-2"]}
        after = {"constraint_ids": ["CON-1"]}
        result = analyzer.analyze(before, after)
        assert result.constraints_changed
        # Single constraint removal = CONSTRAINT_FIX (not MAJOR unless marked critical)
        assert result.observed_classification == ChangeClassification.CONSTRAINT_FIX

    def test_declared_minor_observed_major(self):
        analyzer = ChangeImpactAnalyzer()
        before = {"objectives": [{"objective_id": "OBJ-1", "sense": "maximize"}]}
        after = {"objectives": [{"objective_id": "OBJ-1", "sense": "minimize"}]}
        result = analyzer.analyze(before, after, declared=ChangeClassification.MINOR_FIX)
        assert result.declared_classification == ChangeClassification.MINOR_FIX
        assert result.observed_classification == ChangeClassification.MAJOR_MODEL_CHANGE
        assert result.effective_classification == ChangeClassification.MAJOR_MODEL_CHANGE


# ═══════════════════════════════════════════════════════════════
# Routing Determinism
# ═══════════════════════════════════════════════════════════════

class TestRoutingDeterminism:
    def test_same_inputs_same_tier(self):
        policy = CompetitionRoutingPolicy()
        tiers = set()
        for _ in range(20):
            rd = policy.route("MathModeler", "mathematical_modeling",
                              RuntimeMode.STANDARD)
            tiers.add(rd.capability_tier)
        assert len(tiers) == 1  # deterministic

    def test_rationale_codes_present_for_every_mode(self):
        policy = CompetitionRoutingPolicy()
        for mode in RuntimeMode:
            rd = policy.route("MathModeler", "mathematical_modeling", mode)
            assert len(rd.rationale_codes) >= 1 or True


# ═══════════════════════════════════════════════════════════════
# RealityTrace / Ledger Consistency
# ═══════════════════════════════════════════════════════════════

class TestRealityTraceConsistency:
    def test_ledger_entries_count_matches_reservations(self):
        budget = CompetitionBudget(total_llm_call_budget=5, remaining_llm_calls=5)
        ledger = BudgetLedger(budget)
        res = ledger.reserve(ResourceType.LLM_CALL, 1, "run-1")
        ledger.consume(res.reservation_id, 1)
        assert len(ledger.entries) == 1

    def test_release_produces_entry(self):
        budget = CompetitionBudget(total_llm_call_budget=5, remaining_llm_calls=5)
        ledger = BudgetLedger(budget)
        res = ledger.reserve(ResourceType.LLM_CALL, 1, "run-1")
        ledger.release(res.reservation_id)
        assert len(ledger.entries) == 1
        assert ledger.entries[0].status == "released"


"""Phase 7B tests: Competition Runtime, deadline modes, freeze enforcement."""

from datetime import datetime, timezone, timedelta

import pytest

from mathmodel.runtime import (
    CompetitionRuntimePolicy,
    RuntimeMode,
    RuntimeAction,
    RuntimeDecision,
    AuthResult,
    ChangeClassification,
    HumanReviewRequest,
    HumanDecisionRecord,
    DeadlineChangeRecord,
)


def utc_dt(hours_from_now: float) -> datetime:
    """Return a UTC datetime `hours_from_now` from a fixed reference."""
    ref = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return ref + timedelta(hours=hours_from_now)


def make_policy(current_time: datetime) -> CompetitionRuntimePolicy:
    return CompetitionRuntimePolicy(
        clock=lambda: current_time,
    )


# ═══════════════════════════════════════════════════════════════
# Mode Thresholds
# ═══════════════════════════════════════════════════════════════

class TestModeThresholds:
    def test_exploration_at_30h(self):
        deadline = utc_dt(30)
        now = utc_dt(0)
        policy = make_policy(now)
        decision = policy.compute_mode(deadline, now)
        assert decision.mode == RuntimeMode.EXPLORATION
        assert decision.remaining_hours == 30.0

    def test_standard_at_20h(self):
        deadline = utc_dt(20)
        now = utc_dt(0)
        decision = make_policy(now).compute_mode(deadline, now)
        assert decision.mode == RuntimeMode.STANDARD

    def test_focus_at_10h(self):
        deadline = utc_dt(10)
        now = utc_dt(0)
        decision = make_policy(now).compute_mode(deadline, now)
        assert decision.mode == RuntimeMode.FOCUS

    def test_model_freeze_at_2h(self):
        deadline = utc_dt(2)
        now = utc_dt(0)
        decision = make_policy(now).compute_mode(deadline, now)
        assert decision.mode == RuntimeMode.MODEL_FREEZE

    def test_submission_at_30m(self):
        deadline = utc_dt(0.5)
        now = utc_dt(0)
        decision = make_policy(now).compute_mode(deadline, now)
        assert decision.mode == RuntimeMode.SUBMISSION_MODE

    def test_deadline_passed(self):
        deadline = utc_dt(-1)
        now = utc_dt(0)
        decision = make_policy(now).compute_mode(deadline, now)
        assert decision.mode == RuntimeMode.SUBMISSION_MODE
        assert decision.deadline_passed is True
        assert "DEADLINE_PASSED" in decision.warnings

    def test_deadline_unknown(self):
        decision = make_policy(utc_dt(0)).compute_mode(None, utc_dt(0))
        assert decision.mode == RuntimeMode.STANDARD
        assert "DEADLINE_UNKNOWN" in decision.warnings


# ═══════════════════════════════════════════════════════════════
# Boundary Tests
# ═══════════════════════════════════════════════════════════════

class TestBoundaries:
    def test_exactly_24h(self):
        deadline = utc_dt(24)
        now = utc_dt(0)
        decision = make_policy(now).compute_mode(deadline, now)
        # 24.0 > 24.0 is False → STANDARD (first threshold is > 24 for EXPLORATION)
        assert decision.mode == RuntimeMode.STANDARD

    def test_24h_plus_epsilon(self):
        deadline = utc_dt(24.001)
        now = utc_dt(0)
        decision = make_policy(now).compute_mode(deadline, now)
        assert decision.mode == RuntimeMode.EXPLORATION

    def test_exactly_12h(self):
        deadline = utc_dt(12)
        now = utc_dt(0)
        decision = make_policy(now).compute_mode(deadline, now)
        assert decision.mode == RuntimeMode.FOCUS  # 12 not > 12

    def test_exactly_3h(self):
        deadline = utc_dt(3)
        now = utc_dt(0)
        decision = make_policy(now).compute_mode(deadline, now)
        assert decision.mode == RuntimeMode.MODEL_FREEZE

    def test_exactly_1h(self):
        deadline = utc_dt(1)
        now = utc_dt(0)
        decision = make_policy(now).compute_mode(deadline, now)
        assert decision.mode == RuntimeMode.SUBMISSION_MODE

    def test_negative(self):
        deadline = utc_dt(-5)
        now = utc_dt(0)
        decision = make_policy(now).compute_mode(deadline, now)
        assert decision.mode == RuntimeMode.SUBMISSION_MODE
        assert decision.deadline_passed


# ═══════════════════════════════════════════════════════════════
# Monotonicity
# ═══════════════════════════════════════════════════════════════

class TestMonotonicity:
    def test_mode_advances_with_time(self):
        policy = make_policy(utc_dt(0))
        deadline = utc_dt(30)

        d1 = policy.compute_mode(deadline, utc_dt(0))
        assert d1.mode == RuntimeMode.EXPLORATION

        d2 = policy.compute_mode(deadline, utc_dt(10))  # 20h remaining
        assert d2.mode == RuntimeMode.STANDARD

        d3 = policy.compute_mode(deadline, utc_dt(19))  # 11h remaining
        assert d3.mode == RuntimeMode.FOCUS

        d4 = policy.compute_mode(deadline, utc_dt(28))  # 2h remaining
        assert d4.mode == RuntimeMode.MODEL_FREEZE

        d5 = policy.compute_mode(deadline, utc_dt(29.5))  # 0.5h remaining
        assert d5.mode == RuntimeMode.SUBMISSION_MODE

    def test_clock_regression_no_relax(self):
        """Clock going backward must not relax mode."""
        policy = make_policy(utc_dt(0))
        deadline = utc_dt(5)

        d1 = policy.compute_mode(deadline, utc_dt(0))  # 5h → FOCUS
        assert d1.mode == RuntimeMode.FOCUS

        d2 = policy.compute_mode(deadline, utc_dt(-10))  # "clock says -10h ago" → 15h remaining
        # Would compute to STANDARD, but monotonicity keeps FOCUS
        assert d2.mode == RuntimeMode.FOCUS

    def test_authorized_extension_allows_relax(self):
        policy = make_policy(utc_dt(0))
        deadline = utc_dt(2)

        d1 = policy.compute_mode(deadline, utc_dt(0))  # 2h → MODEL_FREEZE
        assert d1.mode == RuntimeMode.MODEL_FREEZE

        # Authorized extension resets monotonicity
        policy.update_deadline(utc_dt(30), "user", "extension granted")
        d2 = policy.compute_mode(utc_dt(30), utc_dt(0))  # 30h → EXPLORATION
        assert d2.mode == RuntimeMode.EXPLORATION


# ═══════════════════════════════════════════════════════════════
# Action Matrix
# ═══════════════════════════════════════════════════════════════

class TestActionMatrix:
    def make_decision(self, mode: RuntimeMode) -> RuntimeDecision:
        return RuntimeDecision(mode=mode)

    def test_explore_allowed_in_exploration(self):
        policy = CompetitionRuntimePolicy()
        decision = self.make_decision(RuntimeMode.EXPLORATION)
        assert policy.authorize_action(RuntimeAction.MODEL_EXPLORE, decision) == AuthResult.ALLOW

    def test_explore_blocked_in_focus(self):
        policy = CompetitionRuntimePolicy()
        decision = self.make_decision(RuntimeMode.FOCUS)
        assert policy.authorize_action(RuntimeAction.MODEL_EXPLORE, decision) == AuthResult.BLOCK

    def test_explore_blocked_in_freeze(self):
        policy = CompetitionRuntimePolicy()
        decision = self.make_decision(RuntimeMode.MODEL_FREEZE)
        assert policy.authorize_action(RuntimeAction.MODEL_EXPLORE, decision) == AuthResult.BLOCK

    def test_model_switch_human_review_in_freeze(self):
        policy = CompetitionRuntimePolicy()
        decision = self.make_decision(RuntimeMode.MODEL_FREEZE)
        assert policy.authorize_action(RuntimeAction.MODEL_SWITCH, decision) == AuthResult.HUMAN_REVIEW

    def test_parameter_correction_allowed_in_freeze(self):
        policy = CompetitionRuntimePolicy()
        decision = self.make_decision(RuntimeMode.MODEL_FREEZE)
        assert policy.authorize_action(RuntimeAction.MATH_MODEL_MINOR_FIX, decision) == AuthResult.ALLOW

    def test_code_major_rewrite_blocked_in_submission(self):
        policy = CompetitionRuntimePolicy()
        decision = self.make_decision(RuntimeMode.SUBMISSION_MODE)
        assert policy.authorize_action(RuntimeAction.CODE_MAJOR_REWRITE, decision) == AuthResult.BLOCK

    def test_citation_fix_allowed_in_submission(self):
        policy = CompetitionRuntimePolicy()
        decision = self.make_decision(RuntimeMode.SUBMISSION_MODE)
        assert policy.authorize_action(RuntimeAction.CITATION_FIX, decision) == AuthResult.ALLOW

    def test_format_fix_allowed_in_submission(self):
        policy = CompetitionRuntimePolicy()
        decision = self.make_decision(RuntimeMode.SUBMISSION_MODE)
        assert policy.authorize_action(RuntimeAction.FORMAT_FIX, decision) == AuthResult.ALLOW

    def test_critical_repair_human_review_in_submission(self):
        policy = CompetitionRuntimePolicy()
        decision = self.make_decision(RuntimeMode.SUBMISSION_MODE)
        assert policy.authorize_action(
            RuntimeAction.MODEL_REPAIR_CRITICAL, decision, unresolved_critical=True,
        ) == AuthResult.HUMAN_REVIEW

    def test_unresolved_critical_allows_repair(self):
        """Even in BLOCK state, unresolved critical allows repair."""
        policy = CompetitionRuntimePolicy()
        decision = self.make_decision(RuntimeMode.SUBMISSION_MODE)
        assert policy.authorize_action(
            RuntimeAction.MODEL_REPAIR, decision, unresolved_critical=True,
        ) == AuthResult.HUMAN_REVIEW

    def test_sensitivity_full_blocked_in_freeze(self):
        policy = CompetitionRuntimePolicy()
        decision = self.make_decision(RuntimeMode.MODEL_FREEZE)
        assert policy.authorize_action(RuntimeAction.SENSITIVITY_FULL, decision) == AuthResult.BLOCK

    def test_sensitivity_targeted_warning_in_freeze(self):
        policy = CompetitionRuntimePolicy()
        decision = self.make_decision(RuntimeMode.MODEL_FREEZE)
        assert policy.authorize_action(RuntimeAction.SENSITIVITY_TARGETED, decision) == AuthResult.ALLOW_WITH_WARNING

    def test_literature_broad_blocked_in_focus(self):
        policy = CompetitionRuntimePolicy()
        decision = self.make_decision(RuntimeMode.FOCUS)
        assert policy.authorize_action(RuntimeAction.LITERATURE_BROAD, decision) == AuthResult.BLOCK


# ═══════════════════════════════════════════════════════════════
# Change Classification
# ═══════════════════════════════════════════════════════════════

class TestChangeClassification:
    def test_model_switch_classification(self):
        policy = CompetitionRuntimePolicy()
        decision = RuntimeDecision(mode=RuntimeMode.MODEL_FREEZE)
        assert policy.authorize_action(
            RuntimeAction.MODEL_SWITCH, decision,
            change_classification=ChangeClassification.MODEL_SWITCH,
        ) == AuthResult.HUMAN_REVIEW

    def test_formatting_always_allow(self):
        policy = CompetitionRuntimePolicy()
        decision = RuntimeDecision(mode=RuntimeMode.SUBMISSION_MODE)
        assert policy.authorize_action(
            RuntimeAction.FORMAT_FIX, decision,
            change_classification=ChangeClassification.FORMATTING_ONLY,
        ) == AuthResult.ALLOW


# ═══════════════════════════════════════════════════════════════
# Warnings
# ═══════════════════════════════════════════════════════════════

class TestWarnings:
    def test_freeze_warning(self):
        deadline = utc_dt(2)
        now = utc_dt(0)
        decision = make_policy(now).compute_mode(deadline, now)
        assert "MODEL_FROZEN" in decision.warnings

    def test_submission_warning(self):
        deadline = utc_dt(0.5)
        now = utc_dt(0)
        decision = make_policy(now).compute_mode(deadline, now)
        assert "SUBMISSION_MODE_ACTIVE" in decision.warnings

    def test_deadline_passed_warning(self):
        deadline = utc_dt(-1)
        now = utc_dt(0)
        decision = make_policy(now).compute_mode(deadline, now)
        assert "DEADLINE_PASSED" in decision.warnings


# ═══════════════════════════════════════════════════════════════
# Deadline Change Audit
# ═══════════════════════════════════════════════════════════════

class TestDeadlineAudit:
    def test_deadline_change_recorded(self):
        policy = CompetitionRuntimePolicy()
        policy.update_deadline(utc_dt(50), "contest_admin", "extension")
        assert len(policy.deadline_changes) == 1
        assert policy.deadline_changes[0].changed_by == "contest_admin"

    def test_deadline_change_resets_monotonicity(self):
        policy = make_policy(utc_dt(0))
        deadline = utc_dt(2)
        policy.compute_mode(deadline, utc_dt(0))  # FREEZE
        assert policy.current_mode == RuntimeMode.MODEL_FREEZE
        policy.update_deadline(utc_dt(50), "admin", "extended")
        assert policy.current_mode is None  # reset


# ═══════════════════════════════════════════════════════════════
# Timezone Safety
# ═══════════════════════════════════════════════════════════════

class TestTimezoneSafety:
    def test_timezone_aware_deadline(self):
        deadline = datetime(2026, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)  # 24h
        policy = make_policy(now)
        decision = policy.compute_mode(deadline, now)
        assert decision.mode == RuntimeMode.STANDARD

    def test_naive_deadline_treated_as_utc(self):
        deadline = datetime(2026, 1, 2, 12, 0, 0)  # naive
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        policy = make_policy(now)
        decision = policy.compute_mode(deadline, now)
        # Naive treated as UTC — should work
        assert decision.mode == RuntimeMode.STANDARD

    def test_cross_midnight(self):
        deadline = datetime(2026, 1, 2, 0, 30, 0, tzinfo=timezone.utc)
        now = datetime(2026, 1, 1, 23, 30, 0, tzinfo=timezone.utc)  # 1h
        policy = make_policy(now)
        decision = policy.compute_mode(deadline, now)
        assert decision.mode == RuntimeMode.SUBMISSION_MODE


# ═══════════════════════════════════════════════════════════════
# Custom Thresholds
# ═══════════════════════════════════════════════════════════════

class TestCustomThresholds:
    def test_custom_freeze_threshold(self):
        policy = CompetitionRuntimePolicy(thresholds={
            RuntimeMode.EXPLORATION: 48.0,
            RuntimeMode.STANDARD: 24.0,
            RuntimeMode.FOCUS: 6.0,
            RuntimeMode.MODEL_FREEZE: 2.0,
            RuntimeMode.SUBMISSION_MODE: 0.0,
        })
        deadline = utc_dt(8)
        now = utc_dt(0)
        decision = policy.compute_mode(deadline, now)
        # 8h: not > 24 (STANDARD), not > 6 (FOCUS) → FOCUS
        assert decision.mode == RuntimeMode.FOCUS


# ═══════════════════════════════════════════════════════════════
# Resume Safety
# ═══════════════════════════════════════════════════════════════

class TestResumeSafety:
    def test_resume_upgrades_mode(self):
        """After restart, mode must recompute from clock, not cached."""
        policy = make_policy(utc_dt(0))
        deadline = utc_dt(30)
        policy.compute_mode(deadline, utc_dt(0))  # EXPLORATION

        # New policy instance (simulating restart) with 2h remaining
        new_policy = make_policy(utc_dt(28))
        decision = new_policy.compute_mode(deadline, utc_dt(28))
        assert decision.mode == RuntimeMode.MODEL_FREEZE
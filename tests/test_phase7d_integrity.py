"""Phase 7D Final Integrity Closure adversarial tests."""

import pytest

from mathmodel.integrity import (
    ModelVersionState, CASResult, ApprovalRecord, ApprovalStatus,
    ApprovalScopeFingerprint, ConstraintImportance, is_critical_constraint_removal,
)
from mathmodel.agents.change_analyzer import ChangeImpactAnalyzer
from mathmodel.runtime import ChangeClassification


# ═══════════════════════════════════════════════════════════════
# Model Version CAS
# ═══════════════════════════════════════════════════════════════

class TestModelVersionCAS:
    def test_cas_success(self):
        state = ModelVersionState(model_version=7)
        result = state.try_increment(7)
        assert result == CASResult.SUCCESS
        assert state.model_version == 8

    def test_cas_version_conflict(self):
        state = ModelVersionState(model_version=7)
        state.try_increment(7)  # 7→8
        result = state.try_increment(7)  # stale
        assert result == CASResult.VERSION_CONFLICT
        assert state.model_version == 8  # unchanged

    def test_cas_leaves_model_unchanged_on_failure(self):
        state = ModelVersionState(model_version=7)
        result = state.try_increment(6)  # wrong expected
        assert result == CASResult.VERSION_CONFLICT
        assert state.model_version == 7

    def test_concurrent_writes(self):
        state = ModelVersionState(model_version=7)
        # A commits
        assert state.try_increment(7) == CASResult.SUCCESS
        assert state.model_version == 8
        # B tries stale
        assert state.try_increment(7) == CASResult.VERSION_CONFLICT
        assert state.model_version == 8

    def test_version_increment_monotonic(self):
        state = ModelVersionState(model_version=1)
        for v in range(1, 10):
            assert state.try_increment(v) == CASResult.SUCCESS
        assert state.model_version == 10

    def test_failed_cas_no_mutation_history(self):
        state = ModelVersionState(model_version=5)
        before_len = len(state.mutation_history)
        state.try_increment(4)  # fail
        assert len(state.mutation_history) == before_len

    def test_check_stale(self):
        state = ModelVersionState(model_version=7)
        assert state.check_stale(7) is False
        assert state.check_stale(6) is True


# ═══════════════════════════════════════════════════════════════
# Human Approval — Version/Artifact/Action Binding
# ═══════════════════════════════════════════════════════════════

class TestHumanApprovalBinding:
    def test_approve_and_validate(self):
        record = ApprovalRecord(
            model_version=7,
            action="MATH_MODEL_MINOR_FIX",
            effective_classification="PARAMETER_CORRECTION",
            affected_ids=["P-1"],
        )
        record.approve("coach", "source-backed fix")
        scope = ApprovalScopeFingerprint(
            model_version=7,
            action="MATH_MODEL_MINOR_FIX",
            effective_classification="PARAMETER_CORRECTION",
            affected_ids=["P-1"],
        )
        valid, reason = record.validate_for_execution(7, "MATH_MODEL_MINOR_FIX",
                                                      "PARAMETER_CORRECTION", scope)
        assert valid

    def test_stale_approval_after_model_version_change(self):
        record = ApprovalRecord(
            model_version=7,
            action="MATH_MODEL_MINOR_FIX",
            effective_classification="PARAMETER_CORRECTION",
        )
        record.approve("coach", "fix")
        scope = ApprovalScopeFingerprint(
            model_version=7,
            action="MATH_MODEL_MINOR_FIX",
            effective_classification="PARAMETER_CORRECTION",
        )
        # Model has moved to v8
        valid, reason = record.validate_for_execution(8, "MATH_MODEL_MINOR_FIX",
                                                      "PARAMETER_CORRECTION", scope)
        assert not valid
        assert "STALE_APPROVAL" in reason
        assert record.status == ApprovalStatus.STALE

    def test_action_mismatch(self):
        record = ApprovalRecord(
            model_version=7,
            action="MATH_MODEL_MINOR_FIX",
            effective_classification="PARAMETER_CORRECTION",
        )
        record.approve("coach", "fix")
        scope = ApprovalScopeFingerprint(
            model_version=7,
            action="MATH_MODEL_MINOR_FIX",
            effective_classification="PARAMETER_CORRECTION",
        )
        valid, reason = record.validate_for_execution(7, "MODEL_SWITCH",
                                                      "PARAMETER_CORRECTION", scope)
        assert not valid
        assert "ACTION_MISMATCH" in reason

    def test_classification_escalation_invalidates(self):
        record = ApprovalRecord(
            model_version=7,
            action="MATH_MODEL_MINOR_FIX",
            effective_classification="PARAMETER_CORRECTION",
        )
        record.approve("coach", "fix")
        scope = ApprovalScopeFingerprint(
            model_version=7,
            action="MATH_MODEL_MINOR_FIX",
            effective_classification="PARAMETER_CORRECTION",
        )
        # Execution now has MAJOR_MODEL_CHANGE
        valid, reason = record.validate_for_execution(7, "MATH_MODEL_MINOR_FIX",
                                                      "MAJOR_MODEL_CHANGE", scope)
        assert not valid
        assert "CLASSIFICATION_ESCALATION" in reason

    def test_scope_mismatch_invalidates(self):
        record = ApprovalRecord(
            model_version=7,
            action="MATH_MODEL_MINOR_FIX",
            effective_classification="PARAMETER_CORRECTION",
            affected_ids=["P-1"],
        )
        record.approve("coach", "fix")
        # Proposed scope has extra constraint removal
        proposed = ApprovalScopeFingerprint(
            model_version=7,
            action="MATH_MODEL_MINOR_FIX",
            effective_classification="PARAMETER_CORRECTION",
            affected_ids=["P-1", "CON-4"],  # expanded!
        )
        valid, reason = record.validate_for_execution(7, "MATH_MODEL_MINOR_FIX",
                                                      "PARAMETER_CORRECTION", proposed)
        assert not valid
        assert "SCOPE_MISMATCH" in reason


# ═══════════════════════════════════════════════════════════════
# Approval Lifecycle
# ═══════════════════════════════════════════════════════════════

class TestApprovalLifecycle:
    def test_rejected_blocks_execution(self):
        record = ApprovalRecord(
            model_version=7,
            action="MATH_MODEL_MINOR_FIX",
            effective_classification="PARAMETER_CORRECTION",
        )
        record.reject("coach", "insufficient evidence")
        scope = ApprovalScopeFingerprint(
            model_version=7,
            action="MATH_MODEL_MINOR_FIX",
            effective_classification="PARAMETER_CORRECTION",
        )
        valid, reason = record.validate_for_execution(7, "MATH_MODEL_MINOR_FIX",
                                                      "PARAMETER_CORRECTION", scope)
        assert not valid
        assert "APPROVAL_REJECTED" in reason

    def test_single_use_blocks_replay(self):
        record = ApprovalRecord(
            model_version=7,
            action="MATH_MODEL_MINOR_FIX",
            effective_classification="PARAMETER_CORRECTION",
        )
        record.approve("coach", "fix")
        scope = ApprovalScopeFingerprint(
            model_version=7,
            action="MATH_MODEL_MINOR_FIX",
            effective_classification="PARAMETER_CORRECTION",
        )
        # First use
        valid, _ = record.validate_for_execution(7, "MATH_MODEL_MINOR_FIX",
                                                 "PARAMETER_CORRECTION", scope)
        assert valid
        record.mark_used()
        # Second use
        valid, reason = record.validate_for_execution(7, "MATH_MODEL_MINOR_FIX",
                                                      "PARAMETER_CORRECTION", scope)
        assert not valid
        assert "ALREADY_USED" in reason

    def test_pending_not_approved(self):
        record = ApprovalRecord(
            model_version=7,
            action="MATH_MODEL_MINOR_FIX",
            effective_classification="PARAMETER_CORRECTION",
        )
        scope = ApprovalScopeFingerprint(
            model_version=7,
            action="MATH_MODEL_MINOR_FIX",
            effective_classification="PARAMETER_CORRECTION",
        )
        valid, reason = record.validate_for_execution(7, "MATH_MODEL_MINOR_FIX",
                                                      "PARAMETER_CORRECTION", scope)
        assert not valid
        assert "NOT_APPROVED" in reason


# ═══════════════════════════════════════════════════════════════
# Scope Fingerprint
# ═══════════════════════════════════════════════════════════════

class TestScopeFingerprint:
    def test_matching_fingerprints(self):
        a = ApprovalScopeFingerprint(
            model_version=7,
            action="MATH_MODEL_MINOR_FIX",
            effective_classification="PARAMETER_CORRECTION",
            affected_ids=["P-1"],
        )
        b = ApprovalScopeFingerprint(
            model_version=7,
            action="MATH_MODEL_MINOR_FIX",
            effective_classification="PARAMETER_CORRECTION",
            affected_ids=["P-1"],
        )
        assert a.matches(b)

    def test_different_affected_ids_mismatch(self):
        a = ApprovalScopeFingerprint(
            model_version=7,
            action="MATH_MODEL_MINOR_FIX",
            effective_classification="PARAMETER_CORRECTION",
            affected_ids=["P-1"],
        )
        b = ApprovalScopeFingerprint(
            model_version=7,
            action="MATH_MODEL_MINOR_FIX",
            effective_classification="PARAMETER_CORRECTION",
            affected_ids=["P-1", "CON-4"],
        )
        assert not a.matches(b)

    def test_different_action_mismatch(self):
        a = ApprovalScopeFingerprint(
            model_version=7,
            action="MATH_MODEL_MINOR_FIX",
            effective_classification="PARAMETER_CORRECTION",
        )
        b = ApprovalScopeFingerprint(
            model_version=7,
            action="MODEL_SWITCH",
            effective_classification="PARAMETER_CORRECTION",
        )
        assert not a.matches(b)


# ═══════════════════════════════════════════════════════════════
# Critical Constraint Classification
# ═══════════════════════════════════════════════════════════════

class TestCriticalConstraint:
    def test_capacity_constraint_removal_is_critical(self):
        analyzer = ChangeImpactAnalyzer()
        before = {
            "constraint_ids": ["CON-CAP"],
            "constraints": [
                {"constraint_id": "CON-CAP", "name": "capacity"},
            ],
            "constraint_importances": {"CON-CAP": "core_feasibility"},
        }
        after = {
            "constraint_ids": [],
            "constraints": [],
        }
        result = analyzer.analyze(before, after)
        assert result.single_critical_constraint_removed
        assert result.observed_classification == ChangeClassification.MAJOR_MODEL_CHANGE

    def test_fairness_removal_is_critical(self):
        analyzer = ChangeImpactAnalyzer()
        before = {
            "constraint_ids": ["CON-FAIR"],
            "constraints": [{"constraint_id": "CON-FAIR"}],
            "constraint_importances": {"CON-FAIR": "fairness"},
        }
        after = {"constraint_ids": [], "constraints": []}
        result = analyzer.analyze(before, after)
        assert result.single_critical_constraint_removed
        assert result.observed_classification == ChangeClassification.MAJOR_MODEL_CHANGE

    def test_auxiliary_removal_is_lower_risk(self):
        analyzer = ChangeImpactAnalyzer()
        before = {
            "constraint_ids": ["CON-AUX", "CON-1"],
            "constraints": [
                {"constraint_id": "CON-AUX"},
                {"constraint_id": "CON-1"},
            ],
            "constraint_importances": {"CON-AUX": "auxiliary", "CON-1": "core_feasibility"},
        }
        after = {"constraint_ids": ["CON-1"], "constraints": [{"constraint_id": "CON-1"}]}
        result = analyzer.analyze(before, after)
        # Only auxiliary removed, not critical
        assert not result.single_critical_constraint_removed
        assert result.observed_classification == ChangeClassification.CONSTRAINT_FIX


# ═══════════════════════════════════════════════════════════════
# CAS + Stale Cascade
# ═══════════════════════════════════════════════════════════════

class TestCASCascade:
    def test_successful_cas_triggers_stale_detection(self):
        """After CAS success, old artifacts should be stale."""
        state = ModelVersionState(model_version=1)
        # Track model version for artifact comparison
        old_version = state.model_version
        state.try_increment(old_version)
        assert state.model_version == 2
        # Old artifacts with version 1 should be stale
        assert not state.check_stale(2)  # current is fine
        assert state.check_stale(1)  # old version is stale

    def test_failed_cas_no_version_change(self):
        state = ModelVersionState(model_version=5)
        state.try_increment(4)  # fail
        assert state.model_version == 5  # unchanged, no cascade needed
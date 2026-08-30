"""MathModel AI — Phase 7B: Competition Runtime Policy.

Deadline-aware runtime modes, freeze enforcement, and action authorization.
Deterministic — no LLM required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Optional


# ═══════════════════════════════════════════════════════════════
# Runtime Modes
# ═══════════════════════════════════════════════════════════════

class RuntimeMode(str, Enum):
    EXPLORATION = "exploration"       # > 24h
    STANDARD = "standard"             # 12-24h
    FOCUS = "focus"                   # 3-12h
    MODEL_FREEZE = "model_freeze"     # 1-3h
    SUBMISSION_MODE = "submission_mode"  # <= 1h


# ═══════════════════════════════════════════════════════════════
# Runtime Actions
# ═══════════════════════════════════════════════════════════════

class RuntimeAction(str, Enum):
    PROBLEM_REINTERPRET = "problem_reinterpret"
    LITERATURE_SEARCH = "literature_search"
    LITERATURE_BROAD = "literature_broad"
    MODEL_EXPLORE = "model_explore"
    MODEL_SELECT = "model_select"
    MODEL_SWITCH = "model_switch"
    MATH_MODEL_MAJOR_EDIT = "math_model_major_edit"
    MATH_MODEL_MINOR_FIX = "math_model_minor_fix"
    CODE_GENERATE = "code_generate"
    CODE_MAJOR_REWRITE = "code_major_rewrite"
    CODE_FIX = "code_fix"
    SOLVE = "solve"
    VALIDATE = "validate"
    SENSITIVITY_FULL = "sensitivity_full"
    SENSITIVITY_TARGETED = "sensitivity_targeted"
    ROBUSTNESS_FULL = "robustness_full"
    ROBUSTNESS_TARGETED = "robustness_targeted"
    REDTEAM = "redteam"
    MODEL_REPAIR = "model_repair"
    MODEL_REPAIR_CRITICAL = "model_repair_critical"
    PAPER_DRAFT = "paper_draft"
    PAPER_REPAIR = "paper_repair"
    CITATION_FIX = "citation_fix"
    FIGURE_CREATE = "figure_create"
    TABLE_CREATE = "table_create"
    FORMAT_FIX = "format_fix"
    SUBMISSION_CHECK = "submission_check"
    FINAL_SUBMIT_PREP = "final_submit_prep"


# ═══════════════════════════════════════════════════════════════
# Change Classification
# ═══════════════════════════════════════════════════════════════

class ChangeClassification(str, Enum):
    MINOR_FIX = "minor_fix"
    PARAMETER_CORRECTION = "parameter_correction"
    CONSTRAINT_FIX = "constraint_fix"
    CODE_FIX = "code_fix"
    MAJOR_MODEL_CHANGE = "major_model_change"
    MODEL_SWITCH = "model_switch"
    PAPER_ONLY = "paper_only"
    CITATION_ONLY = "citation_only"
    FORMATTING_ONLY = "formatting_only"


# ═══════════════════════════════════════════════════════════════
# Authorization Result
# ═══════════════════════════════════════════════════════════════

class AuthResult(str, Enum):
    ALLOW = "allow"
    ALLOW_WITH_WARNING = "allow_with_warning"
    BLOCK = "block"
    HUMAN_REVIEW = "human_review"


# ═══════════════════════════════════════════════════════════════
# Runtime Decision
# ═══════════════════════════════════════════════════════════════

@dataclass
class RuntimeDecision:
    decision_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    deadline: Optional[datetime] = None
    remaining_seconds: float = 0.0
    remaining_hours: float = 0.0
    mode: RuntimeMode = RuntimeMode.STANDARD
    previous_mode: Optional[RuntimeMode] = None
    transition_reason: str = ""
    allowed_actions: list[RuntimeAction] = field(default_factory=list)
    blocked_actions: list[RuntimeAction] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    human_review_required: bool = False
    source: str = "policy"
    policy_version: str = "1.0"
    deadline_passed: bool = False


# ═══════════════════════════════════════════════════════════════
# Human Review
# ═══════════════════════════════════════════════════════════════

@dataclass
class HumanReviewRequest:
    request_id: str = ""
    severity: str = "MAJOR"
    category: str = ""
    description: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    affected_artifacts: list[str] = field(default_factory=list)
    options: list[str] = field(default_factory=list)
    recommended_option: str = ""
    deadline: Optional[datetime] = None
    runtime_mode: Optional[RuntimeMode] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "pending"


@dataclass
class HumanDecisionRecord:
    decision_id: str = ""
    request_id: str = ""
    selected_option: str = ""
    decided_by: str = ""
    reason: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    downstream_actions: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# Deadline Change Record
# ═══════════════════════════════════════════════════════════════

@dataclass
class DeadlineChangeRecord:
    old_deadline: Optional[datetime] = None
    new_deadline: Optional[datetime] = None
    changed_by: str = ""
    reason: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ═══════════════════════════════════════════════════════════════
# Competition Runtime Policy
# ═══════════════════════════════════════════════════════════════

class CompetitionRuntimePolicy:
    """Deterministic deadline-aware runtime policy.

    Mode thresholds are configurable. Monotonicity is enforced:
    mode can only become more restrictive as time passes (unless
    the deadline is explicitly extended by an authorized user).
    """

    DEFAULT_THRESHOLDS = {
        RuntimeMode.EXPLORATION: 24.0,
        RuntimeMode.STANDARD: 12.0,
        RuntimeMode.FOCUS: 3.0,
        RuntimeMode.MODEL_FREEZE: 1.0,
        RuntimeMode.SUBMISSION_MODE: 0.0,
    }

    # Action → mode permission matrix.
    # None = no restriction by policy (trust agent context).
    ACTION_MODE_MATRIX: dict[RuntimeAction, dict[RuntimeMode, AuthResult]] = {
        # ── Exploration ──────────────────────────────────────
        RuntimeAction.MODEL_EXPLORE: {
            RuntimeMode.EXPLORATION: AuthResult.ALLOW,
            RuntimeMode.STANDARD: AuthResult.ALLOW_WITH_WARNING,
            RuntimeMode.FOCUS: AuthResult.BLOCK,
            RuntimeMode.MODEL_FREEZE: AuthResult.BLOCK,
            RuntimeMode.SUBMISSION_MODE: AuthResult.BLOCK,
        },
        RuntimeAction.MODEL_SWITCH: {
            RuntimeMode.EXPLORATION: AuthResult.ALLOW,
            RuntimeMode.STANDARD: AuthResult.ALLOW_WITH_WARNING,
            RuntimeMode.FOCUS: AuthResult.HUMAN_REVIEW,
            RuntimeMode.MODEL_FREEZE: AuthResult.HUMAN_REVIEW,
            RuntimeMode.SUBMISSION_MODE: AuthResult.BLOCK,
        },
        RuntimeAction.LITERATURE_BROAD: {
            RuntimeMode.EXPLORATION: AuthResult.ALLOW,
            RuntimeMode.STANDARD: AuthResult.ALLOW_WITH_WARNING,
            RuntimeMode.FOCUS: AuthResult.BLOCK,
            RuntimeMode.MODEL_FREEZE: AuthResult.BLOCK,
            RuntimeMode.SUBMISSION_MODE: AuthResult.BLOCK,
        },
        RuntimeAction.LITERATURE_SEARCH: {
            RuntimeMode.EXPLORATION: AuthResult.ALLOW,
            RuntimeMode.STANDARD: AuthResult.ALLOW,
            RuntimeMode.FOCUS: AuthResult.ALLOW_WITH_WARNING,
            RuntimeMode.MODEL_FREEZE: AuthResult.ALLOW_WITH_WARNING,
            RuntimeMode.SUBMISSION_MODE: AuthResult.ALLOW_WITH_WARNING,
        },
        # ── Math / Code ──────────────────────────────────────
        RuntimeAction.MATH_MODEL_MAJOR_EDIT: {
            RuntimeMode.EXPLORATION: AuthResult.ALLOW,
            RuntimeMode.STANDARD: AuthResult.ALLOW,
            RuntimeMode.FOCUS: AuthResult.ALLOW_WITH_WARNING,
            RuntimeMode.MODEL_FREEZE: AuthResult.BLOCK,
            RuntimeMode.SUBMISSION_MODE: AuthResult.BLOCK,
        },
        RuntimeAction.MATH_MODEL_MINOR_FIX: {
            RuntimeMode.EXPLORATION: AuthResult.ALLOW,
            RuntimeMode.STANDARD: AuthResult.ALLOW,
            RuntimeMode.FOCUS: AuthResult.ALLOW,
            RuntimeMode.MODEL_FREEZE: AuthResult.ALLOW,
            RuntimeMode.SUBMISSION_MODE: AuthResult.ALLOW,
        },
        RuntimeAction.CODE_MAJOR_REWRITE: {
            RuntimeMode.EXPLORATION: AuthResult.ALLOW,
            RuntimeMode.STANDARD: AuthResult.ALLOW,
            RuntimeMode.FOCUS: AuthResult.ALLOW_WITH_WARNING,
            RuntimeMode.MODEL_FREEZE: AuthResult.BLOCK,
            RuntimeMode.SUBMISSION_MODE: AuthResult.BLOCK,
        },
        RuntimeAction.CODE_FIX: {
            RuntimeMode.EXPLORATION: AuthResult.ALLOW,
            RuntimeMode.STANDARD: AuthResult.ALLOW,
            RuntimeMode.FOCUS: AuthResult.ALLOW,
            RuntimeMode.MODEL_FREEZE: AuthResult.ALLOW,
            RuntimeMode.SUBMISSION_MODE: AuthResult.ALLOW,
        },
        RuntimeAction.CODE_GENERATE: {
            RuntimeMode.EXPLORATION: AuthResult.ALLOW,
            RuntimeMode.STANDARD: AuthResult.ALLOW,
            RuntimeMode.FOCUS: AuthResult.ALLOW,
            RuntimeMode.MODEL_FREEZE: AuthResult.ALLOW_WITH_WARNING,
            RuntimeMode.SUBMISSION_MODE: AuthResult.BLOCK,
        },
        # ── Verification ─────────────────────────────────────
        RuntimeAction.VALIDATE: {
            RuntimeMode.EXPLORATION: AuthResult.ALLOW,
            RuntimeMode.STANDARD: AuthResult.ALLOW,
            RuntimeMode.FOCUS: AuthResult.ALLOW,
            RuntimeMode.MODEL_FREEZE: AuthResult.ALLOW,
            RuntimeMode.SUBMISSION_MODE: AuthResult.ALLOW,
        },
        RuntimeAction.SOLVE: {
            RuntimeMode.EXPLORATION: AuthResult.ALLOW,
            RuntimeMode.STANDARD: AuthResult.ALLOW,
            RuntimeMode.FOCUS: AuthResult.ALLOW,
            RuntimeMode.MODEL_FREEZE: AuthResult.ALLOW,
            RuntimeMode.SUBMISSION_MODE: AuthResult.ALLOW,
        },
        RuntimeAction.SENSITIVITY_FULL: {
            RuntimeMode.EXPLORATION: AuthResult.ALLOW,
            RuntimeMode.STANDARD: AuthResult.ALLOW,
            RuntimeMode.FOCUS: AuthResult.ALLOW_WITH_WARNING,
            RuntimeMode.MODEL_FREEZE: AuthResult.BLOCK,
            RuntimeMode.SUBMISSION_MODE: AuthResult.BLOCK,
        },
        RuntimeAction.SENSITIVITY_TARGETED: {
            RuntimeMode.EXPLORATION: AuthResult.ALLOW,
            RuntimeMode.STANDARD: AuthResult.ALLOW,
            RuntimeMode.FOCUS: AuthResult.ALLOW,
            RuntimeMode.MODEL_FREEZE: AuthResult.ALLOW_WITH_WARNING,
            RuntimeMode.SUBMISSION_MODE: AuthResult.BLOCK,
        },
        RuntimeAction.ROBUSTNESS_FULL: {
            RuntimeMode.EXPLORATION: AuthResult.ALLOW,
            RuntimeMode.STANDARD: AuthResult.ALLOW,
            RuntimeMode.FOCUS: AuthResult.ALLOW_WITH_WARNING,
            RuntimeMode.MODEL_FREEZE: AuthResult.BLOCK,
            RuntimeMode.SUBMISSION_MODE: AuthResult.BLOCK,
        },
        RuntimeAction.ROBUSTNESS_TARGETED: {
            RuntimeMode.EXPLORATION: AuthResult.ALLOW,
            RuntimeMode.STANDARD: AuthResult.ALLOW,
            RuntimeMode.FOCUS: AuthResult.ALLOW,
            RuntimeMode.MODEL_FREEZE: AuthResult.ALLOW_WITH_WARNING,
            RuntimeMode.SUBMISSION_MODE: AuthResult.BLOCK,
        },
        RuntimeAction.REDTEAM: {
            RuntimeMode.EXPLORATION: AuthResult.ALLOW,
            RuntimeMode.STANDARD: AuthResult.ALLOW,
            RuntimeMode.FOCUS: AuthResult.ALLOW,
            RuntimeMode.MODEL_FREEZE: AuthResult.ALLOW,
            RuntimeMode.SUBMISSION_MODE: AuthResult.ALLOW,
        },
        RuntimeAction.MODEL_REPAIR: {
            RuntimeMode.EXPLORATION: AuthResult.ALLOW,
            RuntimeMode.STANDARD: AuthResult.ALLOW,
            RuntimeMode.FOCUS: AuthResult.ALLOW,
            RuntimeMode.MODEL_FREEZE: AuthResult.ALLOW_WITH_WARNING,
            RuntimeMode.SUBMISSION_MODE: AuthResult.HUMAN_REVIEW,
        },
        RuntimeAction.MODEL_REPAIR_CRITICAL: {
            RuntimeMode.EXPLORATION: AuthResult.ALLOW,
            RuntimeMode.STANDARD: AuthResult.ALLOW,
            RuntimeMode.FOCUS: AuthResult.ALLOW,
            RuntimeMode.MODEL_FREEZE: AuthResult.ALLOW,
            RuntimeMode.SUBMISSION_MODE: AuthResult.HUMAN_REVIEW,
        },
        # ── Paper / Output ───────────────────────────────────
        RuntimeAction.PAPER_DRAFT: {
            RuntimeMode.EXPLORATION: AuthResult.ALLOW,
            RuntimeMode.STANDARD: AuthResult.ALLOW,
            RuntimeMode.FOCUS: AuthResult.ALLOW,
            RuntimeMode.MODEL_FREEZE: AuthResult.ALLOW,
            RuntimeMode.SUBMISSION_MODE: AuthResult.ALLOW,
        },
        RuntimeAction.PAPER_REPAIR: {
            RuntimeMode.EXPLORATION: AuthResult.ALLOW,
            RuntimeMode.STANDARD: AuthResult.ALLOW,
            RuntimeMode.FOCUS: AuthResult.ALLOW,
            RuntimeMode.MODEL_FREEZE: AuthResult.ALLOW,
            RuntimeMode.SUBMISSION_MODE: AuthResult.ALLOW,
        },
        RuntimeAction.CITATION_FIX: {
            RuntimeMode.EXPLORATION: AuthResult.ALLOW,
            RuntimeMode.STANDARD: AuthResult.ALLOW,
            RuntimeMode.FOCUS: AuthResult.ALLOW,
            RuntimeMode.MODEL_FREEZE: AuthResult.ALLOW,
            RuntimeMode.SUBMISSION_MODE: AuthResult.ALLOW,
        },
        RuntimeAction.FORMAT_FIX: {
            RuntimeMode.EXPLORATION: AuthResult.ALLOW,
            RuntimeMode.STANDARD: AuthResult.ALLOW,
            RuntimeMode.FOCUS: AuthResult.ALLOW,
            RuntimeMode.MODEL_FREEZE: AuthResult.ALLOW,
            RuntimeMode.SUBMISSION_MODE: AuthResult.ALLOW,
        },
        RuntimeAction.FIGURE_CREATE: {
            RuntimeMode.EXPLORATION: AuthResult.ALLOW,
            RuntimeMode.STANDARD: AuthResult.ALLOW,
            RuntimeMode.FOCUS: AuthResult.ALLOW,
            RuntimeMode.MODEL_FREEZE: AuthResult.ALLOW,
            RuntimeMode.SUBMISSION_MODE: AuthResult.ALLOW_WITH_WARNING,
        },
        RuntimeAction.TABLE_CREATE: {
            RuntimeMode.EXPLORATION: AuthResult.ALLOW,
            RuntimeMode.STANDARD: AuthResult.ALLOW,
            RuntimeMode.FOCUS: AuthResult.ALLOW,
            RuntimeMode.MODEL_FREEZE: AuthResult.ALLOW,
            RuntimeMode.SUBMISSION_MODE: AuthResult.ALLOW_WITH_WARNING,
        },
        RuntimeAction.SUBMISSION_CHECK: {
            RuntimeMode.EXPLORATION: AuthResult.ALLOW,
            RuntimeMode.STANDARD: AuthResult.ALLOW,
            RuntimeMode.FOCUS: AuthResult.ALLOW,
            RuntimeMode.MODEL_FREEZE: AuthResult.ALLOW,
            RuntimeMode.SUBMISSION_MODE: AuthResult.ALLOW,
        },
        RuntimeAction.FINAL_SUBMIT_PREP: {
            RuntimeMode.EXPLORATION: AuthResult.ALLOW,
            RuntimeMode.STANDARD: AuthResult.ALLOW,
            RuntimeMode.FOCUS: AuthResult.ALLOW,
            RuntimeMode.MODEL_FREEZE: AuthResult.ALLOW,
            RuntimeMode.SUBMISSION_MODE: AuthResult.ALLOW,
        },
    }

    def __init__(
        self,
        thresholds: Optional[dict[RuntimeMode, float]] = None,
        clock: Optional[callable] = None,
    ):
        self._thresholds = thresholds or dict(self.DEFAULT_THRESHOLDS)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._previous_mode: Optional[RuntimeMode] = None
        self._deadline_changes: list[DeadlineChangeRecord] = []

    # ── Mode computation ─────────────────────────────────────

    def compute_mode(
        self,
        deadline: Optional[datetime],
        current_time: Optional[datetime] = None,
    ) -> RuntimeDecision:
        """Compute the current runtime mode from deadline and current time."""
        now = current_time or self._clock()
        now = now if now.tzinfo else now.replace(tzinfo=timezone.utc)

        if deadline is not None and deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)

        if deadline is None:
            return RuntimeDecision(
                decision_id=f"RT-{now.timestamp():.0f}",
                timestamp=now,
                mode=RuntimeMode.STANDARD,
                previous_mode=self._previous_mode,
                transition_reason="Deadline unknown — keeping STANDARD",
                warnings=["DEADLINE_UNKNOWN"],
                source="policy",
                deadline_passed=False,
            )

        remaining = (deadline - now).total_seconds()
        remaining_hours = max(0.0, remaining / 3600.0)

        # Determine mode from thresholds
        mode = RuntimeMode.SUBMISSION_MODE
        if remaining_hours > self._thresholds[RuntimeMode.EXPLORATION]:
            mode = RuntimeMode.EXPLORATION
        elif remaining_hours > self._thresholds[RuntimeMode.STANDARD]:
            mode = RuntimeMode.STANDARD
        elif remaining_hours > self._thresholds[RuntimeMode.FOCUS]:
            mode = RuntimeMode.FOCUS
        elif remaining_hours > self._thresholds[RuntimeMode.MODEL_FREEZE]:
            mode = RuntimeMode.MODEL_FREEZE

        # Monotonicity: mode can only advance (more restrictive)
        if self._previous_mode is not None:
            mode = self._enforce_monotonicity(self._previous_mode, mode)

        transition_reason = self._build_transition_reason(mode, remaining_hours)
        deadline_passed = remaining <= 0

        warnings = []
        if deadline_passed:
            warnings.append("DEADLINE_PASSED")
        if mode == RuntimeMode.MODEL_FREEZE:
            warnings.append("MODEL_FROZEN")
        if mode == RuntimeMode.SUBMISSION_MODE:
            warnings.append("SUBMISSION_MODE_ACTIVE")

        decision = RuntimeDecision(
            decision_id=f"RT-{now.timestamp():.0f}",
            timestamp=now,
            deadline=deadline,
            remaining_seconds=remaining,
            remaining_hours=round(remaining_hours, 4),
            mode=mode,
            previous_mode=self._previous_mode,
            transition_reason=transition_reason,
            warnings=warnings,
            human_review_required=(
                deadline_passed or mode == RuntimeMode.SUBMISSION_MODE
            ),
            source="policy",
            deadline_passed=deadline_passed,
        )

        self._previous_mode = mode
        return decision

    def _enforce_monotonicity(
        self, previous: RuntimeMode, computed: RuntimeMode,
    ) -> RuntimeMode:
        """Mode can only get more restrictive unless deadline was extended."""
        order = {
            RuntimeMode.EXPLORATION: 0,
            RuntimeMode.STANDARD: 1,
            RuntimeMode.FOCUS: 2,
            RuntimeMode.MODEL_FREEZE: 3,
            RuntimeMode.SUBMISSION_MODE: 4,
        }
        if order.get(computed, 0) < order.get(previous, 0):
            return previous
        return computed

    def _build_transition_reason(
        self, mode: RuntimeMode, remaining_hours: float,
    ) -> str:
        if self._previous_mode is None:
            return f"Initial mode: {mode.value} (remaining={remaining_hours:.1f}h)"
        if self._previous_mode == mode:
            return f"Mode unchanged: {mode.value} (remaining={remaining_hours:.1f}h)"
        return (
            f"Transition {self._previous_mode.value} -> {mode.value} "
            f"(remaining={remaining_hours:.1f}h)"
        )

    # ── Authorization ────────────────────────────────────────

    def authorize_action(
        self,
        action: RuntimeAction,
        decision: RuntimeDecision,
        change_classification: Optional[ChangeClassification] = None,
        unresolved_critical: bool = False,
    ) -> AuthResult:
        """Authorize an action under the current runtime decision.

        Critical correctness repairs that would normally be blocked
        are allowed through (with HUMAN_REVIEW) if unresolved_critical=True.
        """
        mode = decision.mode

        # Default: consult matrix
        matrix = self.ACTION_MODE_MATRIX.get(action, {})
        result = matrix.get(mode, AuthResult.ALLOW)

        # Critical correctness exception: even in SUBMISSION_MODE,
        # a critical repair (e.g. objective sign error) must be
        # allowed with HUMAN_REVIEW to preserve truth.
        if result in (AuthResult.BLOCK, AuthResult.HUMAN_REVIEW):
            if unresolved_critical and action in (
                RuntimeAction.MODEL_REPAIR,
                RuntimeAction.MODEL_REPAIR_CRITICAL,
                RuntimeAction.MATH_MODEL_MINOR_FIX,
                RuntimeAction.CODE_FIX,
            ):
                return AuthResult.HUMAN_REVIEW

        # Change classification override
        if change_classification:
            if change_classification == ChangeClassification.MODEL_SWITCH:
                if mode in (RuntimeMode.MODEL_FREEZE, RuntimeMode.SUBMISSION_MODE):
                    return AuthResult.HUMAN_REVIEW
            if change_classification == ChangeClassification.MAJOR_MODEL_CHANGE:
                if mode == RuntimeMode.SUBMISSION_MODE:
                    return AuthResult.HUMAN_REVIEW
            if change_classification == ChangeClassification.FORMATTING_ONLY:
                return AuthResult.ALLOW

        return result

    # ── Deadline management ──────────────────────────────────

    def update_deadline(
        self, new_deadline: datetime, changed_by: str, reason: str,
    ) -> DeadlineChangeRecord:
        """Record an authorized deadline change. Resets monotonicity."""
        record = DeadlineChangeRecord(
            old_deadline=None,  # set by caller
            new_deadline=new_deadline,
            changed_by=changed_by,
            reason=reason,
            timestamp=self._clock(),
        )
        self._deadline_changes.append(record)
        # Reset monotonicity — authorized extension allows mode relaxation
        self._previous_mode = None
        return record

    @property
    def deadline_changes(self) -> list[DeadlineChangeRecord]:
        return list(self._deadline_changes)

    @property
    def current_mode(self) -> Optional[RuntimeMode]:
        return self._previous_mode
"""MathModel AI — Phase 7D Final Integrity: Model CAS + Approval Scoping.

Production-grade model version CAS, human approval scope binding,
rejected decision lifecycle, and critical constraint classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


# ═══════════════════════════════════════════════════════════════
# Constraint Importance
# ═══════════════════════════════════════════════════════════════

class ConstraintImportance(str, Enum):
    STRUCTURAL = "structural"
    CORE_FEASIBILITY = "core_feasibility"
    FAIRNESS = "fairness"
    BALANCE = "balance"
    CONSERVATION = "conservation"
    REGULATORY = "regulatory"
    AUXILIARY = "auxiliary"


CRITICAL_IMPORTANCE = {
    ConstraintImportance.STRUCTURAL,
    ConstraintImportance.CORE_FEASIBILITY,
    ConstraintImportance.FAIRNESS,
    ConstraintImportance.BALANCE,
    ConstraintImportance.CONSERVATION,
    ConstraintImportance.REGULATORY,
}


# ═══════════════════════════════════════════════════════════════
# Model Version CAS
# ═══════════════════════════════════════════════════════════════

class CASResult(str, Enum):
    SUCCESS = "success"
    VERSION_CONFLICT = "version_conflict"
    STALE_WRITE = "stale_write"


@dataclass
class ModelVersionState:
    """Mutable model version holder with CAS support."""
    model_version: int = 1
    mutation_history: list[dict] = field(default_factory=list)

    def try_increment(self, expected: int) -> CASResult:
        """Compare-and-swap version increment. Rejects stale writes."""
        if self.model_version != expected:
            return CASResult.VERSION_CONFLICT
        self.model_version += 1
        self.mutation_history.append({
            "from_version": expected,
            "to_version": self.model_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return CASResult.SUCCESS

    def check_stale(self, expected: int) -> bool:
        """Check if expected version is stale without mutating."""
        return self.model_version != expected


# ═══════════════════════════════════════════════════════════════
# Approval Scope Fingerprint
# ═══════════════════════════════════════════════════════════════

@dataclass
class ApprovalScopeFingerprint:
    model_version: int
    action: str
    effective_classification: str
    affected_ids: list[str] = field(default_factory=list)
    artifact_versions: dict[str, int] = field(default_factory=dict)
    scope_hash: str = ""

    def __post_init__(self):
        if not self.scope_hash:
            self.scope_hash = self._compute()

    def _compute(self) -> str:
        import hashlib
        parts = [
            str(self.model_version),
            self.action,
            self.effective_classification,
            ",".join(sorted(self.affected_ids)),
            ",".join(f"{k}:{v}" for k, v in sorted(self.artifact_versions.items())),
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]

    def matches(self, other: ApprovalScopeFingerprint) -> bool:
        return self.scope_hash == other.scope_hash


# ═══════════════════════════════════════════════════════════════
# Human Approval State
# ═══════════════════════════════════════════════════════════════

class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    USED = "used"
    STALE = "stale"


@dataclass
class ApprovalRecord:
    request_id: str = field(default_factory=lambda: f"HR-{uuid4().hex[:8]}")
    model_version: int = 0
    action: str = ""
    effective_classification: str = ""
    affected_ids: list[str] = field(default_factory=list)
    artifact_versions: dict[str, int] = field(default_factory=dict)
    scope_fingerprint: Optional[ApprovalScopeFingerprint] = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    decision_id: str = ""
    decided_by: str = ""
    reason: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    decided_at: Optional[datetime] = None

    def approve(self, decided_by: str, reason: str) -> ApprovalRecord:
        self.status = ApprovalStatus.APPROVED
        self.decision_id = f"HD-{uuid4().hex[:8]}"
        self.decided_by = decided_by
        self.reason = reason
        self.decided_at = datetime.now(timezone.utc)
        if not self.scope_fingerprint:
            self.scope_fingerprint = ApprovalScopeFingerprint(
                model_version=self.model_version,
                action=self.action,
                effective_classification=self.effective_classification,
                affected_ids=self.affected_ids,
                artifact_versions=self.artifact_versions,
            )
        return self

    def reject(self, decided_by: str, reason: str) -> ApprovalRecord:
        self.status = ApprovalStatus.REJECTED
        self.decision_id = f"HD-{uuid4().hex[:8]}"
        self.decided_by = decided_by
        self.reason = reason
        self.decided_at = datetime.now(timezone.utc)
        return self

    def mark_used(self) -> None:
        if self.status == ApprovalStatus.APPROVED:
            self.status = ApprovalStatus.USED

    def mark_stale(self) -> None:
        if self.status in (ApprovalStatus.APPROVED, ApprovalStatus.PENDING):
            self.status = ApprovalStatus.STALE

    def validate_for_execution(
        self,
        current_model_version: int,
        action: str,
        effective_classification: str,
        proposed_scope: ApprovalScopeFingerprint,
    ) -> tuple[bool, str]:
        """Validate an approval for a proposed execution. Returns (valid, reason)."""
        if self.status == ApprovalStatus.REJECTED:
            return False, "APPROVAL_REJECTED"
        if self.status == ApprovalStatus.USED:
            return False, "APPROVAL_ALREADY_USED"
        if self.status == ApprovalStatus.STALE:
            return False, "APPROVAL_STALE"
        if self.status != ApprovalStatus.APPROVED:
            return False, "APPROVAL_NOT_APPROVED"

        if self.model_version != current_model_version:
            self.mark_stale()
            return False, "STALE_APPROVAL_MODEL_VERSION"

        if self.action != action:
            return False, "ACTION_MISMATCH"

        if self.effective_classification != effective_classification:
            return False, "CLASSIFICATION_ESCALATION"

        if self.scope_fingerprint and not self.scope_fingerprint.matches(proposed_scope):
            return False, "SCOPE_MISMATCH"

        return True, "VALID"


# ═══════════════════════════════════════════════════════════════
# Critical Constraint Checker
# ═══════════════════════════════════════════════════════════════

def is_critical_constraint_removal(
    before: list[dict],
    after: list[dict],
    constraint_importances: Optional[dict[str, ConstraintImportance]] = None,
) -> bool:
    """Check if a removed constraint is critical (capacity/fairness/conservation)."""
    before_ids = set()
    for c in before:
        if isinstance(c, dict):
            cid = c.get("constraint_id", c.get("id", ""))
            before_ids.add(cid)
    after_ids = set()
    for c in after:
        if isinstance(c, dict):
            cid = c.get("constraint_id", c.get("id", ""))
            after_ids.add(cid)

    removed = before_ids - after_ids
    if not removed:
        return False

    if not constraint_importances:
        return False

    for cid in removed:
        if cid in constraint_importances:
            if constraint_importances[cid] in CRITICAL_IMPORTANCE:
                return True
    return False
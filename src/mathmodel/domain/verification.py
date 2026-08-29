"""MathModel AI — Phase 5 domain schemas.

ValidationReport, SensitivityExperiment/Report, RobustnessExperiment/Report,
RedTeamIssue/Report, RepairPlan, ModelRevision, VerificationStatus,
CodeMapping, ValidationGateResult.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    VERIFIED_WITH_WARNINGS = "verified_with_warnings"
    REPAIR_REQUIRED = "repair_required"
    HUMAN_REVIEW = "human_review"
    FAILED = "failed"


class GateStatus(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


class IssueSeverity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


# ═══════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════

class ValidationIssue(BaseModel):
    """A single validation issue."""
    issue_id: str = Field(default_factory=lambda: f"VAL-{uuid4().hex[:8]}")
    severity: IssueSeverity = IssueSeverity.MAJOR
    category: str = ""
    message: str = Field(..., min_length=1)
    evidence: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    """Complete validation report for a model + solver result."""

    validation_id: str = Field(default_factory=lambda: f"VR-{uuid4().hex[:8]}")
    model_id: str = ""
    model_version: Optional[int] = None
    solver_run_id: str = ""
    overall_status: GateStatus = GateStatus.FAIL

    # Sub-validations
    mathematical_validation: dict[str, Any] = Field(default_factory=dict)
    data_validation: dict[str, Any] = Field(default_factory=dict)
    solver_validation: dict[str, Any] = Field(default_factory=dict)
    constraint_validation: dict[str, Any] = Field(default_factory=dict)
    result_validation: dict[str, Any] = Field(default_factory=dict)
    extreme_case_validation: dict[str, Any] = Field(default_factory=dict)
    evidence_validation: dict[str, Any] = Field(default_factory=dict)

    issues: list[ValidationIssue] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ═══════════════════════════════════════════════════════════════
# Sensitivity
# ═══════════════════════════════════════════════════════════════

class SensitivityRun(BaseModel):
    """A single perturbation run."""
    parameter_id: str
    baseline: float
    perturbed: float
    delta_pct: float
    objective_value: Optional[float] = None
    variable_values: dict[str, float] = Field(default_factory=dict)
    solver_status: str = ""
    execution_real: bool = False
    effective: bool = True  # False if perturbation did not change the compiled model


class SensitivityExperiment(BaseModel):
    """Sensitivity analysis for one parameter."""
    experiment_id: str = Field(default_factory=lambda: f"SE-{uuid4().hex[:8]}")
    parameter_id: str
    baseline: float
    perturbations: list[float] = Field(default_factory=list)
    metric: str = "objective_value"
    runs: list[SensitivityRun] = Field(default_factory=list)
    elasticity: Optional[float] = None
    interpretation: str = ""
    status: GateStatus = GateStatus.PASS


class SensitivityReport(BaseModel):
    """Complete sensitivity report."""
    report_id: str = Field(default_factory=lambda: f"SR-{uuid4().hex[:8]}")
    model_id: str = ""
    model_version: Optional[int] = None
    experiments: list[SensitivityExperiment] = Field(default_factory=list)
    most_sensitive_parameters: list[str] = Field(default_factory=list)
    least_sensitive_parameters: list[str] = Field(default_factory=list)
    potential_instability: list[str] = Field(default_factory=list)
    summary: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ═══════════════════════════════════════════════════════════════
# Robustness
# ═══════════════════════════════════════════════════════════════

class RobustnessMethod(str, Enum):
    MONTE_CARLO = "monte_carlo"
    BOOTSTRAP = "bootstrap"
    SCENARIO_ANALYSIS = "scenario_analysis"
    NOISE_PERTURBATION = "noise_perturbation"
    WORST_CASE = "worst_case"
    CONSTRAINT_STRESS = "constraint_stress"


class RobustnessExperiment(BaseModel):
    """A single robustness experiment."""
    experiment_id: str = Field(default_factory=lambda: f"RE-{uuid4().hex[:8]}")
    method: RobustnessMethod = RobustnessMethod.SCENARIO_ANALYSIS
    uncertain_inputs: list[str] = Field(default_factory=list)
    scenarios: list[dict[str, Any]] = Field(default_factory=list)
    sample_count: int = 0
    seed: int = 42
    metrics: dict[str, Any] = Field(default_factory=dict)
    results: list[dict[str, Any]] = Field(default_factory=list)
    failure_rate: float = 0.0
    confidence_interval: Optional[dict[str, float]] = None
    summary: str = ""


class RobustnessReport(BaseModel):
    """Complete robustness report."""
    report_id: str = Field(default_factory=lambda: f"RR-{uuid4().hex[:8]}")
    model_id: str = ""
    model_version: Optional[int] = None
    experiments: list[RobustnessExperiment] = Field(default_factory=list)
    summary: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ═══════════════════════════════════════════════════════════════
# Red Team
# ═══════════════════════════════════════════════════════════════

class RedTeamIssue(BaseModel):
    """A single red team finding."""
    issue_id: str = Field(default_factory=lambda: f"RT-{uuid4().hex[:8]}")
    severity: IssueSeverity = IssueSeverity.MAJOR
    category: str = ""
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    impact: str = ""
    suggested_fix: str = ""
    target_component: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class RedTeamReport(BaseModel):
    """Complete red team report."""
    report_id: str = Field(default_factory=lambda: f"RTR-{uuid4().hex[:8]}")
    model_id: str = ""
    issues: list[RedTeamIssue] = Field(default_factory=list)
    summary: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.CRITICAL)


# ═══════════════════════════════════════════════════════════════
# Model Repair
# ═══════════════════════════════════════════════════════════════

class RepairType(str, Enum):
    PARAMETER_FIX = "parameter_fix"
    CONSTRAINT_FIX = "constraint_fix"
    OBJECTIVE_FIX = "objective_fix"
    ASSUMPTION_FIX = "assumption_fix"
    EQUATION_FIX = "equation_fix"
    CODE_FIX = "code_fix"
    MODEL_STRUCTURE_FIX = "model_structure_fix"
    MODEL_SWITCH_REQUIRED = "model_switch_required"


class RepairAction(BaseModel):
    """A single structured repair operation with source provenance.

    Repairs that introduce new numeric values must cite a source_reference.
    """

    action_id: str = Field(default_factory=lambda: f"RA-{uuid4().hex[:8]}")
    kind: str = Field(..., description="parameter_set | constraint_add | constraint_update | objective_update")
    parameter_id: Optional[str] = None
    new_value: Optional[float] = None
    constraint_id: Optional[str] = None
    constraint_expression: Optional[str] = None
    constraint_relation: Optional[str] = None
    constraint_rhs: Optional[float] = None
    source_reference: str = Field(
        default="",
        description="Evidence/data reference justifying this action (required for new values)",
    )
    reason: str = ""


class RepairPlan(BaseModel):
    """A plan for repairing a model."""
    plan_id: str = Field(default_factory=lambda: f"RP-{uuid4().hex[:8]}")
    issue_ids: list[str] = Field(default_factory=list)
    root_causes: list[str] = Field(default_factory=list)
    proposed_changes: list[str] = Field(default_factory=list)
    repair_type: RepairType = RepairType.CONSTRAINT_FIX
    affected_variables: list[str] = Field(default_factory=list)
    affected_parameters: list[str] = Field(default_factory=list)
    affected_equations: list[str] = Field(default_factory=list)
    affected_constraints: list[str] = Field(default_factory=list)
    affected_code: list[str] = Field(default_factory=list)
    expected_effect: str = ""
    risk: str = "low"
    requires_model_switch: bool = False
    actions: list[RepairAction] = Field(
        default_factory=list,
        description="Structured repair operations; applied in order",
    )


class ModelRevision(BaseModel):
    """A revision of a model."""
    revision_id: str = Field(default_factory=lambda: f"REV-{uuid4().hex[:8]}")
    model_id: str
    version: int
    previous_model_id: Optional[str] = None
    repair_plan_id: Optional[str] = None
    redteam_issue_ids: list[str] = Field(default_factory=list)
    validation_failure_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ═══════════════════════════════════════════════════════════════
# Code Mapping
# ═══════════════════════════════════════════════════════════════

class CodeMapping(BaseModel):
    """Maps mathematical objects to generated code."""
    mapping_id: str = Field(default_factory=lambda: f"CM-{uuid4().hex[:8]}")
    model_id: str = ""
    equation_id: Optional[str] = None
    objective_id: Optional[str] = None
    constraint_id: Optional[str] = None
    file: str = ""
    function: str = ""
    generated_representation: str = ""
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    code_hash: str = ""


# ═══════════════════════════════════════════════════════════════
# Validation Gate Result
# ═══════════════════════════════════════════════════════════════

class ValidationCheckResult(BaseModel):
    """Result of a single validation check."""
    check_name: str
    status: GateStatus = GateStatus.PASS
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class ValidationGateResult(BaseModel):
    """Complete validation gate result."""
    gate_id: str = Field(default_factory=lambda: f"VG-{uuid4().hex[:8]}")
    status: GateStatus = GateStatus.PASS
    checks: list[ValidationCheckResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    blocking_failures: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    override_required: bool = False
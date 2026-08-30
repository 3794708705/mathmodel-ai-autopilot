"""MathModel AI — Phase 7A: Semantic RedTeam, ClaimSupportVerifier, SemanticPaperReviewer.

Real LLM-powered semantic verification. Respects the RealityContext:
when real LLM is unavailable, these agents cannot be used and must
honestly report that fact — never silently fall back to Mock.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from mathmodel.domain.math_model import MathematicalModel
from mathmodel.domain.verification import (
    ValidationReport, SensitivityReport, RobustnessReport,
    RedTeamIssue, RedTeamReport, IssueSeverity,
)
from mathmodel.evidence import Claim, EvidenceStore, EvidenceSourceType
from mathmodel.literature import LiteratureRecord, CitationStatus
from mathmodel.paper import PaperIR

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Semantic RedTeam
# ═══════════════════════════════════════════════════════════════

class SemanticIssueOut(BaseModel):
    """Structured semantic issue from the LLM."""
    severity: str = "MAJOR"  # CRITICAL | MAJOR | MINOR
    category: str = ""
    title: str = ""
    description: str = ""
    evidence: list[dict] = Field(default_factory=list)
    impact: str = ""
    suggested_fix: str = ""
    target_component: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SemanticReviewOutput(BaseModel):
    """Structured output of semantic red team review."""
    issues: list[SemanticIssueOut] = Field(default_factory=list)
    summary: str = ""


class SemanticRedTeamAgent:
    """Real LLM-powered semantic attack on the model.

    Goes beyond deterministic checks: examines problem interpretation,
    unjustified assumptions, missing constraints, objective mismatch,
    implausible modeling choices, unsupported conclusions.
    """

    name = "SemanticRedTeam"

    def __init__(self, router=None):
        self._router = router

    async def review(
        self,
        model: MathematicalModel,
        result: Any,
        validation: ValidationReport,
        analysis: Optional[Any] = None,
        sensitivity: Optional[SensitivityReport] = None,
        robustness: Optional[RobustnessReport] = None,
        evidence: Optional[EvidenceStore] = None,
        problem_context: Optional[str] = None,
    ) -> RedTeamReport:
        """Execute semantic review using real LLM.

        problem_context: additional requirements from the problem
        statement that the model must satisfy (e.g., fairness).
        """
        if self._router is None:
            return RedTeamReport(
                model_id=model.model_id,
                issues=[],
                summary="SemanticRedTeam: NO REAL LLM AVAILABLE — semantic review "
                        "not performed. Deterministic checks only.",
            )

        prompt = self._build_prompt(
            model, result, validation, sensitivity, robustness,
            problem_context=problem_context,
        )

        # Structured generation with real LLM
        from mathmodel.routing.profile import TaskProfile, TaskType

        try:
            profile = TaskProfile.for_task_type(TaskType.RED_TEAM)
            output = await self._router.route_structured_generate(
                profile=profile,
                prompt=prompt,
                output_schema=SemanticReviewOutput,
                system_prompt=(
                    "You are an adversarial mathematical modeling competition "
                    "reviewer. Attack the model honestly; do not fabricate issues."
                ),
            )

            if isinstance(output, SemanticReviewOutput):
                # Validate issue references
                valid_issues = []
                var_ids = {v.variable_id for v in model.variables}
                con_ids = {c.constraint_id for c in model.constraints}
                obj_ids = {o.objective_id for o in model.objectives}
                for issue in output.issues:
                    refs = issue.evidence
                    valid_refs = [r for r in refs if self._reference_valid(r, var_ids, con_ids, obj_ids)]
                    issue.evidence = valid_refs
                    if issue.severity in (IssueSeverity.CRITICAL.value, IssueSeverity.MAJOR.value, IssueSeverity.MINOR.value):
                        valid_issues.append(issue)

                # If the LLM identified flaws in the summary but did not
                # populate the structured issues list, do not lose the signal.
                if not valid_issues and output.summary:
                    summary_lower = output.summary.lower()
                    negative_markers = (
                        "flaw", "missing", "omits", "does not", "fails to",
                        "invalid", "incorrect", "must be revised", "critically",
                    )
                    if any(m in summary_lower for m in negative_markers):
                        valid_issues.append(SemanticIssueOut(
                            severity=IssueSeverity.MAJOR.value,
                            category="semantic",
                            title="Semantic issue identified in summary",
                            description=output.summary[:500],
                            evidence=[],
                            impact="Model may not satisfy problem requirements",
                            suggested_fix="Revise model per the identified issue",
                            target_component="model",
                            confidence=0.7,
                        ))

                return RedTeamReport(
                    model_id=model.model_id,
                    issues=[RedTeamIssue(
                        severity=IssueSeverity(i.severity),
                        category=i.category,
                        title=i.title,
                        description=i.description,
                        evidence=i.evidence,
                        impact=i.impact,
                        suggested_fix=i.suggested_fix,
                        target_component=i.target_component,
                        confidence=i.confidence,
                    ) for i in valid_issues],
                    summary=output.summary,
                )
            else:
                return RedTeamReport(
                    model_id=model.model_id,
                    issues=[],
                    summary=f"SemanticRedTeam: unexpected output type {type(output).__name__}",
                )
        except Exception as e:
            return RedTeamReport(
                model_id=model.model_id,
                issues=[],
                summary=f"SemanticRedTeam LLM call failed: {str(e)[:200]}",
            )

    @staticmethod
    def _reference_valid(ref, var_ids, con_ids, obj_ids) -> bool:
        """Check that an evidence reference points to an existing artifact."""
        if not isinstance(ref, dict):
            return True  # free-form evidence allowed
        for key in ("variable_id", "constraint_id", "objective_id", "equation_id"):
            if key in ref:
                value = ref[key]
                if key == "variable_id" and value not in var_ids:
                    return False
                if key == "constraint_id" and value not in con_ids:
                    return False
                if key == "objective_id" and value not in obj_ids:
                    return False
        return True

    def _build_prompt(
        self, model, result, validation, sensitivity=None, robustness=None,
        problem_context: Optional[str] = None,
    ) -> str:
        lines = [
            "You are an adversarial reviewer for a mathematical modeling competition.",
            "Attack the model from these angles:",
            "- Problem interpretation: does the model solve the right problem?",
            "- Unjustified assumptions: any assumptions without evidence?",
            "- Missing constraints: any constraints the problem requires but the model omits?",
            "- Missing variables: any variables the problem clearly needs but the model lacks?",
            "- Objective mismatch: does the objective reflect the problem's goals?",
            "- Implausible modeling choices: any approach that doesn't fit the problem?",
            "- Unsupported conclusions: any claims that go beyond the evidence?",
            "- Competition relevance: is this approach feasible in competition time?",
            "",
        ]
        if problem_context:
            lines.append("## PROBLEM REQUIREMENTS (from problem statement)")
            lines.append(problem_context)
            lines.append("")
        lines.append("## MODEL STRUCTURE")
        lines.append(f"Model: {model.name} (id={model.model_id})")
        lines.append("Variables:")
        for v in model.variables:
            lines.append(f"  - {v.variable_id} {v.symbol}: {v.name or v.meaning}")
        lines.append("Objectives:")
        for o in model.objectives:
            lines.append(f"  - {o.objective_id}: {o.expression} ({o.sense.value})")
        lines.append("Constraints:")
        for c in model.constraints:
            lines.append(f"  - {c.constraint_id}: {c.expression} {c.relation.value} {c.rhs}")
        if result is not None:
            lines.append(f"Solver status: {getattr(result, 'status', 'unknown')}")
            lines.append(f"Objective value: {getattr(result, 'objective_value', None)}")
        lines.append("")
        lines.append(
            "For each issue, provide: severity (CRITICAL/MAJOR/MINOR), category, "
            "title, description, evidence (reference existing constraint_id/"
            "variable_id/objective_id where applicable), impact, suggested_fix, "
            "target_component, confidence."
        )
        lines.append(
            "IMPORTANT: If the problem requires goals the model does not cover "
            "(e.g. fairness when only profit is optimized), flag it as an "
            "objective mismatch / missing constraint issue."
        )
        lines.append("If the model is genuinely sound, return an empty issues list — do NOT fabricate issues.")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Claim Support Verifier
# ═══════════════════════════════════════════════════════════════

class ClaimSupportStatus(str):
    SUPPORTS = "SUPPORTS"
    PARTIALLY_SUPPORTS = "PARTIALLY_SUPPORTS"
    DOES_NOT_SUPPORT = "DOES_NOT_SUPPORT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICT = "CONFLICT"


class ClaimSupportResult(BaseModel):
    """Result of semantic claim-support verification."""
    claim_id: str
    literature_id: str
    support_status: str = ClaimSupportStatus.INSUFFICIENT_EVIDENCE
    reason: str = ""
    evidence_reference: str = ""  # What part of the literature supports/contradicts
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    provider: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ClaimSupportOutput(BaseModel):
    """Structured claim-support judgment from the LLM."""
    support_status: str = "INSUFFICIENT_EVIDENCE"
    reason: str = ""
    evidence_reference: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ClaimSupportVerifier:
    """Verifies whether a literature record supports a claim.

    NEVER relies on LLM memory — requires actual retrieved evidence
    (abstract, metadata, excerpt). Without retrieved evidence, the
    best answer is INSUFFICIENT_EVIDENCE.
    """

    name = "ClaimSupportVerifier"

    def __init__(self, router=None):
        self._router = router

    async def verify(
        self,
        claim: Claim,
        record: LiteratureRecord,
    ) -> ClaimSupportResult:
        """Verify semantic support of a claim by a literature record.

        Requires actual retrieved evidence (abstract, metadata).
        Without a real LLM, reports INSUFFICIENT_EVIDENCE honestly.
        """
        if self._router is None:
            return ClaimSupportResult(
                claim_id=claim.claim_id,
                literature_id=record.literature_id,
                support_status=ClaimSupportStatus.INSUFFICIENT_EVIDENCE,
                reason="No real LLM available — semantic claim-support verification not performed.",
                provider="none",
            )

        # Deterministic enforcement: no retrieved evidence → INSUFFICIENT
        evidence_text = (record.abstract or "").strip()
        if not evidence_text:
            return ClaimSupportResult(
                claim_id=claim.claim_id,
                literature_id=record.literature_id,
                support_status=ClaimSupportStatus.INSUFFICIENT_EVIDENCE,
                reason="No retrieved abstract/evidence available — cannot verify claim support.",
                provider="deepseek",
            )

        # Build prompt with actual retrieved evidence and call the real LLM
        from mathmodel.routing.profile import TaskProfile, TaskType

        prompt = self._build_prompt(claim, record)
        try:
            profile = TaskProfile.for_task_type(TaskType.CITATION_VERIFICATION)
            output = await self._router.route_structured_generate(
                profile=profile,
                prompt=prompt,
                output_schema=ClaimSupportOutput,
                system_prompt=(
                    "You judge whether a literature record's retrieved abstract "
                    "supports a claim. Base your judgment ONLY on the provided "
                    "abstract/metadata, never on training-data memory."
                ),
            )
            if isinstance(output, ClaimSupportOutput):
                status = output.support_status
                if status not in (v.value for v in ClaimSupportStatus):
                    status = ClaimSupportStatus.INSUFFICIENT_EVIDENCE
                return ClaimSupportResult(
                    claim_id=claim.claim_id,
                    literature_id=record.literature_id,
                    support_status=status,
                    reason=output.reason,
                    evidence_reference=output.evidence_reference,
                    confidence=output.confidence,
                    provider="deepseek",
                )
            return ClaimSupportResult(
                claim_id=claim.claim_id,
                literature_id=record.literature_id,
                support_status=ClaimSupportStatus.INSUFFICIENT_EVIDENCE,
                reason=f"Unexpected LLM output type: {type(output).__name__}",
                provider="deepseek",
            )
        except Exception as e:
            return ClaimSupportResult(
                claim_id=claim.claim_id,
                literature_id=record.literature_id,
                support_status=ClaimSupportStatus.INSUFFICIENT_EVIDENCE,
                reason=f"LLM verification failed: {str(e)[:200]}",
                provider="deepseek",
            )

    def _build_prompt(self, claim: Claim, record: LiteratureRecord) -> str:
        return "\n".join([
            "Determine whether this literature record supports the claim.",
            f"## CLAIM: {claim.text}",
            f"## LITERATURE:",
            f"Title: {record.title}",
            f"Authors: {', '.join(record.authors)}",
            f"Year: {record.year}",
            f"Venue: {record.venue}",
            f"Abstract: {record.abstract}",
            "",
            "Respond with:",
            "- support_status: SUPPORTS | PARTIALLY_SUPPORTS | DOES_NOT_SUPPORT | INSUFFICIENT_EVIDENCE | CONFLICT",
            "- reason: one-sentence explanation",
            "- evidence_reference: what part of the abstract/metadata supports your judgment",
            "- confidence: 0.0-1.0",
            "",
            "CRITICAL: You must base your judgment ONLY on the provided abstract/metadata.",
            "Do NOT rely on your knowledge of this paper from training data.",
            "If the abstract is empty or insufficient, respond INSUFFICIENT_EVIDENCE.",
        ])


# ═══════════════════════════════════════════════════════════════
# Semantic Paper Reviewer
# ═══════════════════════════════════════════════════════════════

class SemanticPaperReviewer:
    """Real LLM-powered semantic paper review.

    Checks for overclaim, assumption-as-fact drift, conclusion stronger
    than evidence, method/result mismatch, validation overstatement,
    citation misuse.

    NEVER overrides deterministic gates (Numerical FAIL, Equation FAIL,
    Citation identity FAIL, Submission BLOCKED).
    """

    name = "SemanticPaperReviewer"

    def __init__(self, router=None):
        self._router = router

    async def review(self, paper: PaperIR, evidence: EvidenceStore) -> list[str]:
        """Review paper for semantic issues.

        Returns list of issue descriptions (empty = clean).
        """
        if self._router is None:
            return ["SemanticPaperReviewer: NO REAL LLM AVAILABLE — semantic review not performed."]

        # Collect paper text for review
        sections_text = []
        for sec in paper.sections:
            for block in sec.content_blocks:
                if block.text:
                    sections_text.append(f"[{sec.section_id}] {block.text[:500]}")

        prompt = "\n".join([
            "Review this paper for semantic issues. Do NOT flag deterministic "
            "formatting issues (those are handled separately). Focus on:",
            "- Overclaim: any claim stronger than the evidence supports",
            "- Assumption presented as fact",
            "- Conclusion stronger than evidence",
            "- Method/result mismatch",
            "- Validation overstatement",
            "- Citation misuse",
            "",
            "## PAPER CONTENT",
            *sections_text[:20],
            "",
            "If the paper is clean, say so. Do NOT fabricate issues.",
            "Respond with a JSON object: {issues: [string descriptions], clean: bool}",
        ])

        from mathmodel.routing.profile import TaskProfile, TaskType

        class ReviewOutput(BaseModel):
            issues: list[str] = Field(default_factory=list)
            clean: bool = True

        try:
            profile = TaskProfile.for_task_type(TaskType.PAPER_GENERATION)
            output = await self._router.route_structured_generate(
                profile=profile,
                prompt=prompt,
                output_schema=ReviewOutput,
                system_prompt=(
                    "You are a semantic paper reviewer. Flag overclaims and "
                    "unsupported conclusions honestly; do not fabricate issues."
                ),
            )
            if isinstance(output, ReviewOutput):
                return list(output.issues)
            return [f"Unexpected reviewer output type: {type(output).__name__}"]
        except Exception as e:
            return [f"Semantic review LLM call failed: {str(e)[:200]}"]
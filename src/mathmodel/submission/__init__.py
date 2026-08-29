"""MathModel AI — Phase 6F: Competition profile, submission check, final jury.

Deterministic checks gate the paper. FinalJury cannot override hard failures.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from mathmodel.paper import PaperIR
from mathmodel.evidence import Claim, ClaimGate, EvidenceStore
from mathmodel.documents import FigureRegistry, TableRegistry, verify_figure, verify_table
from mathmodel.literature import CitationVerifier


class CompetitionProfile(BaseModel):
    """Configurable competition rules — never hardcoded."""
    competition_name: str = "generic"
    template: str = "default"
    page_limit: Optional[int] = None
    font_rules: dict[str, Any] = Field(default_factory=dict)
    anonymity_rules: dict[str, Any] = Field(default_factory=dict)
    required_sections: list[str] = Field(default_factory=list)
    forbidden_content: list[str] = Field(default_factory=list)
    reference_style: str = "bibtex"
    appendix_rules: dict[str, Any] = Field(default_factory=dict)
    submission_format: dict[str, Any] = Field(default_factory=dict)


class SubmissionStatus(str, Enum):
    READY_TO_SUBMIT = "ready_to_submit"
    READY_WITH_WARNINGS = "ready_with_warnings"
    REPAIR_REQUIRED = "repair_required"
    BLOCKED = "blocked"


class SubmissionCheckResult(BaseModel):
    check_id: str = Field(default_factory=lambda: f"SUB-{uuid4().hex[:8]}")
    status: SubmissionStatus = SubmissionStatus.BLOCKED
    checks: list[dict[str, Any]] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UnsupportedClaimChecker:
    """Scans PaperIR for unsupported claims (percentages, values, etc.)."""

    PATTERNS = [
        r"\d+\.?\d*\s*%",           # percentages
        r"improve[dment]* by",       # improvement claims
        r"significan\w+",            # significance
        r"robust\w+",                # robustness
        r"accuracy of",              # accuracy
        r"confidence of",            # confidence
    ]

    def check(self, paper: PaperIR, evidence: EvidenceStore) -> list[str]:
        """Return blocking issues for unsupported claim-like text."""
        issues = []
        all_claims = {c.claim_id: c for c in evidence.list_claims()}
        paper_claim_ids = set(paper.collect_claims())

        for section in paper.sections:
            for block in section.content_blocks:
                if block.block_type.value not in ("paragraph", "note"):
                    continue
                text = block.text
                for pattern in self.PATTERNS:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        snippet = text[max(0, match.start() - 40):match.end() + 40]
                        # If this block has no claim_ids at all, the claim-like text
                        # is unsupported
                        if not block.claim_ids:
                            issues.append(
                                f"[{section.section_id}] claim-like text without "
                                f"claim reference: \"...{snippet.strip()}...\""
                            )
                            break  # one issue per pattern per block

        # Verify all referenced claims exist
        for claim_id in paper_claim_ids:
            if claim_id not in all_claims:
                issues.append(f"Paper references unknown claim: {claim_id}")

        return issues


class NumericalConsistencyChecker:
    """Verifies paper numbers match source numbers.

    Policy:
    - Formatting: 700.0 == 700 == 700.00 (absolute match)
    - Rounding: paper 0.333 matches source 0.333333333 (tolerance derived
      from the paper's decimal precision — NOT arbitrary closeness)
    - Percentage trap: source 0.35 matches paper "35%" (rate semantics);
      source 35 does NOT match paper "35%" (no implicit /100 for values > 1)
    """

    @staticmethod
    def check_consistency(source_value: float, paper_text: str) -> bool:
        """Check whether a source value appears in the paper text."""
        for match in re.finditer(r"\d+\.?\d*\s*%?", paper_text):
            token = match.group()
            is_percent = token.endswith("%")
            num_part = token.rstrip("%")
            try:
                paper_val = float(num_part)
            except ValueError:
                continue

            if is_percent:
                # Percentage semantics: paper value = paper_val / 100
                # Only valid when the source is itself a rate in [0, 1]
                if 0.0 <= source_value <= 1.0:
                    paper_rate = paper_val / 100.0
                    if abs(paper_rate - source_value) < 1e-9:
                        return True
                # Source > 1 does NOT auto-match a percentage
                continue

            # Plain number: absolute formatting match
            if abs(paper_val - source_value) < 1e-9:
                return True

            # Rounding tolerance based on paper decimal precision
            decimals = len(num_part.split(".")[1]) if "." in num_part else 0
            tol = 0.5 * (10 ** (-decimals)) + 1e-9
            if abs(paper_val - source_value) <= tol:
                return True

        return False

    def check_all(
        self,
        paper: PaperIR,
        source_values: dict[str, float],
    ) -> list[str]:
        """Check all source values appear correctly in the paper.

        source_values: {description: value}
        """
        issues = []
        paper_text_parts = []
        for section in paper.sections:
            for block in section.content_blocks:
                paper_text_parts.append(block.text)
        paper_text = " ".join(paper_text_parts)

        for desc, value in source_values.items():
            if not self.check_consistency(value, paper_text):
                issues.append(
                    f"Paper does not contain source value {value} ({desc})"
                )
        return issues

    @staticmethod
    def check_improvement(
        baseline: float,
        improved: float,
        paper_text: str,
    ) -> list[str]:
        """Verify a paper's improvement claim by deterministic calculation.

        baseline=100, improved=120 → true improvement = 20%.
        Paper claiming "120% improvement" must FAIL.
        """
        issues = []
        if baseline == 0:
            return ["Improvement baseline is zero — cannot compute percentage"]

        actual_pct = (improved - baseline) / baseline * 100.0

        # Find percentage claims in the text
        for match in re.finditer(r"(\d+\.?\d*)\s*%", paper_text):
            claimed = float(match.group(1))
            if abs(claimed - actual_pct) <= 0.05 + 1e-9:
                # Paper reports the correct improvement
                return []

        issues.append(
            f"Paper improvement claim does not match deterministic calculation: "
            f"expected {actual_pct:.2f}% (from {baseline}→{improved})"
        )
        return issues


class SubmissionCheckAgent:
    """Runs deterministic submission checks."""

    def __init__(
        self,
        profile: CompetitionProfile,
        evidence: Optional[EvidenceStore] = None,
        figures: Optional[FigureRegistry] = None,
        tables: Optional[TableRegistry] = None,
        citation_verifier: Optional[CitationVerifier] = None,
        current_model_version: Optional[int] = None,
        source_values: Optional[dict[str, Any]] = None,
    ):
        self._profile = profile
        self._evidence = evidence or EvidenceStore()
        self._figures = figures or FigureRegistry()
        self._tables = tables or TableRegistry()
        self._citation_verifier = citation_verifier
        self._current_model_version = current_model_version
        self._source_values = source_values or {}

    def check(self, paper: PaperIR) -> SubmissionCheckResult:
        failures: list[str] = []
        warnings: list[str] = []
        checks: list[dict[str, Any]] = []

        # 1. Abstract present
        if not paper.abstract:
            failures.append("Abstract missing")
        checks.append({"check": "abstract", "ok": bool(paper.abstract)})

        # 2. Required sections
        section_titles = {s.title.lower() for s in paper.sections}
        for required in self._profile.required_sections:
            if required.lower() not in section_titles:
                failures.append(f"Required section missing: {required}")
        checks.append({"check": "required_sections", "ok": not failures})

        # 3. Claim gate
        claim_issues = ClaimGate.check(self._evidence.list_claims())
        if claim_issues:
            failures.extend(claim_issues)
        checks.append({"check": "claim_gate", "ok": not claim_issues})

        # 3b. Claim evidence-type matching
        type_mismatch_claims = self._evidence.claims_with_invalid_evidence_type()
        for claim in type_mismatch_claims:
            failures.append(
                f"Claim {claim.claim_id} ({claim.claim_type.value}) lacks "
                f"type-appropriate evidence"
            )
        checks.append({
            "check": "claim_evidence_types",
            "ok": not type_mismatch_claims,
        })

        # 3c. Stale model detection
        paper_model_version = paper.metadata.get("model_version")
        if self._current_model_version is not None and paper_model_version is not None:
            if int(paper_model_version) < int(self._current_model_version):
                failures.append(
                    f"Paper built from model v{paper_model_version} but current "
                    f"model is v{self._current_model_version} — stale paper"
                )
        checks.append({
            "check": "stale_model",
            "ok": not any("stale paper" in f for f in failures),
        })

        # 3d. Anonymity rules
        if self._profile.anonymity_rules:
            forbidden_terms = self._profile.anonymity_rules.get("forbidden_terms", [])
            if forbidden_terms:
                paper_text = paper.abstract + " " + " ".join(
                    b.text for s in paper.sections for b in s.content_blocks
                )
                lower_text = paper_text.lower()
                for term in forbidden_terms:
                    if term.lower() in lower_text:
                        failures.append(
                            f"Anonymity violation: paper contains '{term}'"
                        )
        checks.append({
            "check": "anonymity",
            "ok": not any("Anonymity violation" in f for f in failures),
        })

        # 4. Unsupported claim scan
        unsupported = UnsupportedClaimChecker().check(paper, self._evidence)
        if unsupported:
            failures.extend(unsupported)
        checks.append({"check": "unsupported_claims", "ok": not unsupported})

        # 5. Figure integrity
        for fig_id in paper.collect_figures():
            fig = self._figures.get(fig_id)
            if fig is None:
                failures.append(f"Paper references unknown figure: {fig_id}")
                continue
            issues = verify_figure(fig)
            if issues:
                failures.extend(issues)
        checks.append({"check": "figures", "ok": True})

        # 6. Table integrity
        for tab_id in paper.collect_tables():
            tab = self._tables.get(tab_id)
            if tab is None:
                failures.append(f"Paper references unknown table: {tab_id}")
                continue
            issues = verify_table(tab, source_values=self._source_values)
            if issues:
                failures.extend(issues)
        checks.append({"check": "tables", "ok": True})

        # 7. Citation verification
        if self._citation_verifier:
            citation_issues = self._citation_verifier.verify_all()
            if citation_issues:
                failures.extend(citation_issues)
            checks.append({"check": "citations", "ok": not citation_issues})

        # 8. References: every cited REF must exist in bibliography
        cited = set(paper.collect_citations())
        bib = set(paper.references)
        missing = cited - bib
        if missing:
            failures.append(f"Cited but not in bibliography: {sorted(missing)}")
        checks.append({"check": "bibliography_consistency", "ok": not missing})

        # Status
        if failures:
            status = SubmissionStatus.BLOCKED
        elif warnings:
            status = SubmissionStatus.READY_WITH_WARNINGS
        else:
            status = SubmissionStatus.READY_TO_SUBMIT

        return SubmissionCheckResult(
            status=status,
            checks=checks,
            failures=failures,
            warnings=warnings,
        )


class FinalJuryAgent:
    """Simulates competition jury review. Cannot override deterministic failures."""

    def review(
        self,
        paper: PaperIR,
        submission: SubmissionCheckResult,
    ) -> dict[str, Any]:
        """Review the paper. Deterministic failures take priority."""
        dimension_scores: dict[str, float] = {}
        critical_issues = list(submission.failures)

        # If submission is blocked, the jury cannot give SUBMIT
        if submission.status == SubmissionStatus.BLOCKED:
            return {
                "overall_score": 0.0,
                "dimension_scores": dimension_scores,
                "strengths": [],
                "weaknesses": ["Deterministic submission failures present"],
                "critical_issues": critical_issues,
                "major_issues": [],
                "recommended_fixes": list(submission.failures),
                "submission_recommendation": "DO_NOT_SUBMIT",
            }

        # Basic heuristics (deterministic)
        if paper.abstract:
            dimension_scores["clarity"] = 70.0
        if paper.sections:
            dimension_scores["problem_coverage"] = 70.0

        return {
            "overall_score": sum(dimension_scores.values()) / max(len(dimension_scores), 1),
            "dimension_scores": dimension_scores,
            "strengths": ["Structured paper with verified evidence"],
            "weaknesses": [],
            "critical_issues": [],
            "major_issues": [],
            "recommended_fixes": [],
            "submission_recommendation": "SUBMIT" if submission.status == SubmissionStatus.READY_TO_SUBMIT else "REVIEW",
        }
"""Phase 6 audit adversarial tests.

Covers: DOI normalization, evidence-type matching, evidence deletion
invalidation, LaTeX injection, figure hash mismatch, table tampering,
percentage trap, rounding policy, improvement calculation, anonymity,
stale model detection, jury override protection.
"""

import os
import hashlib
import tempfile

import pytest

from mathmodel.evidence import (
    EvidenceStore, EvidenceRef, EvidenceSourceType, Claim, ClaimType,
    ClaimStatus, ClaimGate,
)
from mathmodel.literature import (
    LiteratureStore, LiteratureRecord, Citation, CitationVerifier,
    CitationStatus, normalize_doi,
)
from mathmodel.documents import (
    FigureRecord, FigureRegistry, verify_figure,
    TableRecord, TableRegistry, TableCell, verify_table,
)
from mathmodel.paper import PaperIR, PaperSection, ContentBlock, BlockType
from mathmodel.paper.renderer import LaTeXRenderer, LaTeXInjectionError
from mathmodel.submission import (
    CompetitionProfile, SubmissionCheckAgent, SubmissionStatus,
    NumericalConsistencyChecker, FinalJuryAgent,
)


# ═══════════════════════════════════════════════════════════════
# DOI Normalization
# ═══════════════════════════════════════════════════════════════

class TestDOINormalization:
    def test_normalize_bare(self):
        assert normalize_doi("10.xxxx/ABC") == "10.xxxx/abc"

    def test_normalize_https(self):
        assert normalize_doi("https://doi.org/10.xxxx/abc") == "10.xxxx/abc"

    def test_normalize_doi_prefix(self):
        assert normalize_doi("doi:10.xxxx/abc") == "10.xxxx/abc"

    def test_duplicate_detected_across_forms(self):
        store = LiteratureStore()
        store.add_record(LiteratureRecord(literature_id="L1", title="A", doi="10.xxxx/ABC"))
        with pytest.raises(ValueError, match="Duplicate DOI"):
            store.add_record(LiteratureRecord(
                literature_id="L2", title="B", doi="https://doi.org/10.xxxx/abc",
            ))


# ═══════════════════════════════════════════════════════════════
# Evidence Type Matching
# ═══════════════════════════════════════════════════════════════

class TestEvidenceTypeMatching:
    def test_numerical_claim_with_fact_rejected(self):
        """Numerical claim backed only by PROBLEM_FACT must not pass."""
        store = EvidenceStore()
        store.register_evidence(EvidenceRef(
            evidence_id="E1", source_type=EvidenceSourceType.PROBLEM_FACT, source_id="F1",
        ))
        claim = Claim(
            claim_id="C1", text="Optimal objective = 700",
            claim_type=ClaimType.NUMERICAL, importance="high",
            evidence_ids=["E1"], verification_status=ClaimStatus.SUPPORTED,
        )
        store.register_claim(claim)
        assert not store.check_evidence_type_match(claim)
        assert claim.claim_id in [c.claim_id for c in store.claims_with_invalid_evidence_type()]

    def test_numerical_claim_with_solver_result_accepted(self):
        store = EvidenceStore()
        store.register_evidence(EvidenceRef(
            evidence_id="E1", source_type=EvidenceSourceType.SOLVER_RESULT, source_id="S1",
        ))
        claim = Claim(
            claim_id="C1", text="Optimal objective = 700",
            claim_type=ClaimType.NUMERICAL, importance="high",
            evidence_ids=["E1"], verification_status=ClaimStatus.SUPPORTED,
        )
        store.register_claim(claim)
        assert store.check_evidence_type_match(claim)

    def test_submission_rejects_type_mismatch(self):
        store = EvidenceStore()
        store.register_evidence(EvidenceRef(
            evidence_id="E1", source_type=EvidenceSourceType.PROBLEM_FACT, source_id="F1",
        ))
        store.register_claim(Claim(
            claim_id="C1", text="objective=700", claim_type=ClaimType.NUMERICAL,
            importance="high", evidence_ids=["E1"],
            verification_status=ClaimStatus.SUPPORTED,
        ))
        profile = CompetitionProfile()
        paper = PaperIR(title="T", abstract="A")
        agent = SubmissionCheckAgent(profile, evidence=store)
        result = agent.check(paper)
        assert result.status == SubmissionStatus.BLOCKED
        assert any("type-appropriate" in f for f in result.failures)


# ═══════════════════════════════════════════════════════════════
# Evidence Deletion Invalidation
# ═══════════════════════════════════════════════════════════════

class TestEvidenceDeletion:
    def test_delete_invalidates_claims(self):
        store = EvidenceStore()
        store.register_evidence(EvidenceRef(
            evidence_id="E1", source_type=EvidenceSourceType.SOLVER_RESULT, source_id="S1",
        ))
        store.register_claim(Claim(
            claim_id="C1", text="x=700", claim_type=ClaimType.NUMERICAL,
            importance="high", evidence_ids=["E1"],
            verification_status=ClaimStatus.SUPPORTED,
        ))
        store.delete_evidence("E1")
        assert store.get_claim("C1").verification_status == ClaimStatus.UNVERIFIED

    def test_delete_unknown_raises(self):
        store = EvidenceStore()
        with pytest.raises(KeyError):
            store.delete_evidence("E-GHOST")


# ═══════════════════════════════════════════════════════════════
# LaTeX Injection
# ═══════════════════════════════════════════════════════════════

class TestLaTeXInjection:
    def _paper_with(self, block_text, block_type=BlockType.EQUATION):
        return PaperIR(title="T", sections=[
            PaperSection(section_id="S", title="X", content_blocks=[
                ContentBlock(block_type=block_type, text=block_text),
            ]),
        ])

    def test_block_input(self):
        with pytest.raises(LaTeXInjectionError):
            LaTeXRenderer().render(self._paper_with(r"\input{/etc/passwd}"))

    def test_block_write18(self):
        with pytest.raises(LaTeXInjectionError):
            LaTeXRenderer().render(self._paper_with(r"\write18{rm -rf /}"))

    def test_block_include(self):
        with pytest.raises(LaTeXInjectionError):
            LaTeXRenderer().render(self._paper_with(r"\include{secret}"))

    def test_normal_equation_ok(self):
        tex = LaTeXRenderer().render(self._paper_with(r"x^2 + y^2 = 1"))
        assert r"\begin{equation}" in tex

    def test_escaped_text_no_injection(self):
        tex = LaTeXRenderer().render(self._paper_with(
            r"literal \input text", block_type=BlockType.PARAGRAPH,
        ))
        # Escaped → no raw \input remains
        assert r"\input" not in tex.replace(r"\textbackslash{}input", "")


# ═══════════════════════════════════════════════════════════════
# Figure Hash / Table Tampering
# ═══════════════════════════════════════════════════════════════

class TestArtifactTampering:
    def test_figure_hash_mismatch(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"original")
            path = f.name
        fig = FigureRecord(
            figure_id="F1", title="x", artifact_path=path,
            hash="deadbeef", source_execution_ids=["E1"],
        )
        issues = verify_figure(fig)
        assert any("hash mismatch" in i for i in issues)
        os.unlink(path)

    def test_figure_hash_match(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"content")
            path = f.name
        real_hash = hashlib.sha256(b"content").hexdigest()[:16]
        fig = FigureRecord(
            figure_id="F1", title="x", artifact_path=path,
            hash=real_hash, source_execution_ids=["E1"],
        )
        issues = verify_figure(fig)
        assert len(issues) == 0
        os.unlink(path)

    def test_table_value_tampering(self):
        table = TableRecord(
            table_id="T1", title="x", headers=["V"],
            rows=[[TableCell(value=750.0, source_id="SRC-1")]],
        )
        issues = verify_table(table, source_values={"SRC-1": 700.0})
        assert any("does not match" in i for i in issues)

    def test_table_value_correct(self):
        table = TableRecord(
            table_id="T1", title="x", headers=["V"],
            rows=[[TableCell(value=700.0, source_id="SRC-1")]],
        )
        issues = verify_table(table, source_values={"SRC-1": 700.0})
        assert len(issues) == 0


# ═══════════════════════════════════════════════════════════════
# Numerical Consistency — percentage, rounding, improvement
# ═══════════════════════════════════════════════════════════════

class TestNumericalTraps:
    def test_700_variants_pass(self):
        assert NumericalConsistencyChecker.check_consistency(700.0, "700")
        assert NumericalConsistencyChecker.check_consistency(700.0, "700.0")
        assert NumericalConsistencyChecker.check_consistency(700.0, "700.00")

    def test_701_fails(self):
        assert not NumericalConsistencyChecker.check_consistency(700.0, "701")
        assert not NumericalConsistencyChecker.check_consistency(700.0, "750")

    def test_percentage_rate_matches(self):
        # source 0.35 (a rate) matches paper "35%"
        assert NumericalConsistencyChecker.check_consistency(0.35, "rate is 35%")

    def test_percentage_trap_rejected(self):
        # source 35 must NOT auto-match paper "35%"
        assert not NumericalConsistencyChecker.check_consistency(35.0, "rate is 35%")

    def test_rounding_policy(self):
        # source 0.333333333 matches paper "0.333" (3-decimal rounding)
        assert NumericalConsistencyChecker.check_consistency(0.333333333, "value 0.333")
        # "0.33" is a valid 2-decimal rounding
        assert NumericalConsistencyChecker.check_consistency(0.333333333, "value 0.33")
        # But "0.35" (2 decimals) is NOT a valid rounding of 0.333333333
        assert not NumericalConsistencyChecker.check_consistency(0.333333333, "value 0.35")
        # "0.4" (1 decimal) is NOT a valid rounding
        assert not NumericalConsistencyChecker.check_consistency(0.333333333, "value 0.4")

    def test_improvement_correct(self):
        issues = NumericalConsistencyChecker.check_improvement(100.0, 120.0, "improved by 20%")
        assert len(issues) == 0

    def test_improvement_wrong(self):
        issues = NumericalConsistencyChecker.check_improvement(100.0, 120.0, "improved by 120%")
        assert len(issues) == 1


# ═══════════════════════════════════════════════════════════════
# Anonymity / Stale Model / Jury
# ═══════════════════════════════════════════════════════════════

class TestProfileRules:
    def test_anonymity_violation(self):
        profile = CompetitionProfile(
            anonymity_rules={"forbidden_terms": ["Beijing University", "Wang"]},
        )
        paper = PaperIR(title="T", abstract="A", sections=[
            PaperSection(section_id="S", title="X", content_blocks=[
                ContentBlock(block_type=BlockType.PARAGRAPH, text="We are from Beijing University"),
            ]),
        ])
        agent = SubmissionCheckAgent(profile)
        result = agent.check(paper)
        assert result.status == SubmissionStatus.BLOCKED
        assert any("Anonymity violation" in f for f in result.failures)

    def test_stale_model_paper(self):
        profile = CompetitionProfile()
        paper = PaperIR(
            title="T", abstract="A",
            metadata={"model_version": 1},
        )
        agent = SubmissionCheckAgent(profile, current_model_version=2)
        result = agent.check(paper)
        assert result.status == SubmissionStatus.BLOCKED
        assert any("stale paper" in f for f in result.failures)

    def test_current_model_paper_ok(self):
        profile = CompetitionProfile()
        paper = PaperIR(title="T", abstract="A", metadata={"model_version": 2})
        agent = SubmissionCheckAgent(profile, current_model_version=2)
        result = agent.check(paper)
        assert result.status != SubmissionStatus.BLOCKED or \
            all("stale" not in f for f in result.failures)

    def test_jury_cannot_override_blocked(self):
        profile = CompetitionProfile()
        paper = PaperIR(title="T")  # no abstract → blocked
        agent = SubmissionCheckAgent(profile)
        result = agent.check(paper)
        assert result.status == SubmissionStatus.BLOCKED
        jury = FinalJuryAgent()
        verdict = jury.review(paper, result)
        assert verdict["submission_recommendation"] == "DO_NOT_SUBMIT"


# ═══════════════════════════════════════════════════════════════
# Full truth chain fixture
# ═══════════════════════════════════════════════════════════════

class TestTruthChain:
    def test_numerical_claim_full_chain(self):
        """objective=700 → SolverResult evidence → claim → paper → check."""
        store = EvidenceStore()
        store.register_evidence(EvidenceRef(
            evidence_id="EVD-SOLVER", source_type=EvidenceSourceType.SOLVER_RESULT,
            source_id="SOLVER-1",
        ))
        store.register_claim(Claim(
            claim_id="C1", text="maximum profit is 700",
            claim_type=ClaimType.NUMERICAL, importance="high",
            evidence_ids=["EVD-SOLVER"], verification_status=ClaimStatus.SUPPORTED,
        ))
        paper = PaperIR(title="T", abstract="A", sections=[
            PaperSection(section_id="S", title="Results", content_blocks=[
                ContentBlock(block_type=BlockType.PARAGRAPH, text="The maximum profit is 700", claim_ids=["C1"]),
            ]),
        ])
        checker = NumericalConsistencyChecker()
        assert len(checker.check_all(paper, {"profit": 700.0})) == 0
        profile = CompetitionProfile()
        agent = SubmissionCheckAgent(profile, evidence=store)
        result = agent.check(paper)
        assert result.status in (SubmissionStatus.READY_TO_SUBMIT, SubmissionStatus.READY_WITH_WARNINGS)
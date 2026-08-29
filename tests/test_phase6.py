"""Phase 6 tests: Evidence, Literature, Figures, Tables, PaperIR, LaTeX,
Submission, adversarial fixtures."""

import pytest

from mathmodel.evidence import (
    EvidenceStore, EvidenceRef, EvidenceSourceType, Claim, ClaimType,
    ClaimStatus, ClaimGate,
)
from mathmodel.documents import (
    DocumentRegistry, FigureRecord, FigureType, FigureRegistry,
    verify_figure, TableRecord, TableType, TableRegistry, TableCell,
    verify_table,
)
from mathmodel.literature import (
    LiteratureRecord, LiteratureStore, Citation, CitationVerifier,
    CitationStatus,
)
from mathmodel.paper import (
    PaperIR, PaperSection, ContentBlock, BlockType, PaperAgent,
)
from mathmodel.paper.renderer import LaTeXRenderer, _latex_escape, CompilationRecord
from mathmodel.submission import (
    CompetitionProfile, SubmissionCheckAgent, SubmissionStatus,
    UnsupportedClaimChecker, NumericalConsistencyChecker, FinalJuryAgent,
)


# ═══════════════════════════════════════════════════════════════
# Evidence
# ═══════════════════════════════════════════════════════════════

class TestEvidence:
    def test_supported_claim(self):
        store = EvidenceStore()
        store.register_evidence(EvidenceRef(
            evidence_id="EVD-1", source_type=EvidenceSourceType.SOLVER_RESULT,
            source_id="SOLVER-1",
        ))
        claim = Claim(
            claim_id="CLAIM-1", text="Total profit is 700",
            claim_type=ClaimType.NUMERICAL, importance="high",
            evidence_ids=["EVD-1"], verification_status=ClaimStatus.SUPPORTED,
        )
        store.register_claim(claim)
        assert store.get_claim("CLAIM-1") is not None
        assert ClaimGate.can_enter_paper(claim)

    def test_unsupported_claim_blocked(self):
        store = EvidenceStore()
        claim = Claim(
            claim_id="CLAIM-2", text="Total profit is 700",
            claim_type=ClaimType.NUMERICAL, importance="high",
            evidence_ids=[],  # No evidence
        )
        store.register_claim(claim)
        # Gate auto-marks as UNSUPPORTED
        assert store.get_claim("CLAIM-2").verification_status == ClaimStatus.UNSUPPORTED
        assert not ClaimGate.can_enter_paper(store.get_claim("CLAIM-2"))

    def test_invalid_evidence_id_rejected(self):
        store = EvidenceStore()
        claim = Claim(
            claim_id="CLAIM-3", text="X", evidence_ids=["EVD-GHOST"],
        )
        with pytest.raises(ValueError, match="unknown evidence"):
            store.register_claim(claim)

    def test_claim_serialization_roundtrip(self):
        store = EvidenceStore()
        store.register_evidence(EvidenceRef(
            evidence_id="EVD-1", source_type=EvidenceSourceType.VALIDATION,
            source_id="VR-1",
        ))
        claim = Claim(
            claim_id="CLAIM-4", text="Validated", importance="normal",
            evidence_ids=["EVD-1"], verification_status=ClaimStatus.SUPPORTED,
        )
        store.register_claim(claim)
        data = claim.model_dump()
        restored = Claim.model_validate(data)
        assert restored.claim_id == "CLAIM-4"


# ═══════════════════════════════════════════════════════════════
# Literature
# ═══════════════════════════════════════════════════════════════

class TestLiterature:
    def test_verified_record(self):
        store = LiteratureStore()
        record = LiteratureRecord(
            literature_id="LIT-1", title="Real Paper",
            authors=["A. Author"], year=2020, doi="10.1234/real",
            verification_status=CitationStatus.VERIFIED,
        )
        store.add_record(record)
        assert store.get_record("LIT-1").verification_status == CitationStatus.VERIFIED

    def test_duplicate_doi_rejected(self):
        store = LiteratureStore()
        store.add_record(LiteratureRecord(
            literature_id="LIT-1", title="A", doi="10.1234/x",
        ))
        with pytest.raises(ValueError, match="Duplicate DOI"):
            store.add_record(LiteratureRecord(
                literature_id="LIT-2", title="B", doi="10.1234/x",
            ))

    def test_fake_record_blocked_from_bibliography(self):
        store = LiteratureStore()
        store.add_record(LiteratureRecord(
            literature_id="LIT-1", title="Fixture paper", is_fixture=True,
            verification_status=CitationStatus.VERIFIED,
        ))
        verifier = CitationVerifier(store)
        assert len(verifier.verifiable_bibliography()) == 0

    def test_unverified_citation_blocked(self):
        store = LiteratureStore()
        store.add_record(LiteratureRecord(
            literature_id="LIT-1", title="Unverified",
            verification_status=CitationStatus.UNVERIFIED,
        ))
        verifier = CitationVerifier(store)
        bib = verifier.verifiable_bibliography()
        assert len(bib) == 0  # UNVERIFIED does not enter bibliography

    def test_citation_unknown_literature_rejected(self):
        store = LiteratureStore()
        with pytest.raises(ValueError, match="unknown literature"):
            store.add_citation(Citation(literature_id="LIT-GHOST"))

    def test_identity_mismatch_flagged(self):
        store = LiteratureStore()
        store.add_record(LiteratureRecord(
            literature_id="LIT-1", title="X",
            verification_status=CitationStatus.IDENTITY_MISMATCH,
        ))
        store.add_citation(Citation(citation_id="CIT-1", literature_id="LIT-1"))
        verifier = CitationVerifier(store)
        issues = verifier.verify_all()
        assert len(issues) > 0


# ═══════════════════════════════════════════════════════════════
# Figures / Tables
# ═══════════════════════════════════════════════════════════════

class TestFigures:
    def test_missing_artifact_detected(self):
        fig = FigureRecord(
            figure_id="FIG-1", title="Test",
            artifact_path="/nonexistent/path.png",
            source_execution_ids=["EXEC-1"],
        )
        issues = verify_figure(fig)
        assert any("does not exist" in i for i in issues)

    def test_no_source_detected(self):
        fig = FigureRecord(figure_id="FIG-2", title="Test")
        issues = verify_figure(fig)
        assert any("no data or execution source" in i for i in issues)

    def test_registry_duplicate_rejected(self):
        reg = FigureRegistry()
        reg.register(FigureRecord(figure_id="FIG-1", title="A"))
        with pytest.raises(ValueError):
            reg.register(FigureRecord(figure_id="FIG-1", title="B"))


class TestTables:
    def test_numeric_cell_without_source(self):
        table = TableRecord(
            table_id="TAB-1", title="Results",
            headers=["Value"],
            rows=[[TableCell(value=700.0)]],  # no source_id
        )
        issues = verify_table(table)
        assert any("no source_id" in i for i in issues)

    def test_cell_with_source_ok(self):
        table = TableRecord(
            table_id="TAB-2", title="Results",
            headers=["Value"],
            rows=[[TableCell(value=700.0, source_id="SOLVER-1")]],
        )
        issues = verify_table(table)
        assert len(issues) == 0

    def test_row_header_mismatch(self):
        table = TableRecord(
            table_id="TAB-3", title="Bad",
            headers=["A", "B"],
            rows=[[TableCell(value=1)]],  # 1 cell, 2 headers
        )
        issues = verify_table(table)
        assert any("row has" in i for i in issues)


# ═══════════════════════════════════════════════════════════════
# Document Registry
# ═══════════════════════════════════════════════════════════════

class TestDocumentRegistry:
    def test_stable_ids(self):
        reg = DocumentRegistry()
        id1 = reg.register("FIG", FigureRecord(title="A"))
        id2 = reg.register("FIG", FigureRecord(title="B"))
        assert id1 == "FIG-001"
        assert id2 == "FIG-002"

    def test_explicit_id(self):
        reg = DocumentRegistry()
        id1 = reg.register("SEC", object(), explicit_id="SEC-007")
        assert id1 == "SEC-007"

    def test_duplicate_explicit_id_rejected(self):
        reg = DocumentRegistry()
        reg.register("SEC", object(), explicit_id="SEC-001")
        with pytest.raises(ValueError):
            reg.register("SEC", object(), explicit_id="SEC-001")


# ═══════════════════════════════════════════════════════════════
# PaperIR
# ═══════════════════════════════════════════════════════════════

class TestPaperIR:
    def _make_paper(self):
        section = PaperSection(
            section_id="SEC-001", title="Results",
            content_blocks=[
                ContentBlock(
                    block_type=BlockType.PARAGRAPH,
                    text="The total profit is 700.",
                    claim_ids=["CLAIM-1"],
                ),
                ContentBlock(
                    block_type=BlockType.EQUATION,
                    text=r"P = 30x_1 + 25x_2 + 20x_3",
                    equation_ids=["EQ-1"],
                ),
            ],
        )
        return PaperIR(
            title="Test Paper", abstract="Test abstract",
            sections=[section],
        )

    def test_collect_claims(self):
        paper = self._make_paper()
        assert "CLAIM-1" in paper.collect_claims()

    def test_roundtrip(self):
        paper = self._make_paper()
        data = paper.model_dump()
        restored = PaperIR.model_validate(data)
        assert restored.title == "Test Paper"
        assert len(restored.sections) == 1


# ═══════════════════════════════════════════════════════════════
# LaTeX
# ═══════════════════════════════════════════════════════════════

class TestLaTeX:
    def test_escape(self):
        assert _latex_escape("a&b") == r"a\&b"
        assert _latex_escape("50%") == r"50\%"

    def test_render_basic(self):
        renderer = LaTeXRenderer()
        paper = PaperIR(title="T", abstract="A", sections=[
            PaperSection(section_id="S1", title="Intro", content_blocks=[
                ContentBlock(block_type=BlockType.PARAGRAPH, text="Hello & welcome"),
            ]),
        ])
        tex = renderer.render(paper)
        assert r"\documentclass" in tex
        assert r"\begin{document}" in tex
        assert r"\&" in tex  # escaped
        assert r"\end{document}" in tex

    def test_render_equation_block(self):
        renderer = LaTeXRenderer()
        paper = PaperIR(title="T", sections=[
            PaperSection(section_id="S1", title="M", content_blocks=[
                ContentBlock(block_type=BlockType.EQUATION, text=r"x^2 + y^2 = 1"),
            ]),
        ])
        tex = renderer.render(paper)
        assert r"\begin{equation}" in tex

    def test_compile_no_compiler(self):
        renderer = LaTeXRenderer()
        record = renderer.compile(r"\documentclass{article}", output_dir=None)
        # Either no compiler available or compile succeeded
        assert record.compiler_available in (True, False)
        if not record.compiler_available:
            assert record.success is False


# ═══════════════════════════════════════════════════════════════
# Numerical Consistency
# ═══════════════════════════════════════════════════════════════

class TestNumericalConsistency:
    def test_700_0_vs_700(self):
        assert NumericalConsistencyChecker.check_consistency(700.0, "profit is 700")

    def test_700_0_vs_700_00(self):
        assert NumericalConsistencyChecker.check_consistency(700.0, "profit is 700.00")

    def test_700_vs_701_fails(self):
        assert not NumericalConsistencyChecker.check_consistency(700.0, "profit is 701")

    def test_check_all(self):
        checker = NumericalConsistencyChecker()
        paper = PaperIR(title="T", sections=[
            PaperSection(section_id="S", title="R", content_blocks=[
                ContentBlock(block_type=BlockType.PARAGRAPH, text="objective is 700"),
            ]),
        ])
        issues = checker.check_all(paper, {"objective": 700.0})
        assert len(issues) == 0

        bad_paper = PaperIR(title="T", sections=[
            PaperSection(section_id="S", title="R", content_blocks=[
                ContentBlock(block_type=BlockType.PARAGRAPH, text="objective is 750"),
            ]),
        ])
        issues = checker.check_all(bad_paper, {"objective": 700.0})
        assert len(issues) == 1


# ═══════════════════════════════════════════════════════════════
# Submission Check + Final Jury
# ═══════════════════════════════════════════════════════════════

class TestSubmission:
    def test_valid_submission(self):
        profile = CompetitionProfile(required_sections=["abstract"])
        paper = PaperIR(title="T", abstract="A", sections=[
            PaperSection(section_id="S", title="Abstract", content_blocks=[
                ContentBlock(block_type=BlockType.PARAGRAPH, text="abstract content"),
            ]),
        ])
        agent = SubmissionCheckAgent(profile)
        result = agent.check(paper)
        assert result.status in (
            SubmissionStatus.READY_TO_SUBMIT,
            SubmissionStatus.READY_WITH_WARNINGS,
        )

    def test_unsupported_claim_fails(self):
        profile = CompetitionProfile()
        evidence = EvidenceStore()
        claim = Claim(
            claim_id="CLAIM-X", text="Result improved by 35%",
            importance="high", evidence_ids=[],
        )
        evidence.register_claim(claim)
        paper = PaperIR(title="T", abstract="A", sections=[
            PaperSection(section_id="S", title="R", content_blocks=[
                ContentBlock(
                    block_type=BlockType.PARAGRAPH,
                    text="Our model improved by 35%",
                    claim_ids=["CLAIM-X"],
                ),
            ]),
        ])
        agent = SubmissionCheckAgent(profile, evidence=evidence)
        result = agent.check(paper)
        assert result.status == SubmissionStatus.BLOCKED

    def test_unknown_figure_fails(self):
        profile = CompetitionProfile()
        figures = FigureRegistry()
        paper = PaperIR(title="T", abstract="A", sections=[
            PaperSection(section_id="S", title="R", content_blocks=[
                ContentBlock(block_type=BlockType.FIGURE, figure_ids=["FIG-999"]),
            ]),
        ])
        agent = SubmissionCheckAgent(profile, figures=figures)
        result = agent.check(paper)
        assert result.status == SubmissionStatus.BLOCKED
        assert any("unknown figure" in f for f in result.failures)

    def test_jury_cannot_override_failure(self):
        profile = CompetitionProfile()
        paper = PaperIR(title="T")  # no abstract, no sections
        agent = SubmissionCheckAgent(profile)
        result = agent.check(paper)
        assert result.status == SubmissionStatus.BLOCKED

        jury = FinalJuryAgent()
        verdict = jury.review(paper, result)
        assert verdict["submission_recommendation"] == "DO_NOT_SUBMIT"


# ═══════════════════════════════════════════════════════════════
# Adversarial Fixtures: broken papers
# ═══════════════════════════════════════════════════════════════

class TestBrokenPaperFixtures:
    def _make_profile(self):
        return CompetitionProfile(required_sections=["abstract"])

    def test_broken_paper_wrong_number(self):
        """Solver says 700, paper claims 750 → FAIL."""
        checker = NumericalConsistencyChecker()
        paper = PaperIR(title="T", abstract="A", sections=[
            PaperSection(section_id="S", title="Results", content_blocks=[
                ContentBlock(block_type=BlockType.PARAGRAPH, text="Total profit is 750"),
            ]),
        ])
        issues = checker.check_all(paper, {"profit": 700.0})
        assert len(issues) == 1

    def test_unsupported_claim_fixture(self):
        """Claim of 35% improvement without robustness evidence → FAIL."""
        evidence = EvidenceStore()
        paper = PaperIR(title="T", abstract="A", sections=[
            PaperSection(section_id="S", title="R", content_blocks=[
                ContentBlock(block_type=BlockType.PARAGRAPH, text="Robustness improved by 35%"),
            ]),
        ])
        issues = UnsupportedClaimChecker().check(paper, evidence)
        assert len(issues) > 0

    def test_fake_citation_fixture(self):
        """Fixture literature must not enter bibliography."""
        store = LiteratureStore()
        store.add_record(LiteratureRecord(
            literature_id="LIT-FAKE", title="Fake Paper", is_fixture=True,
            verification_status=CitationStatus.VERIFIED,
        ))
        verifier = CitationVerifier(store)
        assert len(verifier.verifiable_bibliography()) == 0

    def test_missing_figure_fixture(self):
        profile = self._make_profile()
        figures = FigureRegistry()
        paper = PaperIR(title="T", abstract="A", sections=[
            PaperSection(section_id="S", title="R", content_blocks=[
                ContentBlock(block_type=BlockType.FIGURE, figure_ids=["FIG-404"]),
            ]),
        ])
        agent = SubmissionCheckAgent(profile, figures=figures)
        result = agent.check(paper)
        assert result.status == SubmissionStatus.BLOCKED

    def test_equation_drift_fixture(self):
        """Model has 30*x, paper writes 300*x → detected via numerical check."""
        checker = NumericalConsistencyChecker()
        paper = PaperIR(title="T", abstract="A", sections=[
            PaperSection(section_id="S", title="M", content_blocks=[
                ContentBlock(block_type=BlockType.EQUATION, text=r"300x_1 + 25x_2"),
            ]),
        ])
        # The model coefficient 30 must appear somewhere; drift means 300 shows
        issues = checker.check_all(paper, {"coefficient_p1": 30.0})
        assert len(issues) == 1


# ═══════════════════════════════════════════════════════════════
# E2E Paper pipeline
# ═══════════════════════════════════════════════════════════════

class TestPaperE2E:
    def test_full_pipeline(self):
        # Evidence
        evidence = EvidenceStore()
        evidence.register_evidence(EvidenceRef(
            evidence_id="EVD-SOLVER", source_type=EvidenceSourceType.SOLVER_RESULT,
            source_id="SOLVER-1",
        ))
        evidence.register_claim(Claim(
            claim_id="CLAIM-1",
            text="The maximum total profit is 700.",
            claim_type=ClaimType.NUMERICAL,
            importance="high",
            evidence_ids=["EVD-SOLVER"],
            verification_status=ClaimStatus.SUPPORTED,
        ))

        # Figures (real file)
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"fake-png-content")
            fig_path = f.name
        figures = FigureRegistry()
        figures.register(FigureRecord(
            figure_id="FIG-001", title="Profit vs x1",
            figure_type=FigureType.TREND,
            source_execution_ids=["EXEC-1"],
            artifact_path=fig_path,
            verification_status="verified",
        ))

        # Tables
        tables = TableRegistry()
        tables.register(TableRecord(
            table_id="TAB-001", title="Optimal solution",
            table_type=TableType.OPTIMIZATION_RESULT,
            headers=["Variable", "Value"],
            rows=[
                [TableCell(value="x1", source_id="MODEL-1"), TableCell(value=10.0, source_id="SOLVER-1")],
                [TableCell(value="x2", source_id="MODEL-1"), TableCell(value=0.0, source_id="SOLVER-1")],
                [TableCell(value="x3", source_id="MODEL-1"), TableCell(value=20.0, source_id="SOLVER-1")],
            ],
            source_ids=["SOLVER-1"],
        ))

        # Paper
        section = PaperSection(
            section_id="SEC-001", title="Results",
            content_blocks=[
                ContentBlock(
                    block_type=BlockType.PARAGRAPH,
                    text="The maximum total profit is 700.",
                    claim_ids=["CLAIM-1"],
                ),
                ContentBlock(
                    block_type=BlockType.TABLE,
                    text="Optimal solution",
                    table_ids=["TAB-001"],
                ),
                ContentBlock(
                    block_type=BlockType.FIGURE,
                    text="Profit vs x1",
                    figure_ids=["FIG-001"],
                ),
            ],
        )
        paper = PaperIR(
            title="Optimization of Production Planning",
            abstract="We solve a production planning LP achieving 700 profit.",
            keywords=["linear programming", "optimization"],
            sections=[section],
        )

        # Numerical consistency
        checker = NumericalConsistencyChecker()
        issues = checker.check_all(paper, {"profit": 700.0})
        assert len(issues) == 0

        # Render LaTeX
        renderer = LaTeXRenderer()
        tex = renderer.render(paper)
        assert r"\begin{document}" in tex
        assert "700" in tex

        # Submission
        profile = CompetitionProfile(required_sections=[])
        agent = SubmissionCheckAgent(
            profile, evidence=evidence, figures=figures, tables=tables,
        )
        result = agent.check(paper)
        assert result.status in (
            SubmissionStatus.READY_TO_SUBMIT,
            SubmissionStatus.READY_WITH_WARNINGS,
        )

        # Final jury
        jury = FinalJuryAgent()
        verdict = jury.review(paper, result)
        assert verdict["submission_recommendation"] in ("SUBMIT", "REVIEW")

        os.unlink(fig_path)
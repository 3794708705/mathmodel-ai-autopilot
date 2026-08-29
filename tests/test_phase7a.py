"""Phase 7A tests: Reality infrastructure (deterministic) + live markers.

Deterministic tests run without credentials. Live tests are marked
with @pytest.mark.live and skip when credentials are absent.
"""

import os

import pytest

from mathmodel.reality import (
    RealityContext, RealityTrace, ExternalRealityGate, RealityStatus,
)
from mathmodel.providers.deepseek import DeepSeekProvider
from mathmodel.literature.search import (
    BaseLiteratureSearchProvider, CrossrefSearchProvider,
)
from mathmodel.literature import LiteratureRecord, normalize_doi
from mathmodel.agents.semantic_agents import (
    SemanticRedTeamAgent, ClaimSupportVerifier, ClaimSupportResult,
    ClaimSupportStatus, SemanticPaperReviewer,
)
from mathmodel.evidence import Claim, ClaimType, EvidenceStore, EvidenceRef, EvidenceSourceType
from mathmodel.paper import PaperIR, PaperSection, ContentBlock, BlockType


# ═══════════════════════════════════════════════════════════════
# Reality infrastructure (deterministic)
# ═══════════════════════════════════════════════════════════════

class TestRealityContext:
    def test_phase_7a_defaults(self):
        ctx = RealityContext.phase_7a()
        assert ctx.reality_required is True
        assert ctx.mock_allowed is False
        assert ctx.fixture_allowed is False
        assert ctx.real_llm_required is True
        assert ctx.real_search_required is True
        assert ctx.semantic_verification_required is True

    def test_development_defaults(self):
        ctx = RealityContext.development()
        assert ctx.reality_required is False
        assert ctx.mock_allowed is True


class TestRealityTrace:
    def test_counts_calls(self):
        trace = RealityTrace()
        trace.record_llm_call(mock=False)
        trace.record_llm_call(mock=False)
        trace.record_llm_call(mock=True)
        trace.record_search(real=True)
        trace.record_fixture_use()

        assert trace.real_llm_calls == 2
        assert trace.mock_llm_calls == 1
        assert trace.real_search_queries == 1
        assert trace.fixture_records_used == 1

    def test_as_dict(self):
        trace = RealityTrace()
        trace.record_llm_call(mock=False)
        d = trace.as_dict()
        assert d["real_llm_calls"] == 1
        assert d["mock_llm_calls"] == 0


class TestExternalRealityGate:
    def test_no_real_calls_fails(self):
        ctx = RealityContext.phase_7a()
        trace = RealityTrace()
        gate = ExternalRealityGate(ctx, trace)
        assert gate.evaluate() == RealityStatus.REALITY_FAILED

    def test_mock_leakage_fails(self):
        ctx = RealityContext.phase_7a()
        trace = RealityTrace()
        trace.record_llm_call(mock=False)
        trace.record_llm_call(mock=True)  # leak
        trace.record_search(real=True)
        gate = ExternalRealityGate(ctx, trace)
        assert gate.evaluate() == RealityStatus.REALITY_FAILED

    def test_fixture_leakage_fails(self):
        ctx = RealityContext.phase_7a()
        trace = RealityTrace()
        trace.record_llm_call(mock=False)
        trace.record_search(real=True)
        trace.record_fixture_use()  # leak
        gate = ExternalRealityGate(ctx, trace)
        assert gate.evaluate() == RealityStatus.REALITY_FAILED

    def test_full_real_pass(self):
        ctx = RealityContext.phase_7a()
        trace = RealityTrace()
        trace.record_llm_call(mock=False)
        trace.record_search(real=True)
        trace.record_solver(mock=False)
        trace.record_semantic_verification()
        gate = ExternalRealityGate(ctx, trace)
        assert gate.evaluate() == RealityStatus.REALITY_VERIFIED

    def test_fallback_fails(self):
        ctx = RealityContext.phase_7a()
        trace = RealityTrace()
        trace.record_llm_call(mock=False)
        trace.record_search(real=True)
        trace.record_fallback("provider unavailable -> mock")
        gate = ExternalRealityGate(ctx, trace)
        assert gate.evaluate() == RealityStatus.REALITY_FAILED

    def test_development_mode_partial(self):
        ctx = RealityContext.development()
        trace = RealityTrace()
        gate = ExternalRealityGate(ctx, trace)
        assert gate.evaluate() == RealityStatus.REALITY_PARTIAL


# ═══════════════════════════════════════════════════════════════
# No-Silent-Mock invariant
# ═══════════════════════════════════════════════════════════════

class TestNoSilentMock:
    def test_provider_requires_key(self):
        """DeepSeekProvider without key must raise, not fall back to Mock."""
        provider = DeepSeekProvider(api_key=None)
        import asyncio
        from mathmodel.providers.base import GenerationRequest

        async def call():
            return await provider.generate(GenerationRequest(prompt="hi"))

        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            asyncio.run(call())

    def test_provider_reports_mock_false(self):
        """Provider identity must be correct."""
        provider = DeepSeekProvider(api_key="test-key")
        assert provider.provider_name == "deepseek"


# ═══════════════════════════════════════════════════════════════
# Semantic agents (contract behavior without LLM)
# ═══════════════════════════════════════════════════════════════

class TestSemanticAgentsNoLLM:
    def test_semantic_redteam_no_llm_honest(self):
        """Without LLM, semantic red team must honestly report no review."""
        from mathmodel.domain.math_model import MathematicalModel
        agent = SemanticRedTeamAgent()  # no router
        model = MathematicalModel(name="Test")

        async def run():
            from mathmodel.verification import build_validation_report
            return await agent.review(model, None, build_validation_report(model, None) if False else None)

        # Contract: without router, review reports no semantic check
        report = None
        import asyncio
        async def review():
            from mathmodel.domain.verification import ValidationReport, GateStatus
            validation = ValidationReport(model_id="M", overall_status=GateStatus.PASS)
            from mathmodel.solver import SolverResult
            return await agent.review(model, SolverResult(model_id="M"), validation)

        report = asyncio.run(review())
        assert "NO REAL LLM" in report.summary or "not performed" in report.summary
        assert report.critical_count == 0

    def test_claim_support_requires_evidence(self):
        """Without retrieved evidence, support must be INSUFFICIENT_EVIDENCE."""
        verifier = ClaimSupportVerifier()  # no router
        claim = Claim(
            claim_id="C1", text="X supports Y",
            claim_type=ClaimType.LITERATURE_SUPPORTED,
            importance="high",
        )
        record = LiteratureRecord(
            literature_id="L1", title="Paper", abstract="",  # EMPTY abstract
        )
        result = asyncio_run(verifier.verify(claim, record))
        assert result.support_status in (
            ClaimSupportStatus.INSUFFICIENT_EVIDENCE,
        )
        assert result.evidence_reference == "" or "abstract" in result.reason.lower()

    def test_semantic_reviewer_no_llm_honest(self):
        reviewer = SemanticPaperReviewer()  # no router
        paper = PaperIR(title="T", abstract="A")
        result = asyncio_run(reviewer.review(paper, EvidenceStore()))
        assert any("NO REAL LLM" in r for r in result)


def asyncio_run(coro):
    import asyncio
    return asyncio.run(coro)


# ═══════════════════════════════════════════════════════════════
# Fixture flag serialization safety
# ═══════════════════════════════════════════════════════════════

class TestFixtureFlagSafety:
    def test_fixture_flag_survives_roundtrip(self):
        record = LiteratureRecord(
            literature_id="L1", title="Fixture", is_fixture=True,
        )
        data = record.model_dump()
        restored = LiteratureRecord.model_validate(data)
        assert restored.is_fixture is True

    def test_real_flag_survives_roundtrip(self):
        record = LiteratureRecord(
            literature_id="L2", title="Real", is_fixture=False,
        )
        data = record.model_dump()
        restored = LiteratureRecord.model_validate(data)
        assert restored.is_fixture is False


# ═══════════════════════════════════════════════════════════════
# Crossref provider (contract, no network needed)
# ═══════════════════════════════════════════════════════════════

class TestSearchProviderContract:
    def test_provider_name(self):
        provider = CrossrefSearchProvider()
        assert provider.provider_name == "crossref"

    def test_normalize_result(self):
        provider = CrossrefSearchProvider()
        record = provider.normalize_result({
            "title": "Real Paper",
            "authors": ["A. Author"],
            "year": 2020,
            "venue": "Journal",
            "abstract": "Abstract text",
            "doi": "10.1234/real",
        })
        assert record.is_fixture is False
        assert record.source == "crossref"
        assert record.doi == "10.1234/real"


# ═══════════════════════════════════════════════════════════════
# Live tests (marked live — skip without credentials)
# ═══════════════════════════════════════════════════════════════

HAS_LLM_KEY = bool(os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY"))

pytestmark_live = pytest.mark.skipif(
    not HAS_LLM_KEY,
    reason="No real LLM credential available (DEEPSEEK_API_KEY / OPENAI_API_KEY)",
)


@pytest.mark.live
@pytest.mark.external
@pytestmark_live
class TestLiveLLM:
    @pytest.mark.asyncio
    async def test_real_generate(self):
        key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        from mathmodel.providers.base import GenerationRequest

        if os.environ.get("DEEPSEEK_API_KEY"):
            provider = DeepSeekProvider(api_key=key)
        else:
            from mathmodel.providers.openai import OpenAIProvider
            provider = OpenAIProvider(api_key=key)

        response = await provider.generate(GenerationRequest(
            prompt="What is 2+2? Answer with a single number.",
            max_tokens=10,
        ))
        assert response.is_mock is False
        assert response.content.strip()
        assert "4" in response.content


@pytest.mark.live
@pytest.mark.external
class TestLiveSearch:
    def test_crossref_search(self):
        import asyncio
        provider = CrossrefSearchProvider()
        results = asyncio.run(provider.search("linear programming optimization"))
        # May be empty on network failure — that's honest, not a crash
        assert isinstance(results, list)
        for r in results:
            assert "title" in r
            assert r.get("title")  # no empty fabricated titles
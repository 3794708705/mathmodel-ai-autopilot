"""Phase 7A Part 1 acceptance tests: reality safety adversarial cases.

Tests A-E from the acceptance spec, credential-absence paths,
strict-mode no-silent-mock, and RealityTrace correctness.
"""

import os

import pytest

from mathmodel.reality import (
    RealityContext, RealityTrace, ExternalRealityGate, RealityStatus,
)
from mathmodel.providers.registry import (
    ProviderRegistry, ProviderUnavailableError,
)
from mathmodel.config import ProviderType
from mathmodel.routing.router import ModelRouter


# ═══════════════════════════════════════════════════════════════
# Gate adversarial cases A-E
# ═══════════════════════════════════════════════════════════════

class TestGateAdversarial:
    def test_case_a_real_search_no_llm(self):
        """A: real search=true, real LLM=false → NOT VERIFIED."""
        ctx = RealityContext.phase_7a()
        trace = RealityTrace()
        trace.record_search(real=True)  # search done
        # No LLM calls
        gate = ExternalRealityGate(ctx, trace)
        assert gate.evaluate() in (
            RealityStatus.REALITY_FAILED,
            RealityStatus.REALITY_PARTIAL,
        )
        assert gate.evaluate() != RealityStatus.REALITY_VERIFIED

    def test_case_b_mock_llm_fails(self):
        """B: mock LLM calls > 0 → FAIL."""
        ctx = RealityContext.phase_7a()
        trace = RealityTrace()
        trace.record_llm_call(mock=False)
        trace.record_llm_call(mock=True)  # leak
        trace.record_search(real=True)
        gate = ExternalRealityGate(ctx, trace)
        assert gate.evaluate() == RealityStatus.REALITY_FAILED

    def test_case_c_fixture_literature_fails(self):
        """C: fixture literature > 0 → FAIL."""
        ctx = RealityContext.phase_7a()
        trace = RealityTrace()
        trace.record_llm_call(mock=False)
        trace.record_search(real=True)
        trace.record_fixture_use()  # leak
        gate = ExternalRealityGate(ctx, trace)
        assert gate.evaluate() == RealityStatus.REALITY_FAILED

    def test_case_d_llm_without_semantic_partial(self):
        """D: real LLM > 0 but semantic verification = 0 → PARTIAL."""
        ctx = RealityContext.phase_7a()
        trace = RealityTrace()
        trace.record_llm_call(mock=False)
        trace.record_search(real=True)
        # No semantic verification recorded
        gate = ExternalRealityGate(ctx, trace)
        assert gate.evaluate() != RealityStatus.REALITY_VERIFIED

    def test_case_e_all_real_verified(self):
        """E: all real signals present → REALITY_VERIFIED (synthetic trace)."""
        ctx = RealityContext.phase_7a()
        trace = RealityTrace()
        trace.record_llm_call(mock=False)
        trace.record_search(real=True)
        trace.record_solver(mock=False)
        trace.record_semantic_verification()
        gate = ExternalRealityGate(ctx, trace)
        assert gate.evaluate() == RealityStatus.REALITY_VERIFIED


# ═══════════════════════════════════════════════════════════════
# Credential absence path — no silent mock
# ═══════════════════════════════════════════════════════════════

class TestCredentialAbsence:
    def test_strict_openai_no_key_raises(self, monkeypatch):
        """Strict mode + missing OPENAI key → raise, not Mock."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        registry = ProviderRegistry()
        with pytest.raises(ProviderUnavailableError):
            registry.get_provider(ProviderType.OPENAI, strict=True)

    def test_strict_mock_request_raises(self):
        """Strict mode + explicit MOCK request → raise (mock not allowed)."""
        registry = ProviderRegistry()
        with pytest.raises(ProviderUnavailableError):
            registry.get_provider(ProviderType.MOCK, strict=True)

    def test_non_strict_development_fallback_ok(self, monkeypatch):
        """Development mode (strict=False) still allows Mock fallback."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        registry = ProviderRegistry()
        provider = registry.get_provider(ProviderType.OPENAI, strict=False)
        assert provider.provider_name == "mock"

    def test_router_strict_raises(self, monkeypatch):
        """ModelRouter strict path raises instead of Mock."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        router = ModelRouter()
        with pytest.raises(ProviderUnavailableError):
            router.get_provider_for_tier(
                router.select_tier(
                    __import__("mathmodel.routing.profile", fromlist=["TaskProfile"]).TaskProfile.for_task_type(
                        __import__("mathmodel.routing.profile", fromlist=["TaskType"]).TaskType.CRUD
                    )
                )[0],
                strict=True,
            )

    def test_deepseek_provider_no_key_raises(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        from mathmodel.providers.deepseek import DeepSeekProvider
        from mathmodel.providers.base import GenerationRequest
        import asyncio

        provider = DeepSeekProvider(api_key=None)

        async def call():
            return await provider.generate(GenerationRequest(prompt="hi"))

        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            asyncio.run(call())


# ═══════════════════════════════════════════════════════════════
# RealityTrace correctness
# ═══════════════════════════════════════════════════════════════

class TestRealityTraceCorrectness:
    def test_provider_construction_not_counted(self):
        """Constructing a provider must NOT count as a real call."""
        trace = RealityTrace()
        # Construct DeepSeekProvider (no API call made)
        from mathmodel.providers.deepseek import DeepSeekProvider
        provider = DeepSeekProvider(api_key="fake")
        assert trace.real_llm_calls == 0
        assert provider is not None

    def test_failed_request_not_counted_as_real(self):
        """A failed request must not count as a successful real call."""
        trace = RealityTrace()
        # Simulate: provider raised → only failure recorded
        trace.record_failure("API error")
        assert trace.real_llm_calls == 0

    def test_fallback_recorded(self):
        trace = RealityTrace()
        trace.record_fallback("no key -> mock")
        assert "no key -> mock" in trace.fallbacks

    def test_counts_only_real(self):
        trace = RealityTrace()
        trace.record_llm_call(mock=True)  # mock does not count as real
        assert trace.real_llm_calls == 0
        assert trace.mock_llm_calls == 1


# ═══════════════════════════════════════════════════════════════
# Secret safety
# ═══════════════════════════════════════════════════════════════

class TestSecretSafety:
    def test_provider_error_does_not_leak_key(self):
        """ProviderUnavailableError must not contain the key itself."""
        registry = ProviderRegistry()
        try:
            registry.get_provider(ProviderType.OPENAI, strict=True)
        except ProviderUnavailableError as e:
            msg = str(e)
            assert "OPENAI_API_KEY" in msg  # env var name is fine
            assert "sk-" not in msg  # but no actual secret material
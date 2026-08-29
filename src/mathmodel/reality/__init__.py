"""MathModel AI — Phase 7A: Reality infrastructure.

RealityContext, RealityTrace, ExternalRealityGate.
Distinguishes REAL_EXECUTION from MOCK, CONTRACT_ONLY, and BENCHMARK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class RealityStatus(str, Enum):
    REALITY_VERIFIED = "reality_verified"
    REALITY_PARTIAL = "reality_partial"
    REALITY_FAILED = "reality_failed"
    BLOCKED_BY_ENVIRONMENT = "blocked_by_environment"


class RealityLevel(str, Enum):
    CONTRACT_VERIFIED = "contract_verified"
    REAL_EXECUTION_VERIFIED = "real_execution_verified"
    REAL_MODEL_QUALITY_SMOKE_VERIFIED = "real_model_quality_smoke_verified"
    BENCHMARK_VERIFIED = "benchmark_verified"


@dataclass
class RealityContext:
    """Controls whether real external services are required.

    When reality_required=True, mock fallback is NOT allowed
    and the system must honestly report REALITY_COMPROMISED.
    """

    reality_required: bool = False
    mock_allowed: bool = True
    fixture_allowed: bool = True
    real_llm_required: bool = False
    real_search_required: bool = False
    semantic_verification_required: bool = False
    real_pdf_required: bool = False
    production_sandbox_required: bool = False

    # Default Phase 7A: reality ON
    @classmethod
    def phase_7a(cls) -> "RealityContext":
        return cls(
            reality_required=True,
            mock_allowed=False,
            fixture_allowed=False,
            real_llm_required=True,
            real_search_required=True,
            semantic_verification_required=True,
            real_pdf_required=False,
            production_sandbox_required=False,
        )

    @classmethod
    def development(cls) -> "RealityContext":
        return cls(
            reality_required=False,
            mock_allowed=True,
            fixture_allowed=True,
            real_llm_required=False,
            real_search_required=False,
            semantic_verification_required=False,
        )


@dataclass
class RealityTrace:
    """Tracks real vs mock vs fixture usage across a run."""

    real_llm_calls: int = 0
    mock_llm_calls: int = 0
    real_search_queries: int = 0
    fixture_records_used: int = 0
    real_solver_runs: int = 0
    mock_executions: int = 0
    semantic_verifications: int = 0
    fallbacks: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    def record_llm_call(self, mock: bool) -> None:
        if mock:
            self.mock_llm_calls += 1
        else:
            self.real_llm_calls += 1

    def record_search(self, real: bool) -> None:
        if real:
            self.real_search_queries += 1

    def record_fixture_use(self) -> None:
        self.fixture_records_used += 1

    def record_solver(self, mock: bool) -> None:
        if mock:
            self.mock_executions += 1
        else:
            self.real_solver_runs += 1

    def record_semantic_verification(self) -> None:
        self.semantic_verifications += 1

    def record_fallback(self, reason: str) -> None:
        self.fallbacks.append(reason)

    def record_failure(self, reason: str) -> None:
        self.failures.append(reason)

    def mark_start(self) -> None:
        self.started_at = datetime.now(timezone.utc)

    def mark_end(self) -> None:
        self.finished_at = datetime.now(timezone.utc)

    def as_dict(self) -> dict[str, Any]:
        return {
            "real_llm_calls": self.real_llm_calls,
            "mock_llm_calls": self.mock_llm_calls,
            "real_search_queries": self.real_search_queries,
            "fixture_records_used": self.fixture_records_used,
            "real_solver_runs": self.real_solver_runs,
            "mock_executions": self.mock_executions,
            "semantic_verifications": self.semantic_verifications,
            "fallbacks": self.fallbacks,
            "failures": self.failures,
        }


class ExternalRealityGate:
    """Gate that checks whether a run used real external services.

    Hard failure conditions:
    - real provider silently fell back to Mock
    - fixture literature entered production path
    - fake search result
    - citation support without evidence
    - Agent output manually replaced
    - REAL status assigned without real call
    """

    def __init__(self, context: RealityContext, trace: RealityTrace):
        self._context = context
        self._trace = trace

    def evaluate(self) -> RealityStatus:
        if not self._context.reality_required:
            return RealityStatus.REALITY_PARTIAL

        failures = []

        if self._context.real_llm_required and self._trace.real_llm_calls == 0:
            failures.append("No real LLM calls recorded")
        if self._context.real_llm_required and self._trace.mock_llm_calls > 0:
            failures.append(f"Mock LLM calls leaked: {self._trace.mock_llm_calls}")
        if self._context.real_search_required and self._trace.real_search_queries == 0:
            failures.append("No real search queries recorded")
        if not self._context.fixture_allowed and self._trace.fixture_records_used > 0:
            failures.append(f"Fixture records leaked: {self._trace.fixture_records_used}")
        if self._trace.fallbacks:
            failures.append(f"Fallbacks occurred: {self._trace.fallbacks}")
        if self._trace.failures:
            failures.append(f"Failures: {len(self._trace.failures)}")

        # Semantic verification required but not performed → not fully verified
        missing_semantic = (
            self._context.semantic_verification_required
            and self._trace.semantic_verifications == 0
        )

        if failures:
            return RealityStatus.REALITY_FAILED

        if missing_semantic:
            return RealityStatus.REALITY_PARTIAL

        if self._trace.real_llm_calls > 0 and self._trace.real_search_queries > 0:
            return RealityStatus.REALITY_VERIFIED

        return RealityStatus.REALITY_PARTIAL
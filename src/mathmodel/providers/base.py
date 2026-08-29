"""MathModel AI — Base model provider interface.

All model providers must implement this abstract base class.
Business code never calls provider SDKs directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Type

from pydantic import BaseModel


@dataclass
class UsageInfo:
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class GenerationRequest:
    """Request for text generation."""
    prompt: str
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    stop_sequences: Optional[list[str]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuredGenerationRequest:
    """Request for structured (JSON schema) generation."""
    prompt: str
    output_schema: Type[BaseModel]
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResponse:
    """Response from text generation."""
    content: str
    model: str
    usage: UsageInfo = field(default_factory=UsageInfo)
    finish_reason: str = "stop"
    is_mock: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseModelProvider(ABC):
    """Abstract base class for all model providers.

    Subclasses must implement generate(), structured_generate(),
    stream(), and get_usage().
    """

    def __init__(self, api_key: Optional[str] = None, default_model: str = ""):
        self.api_key = api_key
        self.default_model = default_model
        self._total_usage = UsageInfo()

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name."""
        ...

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text from a prompt."""
        ...

    @abstractmethod
    async def structured_generate(
        self, request: StructuredGenerationRequest
    ) -> BaseModel:
        """Generate structured output conforming to a Pydantic schema."""
        ...

    @abstractmethod
    async def stream(self, request: GenerationRequest) -> Any:
        """Stream generated text. Returns an async iterator."""
        ...

    def get_usage(self) -> UsageInfo:
        """Return cumulative usage for this provider instance."""
        return self._total_usage

    def _record_usage(self, usage: UsageInfo) -> None:
        """Record token usage."""
        self._total_usage.prompt_tokens += usage.prompt_tokens
        self._total_usage.completion_tokens += usage.completion_tokens
        self._total_usage.total_tokens += usage.total_tokens
        self._total_usage.cost_usd += usage.cost_usd
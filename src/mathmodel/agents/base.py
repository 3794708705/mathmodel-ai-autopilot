"""MathModel AI — Base agent class.

All agents inherit from BaseAgent. Provides standard lifecycle:
- Input/output schema validation
- run() with state mutation
- retry() on failure
- validate_output() for self-checking
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Type

from pydantic import BaseModel

from mathmodel.models.problem_state import ProblemState

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Status of an agent run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class AgentError:
    """Error information from an agent run."""
    message: str
    error_type: str = "unknown"
    details: dict[str, Any] = field(default_factory=dict)
    recoverable: bool = True


@dataclass
class AgentResult:
    """Result of an agent run."""
    agent_name: str
    run_id: str
    status: AgentStatus
    output: Optional[dict[str, Any]] = None
    errors: list[AgentError] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for all agents.

    Subclasses must define these as class-level attributes:
        name: str
        role: str
        input_schema: Type[BaseModel]
        output_schema: Type[BaseModel]
        capabilities: list[str]

    And implement:
        run(state) -> AgentResult
        validate_output(output) -> list[AgentError]
    """

    # These are declared here for type checking.
    # Subclasses override them as simple class attributes.
    name: str
    role: str
    input_schema: Type[BaseModel]
    output_schema: Type[BaseModel]
    capabilities: list[str]

    def __init__(self):
        self._run_history: list[AgentResult] = []
        self._validate_contract()

    def _validate_contract(self) -> None:
        """Ensure all required attributes are set by the subclass."""
        required = ["name", "role", "input_schema", "output_schema", "capabilities"]
        for attr in required:
            value = getattr(self, attr, None)
            if value is None:
                raise TypeError(
                    f"{self.__class__.__name__} must define '{attr}' as a "
                    f"class-level attribute. Got None."
                )

    @abstractmethod
    async def run(self, state: ProblemState) -> AgentResult:
        """Execute the agent's core logic on the given state.

        The agent reads from and writes to the ProblemState.
        Must return an AgentResult with validated output.
        """
        ...

    @abstractmethod
    def validate_output(self, output: BaseModel) -> list[AgentError]:
        """Validate the agent's output against its output schema.

        Returns a list of validation errors (empty if valid).
        """
        ...

    async def retry(self, state: ProblemState, error: AgentError) -> AgentResult:
        """Retry the agent after a failure.

        Default implementation calls run() again. Override for
        custom retry logic.
        """
        logger.warning(
            "Agent %s retrying after error: %s", self.name, error.message
        )
        return await self.run(state)

    def _make_run_id(self) -> str:
        """Generate a unique run ID."""
        return f"{self.name}-{uuid.uuid4().hex[:12]}"

    def _start_result(self) -> AgentResult:
        """Create a result with running status."""
        return AgentResult(
            agent_name=self.name,
            run_id=self._make_run_id(),
            status=AgentStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
        )

    def _finish_result(
        self,
        result: AgentResult,
        status: AgentStatus,
        output: Optional[dict[str, Any]] = None,
        errors: Optional[list[AgentError]] = None,
    ) -> AgentResult:
        """Finalize a result with completion data."""
        result.status = status
        result.completed_at = datetime.now(timezone.utc)
        if output is not None:
            result.output = output
        if errors is not None:
            result.errors = errors
        self._run_history.append(result)
        return result
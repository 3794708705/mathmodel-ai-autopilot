"""Tests for BaseAgent."""

import asyncio
from typing import Type

import pytest
from pydantic import BaseModel

from mathmodel.agents.base import (
    AgentError,
    AgentResult,
    AgentStatus,
    BaseAgent,
)
from mathmodel.models.problem_state import ProblemState


# ── Concrete test agent ──────────────────────────────────────


class _InputSchema(BaseModel):
    message: str = ""


class _OutputSchema(BaseModel):
    result: str = ""
    confidence: float = 0.0


class ConcreteTestAgent(BaseAgent):
    """Concrete agent for testing BaseAgent."""

    name = "test_agent"
    role = "Test agent for unit testing"
    input_schema = _InputSchema
    output_schema = _OutputSchema
    capabilities = ["testing", "validation"]

    async def run(self, state: ProblemState) -> AgentResult:
        result = self._start_result()

        try:
            output = _OutputSchema(
                result=f"Processed: {state.title or 'untitled'}",
                confidence=0.95,
            )

            errors = self.validate_output(output)
            if errors:
                return self._finish_result(
                    result, AgentStatus.FAILED, errors=errors
                )

            return self._finish_result(
                result,
                AgentStatus.COMPLETED,
                output=output.model_dump(),
            )
        except Exception as e:
            return self._finish_result(
                result,
                AgentStatus.FAILED,
                errors=[AgentError(message=str(e), error_type="runtime")],
            )

    def validate_output(self, output: BaseModel) -> list[AgentError]:
        errors = []
        if not isinstance(output, _OutputSchema):
            errors.append(AgentError(
                message="Output is not _OutputSchema",
                error_type="schema",
            ))
        return errors


class FailingAgent(BaseAgent):
    """Agent that always fails."""

    name = "failing_agent"
    role = "Always fails"
    input_schema = _InputSchema
    output_schema = _OutputSchema
    capabilities = ["failing"]

    async def run(self, state: ProblemState) -> AgentResult:
        result = self._start_result()
        return self._finish_result(
            result,
            AgentStatus.FAILED,
            errors=[AgentError(message="Intentional failure", error_type="test")],
        )

    def validate_output(self, output: BaseModel) -> list[AgentError]:
        return [AgentError(message="Always invalid", error_type="validation")]


# ── Tests ────────────────────────────────────────────────────


class TestAgentError:
    """Test AgentError dataclass."""

    def test_default_error(self):
        error = AgentError(message="Something went wrong")
        assert error.message == "Something went wrong"
        assert error.error_type == "unknown"
        assert error.recoverable is True

    def test_non_recoverable_error(self):
        error = AgentError(
            message="Fatal error",
            error_type="fatal",
            recoverable=False,
            details={"code": 500},
        )
        assert error.recoverable is False
        assert error.details["code"] == 500


class TestAgentResult:
    """Test AgentResult dataclass."""

    def test_default_result(self):
        result = AgentResult(
            agent_name="test",
            run_id="test-001",
            status=AgentStatus.PENDING,
        )
        assert result.agent_name == "test"
        assert result.run_id == "test-001"
        assert result.status == AgentStatus.PENDING
        assert result.output is None
        assert result.errors == []

    def test_result_with_output(self):
        result = AgentResult(
            agent_name="test",
            run_id="test-002",
            status=AgentStatus.COMPLETED,
            output={"result": "success", "confidence": 0.9},
        )
        assert result.output["result"] == "success"
        assert result.output["confidence"] == 0.9


class TestBaseAgent:
    """Test BaseAgent through concrete implementation."""

    def test_agent_run(self):
        agent = ConcreteTestAgent()
        state = ProblemState(title="Test Problem")
        result = asyncio.run(agent.run(state))

        assert result.status == AgentStatus.COMPLETED
        assert result.agent_name == "test_agent"
        assert result.run_id is not None
        assert result.output is not None
        assert result.output["result"] == "Processed: Test Problem"
        assert result.output["confidence"] == 0.95

    def test_agent_run_without_title(self):
        agent = ConcreteTestAgent()
        state = ProblemState()
        result = asyncio.run(agent.run(state))

        assert result.status == AgentStatus.COMPLETED
        assert result.output["result"] == "Processed: untitled"

    def test_agent_run_history(self):
        agent = ConcreteTestAgent()
        state = ProblemState()

        assert len(agent._run_history) == 0
        asyncio.run(agent.run(state))
        assert len(agent._run_history) == 1
        asyncio.run(agent.run(state))
        assert len(agent._run_history) == 2

    def test_failing_agent(self):
        agent = FailingAgent()
        state = ProblemState()
        result = asyncio.run(agent.run(state))

        assert result.status == AgentStatus.FAILED
        assert len(result.errors) == 1
        assert result.errors[0].message == "Intentional failure"

    def test_agent_retry(self):
        agent = ConcreteTestAgent()
        state = ProblemState(title="Retry Test")
        error = AgentError(message="Test error")

        result = asyncio.run(agent.retry(state, error))
        assert result.status == AgentStatus.COMPLETED
        assert result.output["result"] == "Processed: Retry Test"

    def test_agent_properties(self):
        agent = ConcreteTestAgent()
        assert agent.name == "test_agent"
        assert agent.role == "Test agent for unit testing"
        assert agent.capabilities == ["testing", "validation"]

    def test_run_id_uniqueness(self):
        agent = ConcreteTestAgent()
        id1 = agent._make_run_id()
        id2 = agent._make_run_id()
        assert id1 != id2
        assert id1.startswith("test_agent-")

    def test_start_result(self):
        agent = ConcreteTestAgent()
        result = agent._start_result()
        assert result.status == AgentStatus.RUNNING
        assert result.started_at is not None
        assert result.agent_name == "test_agent"

    def test_finish_result(self):
        agent = ConcreteTestAgent()
        result = agent._start_result()
        finished = agent._finish_result(
            result,
            AgentStatus.COMPLETED,
            output={"key": "value"},
        )
        assert finished.status == AgentStatus.COMPLETED
        assert finished.completed_at is not None
        assert finished.output == {"key": "value"}

    def test_validate_output_valid(self):
        agent = ConcreteTestAgent()
        output = _OutputSchema(result="test", confidence=0.8)
        errors = agent.validate_output(output)
        assert len(errors) == 0

    def test_validate_output_invalid_type(self):
        agent = ConcreteTestAgent()
        class WrongSchema(BaseModel):
            x: int = 0

        output = WrongSchema()
        errors = agent.validate_output(output)
        assert len(errors) == 1
        assert errors[0].error_type == "schema"
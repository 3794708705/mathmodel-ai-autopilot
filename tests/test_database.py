"""Tests for database layer."""

import uuid

import pytest
from sqlalchemy import select, text

from mathmodel.database import Base
from mathmodel.models.problem_state import (
    ProblemState,
    ProblemStateStage,
    ProblemStateStatus,
)


class TestDatabaseConnection:
    """Test database connectivity and session management."""

    def test_session_creation(self, db_session):
        """Test that we can create and use a database session."""
        result = db_session.execute(text("SELECT 1"))
        assert result.scalar() == 1

    def test_tables_created(self, db_session):
        """Test that tables are created from metadata."""
        result = db_session.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='problem_states'")
        )
        assert result.scalar() == "problem_states"


class TestProblemStateModel:
    """Test the ProblemState ORM model."""

    def test_create_problem_state(self, db_session):
        """Test creating a ProblemState record."""
        problem = ProblemState(
            title="Test Problem",
            competition="Test Competition",
            raw_problem="This is a test problem statement.",
        )
        db_session.add(problem)
        db_session.flush()
        db_session.refresh(problem)

        assert problem.id is not None
        assert isinstance(problem.id, uuid.UUID)
        assert problem.title == "Test Problem"
        assert problem.competition == "Test Competition"
        assert problem.current_stage == ProblemStateStage.INGEST
        assert problem.status == ProblemStateStatus.PENDING
        assert problem.retry_count == 0

    def test_problem_state_defaults(self, db_session):
        """Test default values on ProblemState."""
        problem = ProblemState()
        db_session.add(problem)
        db_session.flush()

        assert problem.current_stage == ProblemStateStage.INGEST
        assert problem.status == ProblemStateStatus.PENDING
        assert problem.retry_count == 0
        assert problem.created_at is not None
        assert problem.updated_at is not None

    def test_problem_state_jsonb_fields(self, db_session):
        """Test JSON fields on ProblemState."""
        problem = ProblemState(
            title="JSONB Test",
            objectives=["Objective 1", "Objective 2"],
            assumptions=[
                {"text": "Assumption 1", "type": "simplification"},
                {"text": "Assumption 2", "type": "boundary"},
            ],
            candidate_models=[
                {"name": "Linear Regression", "score": 0.8},
                {"name": "Neural Network", "score": 0.6},
            ],
        )
        db_session.add(problem)
        db_session.flush()
        db_session.refresh(problem)

        assert problem.objectives == ["Objective 1", "Objective 2"]
        assert len(problem.assumptions) == 2
        assert problem.assumptions[0]["text"] == "Assumption 1"
        assert len(problem.candidate_models) == 2
        assert problem.candidate_models[0]["name"] == "Linear Regression"

    def test_problem_state_stage_transition(self, db_session):
        """Test transitioning between stages."""
        problem = ProblemState(
            current_stage=ProblemStateStage.INGEST,
            status=ProblemStateStatus.COMPLETED,
        )
        db_session.add(problem)
        db_session.flush()

        problem.current_stage = ProblemStateStage.UNDERSTAND
        problem.status = ProblemStateStatus.RUNNING
        db_session.flush()
        db_session.refresh(problem)

        assert problem.current_stage == ProblemStateStage.UNDERSTAND
        assert problem.status == ProblemStateStatus.RUNNING

    def test_problem_state_stage_history(self, db_session):
        """Test recording stage history."""
        problem = ProblemState(
            stage_history=[
                {"stage": "ingest", "status": "completed", "timestamp": "2024-01-01T00:00:00Z"},
            ]
        )
        db_session.add(problem)
        db_session.flush()
        db_session.refresh(problem)

        assert len(problem.stage_history) == 1
        assert problem.stage_history[0]["stage"] == "ingest"

        history = list(problem.stage_history)
        history.append({"stage": "understand", "status": "running", "timestamp": "2024-01-01T01:00:00Z"})
        problem.stage_history = history
        db_session.flush()
        db_session.refresh(problem)

        assert len(problem.stage_history) == 2

    def test_problem_state_retry(self, db_session):
        """Test retry count and error tracking."""
        problem = ProblemState(
            retry_count=2,
            error_message="Test error",
        )
        db_session.add(problem)
        db_session.flush()
        db_session.refresh(problem)

        assert problem.retry_count == 2
        assert problem.error_message == "Test error"

    def test_query_by_stage(self, db_session):
        """Test querying problems by stage."""
        p1 = ProblemState(current_stage=ProblemStateStage.INGEST)
        p2 = ProblemState(current_stage=ProblemStateStage.MODEL)
        db_session.add_all([p1, p2])
        db_session.flush()

        result = db_session.execute(
            select(ProblemState).where(
                ProblemState.current_stage == ProblemStateStage.INGEST
            )
        )
        ingest_problems = result.scalars().all()
        assert len(ingest_problems) == 1
        assert ingest_problems[0].current_stage == ProblemStateStage.INGEST

    def test_all_stages_enum(self, db_session):
        """Test that all workflow stages are defined."""
        stages = list(ProblemStateStage)
        assert len(stages) == 16
        assert ProblemStateStage.INGEST in stages
        assert ProblemStateStage.FINAL in stages

    def test_all_statuses_enum(self, db_session):
        """Test that all statuses are defined."""
        statuses = list(ProblemStateStatus)
        assert ProblemStateStatus.PENDING in statuses
        assert ProblemStateStatus.RUNNING in statuses
        assert ProblemStateStatus.COMPLETED in statuses
        assert ProblemStateStatus.FAILED in statuses
        assert ProblemStateStatus.SKIPPED in statuses
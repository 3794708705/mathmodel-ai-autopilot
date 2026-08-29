"""MathModel AI — FastAPI routes.

Phase 1 provides health check, configuration info, and basic
ProblemState CRUD endpoints.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mathmodel.config import ProviderType, get_settings
from mathmodel.database import get_db
from mathmodel.models.problem_state import (
    ProblemState,
    ProblemStateStage,
    ProblemStateStatus,
)
from mathmodel.providers.registry import get_provider_registry

router = APIRouter(prefix="/api/v1", tags=["v1"])


# ── Pydantic Schemas ─────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    timestamp: datetime


class ConfigInfo(BaseModel):
    app_name: str
    app_version: str
    default_provider: str
    model_router_enabled: bool
    available_providers: list[str]


class ProblemStateCreate(BaseModel):
    project_id: Optional[str] = None
    title: Optional[str] = None
    competition: Optional[str] = None
    deadline: Optional[datetime] = None
    raw_problem: Optional[str] = None


class ProblemStateUpdate(BaseModel):
    title: Optional[str] = None
    competition: Optional[str] = None
    deadline: Optional[datetime] = None
    raw_problem: Optional[str] = None
    current_stage: Optional[ProblemStateStage] = None
    status: Optional[ProblemStateStatus] = None


class ProblemStateResponse(BaseModel):
    id: uuid.UUID
    project_id: Optional[str]
    title: Optional[str]
    competition: Optional[str]
    deadline: Optional[datetime]
    current_stage: str
    status: str
    retry_count: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ProblemStateListResponse(BaseModel):
    items: list[ProblemStateResponse]
    total: int


class ProviderInfo(BaseModel):
    name: str
    default_model: str
    available: bool


# ── Health ───────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        timestamp=datetime.now(timezone.utc),
    )


# ── Configuration ────────────────────────────────────────────


@router.get("/config", response_model=ConfigInfo)
async def get_config():
    """Return current configuration (non-sensitive)."""
    settings = get_settings()
    registry = get_provider_registry()
    available = [
        p.value for p in ProviderType
        if p != ProviderType.MOCK
    ]
    return ConfigInfo(
        app_name=settings.app_name,
        app_version=settings.app_version,
        default_provider=settings.default_provider.value,
        model_router_enabled=settings.model_router_enabled,
        available_providers=available,
    )


@router.get("/providers", response_model=list[ProviderInfo])
async def list_providers():
    """List available model providers and their status."""
    settings = get_settings()
    return [
        ProviderInfo(
            name="openai",
            default_model=settings.openai_default_model,
            available=settings.get_api_key(ProviderType.OPENAI) is not None,
        ),
        ProviderInfo(
            name="google",
            default_model=settings.google_default_model,
            available=settings.get_api_key(ProviderType.GOOGLE) is not None,
        ),
        ProviderInfo(
            name="anthropic",
            default_model=settings.anthropic_default_model,
            available=settings.get_api_key(ProviderType.ANTHROPIC) is not None,
        ),
        ProviderInfo(
            name="mock",
            default_model="mock-model",
            available=True,
        ),
    ]


# ── ProblemState CRUD ────────────────────────────────────────


@router.post("/problems", response_model=ProblemStateResponse, status_code=201)
async def create_problem(
    data: ProblemStateCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new problem state."""
    problem = ProblemState(
        project_id=data.project_id,
        title=data.title,
        competition=data.competition,
        deadline=data.deadline,
        raw_problem=data.raw_problem,
        current_stage=ProblemStateStage.INGEST,
        status=ProblemStateStatus.PENDING,
    )
    db.add(problem)
    await db.flush()
    await db.refresh(problem)
    return problem


@router.get("/problems", response_model=ProblemStateListResponse)
async def list_problems(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """List all problem states."""
    query = select(ProblemState).offset(skip).limit(limit)
    count_query = select(ProblemState)

    result = await db.execute(query)
    items = result.scalars().all()

    count_result = await db.execute(count_query)
    total = len(count_result.scalars().all())

    return ProblemStateListResponse(
        items=[ProblemStateResponse.model_validate(item) for item in items],
        total=total,
    )


@router.get("/problems/{problem_id}", response_model=ProblemStateResponse)
async def get_problem(
    problem_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get a problem state by ID."""
    result = await db.execute(
        select(ProblemState).where(ProblemState.id == problem_id)
    )
    problem = result.scalar_one_or_none()
    if problem is None:
        raise HTTPException(status_code=404, detail="Problem not found")
    return problem


@router.patch("/problems/{problem_id}", response_model=ProblemStateResponse)
async def update_problem(
    problem_id: uuid.UUID,
    data: ProblemStateUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update a problem state."""
    result = await db.execute(
        select(ProblemState).where(ProblemState.id == problem_id)
    )
    problem = result.scalar_one_or_none()
    if problem is None:
        raise HTTPException(status_code=404, detail="Problem not found")

    update_data = data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(problem, key, value)

    await db.flush()
    await db.refresh(problem)
    return problem


@router.delete("/problems/{problem_id}", status_code=204)
async def delete_problem(
    problem_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete a problem state."""
    result = await db.execute(
        select(ProblemState).where(ProblemState.id == problem_id)
    )
    problem = result.scalar_one_or_none()
    if problem is None:
        raise HTTPException(status_code=404, detail="Problem not found")
    await db.delete(problem)
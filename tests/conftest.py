"""MathModel AI — Test configuration and fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from mathmodel.database import Base
from mathmodel.providers.registry import ProviderRegistry


# ── Database fixtures ────────────────────────────────────────


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """Create an in-memory SQLite database for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        echo=False,
    )

    Base.metadata.create_all(engine)

    session_factory = sessionmaker(engine, expire_on_commit=False)
    session = session_factory()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
        engine.dispose()


# ── Environment fixtures ─────────────────────────────────────


@pytest.fixture(autouse=True)
def clean_settings_cache():
    """Clear settings cache between tests."""
    from mathmodel.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def clean_provider_registry():
    """Clear provider registry between tests."""
    registry = ProviderRegistry()
    registry.reset()
    yield
    registry.reset()


# ── Environment variable helpers ─────────────────────────────


@pytest.fixture
def set_env(monkeypatch):
    """Helper to set environment variables for a test."""
    def _set_env(**kwargs):
        for key, value in kwargs.items():
            monkeypatch.setenv(key, str(value))
    return _set_env
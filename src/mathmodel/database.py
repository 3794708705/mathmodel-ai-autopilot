"""MathModel AI — Database layer.

Provides SQLAlchemy engine, session factory, and base model class.
Supports both sync (SQLite for testing) and async (PostgreSQL for production).

Engine creation is lazy: importing this module does not immediately
connect to the database. This allows tests to override the URL before
any connection is attempted.
"""

from __future__ import annotations

from typing import AsyncGenerator, Generator, Optional

from sqlalchemy.orm import DeclarativeBase

from mathmodel.config import get_settings

settings = get_settings()


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# ── Engine management ────────────────────────────────────────

_sync_engine = None
_sync_session_factory = None
_async_engine = None
_async_session_factory = None


def _get_sync_engine():
    """Lazily create sync engine."""
    global _sync_engine, _sync_session_factory
    if _sync_engine is None:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        _sync_engine = create_engine(
            settings.database_url,
            echo=settings.database_echo,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
        )
        _sync_session_factory = sessionmaker(
            _sync_engine,
            expire_on_commit=False,
        )
    return _sync_engine, _sync_session_factory


def _get_async_engine():
    """Lazily create async engine if dependencies are available."""
    global _async_engine, _async_session_factory
    if _async_engine is None:
        try:
            from sqlalchemy.ext.asyncio import (
                AsyncSession,
                async_sessionmaker,
                create_async_engine,
            )

            async_url = _build_async_url(settings.database_url)
            _async_engine = create_async_engine(
                async_url,
                echo=settings.database_echo,
                pool_size=settings.database_pool_size,
                max_overflow=settings.database_max_overflow,
            )
            _async_session_factory = async_sessionmaker(
                _async_engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
        except ImportError:
            raise ImportError(
                "Async database support requires aiosqlite (for SQLite) "
                "or asyncpg (for PostgreSQL). Install with: pip install aiosqlite"
            )
    return _async_engine, _async_session_factory


def _build_async_url(sync_url: str) -> str:
    """Convert a sync database URL to async."""
    if sync_url.startswith("postgresql://"):
        return sync_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if sync_url.startswith("postgresql+psycopg2://"):
        return sync_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
    if sync_url.startswith("sqlite://"):
        return sync_url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    return sync_url


def reset_engines() -> None:
    """Reset engine state (useful for testing with different URLs)."""
    global _sync_engine, _sync_session_factory, _async_engine, _async_session_factory
    if _sync_engine:
        _sync_engine.dispose()
    if _async_engine:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(_async_engine.dispose())
            else:
                loop.run_until_complete(_async_engine.dispose())
        except RuntimeError:
            pass
    _sync_engine = None
    _sync_session_factory = None
    _async_engine = None
    _async_session_factory = None


async def get_db() -> AsyncGenerator:
    """FastAPI dependency: yields an async database session."""
    from sqlalchemy.ext.asyncio import AsyncSession

    _, session_factory = _get_async_engine()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_sync_db() -> Generator:
    """Yield a synchronous database session (for testing and scripts)."""
    from sqlalchemy.orm import Session

    _, session_factory = _get_sync_engine()
    with session_factory() as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
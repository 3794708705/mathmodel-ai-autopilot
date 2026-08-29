"""MathModel AI — FastAPI application entry point.

Phase 1 provides the foundational API server.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from mathmodel.config import get_settings

logger = logging.getLogger(__name__)

# FastAPI is optional in Phase 1 tests — gracefully degrade
try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore
    CORSMiddleware = None  # type: ignore


def create_app():
    """Create and configure the FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is not installed. Install with: pip install fastapi"
        )

    from mathmodel.api.routes import router

    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app):
        """Application lifespan: startup and shutdown events."""
        logger.info(
            "Starting %s v%s on %s:%d",
            settings.app_name,
            settings.app_version,
            settings.host,
            settings.port,
        )
        logger.info("Default provider: %s", settings.default_provider.value)
        logger.info(
            "Model router: %s",
            "enabled" if settings.model_router_enabled else "disabled",
        )
        yield
        logger.info("Shutting down %s", settings.app_name)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    return app


if FASTAPI_AVAILABLE:
    app = create_app()
else:
    app = None


if __name__ == "__main__":
    import uvicorn

    if not FASTAPI_AVAILABLE:
        raise SystemExit("FastAPI is not installed. Install with: pip install fastapi")

    settings = get_settings()
    uvicorn.run(
        "mathmodel.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.value.lower(),
    )
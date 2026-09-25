"""MathModel AI — Configuration management.

Uses pydantic-settings to load configuration from environment variables
and .env files. All settings are validated at startup.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ProviderType(str, Enum):
    OPENAI = "openai"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    MOCK = "mock"


class ModelTier(str, Enum):
    """Model capability tiers for routing."""
    FAST = "fast"
    BALANCED = "balanced"
    FLAGSHIP_HIGH = "flagship_high"
    FLAGSHIP_XHIGH = "flagship_xhigh"
    FLAGSHIP_MAX = "flagship_max"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Application ──────────────────────────────────────────
    app_name: str = "MathModel AI"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO

    # ── Database ─────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql://mathmodel:mathmodel@localhost:5432/mathmodel",
        description="PostgreSQL connection string",
    )
    database_echo: bool = False
    database_pool_size: int = 5
    database_max_overflow: int = 10

    # ── Model Providers ──────────────────────────────────────
    default_provider: ProviderType = ProviderType.MOCK

    # OpenAI
    openai_api_key: Optional[SecretStr] = None
    openai_default_model: str = "gpt-4o"

    # Google
    google_api_key: Optional[SecretStr] = None
    google_default_model: str = "gemini-2.0-flash"

    # Anthropic
    anthropic_api_key: Optional[SecretStr] = None
    anthropic_default_model: str = "claude-sonnet-4-20250514"

    # DeepSeek
    deepseek_api_key: Optional[SecretStr] = None
    deepseek_default_model: str = "deepseek-v4-flash"
    deepseek_reasoning_model: str = "deepseek-v4-pro"
    # These models otherwise spend the whole output budget on reasoning and
    # return nothing usable. Set to "" to let the provider decide.
    deepseek_reasoning_effort: str = "minimal"

    # ── Model Routing ────────────────────────────────────────
    model_router_enabled: bool = True
    model_escalation_enabled: bool = True
    max_auto_retries: int = 3

    # Minimum tiers by task type (JSON strings parsed as dicts)
    minimum_model_tiers: str = Field(
        default='{"mathematical_modeling": "flagship_high", "sandbox_security": "flagship_xhigh", "documentation": "fast", "code_generation": "balanced", "data_analysis": "balanced", "validation": "flagship_high"}',
        description="JSON mapping of task_type to minimum ModelTier",
    )

    # ── Sandbox ──────────────────────────────────────────────
    sandbox_cpu_limit: int = 2
    sandbox_memory_limit_mb: int = 2048
    sandbox_timeout_seconds: int = 300
    sandbox_network_enabled: bool = False

    # ── Server ───────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default=["*"])

    # ── Paths ────────────────────────────────────────────────
    data_dir: Path = Path("./data")
    output_dir: Path = Path("./output")
    generated_dir: Path = Path("./generated")

    def get_api_key(self, provider: ProviderType) -> Optional[str]:
        """Return the API key for a given provider."""
        key_map = {
            ProviderType.OPENAI: self.openai_api_key,
            ProviderType.GOOGLE: self.google_api_key,
            ProviderType.ANTHROPIC: self.anthropic_api_key,
            ProviderType.DEEPSEEK: self.deepseek_api_key,
        }
        secret = key_map.get(provider)
        if secret is None:
            return None
        return secret.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
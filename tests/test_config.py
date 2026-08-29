"""Tests for config module."""

import os

import pytest
from pydantic import SecretStr

from mathmodel.config import (
    LogLevel,
    ModelTier,
    ProviderType,
    Settings,
    get_settings,
)


class TestSettings:
    """Test settings loading and validation."""

    def test_default_settings(self):
        """Test that default settings are loaded correctly."""
        settings = Settings()
        assert settings.app_name == "MathModel AI"
        assert settings.app_version == "0.1.0"
        assert settings.default_provider == ProviderType.MOCK
        assert settings.model_router_enabled is True
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000

    def test_environment_override(self, monkeypatch):
        """Test that environment variables override defaults."""
        monkeypatch.setenv("APP_NAME", "Test App")
        monkeypatch.setenv("DEBUG", "true")
        monkeypatch.setenv("PORT", "9000")

        settings = Settings()
        assert settings.app_name == "Test App"
        assert settings.debug is True
        assert settings.port == 9000

    def test_api_key_retrieval(self):
        """Test API key retrieval with SecretStr."""
        settings = Settings(openai_api_key=SecretStr("test-key-123"))
        assert settings.get_api_key(ProviderType.OPENAI) == "test-key-123"
        assert settings.get_api_key(ProviderType.GOOGLE) is None

    def test_database_url_default(self):
        """Test default database URL."""
        settings = Settings()
        assert "postgresql://" in settings.database_url
        assert "mathmodel" in settings.database_url

    def test_log_level_enum(self):
        """Test log level enum values."""
        assert LogLevel.DEBUG == "DEBUG"
        assert LogLevel.INFO == "INFO"
        assert LogLevel.WARNING == "WARNING"
        assert LogLevel.ERROR == "ERROR"

    def test_model_tier_enum(self):
        """Test model tier enum values."""
        assert ModelTier.FAST == "fast"
        assert ModelTier.BALANCED == "balanced"
        assert ModelTier.FLAGSHIP_HIGH == "flagship_high"
        assert ModelTier.FLAGSHIP_XHIGH == "flagship_xhigh"
        assert ModelTier.FLAGSHIP_MAX == "flagship_max"

    def test_provider_type_enum(self):
        """Test provider type enum values."""
        assert ProviderType.OPENAI == "openai"
        assert ProviderType.GOOGLE == "google"
        assert ProviderType.ANTHROPIC == "anthropic"
        assert ProviderType.MOCK == "mock"

    def test_get_settings_caching(self):
        """Test that get_settings returns the same instance."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_sandbox_defaults(self):
        """Test sandbox configuration defaults."""
        settings = Settings()
        assert settings.sandbox_cpu_limit == 2
        assert settings.sandbox_memory_limit_mb == 2048
        assert settings.sandbox_timeout_seconds == 300
        assert settings.sandbox_network_enabled is False

    def test_minimum_model_tiers_parsing(self):
        """Test minimum model tiers JSON parsing."""
        settings = Settings()
        assert "mathematical_modeling" in settings.minimum_model_tiers
        assert "documentation" in settings.minimum_model_tiers
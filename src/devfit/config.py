"""
Application settings loaded from environment variables and/or a ``.env`` file.

All settings are read once at import time via ``pydantic-settings``.
Access the singleton via ``get_settings()`` — do **not** instantiate
``Settings`` directly in application code.

Environment variables
---------------------
GROQ_API_KEY
    Required for all LLM inference calls.
GITHUB_TOKEN
    Optional.  When supplied, the GitHub client uses authenticated requests
    which provide a higher rate-limit ceiling (5,000 req/hr vs 60 req/hr).
DATABASE_URL
    Optional.  When absent the app runs without a database.  Accepted
    schemes: ``sqlite+aiosqlite://`` or ``postgresql+asyncpg://``.
LOG_LEVEL
    Defaults to ``"INFO"``.  Set to ``"DEBUG"`` during development.
DEVFIT_ENV
    ``"development"`` or ``"production"`` (default ``"development"``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralised configuration for DevFit.

    Parameters
    ----------
    groq_api_key : SecretStr
        Groq inference API key.  Required.
    github_token : SecretStr | None
        Optional GitHub personal-access token for higher API rate limits.
    database_url : str | None
        SQLAlchemy async database URL.  ``None`` disables all DB functionality.
    log_level : str
        Python logging level string.
    devfit_env : Literal["development", "production"]
        Runtime environment flag.
    github_api_base : str
        Base URL for the GitHub REST API.  Override in tests to point at a
        mock server.
    groq_model : str
        Groq model identifier used for all LLM calls.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Required
    groq_api_key: SecretStr

    # Optional infrastructure
    github_token: SecretStr | None = None
    database_url: str | None = None

    # Tunables
    log_level: str = Field(
        default="INFO",
        pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
    )
    devfit_env: Literal["development", "production"] = "development"
    github_api_base: str = "https://api.github.com"
    groq_model: str = "openai/gpt-oss-120b"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached application settings singleton.

    The settings object is constructed once and cached for the lifetime of
    the process.  Call ``get_settings.cache_clear()`` in tests that need to
    inject different values.

    Returns
    -------
    Settings
        The populated settings instance.
    """
    return Settings()  # type: ignore[call-arg]

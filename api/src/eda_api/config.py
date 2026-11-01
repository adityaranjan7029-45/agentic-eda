"""Settings, loaded from environment variables and `.env`.

pydantic-settings validates every value at startup, so a typo in `.env`
fails loudly when the server starts rather than in the middle of a run.
Unknown variables (e.g. v1's settings in the same `.env`) are ignored.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from tempfile import gettempdir
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["dev", "test", "prod"] = "dev"

    # Postgres in production (Neon); a local SQLite file is fine for dev.
    database_url: str = "sqlite:///./agentic_eda_dev.db"

    # Server-side LLM keys (optional). Users can bring their own per request.
    groq_api_key: SecretStr | None = None
    cerebras_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None

    run_token_budget: int = Field(10_000, ge=1_000, le=100_000)
    max_upload_mb: int = Field(50, ge=1, le=500)
    max_rows: int = Field(500_000, ge=100)
    max_columns: int = Field(500, ge=2)
    upload_dir: Path = Path(gettempdir()) / "agentic-eda-uploads"
    stale_upload_minutes: int = 60          # leftovers older than this are purged at startup

    cors_origins: list[str] = ["http://localhost:3000"]

    @field_validator("database_url")
    @classmethod
    def _driver(cls, url: str) -> str:
        """Neon gives postgres:// or postgresql:// URLs; SQLAlchemy needs the psycopg driver named."""
        for prefix in ("postgres://", "postgresql://"):
            if url.startswith(prefix):
                return "postgresql+psycopg://" + url[len(prefix):]
        return url

    @model_validator(mode="after")
    def _production_rules(self) -> Settings:
        if self.app_env == "prod" and not self.database_url.startswith("postgresql"):
            raise ValueError("APP_ENV=prod requires a Postgres DATABASE_URL")
        return self

    @property
    def dev_auth(self) -> bool:
        """Header-based identity (X-User-Email) until real login arrives in step 13. Never in prod."""
        return self.app_env in ("dev", "test")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def server_llm_keys(self) -> dict[str, str]:
        keys = {"groq": self.groq_api_key, "cerebras": self.cerebras_api_key,
                "openrouter": self.openrouter_api_key}
        return {p: k.get_secret_value() for p, k in keys.items() if k and k.get_secret_value().strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()

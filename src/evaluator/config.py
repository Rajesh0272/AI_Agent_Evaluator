"""
Configuration and settings module for the AI Agent Response Evaluator.

All configuration options and secrets (API keys, OpenSearch IDs, etc.) are loaded
from environment variables or an optional local .env file.
No sensitive credentials or environment-specific values are hardcoded in the codebase.
"""

from typing import Optional, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Core execution settings
    evaluator_env: str = "development"
    evaluator_log_level: str = "INFO"
    evaluator_engine_mode: Literal["rule", "llm", "hybrid"] = "rule"

    # LLM Provider settings (Optional - for LLM or Hybrid judge modes)
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # Enterprise Search / OpenSearch / Database credentials (controlled via config)
    opensearch_host: Optional[str] = None
    opensearch_port: int = 9200
    opensearch_app_id: Optional[str] = None
    opensearch_secret_id: Optional[str] = None
    opensearch_index: Optional[str] = None

    # General Integration IDs & Keys
    app_id: Optional[str] = None
    secret_id: Optional[str] = None
    external_api_key: Optional[str] = None


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Return the global application settings instance."""
    return settings

"""
Configuration management for PDF-RAG Backend.
Uses Pydantic Settings for environment variable management.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "PDF-RAG API"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: str = "development"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/pdfrag"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # LLM/Embedding Providers
    llm_provider: str = "openai"
    embedding_provider: str = "openai"

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"
    gemini_embedding_model: str = "text-embedding-004"

    # LlamaIndex / RAG Configuration
    chunk_size: int = 512
    chunk_overlap: int = 50
    similarity_top_k: int = 5

    # Upload
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    upload_dir: str = "./uploads"

    # CORS
    cors_origins: list[str] = ["http://localhost:3000"]

    # JWT Authentication
    jwt_secret_key: str = "change-me-in-production-use-secure-random-key"
    jwt_algorithm: str = "HS256"
    jwt_access_expire_minutes: int = 30
    jwt_refresh_expire_days: int = 7


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()

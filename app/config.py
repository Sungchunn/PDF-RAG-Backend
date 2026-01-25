"""
Configuration management for PDF-RAG Backend.
Uses Pydantic Settings for environment variable management.
"""

from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Embedding Configuration (Cost Optimization)
    embedding_dimensions: int = 512
    """Matryoshka embedding dimension. Options: 256, 512, 1024, 1536.
    Lower = faster search + less storage, slightly lower quality.
    512 recommended for cost/quality balance."""
    use_matryoshka: bool = True
    """Enable Matryoshka dimension reduction."""

    # Hybrid Search Configuration
    use_hybrid_search: bool = True
    """Enable BM25 + vector hybrid search."""
    bm25_candidate_limit: int = 100
    """Number of candidates from BM25 before vector reranking.
    Higher = better recall, slower search. 100-200 recommended."""
    bm25_weight: float = 0.3
    """Weight for BM25 score in hybrid ranking (0.0 to 1.0).
    0.0 = pure vector, 1.0 = pure BM25. 0.3 recommended."""

    # Query Cache Configuration
    query_cache_enabled: bool = True
    """Enable query result caching."""
    query_cache_ttl_seconds: int = 3600
    """Cache TTL in seconds. Default: 1 hour."""
    semantic_cache_threshold: float = 0.95
    """Cosine similarity threshold for semantic cache matching.
    If a cached query embedding is >= this similar, return cached result."""

    # Smart Chunking Configuration
    min_chunk_size: int = 100
    """Minimum characters per chunk. Smaller blocks get merged."""
    max_chunk_size: int = 1000
    """Maximum characters per chunk. Larger blocks get split."""
    skip_headers_footers: bool = True
    """Skip header/footer blocks during chunking."""
    deduplicate_chunks: bool = True
    """Skip chunks with duplicate content hash."""

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

    @model_validator(mode="after")
    def validate_jwt_secret(self) -> "Settings":
        """Validate JWT secret is not the default value in production."""
        default_secret = "change-me-in-production-use-secure-random-key"
        if self.environment == "production" and self.jwt_secret_key == default_secret:
            raise ValueError(
                "JWT_SECRET_KEY must be set to a secure random value in production. "
                "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
            )
        return self


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()

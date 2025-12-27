"""
Application configuration management.

Loads settings from environment variables with defaults from constants.
Validates configuration at startup.
"""

import os
from typing import Optional
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import BaseModel,Field, validator

from core.constants import (
    LOGS_DIR,
    VECTORSTORE_DIR,
    DOCUMENTS_DIR,
    LLMProvider,
    EMBEDDING_MODEL,
    LOG_LEVEL,
)

class EmbeddingsConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # ========================================================================
    # APPLICATION
    # ========================================================================
    
    app_name: str = "Jericho Chatbot"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")

    # ========================================================================
    # DIRECTORIES
    # ========================================================================

    log_dir: Path = Field(default=LOGS_DIR, env="LOG_DIR")
    log_file: str = Field(default="app.log", env="LOG_FILE")
    log_level: str = Field(default=LOG_LEVEL, env="LOG_LEVEL")
    json_logs: bool = Field(default=False, env="JSON_LOGS")

    data_dir: Path = Field(default=Path("data"), env="DATA_DIR")
    documents_dir: Path = Field(default=DOCUMENTS_DIR, env="DOCUMENTS_DIR")
    vectorstore_dir: Path = Field(default=VECTORSTORE_DIR, env="VECTORSTORE_DIR")

    # ========================================================================
    # LLM CONFIGURATION
    # ========================================================================

    llm_provider: str = Field(default=LLMProvider.OLLAMA.value, env="LLM_PROVIDER")
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3:latest", env="OLLAMA_MODEL")
    ollama_timeout: int = Field(default=120, env="OLLAMA_TIMEOUT")

    groq_api_key: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", env="GROQ_MODEL")

    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")

    llm_temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=512, env="LLM_MAX_TOKENS")

    # ========================================================================
    # EMBEDDING CONFIGURATION
    # ========================================================================

    embedding_model: str = Field(default=EMBEDDING_MODEL, env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")
    embedding_batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")

    # ========================================================================
    # CHROMADB CONFIGURATION
    # ========================================================================

    chromadb_collection_name: str = Field(
        default="jericho_documents",
        env="CHROMADB_COLLECTION_NAME"
    )
    chromadb_persist_dir: Path = Field(
        default=VECTORSTORE_DIR,
        env="CHROMADB_PERSIST_DIR"
    )

    # ========================================================================
    # RETRIEVAL CONFIGURATION
    # ========================================================================

    retrieval_mode: str = Field(default="hybrid", env="RETRIEVAL_MODE")
    top_k_retrieval: int = Field(default=5, env="TOP_K_RETRIEVAL")
    top_k_rerank: int = Field(default=3, env="TOP_K_RERANK")
    similarity_threshold: float = Field(default=0.3, env="SIMILARITY_THRESHOLD")

    # ========================================================================
    # CHUNKING CONFIGURATION
    # ========================================================================

    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    chunking_strategy: str = Field(default="hybrid", env="CHUNKING_STRATEGY")

    # ========================================================================
    # DATABASE CONFIGURATION
    # ========================================================================

    metadata_db_path: Path = Field(default=Path("data/metadata.db"), env="METADATA_DB_PATH")

    # ========================================================================
    # API CONFIGURATION
    # ========================================================================

    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    api_reload: bool = Field(default=True, env="API_RELOAD")

    # ========================================================================
    # CORS CONFIGURATION
    # ========================================================================

    cors_origins: list = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="CORS_ORIGINS"
    )
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")

    # ========================================================================
    # CACHE CONFIGURATION
    # ========================================================================

    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")

    # ========================================================================
    # OCR CONFIGURATION
    # ========================================================================

    enable_ocr: bool = Field(default=True, env="ENABLE_OCR")
    ocr_language: str = Field(default="eng", env="OCR_LANGUAGE")

    # ========================================================================
    # VALIDATION
    # ========================================================================

    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of {valid_levels}")
        return v.upper()

    @validator("llm_temperature")
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        return v

    @validator("similarity_threshold")
    def validate_similarity_threshold(cls, v: float) -> float:
        """Validate similarity threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
        return v

    @validator("api_port")
    def validate_port(cls, v: int) -> int:
        """Validate port is in valid range."""
        if not 1024 <= v <= 65535:
            raise ValueError("Port must be between 1024 and 65535")
        return v

    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get application settings.

    Returns:
        Settings instance
    """
    return settings


def validate_settings() -> None:
    """
    Validate critical settings on startup.

    Raises:
        ValueError: If critical settings are missing or invalid
    """
    logger_import = __import__("logging").getLogger(__name__)

    # Validate directories exist and are writable
    for directory in [settings.log_dir, settings.data_dir, settings.vectorstore_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        if not os.access(directory, os.W_OK):
            raise ValueError(f"Directory not writable: {directory}")

    # Validate LLM provider configuration
    if settings.llm_provider == "groq":
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY not set but LLM_PROVIDER=groq")
        logger_import.info(" Groq API configured")

    elif settings.llm_provider == "ollama":
        # Will validate connection at startup
        logger_import.info(f" Ollama configured at {settings.ollama_base_url}")

    elif settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set but LLM_PROVIDER=openai")
        logger_import.info(" OpenAI API configured")

    logger_import.info(" All critical settings validated")

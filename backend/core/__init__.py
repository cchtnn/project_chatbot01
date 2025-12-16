"""Core utilities for Jericho Chatbot."""

from .logger import init_logger, get_logger, LoggerConfig
from .constants import (
    BASE_DIR,
    DATA_DIR,
    DOCUMENTS_DIR,
    VECTORSTORE_DIR,
    DocumentType,
    SUPPORTED_EXTENSIONS,
    ChunkingStrategy,
    RetrievalMode,
    Language,
    LLMProvider,
    CHUNK_SIZE,          # ADD THESE
    CHUNK_OVERLAP,       # ADD THESE
    MIN_CHUNK_SIZE,      # ADD THESE
    CHROMADB_COLLECTION_NAME
)

__all__ = [
    "init_logger",
    "get_logger",
    "LoggerConfig",
    "BASE_DIR",
    "DATA_DIR",
    "DOCUMENTS_DIR",
    "VECTORSTORE_DIR",
    "DocumentType",
    "SUPPORTED_EXTENSIONS",
    "ChunkingStrategy",
    "RetrievalMode",
    "Language",
    "LLMProvider",
    "CHUNK_SIZE",        # ADD THESE
    "CHUNK_OVERLAP",     # ADD THESE
    "MIN_CHUNK_SIZE",    # ADD THESE
    "CHROMADB_COLLECTION_NAME"
]

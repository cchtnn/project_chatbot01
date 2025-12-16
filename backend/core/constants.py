"""
Application constants and configuration values.

Centralized definitions for document types, chunking parameters,
retrieval settings, and model configurations.
"""

from enum import Enum
from pathlib import Path

# ============================================================================
# APPLICATION PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
DOCUMENTS_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ============================================================================
# SUPPORTED DOCUMENT FORMATS
# ============================================================================

class DocumentType(str, Enum):
    """Supported document types."""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    JSON = "json"
    CSV = "csv"
    XLSX = "xlsx"
    MARKDOWN = "md"
    HTML = "html"
    IMAGE = "image"  # PNG, JPG, JPEG
    PPTX = "pptx"


SUPPORTED_EXTENSIONS = {
    ".pdf": DocumentType.PDF,
    ".docx": DocumentType.DOCX,
    ".doc": DocumentType.DOCX,
    ".txt": DocumentType.TXT,
    ".json": DocumentType.JSON,
    ".csv": DocumentType.CSV,
    ".xlsx": DocumentType.XLSX,
    ".xls": DocumentType.XLSX,
    ".md": DocumentType.MARKDOWN,
    ".html": DocumentType.HTML,
    ".htm": DocumentType.HTML,
    ".png": DocumentType.IMAGE,
    ".jpg": DocumentType.IMAGE,
    ".jpeg": DocumentType.IMAGE,
    ".pptx": DocumentType.PPTX,
    ".ppt": DocumentType.PPTX,
}

MAX_FILE_SIZE_MB = 50  # Maximum file upload size
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "application/json",
    "text/csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "text/markdown",
    "text/html",
    "image/png",
    "image/jpeg",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

# ============================================================================
# TEXT PROCESSING & CHUNKING
# ============================================================================

class ChunkingStrategy(str, Enum):
    """Text chunking strategies."""

    FIXED_SIZE = "fixed_size"           # Fixed character length
    SEMANTIC = "semantic"               # Semantic boundaries
    HYBRID = "hybrid"                   # Combination


# Chunking parameters
CHUNK_SIZE = 1000                       # Characters per chunk
CHUNK_OVERLAP = 200                     # Character overlap between chunks
MIN_CHUNK_SIZE = 100                    # Minimum chunk size
MAX_CHUNK_SIZE = 2000                   # Maximum chunk size
CHUNKING_STRATEGY = ChunkingStrategy.HYBRID

# Separator priorities for semantic chunking
SEMANTIC_SEPARATORS = [
    "\n\n",          # Paragraph break
    "\n",            # Line break
    ". ",            # Sentence end
    " ",             # Word boundary
]

# ============================================================================
# EMBEDDING & VECTOR SEARCH
# ============================================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"    # SentenceTransformer model
EMBEDDING_DIMENSION = 384               # Output dimension of embeddings
EMBEDDING_BATCH_SIZE = 32               # Batch size for embedding

# ChromaDB settings
CHROMADB_COLLECTION_NAME = "jericho_documents"
CHROMADB_PERSISTENCE_MODE = True

# ============================================================================
# RETRIEVAL SETTINGS
# ============================================================================

class RetrievalMode(str, Enum):
    """Retrieval strategies."""

    VECTOR_ONLY = "vector_only"         # Pure semantic search
    BM25_ONLY = "bm25_only"             # Keyword-based search
    HYBRID = "hybrid"                   # Combined vector + BM25
    RERANKED = "reranked"               # Hybrid with cross-encoder reranking


RETRIEVAL_MODE = RetrievalMode.HYBRID
TOP_K_RETRIEVAL = 5                     # Number of documents to retrieve
TOP_K_RERANK = 3                        # Top results after reranking
SIMILARITY_THRESHOLD = 0.3              # Minimum similarity score

# BM25 settings
BM25_K1 = 1.5                           # Controls term frequency saturation
BM25_B = 0.75                           # Controls length normalization

# ============================================================================
# LLM SETTINGS
# ============================================================================

class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OLLAMA = "ollama"
    GROQ = "groq"
    OPENAI = "openai"


LLM_PROVIDER = LLMProvider.OLLAMA
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama2:13b"             # or llama2:7b for lighter setup
OLLAMA_TIMEOUT = 120                    # Seconds

# Alternative: GROQ (requires API key)
# LLM_PROVIDER = LLMProvider.GROQ
# GROQ_MODEL = "mixtral-8x7b-32768"
# GROQ_API_KEY = from environment

# LLM generation parameters
LLM_TEMPERATURE = 0.7                   # 0=deterministic, 1=creative
LLM_MAX_TOKENS = 512                    # Max response length
LLM_TOP_P = 0.95                        # Nucleus sampling
LLM_TOP_K = 40                          # Top-k sampling

# ============================================================================
# RAG CHAIN SETTINGS
# ============================================================================

CONTEXT_WINDOW_SIZE = 3000              # Characters of context for LLM
MAX_CONTEXT_DOCUMENTS = 5               # Max documents in context
INCLUDE_METADATA_IN_PROMPT = True       # Include source info in prompt

# ============================================================================
# LANGUAGE SETTINGS
# ============================================================================

class Language(str, Enum):
    """Supported languages."""

    ENGLISH = "en"
    SPANISH = "es"
    NAVAJO = "nv"


SUPPORTED_LANGUAGES = [Language.ENGLISH, Language.SPANISH, Language.NAVAJO]
DEFAULT_LANGUAGE = Language.ENGLISH

# Language detection confidence threshold
LANGUAGE_DETECTION_THRESHOLD = 0.5

# ============================================================================
# API SETTINGS
# ============================================================================

API_TITLE = "Jericho Chatbot API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Enterprise Multi-Domain Chatbot with Open-Source LLM"

# CORS settings
CORS_ORIGINS = ["http://localhost:3000", "http://localhost:8000"]
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]

# ============================================================================
# DATABASE SETTINGS
# ============================================================================

METADATA_DB_PATH = DATA_DIR / "metadata.db"
METADATA_DB_URL = f"sqlite:///{METADATA_DB_PATH}"

# ============================================================================
# CACHE SETTINGS
# ============================================================================

CACHE_ENABLED = True
CACHE_TTL_SECONDS = 3600               # 1 hour
EMBEDDINGS_CACHE_SIZE = 1000            # Max cached embeddings

# ============================================================================
# PERFORMANCE & MONITORING
# ============================================================================

REQUEST_TIMEOUT_SECONDS = 30
LOG_LEVEL = "INFO"                      # DEBUG, INFO, WARNING, ERROR
ENABLE_QUERY_LOGGING = True
ENABLE_PERFORMANCE_METRICS = True

# ============================================================================
# VALIDATION & LIMITS
# ============================================================================

MIN_QUERY_LENGTH = 3                    # Minimum query character length
MAX_QUERY_LENGTH = 1000                 # Maximum query character length
MAX_HISTORY_ITEMS = 10                  # Q&A history to maintain
DOCUMENT_TITLE_MAX_LENGTH = 255         # Max document title length

# ============================================================================
# OCR SETTINGS
# ============================================================================

ENABLE_OCR = True
TESSERACT_PATH = None                   # Auto-detect, or set explicit path
OCR_LANGUAGE = "eng"                    # Tesseract language code
OCR_CONFIG = "--psm 3"                  # Page segmentation mode

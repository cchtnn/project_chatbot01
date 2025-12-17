"""
API request/response Pydantic schemas.

Defines all FastAPI endpoint input/output models with validation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum

from core.constants import (
    DocumentType,
    Language,
    RetrievalMode,
    MIN_QUERY_LENGTH,
    MAX_QUERY_LENGTH,
)


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ChatRequest(BaseModel):
    """Chat query request schema."""
    query: str = Field(..., min_length=MIN_QUERY_LENGTH, max_length=MAX_QUERY_LENGTH)
    language: Optional[Language] = Field(default=Language.ENGLISH)
    retrieval_mode: Optional[RetrievalMode] = Field(default=RetrievalMode.HYBRID)
    top_k: Optional[int] = Field(default=5, ge=1, le=10)
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list, max_items=10)

    @validator("query")
    def validate_query(cls, v: str) -> str:
        """Ensure query is not empty or whitespace."""
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty")
        return v


class ChatResponse(BaseModel):
    """Chat query response schema."""
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    language: Language
    processing_time_ms: float
    retrieval_mode: RetrievalMode
    total_chunks: int = 0


class SourceReference(BaseModel):
    """Single source document reference."""
    document_id: str
    document_name: str
    chunk_id: str
    content: str = Field(..., max_length=500)
    page_number: Optional[int] = None
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = {}


class DocumentUploadRequest(BaseModel):
    """Document upload request schema."""
    files: List[str]  # File paths or URLs
    process_immediately: bool = Field(default=True)


class DocumentUploadResponse(BaseModel):
    """Document upload response schema."""
    success: bool
    documents: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    total_processed: int = 0
    total_failed: int = 0


class DocumentInfo(BaseModel):
    """Document metadata response."""
    document_id: str
    filename: str
    document_type: DocumentType
    size_bytes: int
    status: DocumentStatus
    chunks_count: int = 0
    created_at: datetime
    processed_at: Optional[datetime] = None
    language: Optional[Language] = None
    error_message: Optional[str] = None


class KnowledgeBaseResponse(BaseModel):
    """Knowledge base status response."""
    total_documents: int = 0
    total_chunks: int = 0
    supported_languages: List[Language]
    documents: List[DocumentInfo] = Field(default_factory=list)


class RefreshRequest(BaseModel):
    """Knowledge base refresh request."""
    documents: Optional[List[str]] = None  # Specific document IDs
    full_refresh: bool = Field(default=False)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str
    llm_provider: str
    embedding_model: str
    uptime: float  # seconds
    total_documents: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# models/schemas.py - ADD THESE

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum

class OrchestratorResponse(BaseModel):
    """Final orchestrator output."""
    answer: str                             # Rendered Markdown (text/tables/sections)
    tools_used: List[str]                   # ["TranscriptTool", "GenericRagTool"]
    confidence: float
    sources: List[Dict[str, Any]]           # Your existing source format
    format_type: str                        # "text", "table", "sections"

from typing import Dict, Any, List
from pydantic import BaseModel, Field

from typing import Dict, Any, List
from pydantic import BaseModel, Field

class ToolResult(BaseModel):
    """Generic tool response for orchestrator tools."""
    data: Dict[str, Any]
    explanation: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    # Pydantic v2: use pattern instead of regex
    format_hint: str = Field(..., pattern="^(text|table|sections|list)$")
    citations: List[str] = Field(default_factory=list)
    navajo_segments: List[str] = Field(default_factory=list)




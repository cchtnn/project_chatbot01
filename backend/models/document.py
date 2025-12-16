"""
Document and chunk metadata models for database storage.

SQLAlchemy models for SQLite metadata store + Pydantic for API.
"""

from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pydantic import BaseModel, ConfigDict

from core.constants import DocumentType, Language


Base = declarative_base()


# Pure SQLAlchemy ORM Models (NO Pydantic inheritance)
class DocumentMetadataORM(Base):
    """SQLAlchemy model for document metadata."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(String(255), index=True, unique=True)
    filename = Column(String(500))
    document_type = Column(String(20))
    file_path = Column(String(1000))
    file_hash = Column(String(64))
    size_bytes = Column(Integer)
    language = Column(String(10), nullable=True)
    status = Column(String(20), default="pending")
    error_message = Column(Text, nullable=True)
    total_chunks = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    
    chunks = relationship("ChunkMetadataORM", back_populates="document")


class ChunkMetadataORM(Base):
    """SQLAlchemy model for chunk metadata."""
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(String(255), index=True)
    document_id = Column(String(255), ForeignKey("documents.document_id"), index=True)
    content = Column(Text)
    chunk_index = Column(Integer)
    page_number = Column(Integer, nullable=True)
    chunk_metadata = Column(Text, nullable=True)  # FIXED: was 'metadata'
    embedding_id = Column(String(255), index=True, unique=True, nullable=True)
    
    document = relationship("DocumentMetadataORM", back_populates="chunks")


# Pure Pydantic models for API (NO SQLAlchemy inheritance)
class DocumentMetadataBase(BaseModel):
    """Base document metadata schema."""
    document_id: str
    filename: str
    document_type: DocumentType
    file_path: str
    file_hash: str
    size_bytes: int
    language: Optional[Language] = None
    status: str = "pending"
    error_message: Optional[str] = None
    total_chunks: int = 0
    created_at: datetime
    processed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class ChunkMetadataBase(BaseModel):
    """Base chunk metadata schema."""
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    page_number: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = {}
    embedding_id: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


DocumentMetadata = DocumentMetadataBase
ChunkMetadata = ChunkMetadataBase


class DocumentStatusUpdate(BaseModel):
    """Update document status request."""
    status: str
    total_chunks: Optional[int] = None
    error_message: Optional[str] = None
    processed_at: Optional[datetime] = None

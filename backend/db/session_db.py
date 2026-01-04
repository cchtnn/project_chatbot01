from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any


from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Text,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session


# Base dir is backend/
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "session.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"


engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


Base = declarative_base()


# Add UserFeedback AFTER ChatSession & ChatMessage classes (ORDER MATTERS)

class SessionDocument(Base):
    __tablename__ = 'session_documents'
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey('chatsessions.id'), index=True)
    file_hash = Column(String, index=True)  # Links to ChromaDB
    filename = Column(String)
    filepath = Column(String)  # Full path in data/documents/
    uploaded_at = Column(DateTime, default=datetime.utcnow)

class ChatSession(Base):
    __tablename__ = 'chatsessions'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, index=True)
    sessionname = Column(String, default='New Chat')
    createdat = Column(DateTime, default=datetime.utcnow)
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = 'chatmessages'
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey('chatsessions.id'), index=True)
    question = Column(Text)
    answer = Column(Text)
    createdat = Column(DateTime, default=datetime.utcnow)
    session = relationship("ChatSession", back_populates="messages")


# ADD THIS LAST (after ChatSession/ChatMessage)
class UserFeedback(Base):
    __tablename__ = 'user_feedback'
    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer)
    session_id = Column(Integer, ForeignKey('chatsessions.id'))  # References existing table
    rating = Column(String)  # 'like', 'dislike'
    username = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


def add_feedback(message_id: int, session_id: int, username: str, rating: str):
    db = get_db()
    try:
        fb = UserFeedback(
            message_id=message_id,
            session_id=session_id,
            username=username,
            rating=rating
        )
        db.add(fb)
        db.commit()
    finally:
        db.close()


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    return SessionLocal()


# --- High-level helpers used by routes.py ---


def create_session(username: str, session_name: str = "New Chat") -> int:
    db = get_db()
    try:
        obj = ChatSession(username=username, sessionname=session_name)
        db.add(obj)
        db.commit()
        db.refresh(obj)
        return obj.id
    finally:
        db.close()


def list_sessions_for_user(username: str) -> List[dict]:
    db = get_db()
    try:
        rows = (
            db.query(ChatSession)
            .filter(ChatSession.username == username)
            .order_by(ChatSession.createdat.desc())
            .all()
        )
        return [
            {"session_id": s.id, "session_name": s.sessionname}
            for s in rows
        ]
    finally:
        db.close()


def add_message(session_id: int, question: str, answer: str) -> None:
    db = get_db()
    try:
        msg = ChatMessage(session_id=session_id, question=question, answer=answer)
        db.add(msg)
        db.commit()
    finally:
        db.close()


def get_history(session_id: int) -> List[Tuple[str, str]]:
    """
    FIXED: Changed created_at --> createdat to match ChatMessage model
    """
    db = get_db()
    try:
        msgs = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.createdat.asc())  # FIXED: was created_at
            .all()
        )
        return [(m.question, m.answer) for m in msgs]
    finally:
        db.close()


def count_messages(session_id: int) -> int:
    db = get_db()
    try:
        return (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .count()
        )
    finally:
        db.close()


def rename_session(session_id: int, new_name: str) -> None:
    """
    FIXED: Changed session_name --> sessionname to match ChatSession model
    """
    db = get_db()
    try:
        s = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if s:
            s.sessionname = new_name  # FIXED: was session_name
            db.commit()
    finally:
        db.close()


def delete_session(session_id: int) -> None:
    db = get_db()
    try:
        s = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if s:
            db.delete(s)
            db.commit()
    finally:
        db.close()

def add_document_to_session(session_id: int, file_hash: str, filename: str, filepath: str) -> None:
    """Link uploaded document to session."""
    db = get_db()
    try:
        doc = SessionDocument(
            session_id=session_id,
            file_hash=file_hash,
            filename=filename,
            filepath=filepath
        )
        db.add(doc)
        db.commit()
    finally:
        db.close()

from core import get_logger
logger = get_logger(__name__)

def get_session_file_hashes(session_id: int) -> List[str]:
    """Get all file hashes for a session (for filtering queries)."""
    db = get_db()
    try:
        docs = (
            db.query(SessionDocument)
            .filter(SessionDocument.session_id == session_id)
            .all()
        )
        logger.info(
            f"[SessionDB] Session {session_id} has {len(docs)} linked documents: "
            f"{[d.filename for d in docs]}"
        )
        return [d.file_hash for d in docs]
    finally:
        db.close()


def get_session_documents(session_id: int) -> List[Dict[str, Any]]:
    """Get all document info for a session."""
    db = get_db()
    try:
        docs = (
            db.query(SessionDocument)
            .filter(SessionDocument.session_id == session_id)
            .all()
        )
        return [
            {
                "file_hash": d.file_hash,
                "filename": d.filename,
                "filepath": d.filepath,
                "uploaded_at": d.uploaded_at
            }
            for d in docs
        ]
    finally:
        db.close()


def cleanup_session_documents(session_id: int) -> List[str]:
    """
    Get file hashes and filepaths for cleanup when session is deleted.
    Returns list of (file_hash, filepath) tuples.
    """
    db = get_db()
    try:
        docs = (
            db.query(SessionDocument)
            .filter(SessionDocument.session_id == session_id)
            .all()
        )
        cleanup_data = [(d.file_hash, d.filepath) for d in docs]
        
        # Delete records
        for doc in docs:
            db.delete(doc)
        db.commit()
        
        return cleanup_data
    finally:
        db.close()

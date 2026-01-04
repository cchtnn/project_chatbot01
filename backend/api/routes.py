"""
backend/api/routes.py
Jericho API routes - ENHANCED with Context Resolution

- RAG endpoints (hybrid retrieval + generation)
- UI-compatible endpoints expected by chat.html / login.html / admin.html
- PHASE 2: Context-aware query processing with session history
"""

from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import (
    APIRouter,
    Form,
    UploadFile,
    File,
    HTTPException,
    Depends,
    Response,
)
from pydantic import BaseModel

from api.deps import get_current_user
from core import get_logger
from services.rag_pipeline import RAGPipeline
from core.retrieval import get_hybrid_retriever
from db import session_db

from services.orchestrator import Orchestrator, OrchestratorRequest
from services.context_resolver import ContextResolver  # NEW

logger = get_logger(__name__)
router = APIRouter()

# ---------- RAG PIPELINE INITIALIZATION ----------

rag_pipeline = RAGPipeline()
hybrid_retriever = get_hybrid_retriever()

# Global singleton orchestrator
orchestrator = Orchestrator(rag_pipeline=rag_pipeline)

# Global context resolver (NEW)
context_resolver = ContextResolver()

# ---------- SIMPLE RAG API (TEST / DEBUG) ----------

class SimpleQueryRequest(BaseModel):
    query: str
    session_id: int = 1


@router.get("/query_simple")
async def query_simple(query: str, session_id: int = 1):
    """
    Simple GET query endpoint used for earlier testing.
    Kept for debugging; UI uses POST /query.
    """
    try:
        logger.info(f"[query_simple] q='{query}' session={session_id}")

        result = rag_pipeline.query(question=query, top_k=5)
        answer = result.get("answer", "")
        sources = result.get("sources", [])

        return {
            "answer": answer,
            "sources": [
                {
                    "content": s.get("content", "")[:200] + "...",
                    "filename": s.get("filename", "unknown"),
                }
                for s in sources
            ],
            "session_id": session_id,
        }
    except Exception as e:
        logger.error(f"[query_simple] error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------- UI-COMPATIBLE SESSION ENDPOINTS (DB-BACKED) ----------

@router.post("/new_session")
async def new_session(user=Depends(get_current_user)):
    """
    Create a new chat session for the current user.

    JS usage in chat.html:
    - POST /new_session
    - Returns: {"session_id": int}
    """
    username = getattr(user, "username", "guest")
    session_id = session_db.create_session(username=username)
    logger.info(f"[new_session] user={username} session_id={session_id}")
    return {"session_id": session_id}


@router.get("/user_sessions")
async def user_sessions(user=Depends(get_current_user)):
    """
    Return list of sessions for current user.

    - GET /user_sessions
    - Returns: {"sessions": [{session_id, session_name}, ...]}
    """
    username = getattr(user, "username", "guest")
    sessions = session_db.list_sessions_for_user(username=username)
    return {"sessions": sessions}


@router.get("/history")
async def history(session_id: int, user=Depends(get_current_user)):
    """
    Return Q&A history for a session.

    JS calls: GET /history?session_id=...
    Response: {"history": [{"question": "...", "answer": "..."}, ...]}
    """
    records = session_db.get_history(session_id)
    return {
        "history": [
            {"question": q, "answer": a}
            for (q, a) in records
        ]
    }


@router.post("/rename_session")
async def rename_session(
    session_id: int = Form(...),
    new_name: str = Form(...),
    user=Depends(get_current_user),
):
    """
    Rename a session (used by rename modal in chat.html).
    """
    new_name_clean = new_name.strip() or "New Chat"
    session_db.rename_session(session_id, new_name_clean)
    logger.info(f"[rename_session] session_id={session_id} new_name='{new_name_clean}'")
    return {"success": True}


@router.post("/delete_session")
async def delete_session(
    session_id: int = Form(...),
    user=Depends(get_current_user),
):
    """
    Delete session AND cleanup associated documents.
    """
    try:
        # Get document cleanup data
        cleanup_data = session_db.cleanup_session_documents(session_id)
        
        # Delete from ChromaDB
        file_hashes = [fh for fh, _ in cleanup_data]
        if file_hashes:
            from db.chromadb_manager import ChromaDBManager
            db_manager = ChromaDBManager()
            db_manager.delete_by_file_hashes(file_hashes)
        
        # Delete physical files
        import os
        for _, filepath in cleanup_data:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.info(f"Deleted file: {filepath}")
            except Exception as e:
                logger.error(f"Failed to delete file {filepath}: {e}")
        
        # Delete session-specific directory if empty
        session_dir = Path("data/documents") / f"session_{session_id}"
        if session_dir.exists() and not any(session_dir.iterdir()):
            session_dir.rmdir()
            logger.info(f"Deleted empty session directory: {session_dir}")
        
        # Delete session from database
        session_db.delete_session(session_id)
        
        logger.info(f"[delete_session] Deleted session {session_id} with {len(cleanup_data)} documents")
        return {"success": True, "documents_deleted": len(cleanup_data)}
        
    except Exception as e:
        logger.error(f"[delete_session] error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Session deletion failed")


# ---------- CORE RAG /query USED BY UI (WITH HISTORY + CONTEXT) ----------

@router.post("/query")
async def query_endpoint(
    query: str = Form(...),
    session_id: int = Form(...),
    private: bool = Form(False),
    user=Depends(get_current_user),
):
    """
    Main chat endpoint used by chat.html JS.
    ENHANCED: Context-aware with session history (Phase 2)

    Frontend sends FormData:
      - query
      - session_id
      - private

    Response is expected to contain:
      - answer
      - session_id
      - sources
      - tools_used
      - confidence
      - sessionnameupdated (bool)
    """
    username = getattr(user, "username", "test_user")
    logger.info(f"[query] user={username} q={query!r} session_id={session_id} private={private}")

    try:
        # PHASE 2: Fetch session history and resolve context
        history_records = session_db.get_history(session_id)
        conversation_history = [
            {"question": q, "answer": a}
            for (q, a) in history_records
        ]
        
        # Resolve ambiguous references using context
        enriched_query = context_resolver.resolve(query, conversation_history)

        # PHASE 3: Get session document hashes for filtering
        session_file_hashes = session_db.get_session_file_hashes(session_id)
        # LOG: Session documents being used for filtering
        session_docs_info = session_db.get_session_documents(session_id)
        logger.info(f"[query] Session {session_id} documents:")
        for doc in session_docs_info:
            logger.info(
                f"  - {doc['filename']} (hash: {doc['file_hash'][:16]}...) "
                f"at {doc['filepath']}"
            )
        
        if not session_file_hashes:
            logger.warning(f"[query] No documents in session {session_id}")
            return {
                "answer": "No documents have been uploaded to this session yet. Please upload documents first.",
                "session_id": session_id,
                "sources": [],
                "tools_used": [],
                "confidence": 0.0,
                "sessionnameupdated": False,
            }
        
        logger.info(f"[query] Filtering by {len(session_file_hashes)} session documents")
        
        if enriched_query != query:
            logger.info(f"[query] Context enriched: '{query}' --> '{enriched_query}'")
        
        # Delegate to orchestrator WITH HISTORY AND SESSION FILTERS
        request = OrchestratorRequest(
            query=enriched_query,
            conversation_history=conversation_history,
            session_file_hashes=session_file_hashes  # NEW
        )
        response = orchestrator.handle_query(request)
        answer_text = response.answer

        # Persist message in DB
        before_count = session_db.count_messages(session_id)
        session_db.add_message(session_id=session_id, question=query, answer=answer_text)

        # Auto-rename session on first question
        sessionnameupdated = False
        if before_count == 0:
            words = query.strip().split()
            base = " ".join(words[:6]) if words else "New Chat"
            nice_name = base[:40].rstrip()
            session_db.rename_session(session_id, nice_name)
            sessionnameupdated = True

        return {
            "answer": answer_text,
            "session_id": session_id,
            "sources": response.sources,
            "tools_used": response.tools_used,
            "confidence": response.confidence,
            "sessionnameupdated": sessionnameupdated,
        }
    except Exception as e:
        logger.error(f"[query] error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error while answering query")


# ---------- FILE UPLOAD (USED BY UI) ----------

@router.post("/upload")
async def upload_endpoint(
    files: List[UploadFile] = File(...),
    session_id: int = Form(...),
    private: bool = Form(False),
    user=Depends(get_current_user),
):
    """
    Upload endpoint - SESSION-SCOPED.
    Documents are linked to the session and only accessible within that session.
    """
    try:
        username = getattr(user, 'username', 'guest')
        logger.info(
            f"[upload] user={username} session_id={session_id} "
            f"files={[f.filename for f in files]}"
        )

        # Save files to session-specific subdirectory
        base_dir = Path("data/documents") / f"session_{session_id}"
        base_dir.mkdir(parents=True, exist_ok=True)

        paths: List[str] = []
        for file in files:
            dest = base_dir / file.filename
            with dest.open("wb") as f:
                f.write(await file.read())
            paths.append(str(dest))

        # LOG: Documents being ingested
        logger.info(f"[upload] FILES TO INGEST: {paths}")
        logger.info(f"[upload] Target session directory: {base_dir}")
        # Ingest into RAG pipeline
        stats = rag_pipeline.ingest_documents(paths)

        # Link documents to session in database
        import hashlib
        for path in paths:
            file_path = Path(path)
            # Generate file hash (same as used in ChromaDB)
            with open(path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
                logger.info(
                f"[upload] Linking document to session: "
                f"session_id={session_id}, file_hash={file_hash[:16]}..., "
                f"filename={file_path.name}, filepath={path}"
            )
            
            session_db.add_document_to_session(
                session_id=session_id,
                file_hash=file_hash,
                filename=file_path.name,
                filepath=path
            )

        message = f"Processed {stats.get('processed', 0)} file(s), failed {stats.get('failed', 0)}."
        return {
            "success": True,
            "message": message,
            "processed_files": [Path(p).name for p in paths],
            "errors": [] if stats.get("failed", 0) == 0 else ["Some files failed to ingest."],
        }
    except Exception as e:
        logger.error(f"[upload] error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Upload/ingest failed")


# ---------- REACT-FACING QUERY (ORCHESTRATOR + CONTEXT) ----------

class ReactChatRequest(BaseModel):
    query: str
    sessionid: int = 1
    private: bool = False


@router.post("/react-query")
async def react_query(
    req: ReactChatRequest,
    user=Depends(get_current_user),
):
    """
    React-facing query endpoint.
    ENHANCED: Context-aware with session history (Phase 2)

    Uses the Orchestrator so queries are routed to the right domain tool,
    and persists Q&A into session_db keyed by sessionid.
    """
    try:
        username = getattr(user, "username", "admin")
        logger.info(
            f"[react-query] user={username} q={req.query!r} session={req.sessionid}"
        )

        # PHASE 2: Fetch session history and resolve context
        history_records = session_db.get_history(req.sessionid)
        conversation_history = [
            {"question": q, "answer": a}
            for (q, a) in history_records
        ]
        
        # Resolve ambiguous references using context
        enriched_query = context_resolver.resolve(req.query, conversation_history)

        # PHASE 3: Get session document hashes for filtering
        session_file_hashes = session_db.get_session_file_hashes(req.sessionid)
        
        if not session_file_hashes:
            logger.warning(f"[react-query] No documents in session {req.sessionid}")
            return {
                "answer": "No documents have been uploaded to this session yet. Please upload documents first.",
                "sessionid": req.sessionid,
                "sources": [],
                "tools_used": [],
                "confidence": 0.0,
                "sessionnameupdated": False,
            }
        
        logger.info(f"[react-query] Filtering by {len(session_file_hashes)} session documents")
        
        if enriched_query != req.query:
            logger.info(f"[react-query] Context enriched: '{req.query}' --> '{enriched_query}'")

        # Orchestrator call WITH HISTORY AND SESSION FILTERS
        orch_request = OrchestratorRequest(
            query=enriched_query,
            conversation_history=conversation_history,
            session_file_hashes=session_file_hashes  # NEW
        )
        orch_response = orchestrator.handle_query(orch_request)
        answer_text = orch_response.answer

        # Persist message in DB for this session
        before_count = session_db.count_messages(req.sessionid)
        session_db.add_message(
            session_id=req.sessionid,
            question=req.query,
            answer=answer_text,
        )

        # Auto-rename session on first question
        sessionnameupdated = False
        if before_count == 0:
            words = req.query.strip().split()
            base = " ".join(words[:6]) if words else "New Chat"
            nice_name = base[:40].rstrip()
            session_db.rename_session(req.sessionid, nice_name)
            sessionnameupdated = True

        return {
            "answer": answer_text,
            "sessionid": req.sessionid,
            "tools_used": orch_response.tools_used,
            "confidence": orch_response.confidence,
            "sources": orch_response.sources,
            "sessionnameupdated": sessionnameupdated,
        }
    except Exception as e:
        logger.error(f"[react-query] orchestrator error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Query failed")


# ---------- LOGOUT ----------

@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie("access_token")
    return {"success": True}


@router.post("/feedback")
async def submit_feedback(
    message_id: int = Form(...),
    rating: str = Form(...),
    session_id: int = Form(...),
    user: dict = Depends(get_current_user),
):
    """Submit user feedback (like/dislike) for answer"""
    username = getattr(user, 'username', 'guest')
    try:
        session_db.add_feedback(message_id, session_id, username, rating)
        logger.info(f"Feedback: {username} rated message {message_id} as {rating}")
        return {"success": True}
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail="Feedback failed")
    
@router.get("/debug/chromadb-contents")
async def debug_chromadb_contents(user=Depends(get_current_user)):
    """DEBUG: Show all documents in ChromaDB with their hashes."""
    try:
        from db.chromadb_manager import ChromaDBManager
        db = ChromaDBManager()
        
        # Get all chunks
        all_data = db.collection.get(include=["metadatas"])
        metadatas = all_data.get("metadatas", [])
        
        # Group by file_hash
        docs_by_hash = {}
        for meta in metadatas:
            fh = meta.get("file_hash", "unknown")
            fn = meta.get("filename", "unknown")
            if fh not in docs_by_hash:
                docs_by_hash[fh] = {"filename": fn, "chunks": 0}
            docs_by_hash[fh]["chunks"] += 1
        
        return {
            "total_chunks": len(metadatas),
            "unique_documents": len(docs_by_hash),
            "documents": [
                {
                    "file_hash": fh[:16] + "...",
                    "filename": info["filename"],
                    "chunk_count": info["chunks"]
                }
                for fh, info in docs_by_hash.items()
            ]
        }
    except Exception as e:
        logger.error(f"[debug] ChromaDB contents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debug/session-documents/{session_id}")
async def debug_session_documents(session_id: int, user=Depends(get_current_user)):
    """DEBUG: Show which documents are linked to a session."""
    try:
        docs = session_db.get_session_documents(session_id)
        hashes = session_db.get_session_file_hashes(session_id)
        
        return {
            "session_id": session_id,
            "document_count": len(docs),
            "documents": docs,
            "file_hashes": [h[:16] + "..." for h in hashes]
        }
    except Exception as e:
        logger.error(f"[debug] Session documents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

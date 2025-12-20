# backend/api/admin_routes.py

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict

from core import get_logger
from api.deps import get_current_user
from services.rag_pipeline import RAGPipeline

logger = get_logger(__name__)
router = APIRouter()

# If you already have a global rag_pipeline somewhere, you can import and reuse it.
# Here we instantiate a lightweight one just for stats if needed.
rag_pipeline = RAGPipeline()


@router.get("/admin/stats")
async def admin_stats(user=Depends(get_current_user)) -> Dict:
    """
    Basic RAG / document stats for the admin console.

    Requires:
    - Authenticated user
    - user.role == 'admin' (adjust according to your user model)
    """
    # Adjust this according to your user object structure
    role = getattr(user, "role", None)
    if role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        # You may already have stats in your pipeline or vector store.
        # Here we use a generic interface; adapt to your actual implementation.

        stats = rag_pipeline.get_stats() if hasattr(rag_pipeline, "get_stats") else {}

        # Expected shape example (you can change internals but keep outer structure):
        # {
        #   "documents": {
        #       "total": int,
        #       "by_type": {"pdf": int, "docx": int, "csv": int, "other": int}
        #   }
        # }

        return {
            "documents": stats.get("documents", {}),
        }
    except Exception as e:
        logger.error(f"[admin_stats] error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load admin stats")

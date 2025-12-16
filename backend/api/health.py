from fastapi import APIRouter
from db.chromadb_manager import ChromaDBManager

router = APIRouter()
db = ChromaDBManager()

@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "chunks": db.collection.count(),
        "providers": ["groq", "ollama"]
    }

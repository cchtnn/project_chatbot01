"""Generic RAG tool - your existing hybrid retrieval."""
from typing import Dict, Any, List
from core.retrieval import get_hybrid_retriever
from services.rag_pipeline import RAGPipeline  # Your existing class
from core import get_logger
from models.schemas import ToolResult

logger = get_logger(__name__)

def answer(question: str, params: Dict[str, Any] = None) -> ToolResult:
    """Generic hybrid RAG over all documents."""
    retriever = get_hybrid_retriever()
    rag_pipeline = RAGPipeline()  # Your existing pipeline
    
    # Retrieve with optional domain filter
    domain_filter = params.get("domain_filter")
    filters = {"domain": domain_filter} if domain_filter else None
    
    results = retriever.retrieve(question, top_k=5, filters=filters)
    
    # Similarity threshold for "not found"
    avg_similarity = sum(r.score for r in results) / len(results)
    if avg_similarity < 0.3:
        return ToolResult(
            data={},
            explanation="The documents do not contain information about this question.",
            confidence=0.1,
            format_hint="text",
            citations=[]
        )
    
    context = "\n\n".join(r.content for r in results)
    answer = rag_pipeline._generate_answer(context, question)  # Your existing method
    
    sources = [{"content": r.content[:200], "filename": r.metadata.get("filename")} for r in results]
    
    return ToolResult(
        data={"retrieved_chunks": len(results)},
        explanation=answer,
        confidence=min(avg_similarity, 1.0),
        format_hint="text",
        citations=[r.metadata.get("filename", "unknown") for r in results]
    )

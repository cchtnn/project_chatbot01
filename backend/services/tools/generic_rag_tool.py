"""
backend/services/tools/generic_rag_tool.py
Generic RAG tool - PRODUCTION READY with normalized similarity.
"""
from typing import Dict, Any, List
from core.retrieval import get_hybrid_retriever
from services.rag_pipeline import RAGPipeline
from core import get_logger
from models.schemas import ToolResult

logger = get_logger(__name__)

def normalize_distance(raw_distance: float) -> float:
    """
    Normalize cross-encoder rerank scores to 0-1 confidence range.
    Cross-encoder scores typically range from -10 (irrelevant) to +10 (highly relevant).
    """
    # Handle cross-encoder rerank scores (from retrieval.py)
    if raw_distance >= 0.5:
        return 0.90  # Highly relevant
    elif raw_distance >= 0.3:
        return 0.75  # Good match
    elif raw_distance >= 0.0:
        return 0.55  # Fair match
    elif raw_distance >= -0.3:
        return 0.35  # Poor match
    else:
        return 0.15  # Irrelevant


def smart_dedup(results: List, max_per_doc: int = 2, max_total: int = 8) -> List:
    """Smart deduplication: max 2/doc, total 8, preserve best chunks."""
    from collections import defaultdict
    
    doc_chunks = defaultdict(list)
    for r in results:
        filename = r.metadata.get("filename", "unknown")
        doc_chunks[filename].append(r)
    
    # Take top N per document (best normalized score first)
    diverse = []
    for filename, chunks in doc_chunks.items():
        sorted_chunks = sorted(chunks, key=lambda r: normalize_distance(r.score), reverse=True)
        diverse.extend(sorted_chunks[:max_per_doc])
    
    # Final top 8 (best overall)
    return sorted(diverse, key=lambda r: normalize_distance(r.score), reverse=True)[:max_total]

def answer(question: str, params: Dict[str, Any] = None) -> ToolResult:
    """Generic hybrid RAG - Optimized for 10+ chunk retrieval."""
    retriever = get_hybrid_retriever()
    rag_pipeline = RAGPipeline()

    params = params or {}
    filters = params.get("filters")
    if filters is not None and len(filters) == 0:
        filters = None  # Convert empty dict to None

    logger.info(f"[GenericRagTool] question='{question}' filters={filters}")

    # Retrieve 10+ chunks
    all_results = retriever.retrieve(question, top_k=12, filters=filters)
    
    logger.info(f"[Retriever] returned {len(all_results)} raw chunks")
    
    # LOG: Document sources BEFORE deduplication
    logger.info(f"[GenericRagTool] Retrieved chunks from sources (before dedup):")
    for idx, chunk in enumerate(all_results):
        meta = chunk.metadata  # It's an attribute, not a dict key
        logger.info(
            f"  [{idx+1}] {meta.get('filename', 'Unknown')} "
            f"(hash: {meta.get('file_hash', 'N/A')[:16]}...)"
        )  

    # Smart deduplication
    results = smart_dedup(all_results, max_per_doc=2, max_total=8)
    filenames = [r.metadata.get("filename", "unknown") for r in results]
    logger.info(f"[GenericRagTool] SMART DEDUP --> {len(results)} chunks: {filenames}")

    if not results:
        return ToolResult(
            data={"retrieved_chunks": 0},
            explanation="The documents do not contain information about this question.",
            confidence=0.1,
            format_hint="text",
            citations=[],
            sources=[],
        )

    # Normalized similarity
    raw_scores = [r.score for r in results]
    raw_avg = sum(raw_scores) / len(results)
    norm_similarity = normalize_distance(raw_avg)

    norm_scores = [normalize_distance(s) for s in raw_scores]
    logger.info(
        f"[Scores] raw_avg={raw_avg:.3f} --> norm={norm_similarity:.3f} "
        f"(range: {min(norm_scores):.3f}-{max(norm_scores):.3f})"
    )

    # QUALITY GATE: Check if document actually contains relevant information
    DOCUMENT_RELEVANCE_THRESHOLD = 0.50  # Minimum confidence that document is relevant
    if norm_similarity < DOCUMENT_RELEVANCE_THRESHOLD:
        logger.warning(
            f"[GenericRagTool] Document relevance too low (norm={norm_similarity:.3f} < {DOCUMENT_RELEVANCE_THRESHOLD}). "
            f"Retrieved content not relevant to query."
        )
        return ToolResult(
            data={"retrieved_chunks": len(results), "norm_similarity": norm_similarity, "relevance_check": "failed"},
            explanation=(
                "The uploaded document does not contain information relevant to your question. "
                "Please ask a question related to the document content, or upload a different document that covers this topic."
            ),
            confidence=0.15,
            format_hint="text",
            citations=[],
            sources=[],
        )

    # Generate answer
    context = "\n\n".join(r.content for r in results)
    answer_text = rag_pipeline._generate_answer(context, question)

    # PHASE 3: Dynamic format_hint detection (NEW)
    format_hint = "text"  # default
    # If query asks for comparison/listing and we have multiple results
    if any(kw in question.lower() for kw in ['compare', 'list all', 'show all', 'table', 'versus', 'vs']) and len(results) >= 5:
        format_hint = "table"

    sources = [
        {
            "content": r.content[:200] + "...",
            "filename": r.metadata.get("filename", "unknown"),
            "raw_score": float(r.score),
            "norm_score": normalize_distance(r.score),
        }
        for r in results
    ]

    return ToolResult(
        data={
            "retrieved_chunks": len(results),
            "raw_avg": raw_avg,
            "norm_similarity": norm_similarity
        },
        explanation=answer_text,
        confidence=min(0.95, norm_similarity + 0.25),
        format_hint=format_hint,  # Now dynamic
        citations=filenames,
        sources=sources,
    )

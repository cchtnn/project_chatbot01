from typing import Dict, Any
from core import get_logger
from services.rag_pipeline import RAGPipeline
from models.schemas import ToolResult

logger = get_logger(__name__)


def answer(question: str, params: Dict[str, Any] | None = None) -> ToolResult:
    params = params or {}

    rag = RAGPipeline()

    # Use existing metadata: filename == 'bor_json.json'
    filters = {"filename": "bor_json.json"}

    try:
        result = rag.query(question, top_k=5, filters=filters)
    except Exception as e:
        logger.error(f"BOR RAG query failed: {e}", exc_info=True)
        return ToolResult(
            data={},
            explanation="Board of Regents schedule data could not be accessed due to an internal error.",
            confidence=0.2,
            format_hint="text",
            citations=["bor_json.json"],
        )

    if isinstance(result, dict):
        answer_text = result.get("answer", "")
        sources = result.get("sources", [])
    else:
        try:
            answer_text = result.get("answer")
            sources = result.get("sources")
        except Exception:
            answer_text = str(result)
            sources = []

    if not answer_text:
        return ToolResult(
            data={"sources": sources},
            explanation=(
                "Board of Regents schedule data is available, but no specific answer "
                "could be found for this question. Please rephrase or add more detail."
            ),
            confidence=0.4,
            format_hint="text",
            citations=["bor_json.json"],
        )

    return ToolResult(
        data={"sources": sources},
        explanation=answer_text,
        confidence=0.85,
        format_hint="text",
        citations=["bor_json.json"],
    )

from core import init_logger
init_logger()
from services.rag_pipeline import RAGPipeline

rag = RAGPipeline()

for q in [
    "What is the check date for Pay period 3?",
    "How many payroll periods are in calendar year 2026?"
]:
    result = rag.query(q, top_k=10)
    print("Q:", q)
    print("  Agent:", result.get("agent"))
    print("  Answer:", result.get("answer", "")[:300])
    print()

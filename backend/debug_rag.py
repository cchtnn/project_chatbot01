from core import init_logger
init_logger()
from services.rag_pipeline import RAGPipeline

rag = RAGPipeline()

queries = [
    "What is the check date for Pay period 3",
    "How many payroll periods do we have in Current year 2026"
]

print("🔍 DEBUG: What chunks is RAG actually retrieving?\n")

for q in queries:
    print(f"QUERY: {q}")
    print("="*80)
    
    result = rag.query(q, top_k=5)  # Get TOP 5 sources
    sources = result.get('sources', [])
    
    print(f"Found {len(sources)} sources:")
    for i, source in enumerate(sources, 1):
        filename = source.get('filename', '?')
        content_preview = source.get('content', '')[:300]
        print(f"\n{i}. {filename}")
        print(f"   Preview: {content_preview}...")
    
    print(f"\nLLM Answer: {result.get('answer', 'No answer')[:200]}...\n")

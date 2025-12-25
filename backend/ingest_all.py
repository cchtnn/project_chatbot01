from pathlib import Path
import sys

# 1) Make sure backend root is on sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# 2) Initialize logging BEFORE importing anything from services/db
from core import init_logger, get_logger

init_logger()  # must run before any get_logger() inside services/*
logger = get_logger(__name__)

# 3) Now it's safe to import RAGPipeline
from services.rag_pipeline import RAGPipeline  # note: ragpipeline, no underscore


def main():
    docs_root = Path(r"C:\chtn\gen_ai\hitesh\jericho\project_chatbot01\data")
    paths = [str(p) for p in docs_root.rglob("*") if p.is_file()]
    logger.info(f"Found {len(paths)} files to ingest under {docs_root}")

    rag = RAGPipeline()
    stats = rag.ingest_documents(paths)
    logger.info(f"INGEST STATS: {stats}")


if __name__ == "__main__":
    main()

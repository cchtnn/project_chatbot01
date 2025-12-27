from pathlib import Path
import sys

# 1) Ensure backend root on sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# 2) Import helpers
from core import init_logger, get_logger
from config import get_settings  # ✅ ADD THIS

# Initialize logger
init_logger()
logger = get_logger(__name__)

# 3) Import RAGPipeline
from services.rag_pipeline import RAGPipeline

def main():
    # ✅ FLEXIBLE PATH - reads from .env
    settings = get_settings()
    docs_root = settings.documentsdir
    
    # Auto-create directory if missing
    if not docs_root.exists():
        docs_root.mkdir(parents=True, exist_ok=True)
        logger.warning(f"Created documents directory: {docs_root}")
    
    # Find all files
    paths = [str(p) for p in docs_root.rglob("*") if p.is_file()]
    logger.info(f"Found {len(paths)} files to ingest under {docs_root}")
    
    if not paths:
        logger.error(f"No files found in {docs_root}. Please add documents first.")
        return
    
    # Ingest
    rag = RAGPipeline()
    stats = rag.ingest_documents(paths)
    logger.info(f"INGEST STATS: {stats}")

if __name__ == "__main__":
    main()

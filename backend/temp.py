#!/usr/bin/env python3
"""
Inspect what was parsed from calendar PDF and stored in ChromaDB.
Saves chunks in page order to temp/ folder.
"""
from core import init_logger
init_logger()

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(__file__))

from db.chromadb_manager import ChromaDBManager

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def inspect_calendar_chunks():
    """Get all calendar PDF chunks in page order."""
    
    print("=" * 80)
    print("CALENDAR PDF - PARSED CHUNKS")
    print("=" * 80)
    
    # Create temp folder
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Initialize ChromaDB
    chroma = ChromaDBManager()
    
    # Get all chunks
    all_results = chroma.collection.get(
        limit=10000,
        include=["documents", "metadatas"]
    )
    
    # Filter calendar PDF only
    calendar_chunks = []
    for doc, meta in zip(all_results['documents'], all_results['metadatas']):
        filename = meta.get('filename', '')
        if '24-25-DC-CatalogFINAL' in filename and filename.endswith('.pdf'):
            calendar_chunks.append((doc, meta))
    
    if not calendar_chunks:
        print("\n❌ No calendar PDF chunks found!")
        return
    
    # Sort by chunk_id (page order)
    calendar_chunks.sort(key=lambda x: int(x[1].get('chunk_id', 0)))
    
    print(f"\n✅ Found {len(calendar_chunks)} chunks\n")
    
    # Save each chunk
    for i, (doc, meta) in enumerate(calendar_chunks, 1):
        page = meta.get('page', 'N/A')
        chunk_id = meta.get('chunk_id', i)
        
        filepath = os.path.join(temp_dir, f"chunk_{i:03d}.txt")
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Chunk {i}/{len(calendar_chunks)}\n")
            f.write(f"Page: {page}\n")
            f.write(f"Chunk ID: {chunk_id}\n")
            f.write("="*80 + "\n\n")
            f.write(doc)
        
        print(f"Chunk {i:3d} | Page {page:3s} | {len(doc):5d} chars → {filepath}")
    
    print(f"\n✅ Saved {len(calendar_chunks)} chunks to {temp_dir}/ folder")


if __name__ == "__main__":
    inspect_calendar_chunks()

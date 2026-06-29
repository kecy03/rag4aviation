from langchain_core.documents import Document

from ingestion.loader import load_documents
from ingestion.metadata import enrich_metadata
from ingestion.splitter import SemanticChunker


def run_ingestion_pipeline(source_dir: str) -> list[Document]:
    """
    1. Load all documents from source_dir
    2. Enrich metadata (document_type, aircraft_model, collection)
    3. Split into semantic chunks
    Returns fully-processed, metadata-rich chunks.
    """
    print(f"\n[1/3] Loading documents from: {source_dir}")
    raw_docs = load_documents(source_dir)
    if not raw_docs:
        print("  No documents found.")
        return []
    print(f"  Total pages/sections: {len(raw_docs)}")

    print("[2/3] Enriching metadata...")
    enriched = enrich_metadata(raw_docs, source_dir)

    print("[3/3] Splitting into semantic chunks...")
    chunker = SemanticChunker()
    chunks = chunker.split_documents(enriched)
    print(f"  Total chunks: {len(chunks)}")

    return chunks

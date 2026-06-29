import argparse
import os
import shutil

from langchain_core.documents import Document

from config.settings import get_settings
from ingestion.pipeline import run_ingestion_pipeline
from storage.chroma_store import get_chroma_manager


def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    """Assign unique IDs like 'source:page:chunk_index' to each chunk."""
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

    return chunks


def clear_collection(collection_name: str):
    """Remove a single collection's persisted data."""
    settings = get_settings()
    persist_dir = os.path.join(settings.CHROMA_PERSIST_DIR, collection_name)
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        print(f"  Cleared collection: {collection_name}")


def populate_collection(collection_name: str, source_dir: str, reset: bool = False):
    """Build or update a named Chroma collection from documents in source_dir."""
    settings = get_settings()
    manager = get_chroma_manager()

    if reset:
        print(f"  Clearing collection: {collection_name}")
        clear_collection(collection_name)

    chunks = run_ingestion_pipeline(source_dir)
    if not chunks:
        print("  No chunks to add.")
        return

    chunks = calculate_chunk_ids(chunks)
    new_chunk_ids = [chunk.metadata["id"] for chunk in chunks]

    existing_ids = set()
    try:
        existing_ids = manager.get_all_ids(collection_name)
    except Exception:
        pass

    new_chunks = [
        chunk for chunk in chunks if chunk.metadata["id"] not in existing_ids
        and chunk.page_content.strip()
    ]

    if new_chunks:
        print(f"  Adding {len(new_chunks)} new chunks to '{collection_name}'...")
        ids = [chunk.metadata["id"] for chunk in new_chunks]
        manager.add_documents(collection_name, new_chunks, ids)
        print(f"  Done. Collection '{collection_name}' now has {len(existing_ids) + len(new_chunks)} chunks.")
    else:
        print("  No new documents to add.")


def main():
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Build aviation RAG knowledge bases.")
    parser.add_argument(
        "--collection",
        required=True,
        choices=list(settings.COLLECTIONS) + ["all"],
        help="Target collection name, or 'all' to build all.",
    )
    parser.add_argument(
        "--source-dir",
        help="Directory containing source documents. Required unless --collection=all.",
    )
    parser.add_argument(
        "--reset", action="store_true", help="Clear the collection before building."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build all collections from data/ subdirectories.",
    )
    args = parser.parse_args()

    if args.collection == "all" or args.all:
        for coll in settings.COLLECTIONS:
            src = os.path.join("data", coll)
            if os.path.exists(src):
                print(f"\n=== Building collection: {coll} ===")
                populate_collection(coll, src, reset=args.reset)
            else:
                print(f"\n  Skipping {coll}: data/{coll} not found")
    else:
        if not args.source_dir:
            parser.error("--source-dir is required when building a single collection.")
        print(f"\n=== Building collection: {args.collection} ===")
        populate_collection(args.collection, args.source_dir, reset=args.reset)


if __name__ == "__main__":
    main()

from langchain_core.documents import Document

from config.settings import get_settings
from storage.chroma_store import get_chroma_manager


class VectorRetriever:
    """Thin wrapper around Chroma's similarity search."""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self._db = get_chroma_manager().get_collection(collection_name)

    def retrieve(
        self, query: str, k: int | None = None
    ) -> list[tuple[Document, float]]:
        settings = get_settings()
        k = k or settings.TOP_K_RETRIEVAL
        results = self._db.similarity_search_with_score(query, k=k)
        # Chroma returns (doc, distance); convert to (doc, similarity)
        return [(doc, 1.0 - score) for doc, score in results]

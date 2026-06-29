from langchain_core.documents import Document

from config.settings import get_settings
from retrieval.bm25 import BM25Retriever
from retrieval.vector import VectorRetriever


class HybridRetriever:
    """Fusion-ranking hybrid retriever: BM25 + Vector → RRF."""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self._bm25 = BM25Retriever(collection_name)
        self._vector = VectorRetriever(collection_name)

    def retrieve(
        self, query: str, k: int | None = None
    ) -> list[tuple[Document, float]]:
        settings = get_settings()
        k = k or settings.TOP_K_RETRIEVAL

        bm25_results = self._bm25.retrieve(query, k=k)
        vector_results = self._vector.retrieve(query, k=k)

        return reciprocal_rank_fusion([bm25_results, vector_results], k=k)

    def rebuild(self):
        self._bm25.rebuild()


def reciprocal_rank_fusion(
    results_list: list[list[tuple[Document, float]]],
    k: int | None = None,
    rrf_k: int = 60,
) -> list[tuple[Document, float]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion."""
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for results in results_list:
        for rank, (doc, _score) in enumerate(results, start=1):
            doc_id = doc.metadata.get("id", doc.page_content[:50])
            doc_map[doc_id] = doc
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1.0 / (rank + rrf_k)

    sorted_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    if k:
        sorted_ids = sorted_ids[:k]
    return [(doc_map[doc_id], fused_scores[doc_id]) for doc_id in sorted_ids]

from langchain_core.documents import Document

from config.settings import get_settings


class Reranker:
    """Cross-encoder re-ranker for retrieved documents. Uses sentence-transformers."""

    def __init__(self):
        self._model = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        try:
            from sentence_transformers import CrossEncoder

            settings = get_settings()
            self._model = CrossEncoder(
                settings.RERANKER_MODEL,
                device="cpu",
            )
            self._loaded = True
        except ImportError:
            print("Warning: sentence-transformers not installed. Reranker disabled.")

    def rerank(
        self,
        query: str,
        documents: list[Document] | list[tuple[Document, float]],
        top_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        self._ensure_loaded()
        settings = get_settings()
        top_k = top_k or settings.TOP_K_RERANK

        if not self._model or not documents:
            return [(doc, 0.0) for doc in documents[:top_k]]

        docs = [d[0] if isinstance(d, tuple) else d for d in documents]
        try:
            pairs = [[query, doc.page_content] for doc in docs]
            scores = self._model.predict(pairs)
            scores = [float(s) for s in scores]
        except Exception:
            return [(doc, 0.0) for doc in docs[:top_k]]

        ranked = sorted(
            zip(docs, scores), key=lambda x: x[1], reverse=True
        )
        return ranked[:top_k]


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker

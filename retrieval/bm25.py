import re

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config.settings import get_settings
from storage.chroma_store import get_chroma_manager


class BM25Retriever:
    """In-memory BM25 retriever per collection, with jieba Chinese tokenization."""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self._corpus: list[str] = []
        self._documents: list[Document] = []
        self._bm25: BM25Okapi | None = None
        self._build_index()

    def _build_index(self):
        db = get_chroma_manager().get_collection(self.collection_name)
        items = db.get(include=["documents", "metadatas"])
        texts = items["documents"] or []
        metadatas = items["metadatas"] or []
        self._documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]
        self._corpus = texts
        if self._corpus:
            tokenized = [_tokenize(text) for text in self._corpus]
            self._bm25 = BM25Okapi(tokenized)

    def retrieve(
        self, query: str, k: int | None = None
    ) -> list[tuple[Document, float]]:
        settings = get_settings()
        k = k or settings.TOP_K_RETRIEVAL
        if not self._bm25:
            return []
        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        # Normalize scores to 0-1
        max_score = max(scores) if len(scores) > 0 else 1.0
        if max_score > 0:
            scores = [s / max_score for s in scores]
        ranked = sorted(
            zip(self._documents, scores), key=lambda x: x[1], reverse=True
        )
        return ranked[:k]

    def rebuild(self):
        self._build_index()


def _tokenize(text: str) -> list[str]:
    """Tokenize text, using jieba for Chinese-heavy content."""
    chinese_chars = len(re.findall(r"[一-鿿]", text))
    if len(text) > 0 and chinese_chars / len(text) > 0.3:
        return [w for w in jieba.cut(text) if w.strip()]
    else:
        return text.lower().split()

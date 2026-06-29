import os
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import get_settings
from embeddings import get_embedding_function


class ChromaStoreManager:
    """Manages multiple Chroma collections, one per knowledge domain."""

    def __init__(self):
        self._settings = get_settings()
        self._collections: dict[str, Chroma] = {}

    def get_collection(self, name: str) -> Chroma:
        if name not in self._collections:
            persist_dir = os.path.join(self._settings.CHROMA_PERSIST_DIR, name)
            self._collections[name] = Chroma(
                persist_directory=persist_dir,
                embedding_function=get_embedding_function(),
                collection_name=name,
            )
        return self._collections[name]

    def list_collections(self) -> list[str]:
        persist_dir = self._settings.CHROMA_PERSIST_DIR
        if not os.path.exists(persist_dir):
            return []
        return [
            d
            for d in os.listdir(persist_dir)
            if os.path.isdir(os.path.join(persist_dir, d))
        ]

    def delete_collection(self, name: str):
        persist_dir = os.path.join(self._settings.CHROMA_PERSIST_DIR, name)
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
        if name in self._collections:
            del self._collections[name]

    def add_documents(
        self, collection_name: str, chunks: list[Document], chunk_ids: list[str]
    ):
        db = self.get_collection(collection_name)
        db.add_documents(chunks, ids=chunk_ids)

    def get_all_ids(self, collection_name: str) -> set[str]:
        db = self.get_collection(collection_name)
        items = db.get(include=[])
        return set(items["ids"])


_chroma_manager: ChromaStoreManager | None = None


def get_chroma_manager() -> ChromaStoreManager:
    global _chroma_manager
    if _chroma_manager is None:
        _chroma_manager = ChromaStoreManager()
    return _chroma_manager

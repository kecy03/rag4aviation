from abc import ABC, abstractmethod

from generation.llm import get_chat_model
from generation.prompts import RAG_PROMPT_TEMPLATE
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import get_reranker


class BaseAgent(ABC):
    """Each agent is bound to a specific knowledge base collection."""

    def __init__(self, collection_name: str):
        self.collection = collection_name
        self._retriever = HybridRetriever(collection_name)
        self._reranker = get_reranker()
        self._llm = get_chat_model()

    @abstractmethod
    def get_system_prompt(self) -> str:
        ...

    def query(self, user_query: str, top_k: int = 20) -> dict:
        candidates = self._retriever.retrieve(user_query, k=top_k)
        final_chunks = self._reranker.rerank(user_query, candidates)

        if not final_chunks:
            return {
                "answer": "No relevant context found to answer this question.",
                "sources": [],
            }

        context_text = "\n\n---\n\n".join(
            doc.page_content for doc, _score in final_chunks
        )

        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context_text, question=user_query
        )
        system_prompt = self.get_system_prompt()

        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        response = self._llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        sources = [
            {
                "chunk_id": doc.metadata.get("id", "unknown"),
                "page": doc.metadata.get("page", "?"),
                "score": round(score, 4),
            }
            for doc, score in final_chunks
        ]

        return {"answer": answer, "sources": sources}

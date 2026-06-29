from langchain_core.prompts import ChatPromptTemplate

from generation.llm import get_chat_model
from generation.prompts import RAG_PROMPT_TEMPLATE, SYSTEM_PROMPT
from retrieval.hybrid import HybridRetriever
from retrieval.query_rewriter import QueryRewriter
from retrieval.reranker import get_reranker
from retrieval.router import DomainRouter


class RAGGenerator:
    """Orchestrates the full single-turn RAG pipeline."""

    def __init__(self):
        self._router = DomainRouter()
        self._rewriter = QueryRewriter()
        self._reranker = get_reranker()
        self._llm = get_chat_model()

        self._retrievers: dict[str, HybridRetriever] = {}

    def _get_retriever(self, collection: str) -> HybridRetriever:
        if collection not in self._retrievers:
            self._retrievers[collection] = HybridRetriever(collection)
        return self._retrievers[collection]

    def generate(self, query: str, top_k: int = 20) -> dict:
        formal_query = self._rewriter.rewrite(query)
        collection = self._router.route(formal_query)

        retriever = self._get_retriever(collection)
        candidates = retriever.retrieve(formal_query, k=top_k)

        final_chunks = self._reranker.rerank(query, candidates)

        if not final_chunks:
            return {
                "answer": "No relevant context found to answer this question.",
                "sources": [],
                "collection_used": collection,
                "rewritten_query": formal_query,
            }

        context_text = "\n\n---\n\n".join(
            doc.page_content for doc, _score in final_chunks
        )

        prompt_template = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query)
        response = self._llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        sources = [
            {
                "chunk_id": doc.metadata.get("id", "unknown"),
                "content_excerpt": doc.page_content[:150],
                "page": doc.metadata.get("page", "?"),
                "score": round(score, 4),
            }
            for doc, score in final_chunks
        ]

        return {
            "answer": answer,
            "sources": sources,
            "collection_used": collection,
            "rewritten_query": formal_query,
        }

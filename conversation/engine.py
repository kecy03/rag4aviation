from conversation.history import ConversationHistory
from generation.generator import RAGGenerator
from retrieval.query_rewriter import QueryRewriter


class MultiTurnRAGEngine:
    """Multi-turn RAG engine supporting follow-up questions and pronoun resolution."""

    def __init__(self):
        self._generator = RAGGenerator()
        self._history = ConversationHistory()
        self._rewriter = QueryRewriter()

    def chat(self, user_query: str) -> dict:
        history_text = self._history.get_history_text()
        if self._history.get_raw_history():
            rewritten = self._rewriter.rewrite_for_multi_turn(
                user_query, self._history.get_raw_history()
            )
        else:
            rewritten = self._rewriter.rewrite(user_query)

        result = self._generator.generate(user_query)
        result["rewritten_query"] = rewritten

        self._history.add_turn(user_query, result["answer"])
        return result

    def reset(self):
        self._history.clear()

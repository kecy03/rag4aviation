from generation.llm import get_chat_model
from generation.prompts import REWRITE_PROMPT_TEMPLATE


class QueryRewriter:
    """Use LLM to rewrite colloquial questions into formal search queries."""

    def __init__(self):
        self._llm = get_chat_model(temperature=0)

    def rewrite(self, original_query: str, conversation_history: str = "") -> str:
        prompt = REWRITE_PROMPT_TEMPLATE.format(
            history=conversation_history or "(none)",
            query=original_query,
        )
        response = self._llm.invoke(prompt)
        result = response.content if hasattr(response, "content") else str(response)
        return result.strip()

    def rewrite_for_multi_turn(
        self, original_query: str, history: list[dict]
    ) -> str:
        """Condense history + current query into a single standalone query."""
        history_text = ""
        for turn in history[-6:]:  # Last 3 Q&A pairs
            role = "User" if turn["role"] == "user" else "Assistant"
            history_text += f"{role}: {turn['content']}\n"
        return self.rewrite(original_query, history_text)

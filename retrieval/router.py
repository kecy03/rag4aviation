from generation.llm import get_chat_model
from generation.prompts import ROUTE_PROMPT_TEMPLATE


class DomainRouter:
    """Route a user query to the appropriate knowledge base collection."""

    VALID_COLLECTIONS = {"aircraft_structure", "flight_ops", "regulations"}

    def __init__(self):
        self._llm = get_chat_model(temperature=0)

    def route(self, query: str) -> str:
        prompt = ROUTE_PROMPT_TEMPLATE.format(query=query)
        response = self._llm.invoke(prompt)
        result = response.content if hasattr(response, "content") else str(response)
        category = result.strip().lower()
        if category in self.VALID_COLLECTIONS:
            return category
        return "aircraft_structure"

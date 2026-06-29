from agent.base import BaseAgent


class RegulationAgent(BaseAgent):
    """Agent for aviation regulations — laws, rules, certification, compliance."""

    def __init__(self):
        super().__init__("regulations")

    def get_system_prompt(self) -> str:
        return (
            "You are an expert aviation regulations advisor. "
            "Answer regulatory questions using ONLY the provided context. "
            "Note that regulations may vary by jurisdiction and date. Always cite sources and dates."
        )

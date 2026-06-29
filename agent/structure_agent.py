from agent.base import BaseAgent


class StructureAgent(BaseAgent):
    """Agent for aircraft structure questions — parts, systems, components, design."""

    def __init__(self):
        super().__init__("aircraft_structure")

    def get_system_prompt(self) -> str:
        return (
            "You are an expert aircraft structure instructor. "
            "Explain aircraft parts, systems, and design using ONLY the provided context. "
            "Use precise terminology and cite sources."
        )

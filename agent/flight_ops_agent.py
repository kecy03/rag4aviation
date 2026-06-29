from agent.base import BaseAgent


class FlightOpsAgent(BaseAgent):
    """Agent for flight operations — procedures, checklists, maneuvers, training."""

    def __init__(self):
        super().__init__("flight_ops")

    def get_system_prompt(self) -> str:
        return (
            "You are an expert flight instructor. "
            "Explain flying procedures, cockpit operations, and checklists using ONLY the provided context. "
            "Emphasize safety and standard operating procedures. Cite sources."
        )

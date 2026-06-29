class ConversationHistory:
    """Stores and manages multi-turn conversation context."""

    def __init__(self, max_turns: int = 10):
        self._turns: list[dict] = []
        self.max_turns = max_turns

    def add_turn(self, user_query: str, assistant_answer: str):
        self._turns.append({"role": "user", "content": user_query})
        self._turns.append({"role": "assistant", "content": assistant_answer})
        if len(self._turns) > self.max_turns * 2:
            self._turns = self._turns[-(self.max_turns * 2):]

    def get_history_text(self) -> str:
        lines = []
        for turn in self._turns:
            role = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role}: {turn['content']}")
        return "\n".join(lines) if lines else "(none)"

    def get_raw_history(self) -> list[dict]:
        return list(self._turns)

    def get_last_answer(self) -> str | None:
        for turn in reversed(self._turns):
            if turn["role"] == "assistant":
                return turn["content"]
        return None

    def clear(self):
        self._turns = []

from generation.llm import get_chat_model

FAITHFULNESS_PROMPT = """Context provided to the model:
{context}

Model's answer:
{answer}

Question: {question}

Rate the faithfulness of the answer on a 1-5 scale:
5: All claims in the answer are directly supported by the context
4: Most claims supported, one minor unsupported detail
3: Some claims supported, some unsupported or contradicted
2: Mostly unsupported claims
1: Completely hallucinated, contradicts context

Output ONLY the number (1-5):"""

RELEVANCE_PROMPT = """Question: {question}
Answer: {answer}

Rate how relevant the answer is to the question on a 1-5 scale:
5: Directly and completely answers the question
4: Mostly answers, one minor gap
3: Partially answers, significant gaps
2: Tangentially related
1: Completely off-topic

Output ONLY the number (1-5):"""

CORRECTNESS_PROMPT = """Expected correct answer: {expected}
Actual model answer: {actual}

Rate semantic correctness of the actual answer on a 1-5 scale:
5: Completely correct, all key facts match
4: Mostly correct, one minor factual error
3: Partially correct, some key facts missing or wrong
2: Mostly incorrect
1: Completely wrong, contradicts expected answer

Output ONLY the number (1-5):"""


class LLMJudge:
    """LLM-as-Judge that produces 1-5 Likert scale scores."""

    def __init__(self):
        self._llm = get_chat_model(temperature=0)

    def _score(self, prompt: str) -> int:
        response = self._llm.invoke(prompt)
        text = (response.content if hasattr(response, "content") else str(response)).strip()
        for char in text:
            if char.isdigit() and 1 <= int(char) <= 5:
                return int(char)
        return 3  # Default middle score on parse failure

    def score_faithfulness(self, question: str, context: str, answer: str) -> int:
        prompt = FAITHFULNESS_PROMPT.format(question=question, context=context, answer=answer)
        return self._score(prompt)

    def score_relevance(self, question: str, answer: str) -> int:
        prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
        return self._score(prompt)

    def score_correctness(self, expected: str, actual: str) -> int:
        prompt = CORRECTNESS_PROMPT.format(expected=expected, actual=actual)
        return self._score(prompt)

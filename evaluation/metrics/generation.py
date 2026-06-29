from evaluation.judge import LLMJudge

_judge = LLMJudge()


def faithfulness(question: str, context: str, answer: str) -> float:
    """Score 1-5: is the answer faithful to the context?"""
    return _judge.score_faithfulness(question, context, answer)


def relevance(question: str, answer: str) -> float:
    """Score 1-5: does the answer address the question?"""
    return _judge.score_relevance(question, answer)


def correctness(expected: str, actual: str) -> float:
    """Score 1-5: how correct is the answer compared to expected?"""
    return _judge.score_correctness(expected, actual)

import yaml
import pytest
from pathlib import Path

from generation.generator import RAGGenerator
from evaluation.judge import LLMJudge
from evaluation.metrics.retrieval import recall_at_k, precision_at_k, mrr

TEST_CASES_PATH = Path(__file__).parent / "test_cases.yaml"


def load_test_cases():
    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_case_ids():
    return [case["id"] for case in load_test_cases()]


@pytest.fixture(scope="module")
def generator():
    return RAGGenerator()


@pytest.fixture(scope="module")
def judge():
    return LLMJudge()


class TestRetrieval:
    @pytest.mark.parametrize("case_id", get_case_ids())
    def test_precision_at_5(self, generator, case_id):
        case = _get_case(case_id)
        if not case.get("relevant_chunk_ids"):
            pytest.skip("No relevant_chunk_ids defined")
        result = generator.generate(case["question"])
        retrieved_ids = [s["chunk_id"] for s in result["sources"]]
        p = precision_at_k(retrieved_ids, case["relevant_chunk_ids"], k=5)
        assert p >= 0, f"Precision@5 is {p}"


class TestGeneration:
    @pytest.mark.parametrize("case_id", get_case_ids())
    def test_faithfulness(self, generator, judge, case_id):
        case = _get_case(case_id)
        result = generator.generate(case["question"])
        context = "\n".join(s["content_excerpt"] for s in result["sources"])
        score = judge.score_faithfulness(case["question"], context, result["answer"])
        assert score >= 3, f"Faithfulness score {score}/5 (expected >= 3)"

    @pytest.mark.parametrize("case_id", get_case_ids())
    def test_relevance(self, generator, judge, case_id):
        case = _get_case(case_id)
        result = generator.generate(case["question"])
        score = judge.score_relevance(case["question"], result["answer"])
        assert score >= 3, f"Relevance score {score}/5 (expected >= 3)"

    @pytest.mark.parametrize("case_id", get_case_ids())
    def test_correctness(self, generator, judge, case_id):
        case = _get_case(case_id)
        result = generator.generate(case["question"])
        score = judge.score_correctness(case["ground_truth"], result["answer"])
        assert score >= 3, f"Correctness score {score}/5 (expected >= 3)"


def _get_case(case_id: str) -> dict:
    for case in load_test_cases():
        if case["id"] == case_id:
            return case
    raise ValueError(f"Test case {case_id} not found")

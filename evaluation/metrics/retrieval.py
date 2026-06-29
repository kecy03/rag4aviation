def recall_at_k(
    retrieved_ids: list[str], relevant_ids: list[str], k: int = 5
) -> float:
    if not relevant_ids:
        return 1.0
    intersection = len(set(retrieved_ids[:k]) & set(relevant_ids))
    return intersection / len(relevant_ids)


def precision_at_k(
    retrieved_ids: list[str], relevant_ids: list[str], k: int = 5
) -> float:
    if k == 0:
        return 0.0
    intersection = len(set(retrieved_ids[:k]) & set(relevant_ids))
    return intersection / min(k, len(retrieved_ids[:k])) if retrieved_ids[:k] else 0.0


def mrr(
    queries_retrieved: list[list[str]], queries_relevant: list[list[str]]
) -> float:
    if not queries_retrieved:
        return 0.0
    reciprocal_ranks = []
    for retrieved, relevant in zip(queries_retrieved, queries_relevant):
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in set(relevant):
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)

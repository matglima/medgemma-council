"""
Retrieval metrics: Mean Reciprocal Rank (MRR) and NDCG@k.

Used to evaluate how well the council's retrieval components
rank relevant clinical documents for patient queries.
"""

import math
from typing import Dict, List, Tuple


def compute_mrr(results: List[List[Tuple[str, bool]]]) -> float:
    """Compute Mean Reciprocal Rank over a set of query results.

    Args:
        results: List of query results. Each query result is a list of
            (item_id, is_relevant) tuples ordered by predicted rank.

    Returns:
        MRR score between 0.0 and 1.0.
    """
    if not results:
        return 0.0

    reciprocal_ranks = []
    for query_results in results:
        rr = 0.0
        for rank, (_, is_relevant) in enumerate(query_results, start=1):
            if is_relevant:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def compute_ndcg(results: List[Dict[str, List[int]]], k: int = 10) -> float:
    """Compute mean Normalized Discounted Cumulative Gain at k.

    Args:
        results: List of dicts, each with:
            - 'retrieved_relevance': relevance scores in retrieved order
            - 'ideal_relevance': relevance scores in ideal (sorted) order
        k: Cutoff rank (default 10).

    Returns:
        Mean NDCG@k score between 0.0 and 1.0.
    """
    if not results:
        return 0.0

    ndcg_scores = []
    for result in results:
        retrieved = result["retrieved_relevance"][:k]
        ideal = result["ideal_relevance"][:k]

        dcg = _dcg(retrieved)
        idcg = _dcg(ideal)

        if idcg == 0.0:
            ndcg_scores.append(0.0)
        else:
            ndcg_scores.append(dcg / idcg)

    return sum(ndcg_scores) / len(ndcg_scores)


def _dcg(relevance_scores: List[int]) -> float:
    """Compute Discounted Cumulative Gain.

    Uses the standard formula: sum(rel_i / log2(i + 1)) for i starting at 1.

    Args:
        relevance_scores: Relevance scores in rank order.

    Returns:
        DCG value.
    """
    dcg = 0.0
    for i, rel in enumerate(relevance_scores, start=1):
        dcg += rel / math.log2(i + 1)
    return dcg

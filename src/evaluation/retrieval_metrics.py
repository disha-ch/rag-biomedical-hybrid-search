"""Binary-relevance metrics over ranked original passage IDs."""

from math import log2


def recall_at_k(retrieved_ids, relevant_ids, k):
    relevant = set(relevant_ids)
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant) if relevant else 0.0


def mrr_at_k(retrieved_ids, relevant_ids, k=10):
    relevant = set(relevant_ids)
    return next((1.0 / rank for rank, pid in enumerate(retrieved_ids[:k], 1)
                 if pid in relevant), 0.0)


def ndcg_at_k(retrieved_ids, relevant_ids, k=10):
    relevant = set(relevant_ids)
    seen = set()
    dcg = 0.0
    for rank, pid in enumerate(retrieved_ids[:k], 1):
        if pid in relevant and pid not in seen:
            dcg += 1.0 / log2(rank + 1)
        seen.add(pid)
    ideal_dcg = sum(1.0 / log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def retrieval_metrics(retrieved_ids, relevant_ids):
    return {
        "Recall@5": recall_at_k(retrieved_ids, relevant_ids, 5),
        "Recall@10": recall_at_k(retrieved_ids, relevant_ids, 10),
        "MRR@10": mrr_at_k(retrieved_ids, relevant_ids, 10),
        "nDCG@10": ndcg_at_k(retrieved_ids, relevant_ids, 10),
    }

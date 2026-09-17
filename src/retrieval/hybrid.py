"""Reciprocal rank fusion of existing BM25 and dense retrievers."""

from src.retrieval.bm25 import BM25Retriever, SearchResult
from src.retrieval.dense import DenseRetriever


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever) -> None:
        """Reuse retrievers built over the same normalized corpus."""
        self.bm25 = bm25
        self.dense = dense

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Fuse top_k from each source, then return top_k; ties use passage ID."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if type(top_k) is not int:
            raise TypeError("top_k must be an integer")
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        if not query.strip() or top_k == 0:
            return []

        fused = {}
        for retriever in (self.bm25, self.dense):
            for result in retriever.search(query, top_k):
                pid = result["passage_id"]
                if pid not in fused:
                    fused[pid] = {"passage_id": pid, "score": 0.0, "rank": 0, "text": result["text"]}
                fused[pid]["score"] += 1.0 / (60 + result["rank"])

        results = sorted(fused.values(), key=lambda row: (-row["score"], row["passage_id"]))[:top_k]
        for rank, result in enumerate(results, start=1):
            result["rank"] = rank
        return results

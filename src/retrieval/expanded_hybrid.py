"""Run hybrid search for the original query and two expansions; fuse with RRF."""

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.query_expansion import expand_query


class ExpandedHybridRetriever:
    def __init__(self, hybrid: HybridRetriever, *, client=None, model=None):
        self.hybrid = hybrid
        self.client = client
        self.model = model

    def search(self, query: str, top_k: int) -> list[dict]:
        return self.search_record(query, top_k)["results"]

    def search_record(self, query: str, top_k: int, *, usage=None) -> dict:
        """Expose the two alternates separately from the original query."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if type(top_k) is not int:
            raise TypeError("top_k must be an integer")
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        record = {"original_query": query, "expanded_queries": [],
                  "retrieval_mode": "Hybrid + Query Expansion", "results": []}
        if not query.strip() or top_k == 0:
            return record
        queries = expand_query(query, client=self.client, model=self.model, usage=usage)
        record["expanded_queries"] = queries[1:]
        fused = {}
        for alternate in queries:
            for result in self.hybrid.search(alternate, top_k):
                pid = result["passage_id"]
                if pid not in fused:
                    fused[pid] = {"passage_id": pid, "score": 0.0, "rank": 0, "text": result["text"]}
                fused[pid]["score"] += 1.0 / (60 + result["rank"])
        record["results"] = sorted(fused.values(), key=lambda r: (-r["score"], r["passage_id"]))[:top_k]
        for rank, result in enumerate(record["results"], 1):
            result["rank"] = rank
        return record

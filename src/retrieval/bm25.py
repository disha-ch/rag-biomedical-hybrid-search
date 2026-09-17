"""In-memory BM25 retrieval over valid normalized passages."""

import re
from collections.abc import Iterable
from typing import TypedDict

from rank_bm25 import BM25Okapi

from src.data.normalize import Passage


class SearchResult(TypedDict):
    passage_id: int
    score: float
    rank: int
    text: str


def tokenize(text: str) -> list[str]:
    """Lowercase Unicode letters/digits; punctuation and underscores separate words."""
    return re.findall(r"[^\W_]+", text.lower())


class BM25Retriever:
    def __init__(self, passages: Iterable[Passage]) -> None:
        # Copy records so caller mutations cannot break the index-to-ID mapping.
        # Keep duplicate text under its original IDs.
        self._passages = [dict(row) for row in passages if row["is_valid"] is True]
        ids = [row["id"] for row in self._passages]
        if any(type(pid) is not int for pid in ids):
            raise ValueError("passage IDs must be integers; IDs are never coerced")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate passage IDs are ambiguous")
        tokens = [tokenize(row["text"]) for row in self._passages]
        # rank-bm25 cannot initialize an empty vocabulary. Keep rows unchanged.
        self._index = (
            BM25Okapi(tokens, k1=1.5, b=0.75, epsilon=0.25)
            if any(tokens) else None
        )

    def __len__(self) -> int:
        return len(self._passages)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return up to top_k rows, ordered by score then original corpus order.

        Zero/negative scores are retained, without a relevance threshold. Empty
        or punctuation-only queries, top_k=0, and empty vocabularies return [].
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if type(top_k) is not int:
            raise TypeError("top_k must be an integer")
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        query_tokens = tokenize(query)
        if not query_tokens or top_k == 0 or self._index is None:
            return []
        scores = self._index.get_scores(query_tokens)
        indices = sorted(range(len(self)), key=lambda i: -float(scores[i]))[:top_k]
        return [
            {
                "passage_id": self._passages[i]["id"],
                "score": float(scores[i]),
                "rank": rank,
                "text": self._passages[i]["text"],
            }
            for rank, i in enumerate(indices, start=1)
        ]

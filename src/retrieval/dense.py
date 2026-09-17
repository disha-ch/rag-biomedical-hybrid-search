"""In-memory cosine retrieval over valid normalized passages."""

from collections.abc import Iterable
from typing import TypedDict

import numpy as np
from sentence_transformers import SentenceTransformer

from src.data.normalize import Passage

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SearchResult(TypedDict):
    passage_id: int
    score: float
    rank: int
    text: str


class DenseRetriever:
    def __init__(self, passages: Iterable[Passage]) -> None:
        # Preserve corpus order, original IDs, text, and duplicate-text rows.
        self._passages = [dict(row) for row in passages if row["is_valid"] is True]
        ids = [row["id"] for row in self._passages]
        if any(type(pid) is not int for pid in ids):
            raise ValueError("passage IDs must be integers; IDs are never coerced")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate passage IDs are ambiguous")
        self._model = None
        self._embeddings = None
        if self._passages:
            self._model = SentenceTransformer(MODEL_NAME)
            # One batched corpus encoding per retriever; reuse it for every query.
            self._embeddings = self._model.encode(
                [row["text"] for row in self._passages],
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )

    def __len__(self) -> int:
        return len(self._passages)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return cosine-ranked rows, with corpus order breaking equal-score ties."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if type(top_k) is not int:
            raise TypeError("top_k must be an integer")
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        if not query.strip() or top_k == 0 or self._model is None:
            return []
        query_embedding = self._model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # Dot product of L2-normalized vectors equals cosine similarity.
        scores = self._embeddings @ query_embedding
        indices = np.argsort(-scores, kind="stable")[:top_k]
        return [
            {
                "passage_id": self._passages[i]["id"],
                "score": float(scores[i]),
                "rank": rank,
                "text": self._passages[i]["text"],
            }
            for rank, i in enumerate(indices, start=1)
        ]

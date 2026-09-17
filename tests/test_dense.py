import unittest
from unittest.mock import patch

import numpy as np

from src.retrieval.dense import DenseRetriever


class DenseTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("src.retrieval.dense.SentenceTransformer")
        self.model_class = patcher.start()
        self.addCleanup(patcher.stop)
        self.model = self.model_class.return_value
        # Known unit vectors give independently checkable cosine scores.
        self.model.encode.side_effect = lambda texts, **kwargs: (
            np.array([[0.6, 0.8], [1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
            if isinstance(texts, list) else np.array([1.0, 0.0], dtype=np.float32)
        )
        self.passages = [
            {"id": 77, "text": "First passage", "is_valid": True},
            {"id": 9000000000000001, "text": "Second passage", "is_valid": True},
            {"id": 42, "text": "Third passage", "is_valid": True},
            {"id": 99, "text": "nan", "is_valid": False},
        ]
        self.retriever = DenseRetriever(self.passages)

    def test_top_k(self):
        for k, expected in ((0, 0), (1, 1), (2, 2), (10, 3)):
            with self.subTest(k=k):
                rows = self.retriever.search("query", k)
                self.assertEqual(len(rows), expected)
                self.assertEqual([r["rank"] for r in rows], list(range(1, expected + 1)))
        with self.assertRaises(ValueError):
            self.retriever.search("query", -1)

    def test_descending_cosine_scores_and_reused_corpus(self):
        rows = self.retriever.search("query", 3)
        np.testing.assert_allclose([r["score"] for r in rows], [1.0, 0.6, -1.0])
        self.assertEqual(self.retriever.search("another query", 3), rows)
        corpus_calls = [c for c in self.model.encode.call_args_list if isinstance(c.args[0], list)]
        self.assertEqual(len(corpus_calls), 1)
        self.assertTrue(all(c.kwargs["normalize_embeddings"] for c in self.model.encode.call_args_list))

    def test_original_ids_text_and_result_structure(self):
        rows = self.retriever.search("query", 3)
        self.assertEqual([r["passage_id"] for r in rows], [9000000000000001, 77, 42])
        original = {p["id"]: p["text"] for p in self.passages}
        for row in rows:
            self.assertEqual(set(row), {"passage_id", "score", "rank", "text"})
            self.assertIs(type(row["passage_id"]), int)
            self.assertIs(type(row["score"]), float)
            self.assertIs(type(row["rank"]), int)
            self.assertEqual(row["text"], original[row["passage_id"]])

    def test_invalid_passages_excluded_from_encoding_and_results(self):
        self.assertEqual(len(self.retriever), 3)
        self.assertEqual(self.model.encode.call_args.args[0], [p["text"] for p in self.passages[:3]])
        self.assertNotIn(99, [r["passage_id"] for r in self.retriever.search("query", 10)])
        self.model_class.reset_mock()
        empty = DenseRetriever([self.passages[-1]])
        self.assertEqual(empty.search("query", 3), [])
        self.model_class.assert_not_called()

    def test_empty_query_does_not_encode(self):
        self.model.encode.reset_mock()
        for query in ("", " ", "\n\t"):
            with self.subTest(query=query):
                self.assertEqual(self.retriever.search(query, 3), [])
        self.model.encode.assert_not_called()


if __name__ == "__main__":
    unittest.main()

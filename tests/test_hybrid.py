import unittest
from unittest.mock import Mock

from src.retrieval.hybrid import HybridRetriever


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.bm25 = Mock()
        self.dense = Mock()
        self.bm25.search.return_value = [
            {"passage_id": 9000000000000001, "score": 100.0, "rank": 1, "text": "BM25 only"},
            {"passage_id": 77, "score": 20.0, "rank": 2, "text": "Shared"},
        ]
        self.dense.search.return_value = [
            {"passage_id": 77, "score": 0.9, "rank": 1, "text": "Shared"},
            {"passage_id": 42, "score": 0.7, "rank": 2, "text": "Dense only"},
        ]
        self.hybrid = HybridRetriever(self.bm25, self.dense)

    def test_score_and_rank_ordering(self):
        rows = self.hybrid.search("query", 3)
        self.assertEqual([r["passage_id"] for r in rows], [77, 9000000000000001, 42])
        self.assertEqual([r["rank"] for r in rows], [1, 2, 3])
        scores = [r["score"] for r in rows]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.bm25.search.assert_called_once_with("query", 3)
        self.dense.search.assert_called_once_with("query", 3)
        self.assertEqual(len(self.hybrid.search("query", 1)), 1)
        self.assertEqual(self.hybrid.search("query", 0), [])

    def test_overlap_combines_rank_contributions_once(self):
        rows = self.hybrid.search("query", 3)
        shared = [r for r in rows if r["passage_id"] == 77]
        self.assertEqual(len(shared), 1)
        self.assertAlmostEqual(shared[0]["score"], 1 / 62 + 1 / 61)
        # Raw scores must have no effect on RRF.
        self.bm25.search.return_value[1]["score"] = -1000.0
        self.assertEqual(self.hybrid.search("query", 3), rows)

    def test_single_source_contributions_and_ties(self):
        rows = {r["passage_id"]: r for r in self.hybrid.search("query", 3)}
        self.assertAlmostEqual(rows[9000000000000001]["score"], 1 / 61)
        self.assertAlmostEqual(rows[42]["score"], 1 / 62)
        self.dense.search.return_value = []
        self.assertEqual(len(self.hybrid.search("query", 3)), 2)
        self.dense.search.return_value = [{"passage_id": 42, "score": 0.7, "rank": 1, "text": "Dense only"}]
        self.assertEqual(self.hybrid.search("query", 3)[0]["passage_id"], 42)

    def test_original_ids_text_and_exact_structure(self):
        originals = {r["passage_id"]: r["text"] for r in self.bm25.search.return_value + self.dense.search.return_value}
        rows = self.hybrid.search("query", 10)
        self.assertEqual({r["passage_id"] for r in rows}, set(originals))
        for row in rows:
            self.assertEqual(set(row), {"passage_id", "score", "rank", "text"})
            self.assertIs(type(row["passage_id"]), int)
            self.assertIs(type(row["score"]), float)
            self.assertIs(type(row["rank"]), int)
            self.assertEqual(row["text"], originals[row["passage_id"]])
        self.assertEqual(self.bm25.search.return_value[0]["score"], 100.0)

    def test_empty_query_skips_both_retrievers(self):
        for query in ("", " ", "\n\t"):
            with self.subTest(query=query):
                self.assertEqual(self.hybrid.search(query, 3), [])
        self.bm25.search.assert_not_called()
        self.dense.search.assert_not_called()


if __name__ == "__main__":
    unittest.main()

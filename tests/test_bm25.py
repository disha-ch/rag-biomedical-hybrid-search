import unittest

from src.retrieval.bm25 import BM25Retriever, tokenize


class BM25Tests(unittest.TestCase):
    def setUp(self):
        self.passages = [
            {"id": 9000000000000001, "text": "dopamine dopamine dopamine", "is_valid": True},
            {"id": 42, "text": "dopamine signaling pathway", "is_valid": True},
            {"id": 7, "text": "insulin glucose pancreas", "is_valid": True},
            {"id": 801, "text": "heart muscle contraction", "is_valid": True},
            {"id": 55, "text": "immune cell response", "is_valid": True},
            {"id": 99, "text": "dopamine " * 100, "is_valid": False},
        ]
        self.retriever = BM25Retriever(self.passages)

    def test_top_k(self):
        for k, expected in ((0, 0), (1, 1), (3, 3), (5, 5), (100, 5)):
            with self.subTest(k=k):
                rows = self.retriever.search("dopamine", k)
                self.assertEqual(len(rows), expected)
                self.assertEqual([row["rank"] for row in rows], list(range(1, expected + 1)))
        with self.assertRaises(ValueError):
            self.retriever.search("dopamine", -1)
        for k in (True, 1.5, "3"):
            with self.subTest(k=k), self.assertRaises(TypeError):
                self.retriever.search("dopamine", k)

    def test_descending_scores_and_term_frequency(self):
        rows = self.retriever.search("dopamine", 5)
        scores = [row["score"] for row in rows]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])
        self.assertEqual([row["passage_id"] for row in rows[:2]], [9000000000000001, 42])

    def test_exact_result_structure_ids_and_text(self):
        originals = {row["id"]: row["text"] for row in self.passages if row["is_valid"]}
        rows = self.retriever.search("dopamine", 100)
        self.assertEqual({row["passage_id"] for row in rows}, set(originals))
        for row in rows:
            self.assertEqual(set(row), {"passage_id", "score", "rank", "text"})
            self.assertIs(type(row["passage_id"]), int)
            self.assertIs(type(row["score"]), float)
            self.assertIs(type(row["rank"]), int)
            self.assertEqual(row["text"], originals[row["passage_id"]])

    def test_invalid_passages_do_not_affect_results_or_scores(self):
        valid_only = BM25Retriever(self.passages[:-1])
        self.assertEqual(len(self.retriever), 5)
        self.assertEqual(self.retriever.search("dopamine", 100), valid_only.search("dopamine", 100))

    def test_empty_queries(self):
        for query in ("", " \n\t", "...!?"):
            with self.subTest(query=query):
                self.assertEqual(self.retriever.search(query, 3), [])
        with self.assertRaises(TypeError):
            self.retriever.search(None, 3)

    def test_empty_corpus_or_vocabulary(self):
        for passages in ([], [self.passages[-1]], [{"id": 1, "text": "...", "is_valid": True}]):
            with self.subTest(passages=passages):
                self.assertEqual(BM25Retriever(passages).search("dopamine", 3), [])

    def test_tokenization(self):
        self.assertEqual(tokenize("EGFR, IL-6 / β-catenin_gene"), ["egfr", "il", "6", "β", "catenin", "gene"])
        self.assertEqual(self.retriever.search("DOPAMINE!", 3), self.retriever.search("dopamine", 3))

    def test_ties_and_out_of_vocabulary_query(self):
        rows = self.retriever.search("zzzzunknown", 5)
        self.assertEqual([row["passage_id"] for row in rows], [row["id"] for row in self.passages[:-1]])
        self.assertTrue(all(row["score"] == 0.0 for row in rows))

    def test_duplicate_text_retained_and_caller_mutation_isolated(self):
        passages = [dict(row) for row in self.passages[:-1]]
        passages.append({"id": 1001, "text": passages[0]["text"], "is_valid": True})
        retriever = BM25Retriever(passages)
        before = retriever.search("dopamine", 10)
        self.assertEqual(len(before), 6)
        self.assertIn(1001, [row["passage_id"] for row in before])
        passages[0]["id"] = -1
        passages[0]["text"] = "changed"
        self.assertEqual(retriever.search("dopamine", 10), before)


if __name__ == "__main__":
    unittest.main()

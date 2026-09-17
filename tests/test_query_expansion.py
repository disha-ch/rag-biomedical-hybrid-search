import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from src.retrieval.query_expansion import expand_query
from src.retrieval.expanded_hybrid import ExpandedHybridRetriever


class ExpansionTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.client.responses.create.return_value = SimpleNamespace(
            status="completed", output_text=json.dumps({
                "expansion_1": "Hirschsprung disease genetic associations",
                "expansion_2": "Congenital aganglionosis associated genes"}))

    def test_original_retained_and_exactly_two_expansions(self):
        query = "  Hirschsprung disease genetics  "
        queries = expand_query(query, client=self.client)
        self.assertEqual(queries[0], query)
        self.assertEqual(len(queries), 3)
        self.assertEqual(len(set(queries)), 3)
        self.client.responses.create.assert_called_once()
        self.assertEqual(self.client.responses.create.call_args.kwargs["input"], query)

    def test_malformed_duplicate_or_empty_expansions_rejected(self):
        for text in ('bad json', '{}', '{"expansion_1": "", "expansion_2": "other"}',
                     '{"expansion_1": " Query ", "expansion_2": "other"}',
                     '{"expansion_1": "other", "expansion_2": "OTHER"}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.client.responses.create.return_value.output_text = text
                expand_query("query", client=self.client)

    def test_expanded_hybrid_ranking_ids_and_record(self):
        hybrid = Mock()
        def row(pid, rank):
            return {"passage_id": pid, "score": 999.0, "rank": rank, "text": f"Evidence {pid}"}
        hybrid.search.side_effect = [
            [row(9000000000000001, 1), row(77, 2)],
            [row(77, 1), row(42, 2)],
            [row(77, 1), row(9000000000000001, 2)],
        ]
        record = ExpandedHybridRetriever(hybrid, client=self.client).search_record("query", 3)
        results = record["results"]
        self.assertEqual([r["passage_id"] for r in results], [77, 9000000000000001, 42])
        self.assertEqual([r["rank"] for r in results], [1, 2, 3])
        self.assertAlmostEqual(results[0]["score"], 1 / 62 + 2 / 61)
        self.assertAlmostEqual(results[1]["score"], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(results[2]["score"], 1 / 62)
        self.assertEqual(record["original_query"], "query")
        self.assertEqual(record["retrieval_mode"], "Hybrid + Query Expansion")
        self.assertEqual(len(record["expanded_queries"]), 2)
        self.assertEqual([c.args for c in hybrid.search.call_args_list],
                         [(q, 3) for q in ["query", *record["expanded_queries"]]])
        for r in results:
            self.assertEqual(set(r), {"passage_id", "score", "rank", "text"})
            self.assertEqual(r["text"], f"Evidence {r['passage_id']}")

    def test_top_k_and_search_list(self):
        hybrid = Mock()
        hybrid.search.return_value = [{"passage_id": 4, "score": 2.0, "rank": 1, "text": "text"}]
        results = ExpandedHybridRetriever(hybrid, client=self.client).search("query", 1)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0]["score"], 3 / 61)

    def test_empty_query_skips_llm_and_retrieval(self):
        hybrid = Mock()
        for query in ("", " \n\t"):
            self.assertEqual(expand_query(query, client=self.client), [])
            self.assertEqual(ExpandedHybridRetriever(hybrid, client=self.client).search(query, 5), [])
        self.client.responses.create.assert_not_called()
        hybrid.search.assert_not_called()

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from scripts.create_eval_set import create_eval_set, load_eval_set, EVAL_PATH
from scripts.evaluate_retrieval import evaluate_queries
from src.evaluation.retrieval_metrics import recall_at_k, mrr_at_k, ndcg_at_k
from src.retrieval.query_expansion import expand_query


class MetricTests(unittest.TestCase):
    def test_recall_and_duplicate_ids(self):
        self.assertEqual(recall_at_k([1, 1, 9, 2, 3], [1, 2, 3, 4], 3), 0.25)
        self.assertEqual(recall_at_k([1, 1, 9, 2, 3], [1, 2, 3, 4], 5), 0.75)
        self.assertEqual(recall_at_k([], [1], 5), 0)
        self.assertEqual(recall_at_k([1], [], 5), 0)

    def test_mrr_at_10_and_cutoff(self):
        self.assertEqual(mrr_at_k([9, 8, 1, 2], [1, 2]), 1 / 3)
        self.assertEqual(mrr_at_k([9] * 10 + [1], [1]), 0)
        self.assertEqual(mrr_at_k([], [1]), 0)

    def test_ndcg_at_10_binary_relevance(self):
        expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(ndcg_at_k([9, 1, 8, 2], [1, 2]), expected)
        self.assertEqual(ndcg_at_k([1, 2], [1, 2]), 1)
        self.assertEqual(ndcg_at_k([1, 1], [1]), 1)
        self.assertEqual(ndcg_at_k([9] * 10 + [1], [1]), 0)
        self.assertEqual(ndcg_at_k([1], []), 0)


class FixedSetTests(unittest.TestCase):
    def test_size_eligibility_reproducibility_and_reuse(self):
        questions = [{"id": i + 1000, "question": f"Question {i}?", "answer": "Answer",
                      "relevant_passage_ids": [7, 99] if i < 120 else [99]} for i in range(125)]
        passages = [{"id": 7, "text": "valid", "is_valid": True},
                    {"id": 99, "text": "nan", "is_valid": False}]
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first.json", Path(directory) / "second.json"
            selected = create_eval_set(first, questions=questions, passages=passages)
            self.assertEqual(len(selected), 100)
            self.assertEqual(len({q["question_id"] for q in selected}), 100)
            self.assertTrue(all(1000 <= q["question_id"] < 1120 for q in selected))
            self.assertTrue(all(q["relevant_passage_ids"] == [7, 99] for q in selected))
            self.assertEqual(selected, create_eval_set(second, questions=list(reversed(questions)), passages=passages))
            before = first.read_bytes()
            self.assertEqual(selected, create_eval_set(first, questions=[], passages=[]))
            self.assertEqual(first.read_bytes(), before)
            self.assertEqual(load_eval_set(first), selected)

    def test_saved_real_set_has_100_unique_original_ids(self):
        queries = load_eval_set(EVAL_PATH)
        self.assertEqual(len(queries), 100)
        self.assertEqual(len({q["question_id"] for q in queries}), 100)
        self.assertTrue(all(set(q) == {"question_id", "question", "answer", "relevant_passage_ids"} for q in queries))


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.queries = [{"question_id": i, "question": f"Query {i}", "answer": "answer",
                         "relevant_passage_ids": [7001, 99]} for i in (44, 9000000000000001)]
        self.rows = [{"passage_id": 7001, "score": 1.5, "rank": 1, "text": "Evidence"}]

    def test_result_structure_and_same_query_ids_for_each_mode(self):
        for _ in range(3):
            retriever = Mock()
            retriever.search.return_value = self.rows
            report = evaluate_queries(self.queries, retriever)
            self.assertEqual([r["query_id"] for r in report["queries"]], [44, 9000000000000001])
            self.assertEqual([c.args for c in retriever.search.call_args_list], [(q["question"], 10) for q in self.queries])
            for row in report["queries"]:
                self.assertEqual(row["retrieved_passage_ids"], [7001])
                self.assertEqual(row["scores"], [1.5])
                self.assertEqual(row["ranks"], [1])
                self.assertEqual(row["Recall@5"], 0.5)  # Invalid reference 99 is not dropped.
                self.assertEqual(row["Recall@10"], 0.5)
                self.assertEqual(row["MRR@10"], 1)
                self.assertAlmostEqual(row["nDCG@10"], 1 / (1 + 1 / math.log2(3)))
                self.assertGreaterEqual(row["latency_seconds"], 0)
                self.assertEqual(row["expansion_calls"], 0)
                self.assertEqual(row["external_api_cost_usd"], 0)
            self.assertEqual(report["summary"]["mean_metrics"]["Recall@10"], 0.5)
            json.dumps(report)

    def test_errors_are_not_reported_as_complete_means(self):
        retriever = Mock()
        retriever.search.side_effect = [self.rows, RuntimeError("failure")]
        report = evaluate_queries(self.queries, retriever)
        self.assertEqual(report["summary"]["status"], "failed_or_partial")
        self.assertIsNone(report["summary"]["mean_metrics"])
        self.assertIsNone(report["queries"][1]["Recall@10"])

    def test_expansion_calls_and_usage_recorded(self):
        client = Mock()
        client.responses.create.return_value = SimpleNamespace(status="completed",
            output_text='{"expansion_1":"Alternate one", "expansion_2":"Alternate two"}',
            usage=SimpleNamespace(input_tokens=10, output_tokens=8, total_tokens=18))
        usage = {}
        expand_query("query", client=client, usage=usage)
        self.assertEqual(usage, {"expansion_calls": 1, "token_usage": {"input_tokens": 10, "output_tokens": 8, "total_tokens": 18}})
        retriever = Mock()
        def search_record(query, top_k, *, usage):
            usage.update(expansion_calls=1, token_usage={"input_tokens": 10, "output_tokens": 8, "total_tokens": 18})
            return {"expanded_queries": ["Alternate one", "Alternate two"], "results": self.rows}
        retriever.search_record.side_effect = search_record
        report = evaluate_queries(self.queries, retriever, expanded=True)
        self.assertEqual(report["summary"]["expansion_calls"], 2)
        self.assertEqual(report["summary"]["token_usage"]["total_tokens"], 36)
        self.assertIsNone(report["summary"]["external_api_cost_usd"])


if __name__ == "__main__":
    unittest.main()

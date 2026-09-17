import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from src.generation.answer import generate_answer, INSUFFICIENT


class AnswerTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.client.responses.create.return_value = SimpleNamespace(
            status="completed", output_text="RET is associated with Hirschsprung disease [23001136].")
        self.passages = [{"passage_id": 23001136, "score": 1.0, "rank": 1,
                          "text": "RET is associated with Hirschsprung disease."},
                         {"passage_id": 42, "score": 0.5, "rank": 2, "text": "Other evidence."}]

    def test_valid_answer_and_only_selected_context_sent(self):
        result = generate_answer("Which gene?", self.passages, client=self.client, context_top_k=1)
        self.assertEqual(result["citation_status"], "valid")
        self.assertEqual(result["cited_passage_ids"], [23001136])
        self.assertFalse(result["insufficient_evidence"])
        self.client.responses.create.assert_called_once()
        payload = json.loads(self.client.responses.create.call_args.kwargs["input"])
        self.assertEqual(payload, {"question": "Which gene?", "evidence": [
            {"passage_id": 23001136, "text": self.passages[0]["text"]}]})

    def test_unsupported_and_unselected_citations_rejected(self):
        for cited in (999, 42):
            with self.subTest(cited=cited):
                self.client.responses.create.return_value.output_text = f"A claim [23001136] [{cited}]."
                result = generate_answer("Q", self.passages, client=self.client, context_top_k=1)
                self.assertEqual(result["citation_status"], "rejected")
                self.assertEqual(result["answer"], "")
                self.assertIn(str(cited), result["warning"])

    def test_missing_citations_rejected(self):
        self.client.responses.create.return_value.output_text = "An unsupported uncited claim."
        self.assertEqual(generate_answer("Q", self.passages, client=self.client)["citation_status"], "rejected")

    def test_model_insufficient_evidence(self):
        self.client.responses.create.return_value.output_text = "INSUFFICIENT_EVIDENCE"
        result = generate_answer("Q", self.passages, client=self.client)
        self.assertEqual(result["answer"], INSUFFICIENT)
        self.assertTrue(result["insufficient_evidence"])
        self.assertEqual(result["citation_status"], "not applicable")

    def test_empty_or_invalid_context_skips_llm(self):
        for passages in ([], [{"passage_id": 1, "text": "nan"}], [{"passage_id": 2, "text": "1."}],
                         [{"passage_id": 3, "text": " \n"}], [{"passage_id": 4, "text": "..."}]):
            with self.subTest(passages=passages):
                self.assertEqual(generate_answer("Q", passages, client=self.client)["answer"], INSUFFICIENT)
        self.assertTrue(generate_answer(" ", self.passages, client=self.client)["insufficient_evidence"])
        self.client.responses.create.assert_not_called()

    def test_incomplete_response_not_accepted(self):
        self.client.responses.create.return_value.status = "incomplete"
        with self.assertRaises(ValueError):
            generate_answer("Q", self.passages, client=self.client)

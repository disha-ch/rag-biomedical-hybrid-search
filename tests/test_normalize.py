import copy
import unittest

from src.data.normalize import (
    analyze_question_passages, is_valid_passage_text, normalize_passages,
    normalize_questions, parse_relevant_ids,
)


class NormalizationTests(unittest.TestCase):
    def test_invalid_text_and_numbered_content(self):
        for text in ("", " \n\t", "nan", " NaN\n", "1.", " 2. ", "10."):
            with self.subTest(text=text):
                self.assertFalse(is_valid_passage_text(text))
        for text in ("1. Introduction", "1.5", "NAN protein", "A", "  Biomedical text\n"):
            with self.subTest(text=text):
                self.assertTrue(is_valid_passage_text(text))

    def test_passage_ids_duplicates_text_and_order_are_preserved(self):
        rows = [{"id": pid, "passage": " same text\n"} for pid in (91, 7, 9000000000000001)]
        rows.append({"id": 42, "passage": None})
        original = copy.deepcopy(rows)
        result = normalize_passages(rows)
        self.assertEqual([row["id"] for row in result], [91, 7, 9000000000000001, 42])
        self.assertEqual([row["text"] for row in result[:3]], [" same text\n"] * 3)
        self.assertEqual(result[-1], {"id": 42, "text": "", "is_valid": False})
        self.assertEqual(rows, original)
        self.assertEqual(normalize_passages(rows), result)

    def test_questions_and_references_are_preserved(self):
        rows = [{"id": 72, "question": " Q? ", "answer": " A. ", "relevant_passage_ids": "[91, 7, 91]"}]
        original = copy.deepcopy(rows)
        result = normalize_questions(rows)
        self.assertEqual(result, [{"id": 72, "question": " Q? ", "answer": " A. ", "relevant_passage_ids": [91, 7, 91]}])
        self.assertEqual(rows, original)
        self.assertEqual(normalize_questions(rows), result)

    def test_malformed_references_are_rejected(self):
        self.assertEqual(parse_relevant_ids("[]"), [])
        for value in ('["1"]', '[true]', '[1.0]', 'null', '{}', 'bad', None):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                parse_relevant_ids(value)
        with self.assertRaisesRegex(ValueError, "question 72"):
            normalize_questions([{"id": 72, "question": "Q", "answer": "A", "relevant_passage_ids": "bad"}])

    def test_ids_are_not_coerced(self):
        for value in ("1", 1.0, True, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_passages([{"id": value, "passage": "text"}])
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_questions([{"id": value, "question": "Q", "answer": "A", "relevant_passage_ids": "[]"}])

    def test_flags_distinguish_mixed_invalid_missing_and_empty_references(self):
        passages = normalize_passages([
            {"id": 10, "passage": "valid"}, {"id": 20, "passage": "nan"},
            {"id": 30, "passage": "1."},
        ])
        questions = normalize_questions([
            {"id": i, "question": "Q", "answer": "A", "relevant_passage_ids": refs}
            for i, refs in enumerate(("[10, 20]", "[20, 30]", "[99, 20]", "[]"))
        ])
        original = copy.deepcopy(questions)
        flags = analyze_question_passages(questions, passages)
        self.assertEqual([f["id"] for f in flags], [0, 1, 2, 3])
        self.assertEqual([f["has_valid_relevant_passage"] for f in flags], [True, False, False, False])
        self.assertEqual([f["all_relevant_passages_invalid"] for f in flags], [False, True, False, False])
        self.assertEqual(flags[0]["invalid_relevant_passage_ids"], [20])
        self.assertEqual(flags[2]["missing_relevant_passage_ids"], [99])
        self.assertEqual(questions, original)

    def test_duplicate_ids_are_not_silently_collapsed_in_analysis(self):
        passages = normalize_passages([{"id": 1, "passage": "text"}] * 2)
        self.assertEqual(len(passages), 2)
        with self.assertRaises(ValueError):
            analyze_question_passages([], passages)


if __name__ == "__main__":
    unittest.main()

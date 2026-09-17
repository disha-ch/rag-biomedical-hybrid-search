"""Normalize raw BioASQ rows without filtering, renumbering, or deduplication."""

import json
import re
from typing import TypedDict


class Passage(TypedDict):
    id: int
    text: str
    is_valid: bool


class Question(TypedDict):
    id: int
    question: str
    answer: str
    relevant_passage_ids: list[int]


class QuestionPassageStatus(TypedDict):
    id: int
    has_valid_relevant_passage: bool
    all_relevant_passages_invalid: bool
    invalid_relevant_passage_ids: list[int]
    missing_relevant_passage_ids: list[int]


def parse_relevant_ids(value: str) -> list[int]:
    """Parse strictly, retaining reference order and duplicates without coercion."""
    ids = json.loads(value)
    if not isinstance(ids, list) or any(type(item) is not int for item in ids):
        raise ValueError("expected a JSON list of integer passage IDs")
    return ids


def is_valid_passage_text(text: str) -> bool:
    stripped = text.strip()
    return (
        bool(stripped)
        and stripped.lower() != "nan"
        and re.fullmatch(r"[0-9]+\.", stripped) is None
    )


def normalize_passages(rows) -> list[Passage]:
    passages = []
    for row in rows:
        if type(row["id"]) is not int:
            raise ValueError("passage ID must be an integer; IDs are never coerced")
        # Null text becomes an empty invalid string; other text stays untouched.
        text = row["passage"] if row["passage"] is not None else ""
        if not isinstance(text, str):
            raise ValueError(f"passage {row['id']}: text must be a string or null")
        passages.append({"id": row["id"], "text": text, "is_valid": is_valid_passage_text(text)})
    return passages


def normalize_questions(rows) -> list[Question]:
    questions = []
    for row in rows:
        if type(row["id"]) is not int:
            raise ValueError("question ID must be an integer; IDs are never coerced")
        if not isinstance(row["question"], str) or not isinstance(row["answer"], str):
            raise ValueError(f"question {row['id']}: question and answer must be strings")
        try:
            ids = parse_relevant_ids(row["relevant_passage_ids"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"question {row['id']}: invalid relevant_passage_ids") from error
        questions.append({
            "id": row["id"], "question": row["question"], "answer": row["answer"],
            "relevant_passage_ids": ids,
        })
    return questions


def analyze_question_passages(
    questions: list[Question], passages: list[Passage]
) -> list[QuestionPassageStatus]:
    """Flag references without changing questions; missing/empty is not all-invalid."""
    validity = {passage["id"]: passage["is_valid"] for passage in passages}
    if len(validity) != len(passages):
        raise ValueError("duplicate passage IDs make validity analysis ambiguous")
    statuses = []
    for question in questions:
        ids = question["relevant_passage_ids"]
        invalid = [pid for pid in ids if pid in validity and not validity[pid]]
        missing = [pid for pid in ids if pid not in validity]
        statuses.append({
            "id": question["id"],
            "has_valid_relevant_passage": any(validity.get(pid, False) for pid in ids),
            "all_relevant_passages_invalid": bool(ids) and len(invalid) == len(ids),
            "invalid_relevant_passage_ids": invalid,
            "missing_relevant_passage_ids": missing,
        })
    return statuses

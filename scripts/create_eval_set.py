"""Create the fixed 100-query set once; existing files are validated and reused."""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.normalize import normalize_passages, normalize_questions

DATASET = "rag-datasets/rag-mini-bioasq"
REVISION = "224a87f64a5c3a720b5bc627cf760f543b2f1e79"
EVAL_PATH = ROOT / "data/eval/queries_100.json"
SEED = 42


def load_eval_set(path=EVAL_PATH):
    queries = json.loads(Path(path).read_text())
    if not isinstance(queries, list) or len(queries) != 100:
        raise ValueError("Evaluation set must contain exactly 100 queries.")
    for q in queries:
        if (type(q["question_id"]) is not int
                or not isinstance(q["question"], str) or not q["question"].strip()
                or not isinstance(q["answer"], str)
                or not isinstance(q["relevant_passage_ids"], list) or not q["relevant_passage_ids"]
                or any(type(pid) is not int for pid in q["relevant_passage_ids"])):
            raise ValueError("Invalid evaluation question fields.")
    if len({q["question_id"] for q in queries}) != 100:
        raise ValueError("Evaluation question IDs must be unique.")
    return queries


def create_eval_set(path=EVAL_PATH, *, questions=None, passages=None):
    path = Path(path)
    if path.exists():
        return load_eval_set(path)
    if questions is None or passages is None:
        from datasets import load_dataset
        questions = normalize_questions(load_dataset(
            DATASET, name="question-answer-passages", split="test", revision=REVISION))
        passages = normalize_passages(load_dataset(
            DATASET, name="text-corpus", split="passages", revision=REVISION))
    valid_ids = {p["id"] for p in passages if p["is_valid"] is True}
    eligible = sorted((q for q in questions if valid_ids.intersection(q["relevant_passage_ids"])),
                      key=lambda q: q["id"])
    if len(eligible) < 100:
        raise ValueError("At least 100 eligible questions are required.")
    selected = random.Random(SEED).sample(eligible, 100)
    records = [{"question_id": q["id"], "question": q["question"], "answer": q["answer"],
                "relevant_passage_ids": q["relevant_passage_ids"]} for q in selected]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        json.dump(records, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return load_eval_set(path)


if __name__ == "__main__":
    existed = EVAL_PATH.exists()
    queries = create_eval_set()
    print(f"{'Reused' if existed else 'Created'} {len(queries)} fixed queries: {EVAL_PATH}")

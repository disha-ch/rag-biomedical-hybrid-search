"""Inspect raw BioASQ data and ID relationships without changing any records."""

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

from datasets import get_dataset_config_names, load_dataset

# Support the README's `python scripts/inspect_dataset.py` command.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.normalize import (
    analyze_question_passages,
    normalize_passages,
    normalize_questions,
    parse_relevant_ids,
)

DATASET_NAME = "rag-datasets/rag-mini-bioasq"


def print_distribution(label, values):
    if not values:
        print(f"{label}: no values")
        return
    print(
        f"{label}: min={min(values)}, mean={mean(values):.2f}, "
        f"median={median(values):g}, max={max(values)}"
    )


def is_blank(value):
    return value is None or (isinstance(value, str) and not value.strip())


def is_nan_text(value):
    # This dataset contains missing text serialized as the literal string "nan".
    return isinstance(value, str) and value.strip().lower() == "nan"


def inspect_text(label, values):
    counts = Counter(value for value in values if not is_blank(value))
    print(f"{label}: null={sum(value is None for value in values)}, "
          f"blank strings={sum(isinstance(value, str) and not value.strip() for value in values)}, "
          f"duplicate nonblank rows beyond first={sum(n - 1 for n in counts.values())}")
    print(f"{label} literal 'nan' strings: {sum(is_nan_text(v) for v in values)}")
    print(f"{label} most frequent nonblank text (80-character previews):",
          [(text[:80], count) for text, count in counts.most_common(3)])
    # Whitespace word counts are descriptive lengths, not model token counts.
    print_distribution(f"{label} characters (nonblank)", [len(v) for v in values if not is_blank(v)])
    print_distribution(f"{label} whitespace words (nonblank)", [len(v.split()) for v in values if not is_blank(v)])


def inspect_ids(label, values):
    counts = Counter(values)
    print(f"{label}: unique nonnull={len(counts) - (None in counts)}, "
          f"null={counts[None]}, duplicate rows beyond first="
          f"{sum(n - 1 for value, n in counts.items() if value is not None)}")


def inspect_normalization(questions, passages):
    normalized_questions = normalize_questions(questions)
    normalized_passages = normalize_passages(passages)
    flags = analyze_question_passages(normalized_questions, normalized_passages)
    # Compare sequences and multiplicities, not sets: no row may disappear.
    assert [row["id"] for row in normalized_questions] == list(questions["id"])
    assert [row["id"] for row in normalized_passages] == list(passages["id"])
    assert Counter(row["text"] for row in normalized_passages) == Counter(
        text if text is not None else "" for text in passages["passage"]
    )
    for raw, normalized in zip(questions, normalized_questions):
        assert normalized["question"] == raw["question"]
        assert normalized["answer"] == raw["answer"]
        assert normalized["relevant_passage_ids"] == json.loads(raw["relevant_passage_ids"])
    print("\nNormalization verification (in memory; no rows removed):")
    print("Valid passages:", sum(row["is_valid"] for row in normalized_passages))
    print("Invalid passages:", sum(not row["is_valid"] for row in normalized_passages))
    print("Questions with at least one valid relevant passage:", sum(f["has_valid_relevant_passage"] for f in flags))
    print("Questions whose relevant passages are all invalid:", sum(f["all_relevant_passages_invalid"] for f in flags))
    print("Questions referencing any invalid passage:", sum(bool(f["invalid_relevant_passage_ids"]) for f in flags))
    print("Questions referencing missing passage IDs:", sum(bool(f["missing_relevant_passage_ids"]) for f in flags))
    print("Questions with empty relevance lists:", sum(not q["relevant_passage_ids"] for q in normalized_questions))
    print("All original IDs, row order, reference lists, and duplicate-text rows preserved: True")
    print("All-invalid question IDs:", [f["id"] for f in flags if f["all_relevant_passages_invalid"]])


def inspect_mapping(questions, passages):
    # Match actual IDs, never corpus row positions. Keep the raw integer type.
    passage_ids = {value for value in passages["id"] if value is not None}
    blank_passage_ids = {row["id"] for row in passages if is_blank(row["passage"])}
    nan_passage_ids = {row["id"] for row in passages if is_nan_text(row["passage"])}
    available_text_ids = passage_ids - blank_passage_ids - nan_passage_ids
    referenced_ids = set()
    missing_ids = set()
    counts = Counter()
    relevance_lengths = []
    error_examples = []
    missing_examples = []

    for row in questions:
        try:
            ids = parse_relevant_ids(row["relevant_passage_ids"])
        except (TypeError, ValueError) as error:
            counts["malformed relevance rows"] += 1
            if len(error_examples) < 3:
                error_examples.append({"question_id": row["id"], "error": str(error)})
            continue

        unique_ids = set(ids)
        missing = unique_ids - passage_ids
        referenced_ids.update(unique_ids)
        missing_ids.update(missing)
        relevance_lengths.append(len(ids))
        counts["parsed relevance rows"] += 1
        counts["questions with no relevant IDs"] += not ids
        counts["duplicate references within questions"] += len(ids) - len(unique_ids)
        counts["total references"] += len(ids)
        counts["matched references"] += sum(pid in passage_ids for pid in ids)
        counts["missing references"] += sum(pid not in passage_ids for pid in ids)
        counts["questions with missing IDs"] += bool(missing)
        counts["questions with all IDs present (nonempty)"] += bool(ids) and not missing
        counts["references to blank passage text"] += sum(pid in blank_passage_ids for pid in ids)
        counts["questions referencing blank passage text"] += bool(unique_ids & blank_passage_ids)
        counts["references to literal 'nan' passage text"] += sum(pid in nan_passage_ids for pid in ids)
        counts["questions referencing literal 'nan' passage text"] += bool(unique_ids & nan_passage_ids)
        counts["questions with only blank/'nan'/missing referenced passages"] += (
            bool(ids) and not (unique_ids & available_text_ids)
        )
        if missing and len(missing_examples) < 3:
            missing_examples.append({"question_id": row["id"], "missing_ids": sorted(missing)})

    print("\nQuestion-to-passage mapping (all question rows):")
    print("Malformed relevance rows:", counts["malformed relevance rows"])
    for label, value in counts.items():
        if label != "malformed relevance rows":
            print(f"{label}: {value}")
    print("Unique referenced IDs:", len(referenced_ids))
    print("Unique missing IDs:", len(missing_ids))
    print("Corpus IDs not referenced by any question:", len(passage_ids - referenced_ids))
    print_distribution("Relevant IDs per question (successfully parsed)", relevance_lengths)
    if error_examples:
        print("Parse error examples:", error_examples)
    if missing_examples:
        print("Missing ID examples:", missing_examples)
    mapping_ok = counts["malformed relevance rows"] == 0 and not missing_ids
    print("All relevance fields parse and every referenced ID exists:", mapping_ok)
    print("ID existence does not imply usable text; see blank/'nan' counts above.")


def main() -> None:
    print("Dataset:", DATASET_NAME, flush=True)
    configs = get_dataset_config_names(DATASET_NAME)
    print("Available configs:", configs, flush=True)
    datasets = {}
    for config in configs:
        dataset = load_dataset(DATASET_NAME, name=config)
        datasets[config] = dataset
        print(f"\nConfig: {config}")
        print(dataset)
        for split_name, split in dataset.items():
            print(f"Split: {split_name}; rows: {len(split)}")
            print("Schema:", json.dumps(split.features.to_dict(), indent=2))
            print("Source/URL metadata columns:", [
                name for name in split.column_names
                if "source" in name.lower() or "url" in name.lower()
            ])

    questions = datasets["question-answer-passages"]["test"]
    passages = datasets["text-corpus"]["passages"]
    for label, split in (("questions", questions), ("passages", passages)):
        print(f"\nSample {label} (first three raw rows):")
        for row in split.select(range(min(3, len(split)))):
            print(json.dumps(row, ensure_ascii=False, indent=2))

    print("\nBasic statistics:")
    inspect_ids("Question IDs", questions["id"])
    inspect_ids("Passage IDs", passages["id"])
    inspect_text("Questions", questions["question"])
    inspect_text("Answers", questions["answer"])
    inspect_text("Passages", passages["passage"])
    inspect_mapping(questions, passages)

    inspect_normalization(questions, passages)


if __name__ == "__main__":
    main()

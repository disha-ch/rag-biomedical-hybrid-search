# Normalized data schema

Implemented in `src/data/normalize.py` for `rag-datasets/rag-mini-bioasq`.
Normalization returns dictionaries in input order, without modifying inputs,
renumbering IDs, filtering rows, or deduplicating passage text.

## Passage

- `id: int`: original corpus ID, unchanged.
- `text: str`: original `passage` text, unchanged; null becomes an empty string.
- `is_valid: bool`: false for empty/whitespace-only text, case-insensitive
  `nan` after trimming, or a bare numbered-list marker matching `[0-9]+\.`
  after trimming (such as `1.` or `2.`).

Trimming and case normalization are used only for validity checks. Numbered
content such as `1. Introduction` and decimals such as `1.5` remain valid.
Validity detects these explicit unusable values; it does not assess biomedical
quality or completeness. No source/URL columns are supplied by the raw dataset.

## Question

- `id: int`: original question ID, unchanged.
- `question: str`: original question text.
- `answer: str`: original answer text.
- `relevant_passage_ids: list[int]`: parsed from the raw JSON string, retaining
  order and repeated references if present.

IDs are checked as integers without coercion. Malformed relevance lists raise
an error identifying the question; they are never replaced with empty lists.
Question and answer text must be strings.

## Separate per-question analysis flags

`analyze_question_passages` returns one status dictionary per question:

- `id: int`: unchanged question ID.
- `has_valid_relevant_passage: bool`.
- `all_relevant_passages_invalid: bool`: true only for a nonempty list whose
  references all exist and are invalid.
- `invalid_relevant_passage_ids: list[int]`.
- `missing_relevant_passage_ids: list[int]`.

Questions with mixed valid/invalid references are flagged, too. Missing IDs and
empty relevance lists are distinguished from all-invalid references. Every
question and its full relevance list remain in normalized output. Duplicate
passage IDs cause analysis to raise an error rather than silently overwrite
validity; duplicate text under different IDs is retained.

Run `python scripts/inspect_dataset.py` to inspect and verify normalization in
memory (dataset config discovery requires network access). Run
`python -m unittest discover -s tests -v` for normalization checks without
network access. No normalized files are exported by this layer.

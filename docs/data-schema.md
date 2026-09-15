# Data Schema Notes

Do not finalize this schema until `scripts/inspect_dataset.py` has been run against the dataset.

## Target normalized entities

### Question
- query_id
- question
- reference_answer
- relevant_passage_ids
- optional question type

### Passage
- passage_id
- text
- source / source_url when available

### Retrieval result
- query_id
- retrieval_mode
- passage_id
- rank
- score
- optional component scores

The same normalized representation should be reused by lexical, dense, hybrid, UI, and evaluation code.

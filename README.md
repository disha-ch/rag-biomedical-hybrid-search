# Biomedical Hybrid Search

A neutral starter repository for the biomedical hybrid search exercise built around `rag-datasets/rag-mini-bioasq`.

## First milestone

Start by understanding and validating the dataset before implementing retrieval.

The first milestone is to:
1. load the dataset,
2. inspect questions, answers, passage IDs, relevant passage IDs, and corpus passages,
3. verify the relationships between questions and passages,
4. record basic dataset statistics,
5. define a stable internal schema that all retrieval methods can share.

This avoids coupling later BM25, dense, hybrid, query-expansion, UI, and evaluation code to assumptions about the raw dataset format.

## Planned project stages

1. Dataset inspection and normalization
2. Lexical retrieval
3. Dense retrieval
4. Hybrid retrieval
5. Query expansion
6. Grounded answer generation with citations
7. Search UI
8. Fixed 100-query offline evaluation
9. Retrieval + answer evaluation
10. System design and experiment report

## Repository structure

```text
biomedical-hybrid-search/
├── data/
│   ├── raw/            # optional local/raw exports (gitignored)
│   ├── processed/      # normalized local data (gitignored)
│   └── eval/           # fixed evaluation query set
├── docs/               # design notes / architecture
├── artifacts/          # experiment outputs (gitignored)
├── scripts/
│   └── inspect_dataset.py
├── src/
│   ├── data/
│   ├── retrieval/
│   ├── generation/
│   ├── evaluation/
│   └── ui/
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## First command

```bash
python scripts/inspect_dataset.py
```

The inspection script is intentionally minimal. Update the dataset configuration only after confirming the exact Hugging Face dataset structure.

## Design principle

Keep one common passage representation and one common retrieval result format across lexical, dense, and hybrid search. This makes the required offline comparison reproducible and fair.

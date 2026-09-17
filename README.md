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

## Lexical retrieval (BM25)

Install `requirements.txt`, then run three sample queries against the normalized
valid corpus:

```bash
python scripts/search_bm25.py
python scripts/search_bm25.py --query "EGFR signaling ligands" --top-k 3
python -m unittest discover -s tests -v
```

The script loads only the `text-corpus` config and normalizes it in memory.
If using the existing project-local cache, set
`HF_HOME="$PWD/data/raw/huggingface"`; add `HF_HUB_OFFLINE=1` to use it offline.

`src.retrieval.bm25.BM25Retriever(normalized_passages)` indexes only rows whose
`is_valid` is `True`. Its `search(query: str, top_k: int)` returns dictionaries
with exactly `passage_id`, `score`, `rank`, and `text`. Original IDs and text are
preserved, including duplicate text under different IDs. Ranks start at 1.

Scoring uses `rank-bm25==0.2.2`'s `BM25Okapi` with `k1=1.5`, `b=0.75`, and
`epsilon=0.25`. Both passage and query tokenization lowercase Unicode text and
extract runs of letters/digits; punctuation, hyphens, and underscores separate
tokens. There is no stemming or stopword removal.

Results are sorted by descending score, with corpus order breaking ties.
`top_k=0`, empty/whitespace/punctuation-only queries, and an empty vocabulary
return no results. Negative `top_k` is rejected; values larger than the corpus
return all indexed rows. No score threshold is applied: zero or negative scores
can be returned, and an unknown-term query returns zero-score rows in corpus
order. BM25 scores are not confidence probabilities.

The index is rebuilt in memory each run and scores the corpus for each query.
This is a simple lexical baseline, without retrieval-quality evaluation.

## Dense retrieval

```bash
python scripts/search_dense.py
python -m unittest discover -s tests -v
```

`src.retrieval.dense.DenseRetriever(normalized_passages)` uses
`sentence-transformers/all-MiniLM-L6-v2` to encode every valid passage once,
in batches of 32. Each `search(query, top_k)` encodes only the query and computes
a NumPy matrix-vector product against the stored L2-normalized passage vectors.
The query is also L2-normalized, so the scores are cosine similarities.

Results have exactly `passage_id`, `score`, `rank`, and `text`, sorted by descending
similarity with corpus-order ties. Original IDs, text, and duplicate-text rows
are retained. Empty/whitespace queries, an empty valid corpus, or `top_k=0`
return `[]`. Negative `top_k` is rejected; oversized values return all valid rows.

The first run downloads the model; subsequent runs can use the Hugging Face
cache. Use the same `HF_HOME` as above for the existing dataset cache. Embeddings
stay in memory and are rebuilt for each retriever instance. There is no vector
database, score threshold, or quality evaluation. The general-purpose model
produces 384-dimensional embeddings and truncates text beyond 256 wordpieces;
long passages are not chunked. Full original text is still returned. Unit tests
use fixed mocked embeddings; the sample script runs the actual model.

## Design principle

Keep one common passage representation and one common retrieval result format across lexical, dense, and hybrid search. This makes the required offline comparison reproducible and fair.

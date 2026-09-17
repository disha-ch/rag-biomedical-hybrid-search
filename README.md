# Biomedical Hybrid Search

Biomedical evidence search over `rag-datasets/rag-mini-bioasq`, with lexical,
semantic, hybrid, and query-expanded retrieval, a Streamlit UI, grounded
answers, and a fixed 100-query retrieval evaluation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set OPENAI_API_KEY in .env for query expansion and answers.
export HF_HOME="$PWD/data/raw/huggingface"
```

`OPENAI_MODEL` defaults to `gpt-4.1-mini`. Never commit credentials. Dataset and
embedding-model downloads happen on first use. Once cached, `HF_HUB_OFFLINE=1`
lets Hugging Face loading work offline; OpenAI features still require network
access and a key.

## Run

```bash
streamlit run app.py
python scripts/inspect_dataset.py
python scripts/search.py --mode lexical --query "EGFR signaling ligands"
python scripts/search.py --mode dense
python scripts/search.py --mode hybrid --top-k 5
python scripts/search.py --mode expanded --query "What genes are associated with Hirschsprung disease?" --answer
python -m unittest discover -s tests -v
```

The single demo script replaces the four former search scripts. Without
`--query`, it runs the same three biomedical examples. `--query` is repeatable;
`--answer` optionally generates a grounded answer from up to five results.

## Repository

```text
src/
  data/                 # normalization
  retrieval/            # BM25, dense, hybrid, expansion
  generation/           # grounded answer + citation checks
  evaluation/           # plain retrieval metric functions
tests/
scripts/
  inspect_dataset.py
  search.py
  create_eval_set.py
  evaluate_retrieval.py
data/
  raw/                  # ignored Hugging Face cache
  processed/            # reserved for local exports
  eval/
    queries_100.json     # fixed, shared question records
    results/            # per-mode details and summary JSON
docs/                   # schema and implementation notes
artifacts/              # reserved for local artifacts
app.py
requirements.txt
.env.example
```

## Retrieval and answers

Normalization preserves original question/passage IDs, relevance lists, row
order, and duplicate-text rows. Only valid passages are indexed: 27,977 of
40,221 in the inspected snapshot. See `docs/data-schema.md` for validity rules.

- **Lexical:** `rank-bm25`'s `BM25Okapi`, k1=1.5, b=0.75, epsilon=0.25.
  Lowercase Unicode letters/digits; punctuation separates tokens. No stemming
  or stopword removal.
- **Dense:** `sentence-transformers/all-MiniLM-L6-v2`, 384-dimensional vectors.
  Cosine similarity is the dot product of L2-normalized vectors. Text beyond
  256 wordpieces is truncated for embedding; full original text is returned.
- **Hybrid:** RRF sums `1 / (60 + rank)` over BM25 and dense candidates.
- **Hybrid + Query Expansion:** one structured OpenAI call generates exactly
  two distinct alternate queries; the original is retained verbatim. Hybrid
  runs on all three, then a second RRF merges their results.

All searches return `passage_id`, `score`, `rank`, and `text`, ordered by score.
Ranks start at 1. Ties use corpus order for BM25/dense and passage ID for RRF.
Empty/whitespace queries or top_k=0 return no results; negative top_k is rejected.
No score threshold is applied. Candidate depth equals the requested top_k.

Streamlit lazily caches retrievers in memory; the first dense search can take
several minutes. It shows original/expanded queries, evidence, a plain search
record, and an answer with inline `[passage_id]` citations. Source URLs are not
provided by the dataset, and the UI does not invent them.

Answer generation sends only the original question and selected context
(default five passages), with an evidence-only prompt and temperature 0.
Unusable context or `INSUFFICIENT_EVIDENCE` displays
`Insufficient evidence in the retrieved passages.` Missing or out-of-context
citations reject the answer. ID membership does not prove factual support;
prompting and temperature 0 cannot guarantee grounding or identical outputs.
API errors keep evidence visible and are not treated as insufficient evidence.

## Fixed 100-query evaluation

```bash
python scripts/create_eval_set.py
python scripts/evaluate_retrieval.py
# Optional rerun of one configuration, still using the exact same query file:
python scripts/evaluate_retrieval.py --modes expanded
```

Creation uses BioASQ revision `224a87f64a5c3a720b5bc627cf760f543b2f1e79`.
Questions with at least one valid relevant passage are sorted by original ID;
`random.Random(42).sample(..., 100)` selects the fixed set. Each saved row contains
`question_id`, `question`, `answer`, and the full original `relevant_passage_ids`.
An existing file is validated and reused without rewriting or resampling.
Evaluation requires this saved file and never generates another set.

Every mode retrieves ten candidates/results per query. The original relevance
lists are the ground truth, **including IDs with invalid passage text**. Such
passages cannot be retrieved, so perfect recall/nDCG may be unattainable.
This preserves the supplied labels instead of silently making the task easier.
With binary relevance and one credit per passage ID:

- **Recall@k:** number of distinct relevant IDs in the first k results / total
  distinct relevant IDs (k=5 and 10).
- **MRR@10:** reciprocal rank of the first relevant result within ten, or zero.
- **nDCG@10:** sum of `relevance / log2(rank + 1)` through rank ten, divided by
  the ideal sum for `min(10, number of relevant IDs)` relevant results.

Metrics are averaged equally over all 100 questions. Each mode's JSON records
query IDs, ranked retrieved IDs, scores, ranks, all four metrics, search latency,
and expansion usage. Metadata includes the fixed-file SHA-256 and dataset
revision. Latency includes query embedding/expansion where applicable, but
excludes loading and index construction; setup and total runtime are separate.
Indexes are built once and reused across modes. Partial/failed modes retain
per-query errors and do not report a misleading complete-set metric mean.

Without `OPENAI_API_KEY`, expanded hybrid is explicitly skipped; no mock
expansions are used in evaluation. Non-LLM retrieval has $0 external API cost
(local compute is not priced). Expanded hybrid records attempted call count
and response token usage when available. Its dollar cost is `null`/unknown;
no pricing estimate is invented. Results are saved under `data/eval/results/`;
rerunning a mode replaces its JSON, and `summary.json` describes that invocation.

Tests use mocks only for unit/UI checks. LLM-judge evaluation, the final report,
and the system-design diagram are not implemented yet.

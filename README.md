# Biomedical Hybrid Search RAG

Biomedical evidence search over `rag-datasets/rag-mini-bioasq`, with lexical,
semantic, hybrid, and query-expanded retrieval, a Streamlit UI, grounded
answers, and a fixed 100-query retrieval evaluation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Use the installed Ollama model configured in .env.
export HF_HOME="$PWD/data/raw/huggingface"
```

`OLLAMA_MODEL` defaults to the installed `hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M`.
Start Ollama locally (use `ollama serve` if it is not already running);
`ollama list` shows installed models. `OLLAMA_HOST` defaults to
`http://127.0.0.1:11434`. No cloud API key is required. Dataset and embedding-model
downloads happen on first use. Once cached, `HF_HUB_OFFLINE=1` enables offline loading.

## Run

```bash
streamlit run app.py
python scripts/inspect_dataset.py
python scripts/search.py --mode lexical --query "EGFR signaling ligands"
python scripts/search.py --mode dense
python scripts/search.py --mode hybrid --top-k 5
```

The single demo script replaces the four former search scripts. Without
`--query`, it runs the same three biomedical examples. `--query` is repeatable.
Use Streamlit for expansion and answers: the legacy demo still has an old
OpenAI-key guard for its LLM options and was left outside this migration scope.

## Local prerequisites and manual verification

For an existing checkout with dependencies and models already installed, use the
following commands from the repository root. No package installation, model
download, dataset regeneration, or benchmark run is needed for this check.

```bash
source .venv/bin/activate
ollama list
curl --fail --max-time 5 http://127.0.0.1:11434/api/tags
```

Confirm that `hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M` appears in the model list. The HTTP
request only lists installed models; it does not generate text. If Ollama is not
running, open another terminal, run `ollama serve`, and leave it running. Do not
start a second Ollama service when one is already reachable.

To start the application yourself, **only if it is not already running**:

```bash
source .venv/bin/activate
export HF_HOME="$PWD/data/raw/huggingface"
export HF_HUB_OFFLINE=1
export OLLAMA_MODEL='hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M'
export OLLAMA_HOST='http://127.0.0.1:11434'
streamlit run app.py
```

The offline setting assumes the dataset and MiniLM model are already cached in
this checkout. Open the URL printed by Streamlit, normally
[http://localhost:8501](http://localhost:8501). Keep the terminal open. If an
existing instance already serves this repository, reuse it instead of starting
another process. Starting the server does not itself build the dense index;
the first semantic or hybrid search does. That first search can take several
minutes because it encodes all valid passages. Subsequent searches reuse the
cached retriever while the process remains alive. Avoid restarting or clearing
the cache during manual validation.

### One end-to-end manual check

1. Open the application and confirm the retrieval selector contains **Lexical**,
   **Dense**, **Hybrid**, and **Hybrid + Query Expansion**.
2. Enter exactly: **What genes are associated with Hirschsprung disease?**
3. Select **Hybrid + Query Expansion**. Leave **Passages to retrieve** and
   **Maximum passages for answer context** at **5**.
4. Click **Search** once. Wait for index preparation, retrieval, and answer
   generation; do not resubmit while the request is running.
5. Confirm **Original query** matches what you entered and exactly **two**
   expanded queries appear. They should be distinct paraphrases about the same
   biomedical intent, not proposed answers.
6. Confirm **Ranked evidence** contains ranks, original passage IDs, scores, and
   full passage text. Ranks should start at 1 and scores should descend. The
   source label should read **Source link: not available in dataset**.
7. Read the **Grounded answer** and confirm inline numeric `[passage_id]`
   citations appear. Confirm **Citation validation: valid** is displayed.
8. Match each cited ID to one of the first five evidence passages. Read those
   passages yourself to check whether they support the associated claims.
   The automatic validator checks ID membership, not factual support.
9. Expand **Search record** to inspect `original_query`, `expanded_queries`,
   `retrieval_mode`, and `results`. Every result has `passage_id`, `score`,
   `rank`, and `text`; IDs need not be consecutive.

This single UI check exercises expansion, both retrieval branches, rank fusion,
answer generation, and deterministic citation validation. It is not a benchmark
or a substitute for a correctness test suite. The separate LLM judge is not
invoked by the UI. An explicit insufficient-evidence result is a supported
application outcome, but does not establish that this particular citation-based
smoke check passed. Likewise, a valid citation status alone does not prove the
answer is complete or medically correct.

For optional manual comparison after the initial check, run the same question
in the other three modes in the same browser session. Only the expanded mode
should display alternate queries. Do not compare the numerical score magnitude
across modes: BM25 scores, cosine similarities, and RRF scores have different
meanings. No new embeddings should be needed while the dense cache is retained.

### Understanding what each component verifies

| Stage | Implementation | Observable evidence |
| --- | --- | --- |
| Normalize | `src/data/normalize.py` | Original IDs/text retained; invalid text excluded only from indexing |
| Lexical search | `src/retrieval/bm25.py` | Term-based ranked passages |
| Semantic search | `src/retrieval/dense.py` | MiniLM cosine-ranked passages |
| Hybrid search | `src/retrieval/hybrid.py` | BM25 and dense rankings fused by passage ID |
| Expansion | `src/retrieval/query_expansion.py` | Original query plus exactly two validated alternates |
| Expanded fusion | `src/retrieval/expanded_hybrid.py` | Three hybrid result lists merged with a second RRF |
| Answer and citations | `src/generation/answer.py` | Evidence-only prompt, final answer, citation status and cited IDs |
| UI | `app.py` | Query, expansions, ranked evidence, answer, and search record |
| Answer-quality judge | `src/evaluation/llm_judge.py` | Separate 1–5 ratings and reasons; not called by the UI |

The complete path is: cached BioASQ → normalization → valid passages → BM25 and
MiniLM indexes → optional local expansion → retrieval and RRF → selected context
→ local Ollama answer → citation validation → displayed answer. The original
question is used for answering even when expanded queries were used to retrieve
evidence. Retrieval never uses the reference answer. Reference answers are only
inputs to offline evaluation or the separate answer-quality judge.

### Troubleshooting without rebuilding the project

- **Cannot reach Ollama:** run `ollama serve` in a separate terminal, then repeat
  the model-list command. Check that `OLLAMA_HOST` matches the running service.
- **Model missing:** check `ollama list` and the exact `OLLAMA_MODEL` value. This
  project assumes the named model is already installed; verification does not
  require downloading a replacement.
- **First search appears slow:** leave Streamlit and Ollama running and watch
  their terminal output. Initial embeddings and local generation take time.
- **Dataset/model unavailable offline:** verify `HF_HOME` points to the existing
  cache. An offline cache error is not a reason to regenerate evaluation data.
- **Answer unavailable:** inspect Ollama's status and terminal errors. Retrieved
  evidence remains visible when generation fails. HTTP calls have a 180-second
  timeout; local model loading and memory pressure can cause a timeout.
- **Citation rejected:** do not treat the answer as verified. Missing citations
  or IDs outside the selected answer context cause rejection.
- **Port already in use:** reuse the existing app if it belongs to this checkout;
  do not kill an unrelated process. Streamlit prints the URL it actually uses.

Keep the server running until manual verification is finished. Use Ctrl+C in its
terminal only when you deliberately want to stop it. Stopping loses in-memory
retrievers, so the next semantic search must rebuild them.

## Repository

```text
src/
  data/                 # normalization
  retrieval/            # BM25, dense, hybrid, expansion
  generation/           # grounded answer + citation checks
  evaluation/           # retrieval metrics + local LLM judge
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
- **Hybrid + Query Expansion:** one structured local Ollama call generates exactly
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

Local Ollama answer generation sends only the original question and selected context
(default five passages), with an evidence-only prompt and temperature 0.
Unusable context or `INSUFFICIENT_EVIDENCE` displays
`Insufficient evidence in the retrieved passages.` Missing or out-of-context
citations reject the answer. ID membership does not prove factual support;
prompting and temperature 0 cannot guarantee grounding or identical outputs.
API errors keep evidence visible and are not treated as insufficient evidence.

## Fixed 100-query evaluation

The saved query set and retrieval results are frozen; they were not regenerated
or rerun during the Ollama migration.

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

Existing measured results from `data/eval/results/summary.json`:

| Mode | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Mean search latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lexical | 0.4293 | 0.5029 | 0.8306 | 0.6228 | 80.5 ms |
| Dense | 0.3040 | 0.3993 | 0.6659 | 0.4829 | 14.2 ms |
| Hybrid | 0.4360 | 0.5268 | 0.7943 | 0.6185 | 92.6 ms |

Hybrid has the strongest measured recall; lexical has the strongest MRR/nDCG.
This is a retrieval trade-off, not dominance across every metric. Expanded-hybrid
100-query metrics were not measured. The frozen evaluation runner still contains
its historical OpenAI-key guard and metadata; its expanded mode has not been
migrated. Existing result files retain their original provenance.

## Local answer evaluation

`src/evaluation/llm_judge.py` provides `judge_answer(question, reference_answer,
retrieved_context, generated_answer)`. One fixed prompt and the configured Ollama
model at temperature 0 return correctness, groundedness, and context relevance,
each with a 1–5 score and short reason. Citation validity remains a separate,
deterministic check. The judge is a heuristic, not a factual guarantee; using the
same model for generation and judging may introduce bias. No 100-query judge run
was performed. Generation, expansion, and judging have **$0 external API cost**;
local inference still consumes compute, memory, and electricity.

The existing tests are frozen and were not run in this pass. Provider-specific
OpenAI mocks and UI assertions remain legacy checks and need a separately
approved update for Ollama.

## Architecture and limitations

See [architecture](docs/architecture.md) for data, indexing, query, and evaluation flows.

- MiniLM is a general-purpose embedding model, not biomedical-specific.
- Dense embeddings are rebuilt in memory; there is no vector database.
- The dataset has no source URLs.
- Citation validation confirms passage-ID membership, not factual entailment.
- Local LLM quality and latency depend on the installed Ollama model and hardware.
- Expanded-hybrid 100-query metrics were not measured.

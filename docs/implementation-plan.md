# Implementation Plan

## Phase 0 — Dataset understanding
- Inspect splits and fields.
- Confirm passage IDs and relevance labels.
- Confirm whether source links are present.
- Normalize questions and passages into stable internal records.
- Capture dataset statistics.

## Phase 1 — Retrieval baselines
- Lexical retrieval.
- Dense retrieval.
- Shared result format and logging.

## Phase 2 — Hybrid retrieval
- Combine lexical and dense signals.
- Keep fusion configurable.

## Phase 3 — Query expansion
- Preserve the original query.
- Generate a small set of alternate queries.
- Reuse the same retrieval pipeline.

## Phase 4 — Generation and UI
- Select retrieved context.
- Generate answers only from selected passages.
- Validate passage-ID citations.
- Support an insufficient-evidence state.

## Phase 5 — Offline evaluation
- Freeze one 100-query evaluation set.
- Run all configurations on exactly the same query IDs.
- Compute required retrieval metrics, latency, and cost.
- Add answer-level judging and manual review.

## Phase 6 — Final report
- Compare configurations.
- Select the strongest variant using measured results.
- Document latency/cost trade-offs, successes, and failures.

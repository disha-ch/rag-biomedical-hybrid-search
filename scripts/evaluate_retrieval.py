"""Evaluate real retrievers against the same saved 100 queries; no answer judging."""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.create_eval_set import DATASET, REVISION, EVAL_PATH, SEED, load_eval_set
from src.evaluation.retrieval_metrics import retrieval_metrics

NAMES = {"lexical": "Lexical", "dense": "Dense", "hybrid": "Hybrid", "expanded": "Hybrid + Query Expansion"}
RESULTS_DIR = ROOT / "data/eval/results"


def evaluate_queries(queries, retriever, *, expanded=False):
    details = []
    for q in queries:
        usage = {}
        start = perf_counter()
        row = {"query_id": q["question_id"], "question": q["question"],
               "relevant_passage_ids": q["relevant_passage_ids"], "expanded_queries": [],
               "retrieved_passage_ids": [], "scores": [], "ranks": []}
        try:
            if expanded:
                record = retriever.search_record(q["question"], 10, usage=usage)
                results = record["results"]
                row["expanded_queries"] = record["expanded_queries"]
            else:
                results = retriever.search(q["question"], 10)
            row["latency_seconds"] = perf_counter() - start
            row.update(retrieved_passage_ids=[r["passage_id"] for r in results],
                       scores=[r["score"] for r in results], ranks=[r["rank"] for r in results])
            row.update(retrieval_metrics(row["retrieved_passage_ids"], q["relevant_passage_ids"]))
        except Exception as error:
            row["latency_seconds"] = perf_counter() - start
            row["error"] = type(error).__name__
            row.update({name: None for name in retrieval_metrics([], [])})
        row.update(expansion_calls=usage.get("expansion_calls", 0),
                   token_usage=usage.get("token_usage"), external_api_cost_usd=None if expanded else 0.0)
        details.append(row)
        if len(details) % 25 == 0:
            print(f"  {len(details)}/{len(queries)} queries processed", flush=True)
    complete = bool(details) and all("error" not in row for row in details)
    tokens = [row["token_usage"] for row in details if row["token_usage"] is not None]
    summary = {
        "status": "complete" if complete else "failed_or_partial",
        "query_count": len(details), "successful_queries": sum("error" not in r for r in details),
        # Never present a mean over fewer queries as a complete 100-query result.
        "mean_metrics": {name: mean(r[name] for r in details) for name in retrieval_metrics([], [])}
        if complete else None,
        "mean_latency_seconds": mean(r["latency_seconds"] for r in details) if details else None,
        "expansion_calls": sum(r["expansion_calls"] for r in details),
        "token_usage": {name: sum(t[name] for t in tokens) for name in tokens[0]} if tokens else None,
        "external_api_cost_usd": None if expanded else 0.0,
    }
    return {"summary": summary, "queries": details}


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=NAMES, default=list(NAMES))
    args = parser.parse_args()
    if not EVAL_PATH.exists():
        parser.error("Run python scripts/create_eval_set.py first; evaluation never regenerates the set.")
    queries = load_eval_set()
    started = perf_counter()
    metadata = {"dataset": DATASET, "dataset_revision": REVISION, "selection_seed": SEED,
                "query_file_sha256": hashlib.sha256(EVAL_PATH.read_bytes()).hexdigest(),
                "query_ids": [q["question_id"] for q in queries], "top_k": 10,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "ground_truth": "full original relevant_passage_ids, including invalid corpus text",
                "latency_scope": "search only; excludes corpus loading and index construction"}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summaries = {}
    modes = list(dict.fromkeys(args.modes))
    if "expanded" in modes and not os.getenv("OPENAI_API_KEY"):
        modes.remove("expanded")
        summaries[NAMES["expanded"]] = {"status": "skipped", "reason": "OPENAI_API_KEY is required", "query_count": 0, "expansion_calls": 0, "external_api_cost_usd": 0.0}
        (RESULTS_DIR / "expanded.json").write_text(json.dumps(
            {"metadata": metadata, "retrieval_mode": NAMES["expanded"],
             "summary": summaries[NAMES["expanded"]], "queries": []}, indent=2) + "\n")
        print("SKIP Hybrid + Query Expansion: OPENAI_API_KEY is required; no mocked results.", flush=True)

    retrievers = {}
    if modes:
        from datasets import load_dataset
        from src.data.normalize import normalize_passages
        from src.retrieval.bm25 import BM25Retriever
        from src.retrieval.dense import DenseRetriever, MODEL_NAME
        from src.retrieval.hybrid import HybridRetriever
        from src.retrieval.expanded_hybrid import ExpandedHybridRetriever
        passages = normalize_passages(load_dataset(DATASET, name="text-corpus", split="passages", revision=REVISION))
        valid_ids = {p["id"] for p in passages if p["is_valid"]}
        if any(not valid_ids.intersection(q["relevant_passage_ids"]) for q in queries):
            raise ValueError("Every evaluation question must have at least one valid relevant passage.")
        metadata.update(indexed_passages=len(valid_ids), embedding_model=MODEL_NAME, rrf_k=60)
        if any(m in modes for m in ("lexical", "hybrid", "expanded")):
            retrievers["lexical"] = BM25Retriever(passages)
        if any(m in modes for m in ("dense", "hybrid", "expanded")):
            retrievers["dense"] = DenseRetriever(passages)
        if any(m in modes for m in ("hybrid", "expanded")):
            retrievers["hybrid"] = HybridRetriever(retrievers["lexical"], retrievers["dense"])
        if "expanded" in modes:
            retrievers["expanded"] = ExpandedHybridRetriever(retrievers["hybrid"])
            metadata["expansion_model"] = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    metadata["setup_seconds"] = perf_counter() - started
    for mode in modes:
        print(f"Evaluating {NAMES[mode]} on the same {len(queries)} queries…", flush=True)
        report = evaluate_queries(queries, retrievers[mode], expanded=mode == "expanded")
        report.update(metadata=metadata, retrieval_mode=NAMES[mode])
        summaries[NAMES[mode]] = report["summary"]
        (RESULTS_DIR / f"{mode}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    metadata["total_seconds"] = perf_counter() - started
    (RESULTS_DIR / "summary.json").write_text(json.dumps({"metadata": metadata, "configurations": summaries}, indent=2) + "\n")
    print(f"\n{'Mode':26} {'R@5':>8} {'R@10':>8} {'MRR@10':>8} {'nDCG@10':>8} {'Mean ms':>10}")
    for name in NAMES.values():
        if name not in summaries:
            continue
        row = summaries[name]
        if row["status"] != "complete":
            print(f"{name:26} {row['status']} {row.get('reason', '')}")
            continue
        m = row["mean_metrics"]
        print(f"{name:26} {m['Recall@5']:8.4f} {m['Recall@10']:8.4f} {m['MRR@10']:8.4f} {m['nDCG@10']:8.4f} {1000 * row['mean_latency_seconds']:10.1f}")
    print(f"Total runtime: {metadata['total_seconds']:.1f}s; setup: {metadata['setup_seconds']:.1f}s")
    if any(row["status"] == "failed_or_partial" for row in summaries.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""One demo for all retrieval modes, with optional grounded answer generation."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("lexical", "dense", "hybrid", "expanded"), default="lexical")
    parser.add_argument("--query", action="append", help="Repeat for multiple queries.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--answer", action="store_true", help="Also generate a grounded answer (requires an API key).")
    args = parser.parse_args()
    if args.top_k < 0:
        parser.error("--top-k must be nonnegative")
    if (args.mode == "expanded" or args.answer) and not os.getenv("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY in your environment or .env for expansion/answers.")
    from datasets import load_dataset
    from src.data.normalize import normalize_passages
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.expanded_hybrid import ExpandedHybridRetriever
    from src.generation.answer import generate_answer

    passages = normalize_passages(load_dataset("rag-datasets/rag-mini-bioasq", name="text-corpus", split="passages"))
    if args.mode == "lexical":
        retriever = BM25Retriever(passages)
    elif args.mode == "dense":
        retriever = DenseRetriever(passages)
    else:
        retriever = HybridRetriever(BM25Retriever(passages), DenseRetriever(passages))
        if args.mode == "expanded":
            retriever = ExpandedHybridRetriever(retriever)
    print("Indexed valid passages:", sum(p["is_valid"] for p in passages), flush=True)
    queries = args.query if args.query is not None else [
        "Hirschsprung disease genetics", "EGFR signaling ligands", "Parkinson disease dopamine"]
    for query in queries:
        record = retriever.search_record(query, args.top_k) if args.mode == "expanded" else {
            "original_query": query, "expanded_queries": [], "retrieval_mode": args.mode,
            "results": retriever.search(query, args.top_k)}
        print("\nOriginal query:", record["original_query"])
        print("Retrieval mode:", record["retrieval_mode"])
        for i, alternate in enumerate(record["expanded_queries"], 1):
            print(f"Expansion {i}: {alternate}")
        for row in record["results"]:
            preview = " ".join(row["text"].split())[:160]
            print(f"rank={row['rank']} passage_id={row['passage_id']} score={row['score']:.6f} text={preview!r}")
        if not record["results"]:
            print("No results.")
        if args.answer:
            answer = generate_answer(query, record["results"])
            print("Grounded answer:", answer["answer"] or answer["warning"])
            print("Citation validation:", answer["citation_status"])


if __name__ == "__main__":
    main()

"""Build the lexical index and print sample results; no evaluation or answering."""

import argparse
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.normalize import normalize_passages
from src.retrieval.bm25 import BM25Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", action="append", help="Repeat for multiple queries.")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.top_k < 0:
        parser.error("--top-k must be nonnegative")

    raw = load_dataset("rag-datasets/rag-mini-bioasq", name="text-corpus", split="passages")
    passages = normalize_passages(raw)
    retriever = BM25Retriever(passages)
    print(f"Indexed valid passages: {len(retriever)} / {len(passages)}", flush=True)
    queries = args.query if args.query is not None else [
        "Hirschsprung disease genetics",
        "EGFR signaling ligands",
        "Parkinson disease dopamine",
    ]
    for query in queries:
        print(f"\nQuery: {query}")
        results = retriever.search(query, args.top_k)
        if not results:
            print("No results.")
        for result in results:
            preview = " ".join(result["text"].split())[:160]
            print(f"rank={result['rank']} passage_id={result['passage_id']} "
                  f"score={result['score']:.6f} text={preview!r}")


if __name__ == "__main__":
    main()

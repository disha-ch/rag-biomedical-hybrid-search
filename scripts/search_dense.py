"""Encode the valid corpus once and run three sample semantic searches."""

import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.normalize import normalize_passages
from src.retrieval.dense import DenseRetriever


def main() -> None:
    raw = load_dataset("rag-datasets/rag-mini-bioasq", name="text-corpus", split="passages")
    retriever = DenseRetriever(normalize_passages(raw))
    print(f"Indexed valid passages: {len(retriever)} / {len(raw)}", flush=True)
    for query in (
        "Hirschsprung disease genetics",
        "EGFR signaling ligands",
        "Parkinson disease dopamine",
    ):
        print(f"\nQuery: {query}", flush=True)
        for result in retriever.search(query, top_k=3):
            preview = " ".join(result["text"].split())[:160]
            print(f"rank={result['rank']} passage_id={result['passage_id']} "
                  f"score={result['score']:.6f} text={preview!r}", flush=True)


if __name__ == "__main__":
    main()

"""Small Streamlit interface for biomedical retrieval and evidence-based answers."""

from pathlib import Path

import streamlit as st
from datasets import load_dataset
from dotenv import load_dotenv

from src.data.normalize import normalize_passages
from src.generation.answer import generate_answer
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.expanded_hybrid import ExpandedHybridRetriever

load_dotenv(Path(__file__).resolve().parent / ".env")


@st.cache_resource(show_spinner="Loading biomedical passages…")
def get_passages():
    return normalize_passages(load_dataset(
        "rag-datasets/rag-mini-bioasq", name="text-corpus", split="passages"))


@st.cache_resource(show_spinner="Building lexical index…")
def get_bm25():
    return BM25Retriever(get_passages())


@st.cache_resource(show_spinner="Preparing semantic search. The first run can take several minutes…")
def get_dense():
    return DenseRetriever(get_passages())


def run_search(query, mode, top_k):
    if mode == "Lexical":
        retriever = get_bm25()
    elif mode == "Dense":
        retriever = get_dense()
    else:
        retriever = HybridRetriever(get_bm25(), get_dense())
        if mode == "Hybrid + Query Expansion":
            return ExpandedHybridRetriever(retriever).search_record(query, top_k)
    return {"original_query": query, "expanded_queries": [], "retrieval_mode": mode,
            "results": retriever.search(query, top_k)}


def main():
    st.set_page_config(page_title="Biomedical Evidence Search", page_icon="🔎", layout="centered")
    st.title("Biomedical Evidence Search")
    st.caption("Search biomedical passages, inspect the evidence, and read an answer with passage-ID citations.")
    with st.form("search"):
        query = st.text_input("Ask a biomedical question")
        mode = st.selectbox("Retrieval mode", ["Lexical", "Dense", "Hybrid", "Hybrid + Query Expansion"])
        top_k = st.number_input("Passages to retrieve", min_value=1, max_value=20, value=5)
        context_k = st.number_input("Maximum passages for answer context", min_value=1, max_value=10, value=5)
        submitted = st.form_submit_button("Search", type="primary")

    if submitted:
        # A failed or empty new search must never show the previous answer.
        for key in ("record", "answer", "answer_error", "context_k"):
            st.session_state.pop(key, None)
        if not query.strip():
            st.warning("Enter a biomedical question.")
        else:
            try:
                with st.spinner("Retrieving evidence…"):
                    st.session_state.record = run_search(query, mode, top_k)
                st.session_state.context_k = context_k
            except Exception as error:
                st.error(f"Search failed ({type(error).__name__}). Check dataset/model access and local Ollama, then retry.")
            if "record" in st.session_state:
                try:
                    with st.spinner("Writing an answer from the retrieved evidence…"):
                        st.session_state.answer = generate_answer(
                            query, st.session_state.record["results"], context_top_k=context_k)
                except Exception as error:
                    st.session_state.answer_error = (
                        f"Answer generation unavailable ({type(error).__name__}). "
                        "Check the local Ollama service and OLLAMA_MODEL, then retry. Retrieved evidence is shown below."
                    )

    record = st.session_state.get("record")
    st.subheader("Original query")
    st.text(record["original_query"] if record else query or "Enter a question above.")
    if not record:
        return
    st.caption(f"Retrieval mode: {record['retrieval_mode']}")
    for i, alternate in enumerate(record["expanded_queries"], 1):
        st.write(f"Expansion {i}")
        st.text(alternate)
    st.subheader("Ranked evidence")
    if not record["results"]:
        st.info("No passages retrieved.")
    for result in record["results"]:
        with st.container(border=True):
            st.write(f"Rank {result['rank']} · Passage ID {result['passage_id']} · Score {result['score']:.6f}")
            st.text(result["text"])
            st.caption("Source link: not available in dataset")
    st.subheader("Grounded answer")
    st.caption(f"Answer context: up to the first {st.session_state.context_k} retrieved passages.")
    answer = st.session_state.get("answer")
    if answer:
        if answer["insufficient_evidence"]:
            st.warning(answer["answer"])
        elif answer["warning"]:
            st.error(answer["warning"])
        else:
            st.markdown(answer["answer"])
        st.caption(f"Citation validation: {answer['citation_status']}")
        if answer["citation_status"] == "valid":
            st.caption("Cited IDs are in the answer context; this check does not verify that each claim is supported.")
    else:
        st.warning(st.session_state.get("answer_error", "Answer unavailable."))
        st.caption("Citation validation: not run")
    with st.expander("Search record"):
        st.json(record)


if __name__ == "__main__":
    main()

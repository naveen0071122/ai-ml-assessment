"""
app.py -- Streamlit UI for Task 1.

Run:
    streamlit run app.py

Lets the user:
  - pick a text query OR upload an image query
  - choose retrieval mode: Multimodal vs Text-only baseline
  - choose Top-K
  - see, per result: doc title/id, page, snippet or image, relevance score
    (explicitly labelled as "not a calibrated probability"), match reason,
    and per-query latency broken down by pipeline stage.

Kept intentionally plain (per the brief: "do not spend excessive time on
UI styling; functionality is more important").
"""
import os
import sys
import tempfile
import time

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion import ingest_corpus, _load_manifest  # noqa: E402
from baseline import TextOnlyRAG  # noqa: E402
from retrieval import MultimodalRAG  # noqa: E402
from multimodal import USE_OFFLINE_MOCK, RESOLVED_BACKEND  # noqa: E402

st.set_page_config(page_title="Intranet Multimodal Search", layout="wide")


@st.cache_resource(show_spinner="Ingesting corpus and building indexes (first run only)...")
def load_pipelines():
    chunks, images = ingest_corpus()
    manifest = _load_manifest()
    text_rag = TextOnlyRAG(chunks)
    mm_rag = MultimodalRAG(chunks, images, manifest)
    return text_rag, mm_rag, chunks, images, manifest


text_rag, mm_rag, chunks, images, manifest = load_pipelines()

st.title("Intranet Multimodal Document Search")
st.caption(
    "Prototype: Qwen-VL-family multimodal retrieval vs. a text-only RAG baseline, "
    "over a synthetic 12-document intranet corpus."
)

if RESOLVED_BACKEND == "mock":
    st.info(
        "Running with the **offline mock VLM client** (no `DASHSCOPE_API_KEY` set, "
        "`QWEN_VL_BACKEND` not set to `local`). Image captions/reranking are "
        "deterministic stand-ins grounded in this corpus's own ground truth, not "
        "live Qwen-VL calls. Set `DASHSCOPE_API_KEY` for the hosted API, or "
        "`QWEN_VL_BACKEND=local` to run Qwen2-VL on this machine. See README.",
        icon="ℹ️",
    )
elif RESOLVED_BACKEND == "local":
    st.success(
        "Running with a **local Qwen2-VL model** (`QWEN_VL_BACKEND=local`) — "
        "captions/reranking come from a real VLM running on this machine. First "
        "query after startup may be slow while the model loads.",
        icon="🖥️",
    )
elif RESOLVED_BACKEND == "hosted":
    st.success(
        "Running with the **hosted Qwen-VL API** (DashScope, `qwen-vl-max`) — "
        "captions/reranking are real live model calls.",
        icon="☁️",
    )

with st.sidebar:
    st.header("Query")
    query_mode = st.radio("Query type", ["Text", "Image"], horizontal=True)
    text_query = None
    uploaded_image_path = None

    if query_mode == "Text":
        text_query = st.text_input("Enter your query", placeholder="e.g. What was the highest-revenue region in Q3?")
    else:
        uploaded = st.file_uploader("Upload an image to find similar documents", type=["png", "jpg", "jpeg"])
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.write(uploaded.getvalue())
            tmp.close()
            uploaded_image_path = tmp.name
            st.image(uploaded.getvalue(), caption="Query image", use_container_width=True)

    st.header("Settings")
    retrieval_mode = st.radio("Retrieval mode", ["Multimodal", "Text-only baseline"], horizontal=True)
    top_k = st.slider("Top-K", min_value=1, max_value=10, value=5)

    st.header("Corpus")
    st.caption(f"{len(set(c.doc_id for c in chunks) | set(i.doc_id for i in images))} documents, "
               f"{len(chunks)} text chunks, {len(images)} images indexed.")
    with st.expander("Browse documents"):
        for doc in manifest:
            st.write(f"**{doc['doc_id']}** ({doc['kind']}) — {os.path.basename(doc['path'])}")

    run = st.button("Search", type="primary", use_container_width=True)

if run:
    if retrieval_mode == "Text-only baseline" and query_mode == "Image":
        st.error("The text-only baseline cannot accept image queries by design — it has no access to visual content. Switch to Multimodal mode, or enter a text query.")
    elif not text_query and not uploaded_image_path:
        st.warning("Enter a text query or upload an image first.")
    else:
        t0 = time.time()
        if retrieval_mode == "Multimodal":
            results, timings = mm_rag.search(text_query=text_query, image_query_path=uploaded_image_path, top_k=top_k)
            total_latency = timings["total_ms"]
        else:
            results, latency_ms = text_rag.search(text_query or "", top_k=top_k)
            timings = None
            total_latency = latency_ms

        st.subheader(f"Results ({retrieval_mode}) — {total_latency:.1f} ms total")

        if timings:
            c1, c2, c3 = st.columns(3)
            c1.metric("Query understanding", f"{timings['query_understanding_ms']:.1f} ms")
            c2.metric("Candidate retrieval", f"{timings['candidate_retrieval_ms']:.1f} ms")
            c3.metric("Qwen-VL rerank", f"{timings['vlm_rerank_ms']:.1f} ms")
            st.caption(timings["understanding_note"])
            st.caption(
                f"{timings['num_candidates']} candidates retrieved, "
                f"{timings['num_image_candidates_reranked']} image candidate(s) sent to the VLM reranker."
            )

        if not results:
            st.write("No results above the relevance threshold.")

        for r in results:
            with st.container(border=True):
                cols = st.columns([3, 1]) if getattr(r, "kind", "text") == "image" else [st.container()]
                title = getattr(r, "title", None) or ""
                doc_id = getattr(r, "doc_id", "")
                page = getattr(r, "page", None)
                # NOTE: do not use `or` to chain these -- a real (correctly
                # computed) score of exactly 0.0 is falsy in Python, so
                # `final_score or getattr(...)` would wrongly fall through
                # to None whenever the VLM legitimately scored something 0
                # relevance. Confirmed by a real user hitting this crash.
                score = getattr(r, "final_score", None)
                if score is None:
                    score = getattr(r, "score", None)
                if score is None:
                    score = 0.0
                kind = getattr(r, "kind", "text")

                st.markdown(f"**{title}**  ·  `{doc_id}`  ·  page {page if page is not None else '—'}  ·  kind: `{kind}`")
                st.progress(min(1.0, max(0.0, score)), text=f"Relevance indicator: {score:.3f} (not a calibrated probability)")
                if hasattr(r, "match_reason"):
                    st.caption(f"Why it matched: {r.match_reason}")

                if kind == "image" and getattr(r, "image_path", None) and os.path.exists(r.image_path):
                    col_img, col_text = st.columns([1, 2])
                    with col_img:
                        st.image(r.image_path, use_container_width=True)
                    with col_text:
                        st.write(r.snippet)
                else:
                    st.write(r.snippet)

st.divider()
st.caption(
    "See evaluation.py / data/eval_results.json for quantitative Precision@K, Recall@K, "
    "MRR and latency comparisons against the text-only baseline."
)

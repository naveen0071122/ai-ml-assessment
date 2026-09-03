"""
tests/test_retrieval.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion import ingest_corpus, _load_manifest  # noqa: E402
from baseline import TextOnlyRAG  # noqa: E402
from retrieval import MultimodalRAG  # noqa: E402
from evaluation import precision_at_k, recall_at_k, hit_rate_at_k, reciprocal_rank  # noqa: E402


def _build_pipelines():
    chunks, images = ingest_corpus()
    manifest = _load_manifest()
    return TextOnlyRAG(chunks), MultimodalRAG(chunks, images, manifest)


def test_text_only_baseline_finds_relevant_doc_for_text_query():
    text_rag, _mm_rag = _build_pipelines()
    results, _latency = text_rag.search("What should new engineers do in their first week?", top_k=3)
    assert any(r.doc_id == "doc02" for r in results)


def test_text_only_baseline_has_no_image_search_capability():
    """Guardrail against 'faking' multimodal by construction: baseline
    module must not expose any way to pass an image query."""
    import inspect
    from baseline import TextOnlyRAG
    sig = inspect.signature(TextOnlyRAG.search)
    assert "image" not in " ".join(sig.parameters.keys()).lower()


def test_multimodal_finds_chart_only_fact():
    """The core claim of the pipeline: it should surface doc01 for a query
    whose answer lives only in the revenue bar chart."""
    _text_rag, mm_rag = _build_pipelines()
    results, _timings = mm_rag.search(text_query="What was the highest-revenue region in Q3?", top_k=3)
    assert any(r.doc_id == "doc01" for r in results)
    # and specifically via the image unit, not just the surrounding text
    image_hits = [r for r in results if r.doc_id == "doc01" and r.kind == "image"]
    assert image_hits, "expected the chart image itself to be a retrieved candidate"


def test_multimodal_standalone_image_findable_with_no_text_at_all():
    _text_rag, mm_rag = _build_pipelines()
    results, _timings = mm_rag.search(text_query="Which rack is near the cooling unit?", top_k=3)
    assert any(r.doc_id == "doc05" for r in results)


def test_text_only_baseline_cannot_find_standalone_image_doc():
    """Fairness check: doc05 has ZERO extractable text, so the text-only
    baseline must never be able to retrieve it -- if it could, something
    would be leaking visual info into the 'text-only' index."""
    text_rag, _mm_rag = _build_pipelines()
    results, _latency = text_rag.search("Which rack is near the cooling unit?", top_k=10)
    assert not any(r.doc_id == "doc05" for r in results)


# ---- metric unit tests ----

def test_precision_recall_hit_perfect_match():
    retrieved = ["docA", "docB", "docC"]
    relevant = {"docA"}
    assert precision_at_k(retrieved, relevant, 3) == 1 / 3
    assert recall_at_k(retrieved, relevant, 3) == 1.0
    assert hit_rate_at_k(retrieved, relevant, 3) == 1.0


def test_precision_recall_hit_no_match():
    retrieved = ["docX", "docY"]
    relevant = {"docA"}
    assert precision_at_k(retrieved, relevant, 2) == 0.0
    assert recall_at_k(retrieved, relevant, 2) == 0.0
    assert hit_rate_at_k(retrieved, relevant, 2) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["docB", "docA"], {"docA"}) == 0.5
    assert reciprocal_rank(["docA", "docB"], {"docA"}) == 1.0
    assert reciprocal_rank(["docX"], {"docA"}) == 0.0


def test_multimodal_includes_image_candidates_even_with_zero_lexical_overlap():
    """Regression test for a real bug found during manual testing with the
    local Qwen2-VL backend: a real VLM caption can legitimately share ZERO
    vocabulary with the query (e.g. captioning a bar chart as "a chart with
    colored bars" instead of using the word "revenue"). Before the fix, the
    TF-IDF candidate filter (sims[i] > 0) dropped such images before the
    VLM reranker -- whose whole job is to judge relevance by looking at the
    actual image -- ever got a chance to score them. This test simulates
    that exact scenario with a synthetic zero-overlap caption and asserts
    the image still reaches the candidate pool."""
    chunks, images = ingest_corpus()
    manifest = _load_manifest()
    mm_rag = MultimodalRAG(chunks, images, manifest)

    # Overwrite doc01's caption with something that shares NO words with
    # the query below, simulating a real captioner's vocabulary mismatch.
    doc01_image_id = next(i.image_id for i in images if i.doc_id == "doc01")
    for entry in mm_rag._entries:
        if entry["kind"] == "image" and entry["ref"].image_id == doc01_image_id:
            entry["text"] = "zzz qqq xyz unrelated placeholder words nonsense"
    # Rebuild the TF-IDF index so the corrupted caption is reflected in it
    # (mirrors what caption_all_images + __init__ do at construction time).
    from sklearn.feature_extraction.text import TfidfVectorizer
    mm_rag.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    mm_rag.matrix = mm_rag.vectorizer.fit_transform([e["text"] for e in mm_rag._entries])

    results, timings = mm_rag.search(text_query="What was the highest-revenue region in Q3?", top_k=5)
    assert timings["num_image_candidates_reranked"] >= 1, (
        "expected at least one image candidate to reach the VLM reranker "
        "even with zero TF-IDF overlap -- the guaranteed-image-slots fix regressed"
    )


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

"""
baseline.py
A genuinely text-only RAG pipeline: PDF text extraction (already done in
ingestion.py) -> chunk -> TF-IDF embedding -> cosine similarity search.

This module NEVER touches image_units or any VLM caption. That's the
whole point of it being the baseline: doc07 (expense report) and doc05
(floor plan) and every other visual-only-fact document should be
UNFINDABLE here for queries whose answer lives only in the image -- if
this baseline can somehow answer those, the comparison in evaluation.py
would not be fair, so this constraint is enforced by construction (no
image_units parameter exists on this module's functions at all).

Embedding choice: TF-IDF (scikit-learn), not a neural sentence embedder.
See README "known limitations" -- this sandbox has no network access to
download a HuggingFace sentence-transformer model, so TF-IDF is the
honest, runnable choice for this environment, not a claim that it's
state-of-the-art. It is explainable, has zero model-download dependency,
and is a completely standard first baseline for lexical retrieval.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ingestion import TextChunk


@dataclass
class TextResult:
    chunk_id: str
    doc_id: str
    title: str
    page: int
    snippet: str
    score: float


class TextOnlyRAG:
    def __init__(self, chunks: list[TextChunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        corpus_texts = [c.text for c in chunks]
        self.matrix = self.vectorizer.fit_transform(corpus_texts)

    def search(self, query: str, top_k: int = 5) -> tuple[list[TextResult], float]:
        t0 = time.time()
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        ranked_idx = sims.argsort()[::-1][:top_k]
        results = [
            TextResult(
                chunk_id=self.chunks[i].chunk_id,
                doc_id=self.chunks[i].doc_id,
                title=self.chunks[i].title,
                page=self.chunks[i].page,
                snippet=self.chunks[i].text[:220],
                score=float(sims[i]),
            )
            for i in ranked_idx if sims[i] > 0
        ]
        latency_ms = (time.time() - t0) * 1000
        return results, latency_ms


if __name__ == "__main__":
    from ingestion import ingest_corpus

    chunks, _images = ingest_corpus()
    rag = TextOnlyRAG(chunks)
    for q in ["What was the highest-revenue region in Q3?", "onboarding checklist for new engineers"]:
        results, latency = rag.search(q, top_k=3)
        print(f"\nQuery: {q}  ({latency:.1f}ms)")
        for r in results:
            print(f"  [{r.score:.3f}] {r.title} (p{r.page}): {r.snippet[:100]}...")

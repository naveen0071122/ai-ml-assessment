"""
retrieval.py
The multimodal retrieval pipeline:

    User Query (text OR image)
        |
    Query Understanding      -- image query: VLM captions the uploaded image
        |                        into a text description (this IS the
        |                        "multimodal query understanding" step);
        |                        text query: used as-is.
        v
    Candidate Retrieval      -- TF-IDF cosine similarity over a combined
        |                        index of {text chunks} U {image captions},
        |                        same mechanism as baseline.py's index, so
        |                        the two pipelines are comparable.
        v
    Qwen-VL Ranking          -- for candidates that are IMAGES, re-score by
        |                        actually looking at the image (real
        |                        backend) or via the offline mock reranker
        |                        (see multimodal.py). Text candidates keep
        |                        their retrieval score.
        v
    Top-K results, each tagged: doc/page/region, snippet-or-caption,
    combined relevance score, and which stage produced the match.

IMPORTANT (candidate retrieval vs. VLM ranking are DIFFERENT jobs):
TF-IDF does the large-scale narrowing (fast, cheap, works over hundreds of
chunks); Qwen-VL only ever looks at the handful of image candidates that
already made the shortlist. This mirrors how you'd actually want to run a
VLM in production -- it's too slow/expensive to call per-document at
index scale, so it's a reranker over pre-filtered candidates, not a
retriever itself. See multimodal.py's docstring for the fuller version of
this argument.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ingestion import TextChunk, ImageUnit
from multimodal import get_multimodal_client, CaptionResult

RERANK_CANDIDATE_POOL = 8      # how many top TEXT candidates go into the pool
VLM_WEIGHT = 0.6                # final_score = (1-w)*retrieval_score + w*vlm_score, for image candidates only
MIN_IMAGE_CANDIDATES = 3        # fallback cap used only if the corpus has MORE images than
                                 # MAX_IMAGES_TO_ALWAYS_INCLUDE (see below)
MAX_IMAGES_TO_ALWAYS_INCLUDE = 12   # for a corpus this small, just rerank every image every
                                     # time rather than picking an arbitrary top-N by TF-IDF --
                                     # a real caption can score 0 against a query it actually
                                     # answers (see MIN_IMAGE_CANDIDATES commit history / README),
                                     # so any TF-IDF-based pre-filter on images risks silently
                                     # dropping the correct one. This does NOT scale to a large
                                     # corpus -- see README "known limitations" for the
                                     # production fix (a real image embedding index).


@dataclass
class MultimodalResult:
    unit_id: str
    doc_id: str
    title: str
    page: int | None
    kind: str            # "text" or "image"
    snippet: str          # text excerpt OR image caption
    image_path: str | None
    retrieval_score: float
    vlm_score: float | None
    final_score: float
    match_reason: str


class MultimodalRAG:
    def __init__(self, chunks: list[TextChunk], images: list[ImageUnit], manifest: list[dict]):
        self.chunks = chunks
        self.images = images
        self.captions: dict[str, CaptionResult] = {}
        self.vlm_client = get_multimodal_client(manifest)

        from multimodal import caption_all_images
        self.captions = caption_all_images(images, manifest)

        # Build one combined corpus: text chunks + image captions, each
        # entry tagged with its kind so results can be told apart later.
        self._entries = []  # list of dicts: {kind, ref (TextChunk|ImageUnit), text}
        for c in chunks:
            self._entries.append({"kind": "text", "ref": c, "text": c.text})
        for img in images:
            cap = self.captions[img.image_id].caption
            self._entries.append({"kind": "image", "ref": img, "text": cap})

        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.matrix = self.vectorizer.fit_transform([e["text"] for e in self._entries])

    def _query_understanding(self, text_query: str | None, image_query_path: str | None) -> tuple[str, str]:
        """Returns (search_text, understanding_note)."""
        if image_query_path:
            cap = self.vlm_client.caption_image(
                image_query_path,
                instruction=(
                    "Describe this image in detail as a search query: what kind of "
                    "document/content would this image be similar to? Mention any "
                    "visible text, numbers, or diagram elements."
                ),
            )
            note = f"Image query captioned by VLM ({cap.source}): \"{cap.caption[:120]}...\""
            return cap.caption, note
        return text_query or "", "Text query used as-is."

    def search(
        self, text_query: str | None = None, image_query_path: str | None = None, top_k: int = 5
    ) -> tuple[list[MultimodalResult], dict]:
        t0 = time.time()
        search_text, understanding_note = self._query_understanding(text_query, image_query_path)
        t_understanding = time.time()

        q_vec = self.vectorizer.transform([search_text])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        ranked_idx = sims.argsort()[::-1]
        t_retrieval = time.time()

        # Take a wider candidate pool than top_k so the VLM rerank step has
        # something to work with, then trim to top_k after reranking.
        #
        # IMPORTANT: text candidates are filtered to sims[i] > 0 (no point
        # reranking something with zero lexical overlap), but IMAGES are
        # NOT filtered by TF-IDF score at all when the corpus is small
        # enough (see MAX_IMAGES_TO_ALWAYS_INCLUDE). A real VLM caption (as
        # opposed to the offline mock's hand-written ground truth) may
        # legitimately share zero vocabulary with the query even when the
        # image itself answers the question -- e.g. a small local model's
        # caption of a bar chart might say "a chart with colored bars"
        # without using the word "revenue" at all, and different images
        # can tie at zero score, so picking an arbitrary top-N by TF-IDF
        # risks silently excluding the one image that actually answers the
        # query (confirmed happening in practice with a real user's local
        # Qwen2-VL run). So: for a corpus this small, just rerank every
        # image, every query, and let the VLM be the real judge -- see
        # MAX_IMAGES_TO_ALWAYS_INCLUDE's docstring for why this doesn't
        # scale to a large corpus and what the production fix would be.
        text_ranked = [i for i in ranked_idx if self._entries[i]["kind"] == "text" and sims[i] > 0]
        image_ranked_all = [i for i in ranked_idx if self._entries[i]["kind"] == "image"]

        if len(image_ranked_all) <= MAX_IMAGES_TO_ALWAYS_INCLUDE:
            image_slots = image_ranked_all  # rerank all of them, no pre-filtering
        else:
            image_slots = image_ranked_all[: max(MIN_IMAGE_CANDIDATES, RERANK_CANDIDATE_POOL // 2)]

        pool_size = max(top_k, RERANK_CANDIDATE_POOL)
        text_slots = text_ranked[:pool_size]

        # Re-sort the combined pool back into score order for readability;
        # final ranking is decided after VLM reranking below anyway.
        candidate_idx = sorted(set(text_slots) | set(image_slots), key=lambda i: -sims[i])

        image_candidates = [
            {
                "image_id": self._entries[i]["ref"].image_id,
                "image_path": self._entries[i]["ref"].image_path,
                "caption": self._entries[i]["text"],
            }
            for i in candidate_idx
            if self._entries[i]["kind"] == "image"
        ]
        vlm_scores: dict[str, tuple[float, str]] = {}
        if image_candidates:
            rerank_results = self.vlm_client.rerank(search_text, image_candidates)
            vlm_scores = {r.image_id: (r.score, r.reason) for r in rerank_results}
        t_vlm = time.time()

        results = []
        for i in candidate_idx:
            entry = self._entries[i]
            retrieval_score = float(sims[i])
            if entry["kind"] == "image":
                vlm_score, reason = vlm_scores.get(entry["ref"].image_id, (None, ""))
                final = (
                    (1 - VLM_WEIGHT) * retrieval_score + VLM_WEIGHT * vlm_score
                    if vlm_score is not None
                    else retrieval_score
                )
                results.append(
                    MultimodalResult(
                        unit_id=entry["ref"].image_id,
                        doc_id=entry["ref"].doc_id,
                        title=entry["ref"].title,
                        page=entry["ref"].page,
                        kind="image",
                        snippet=entry["text"],
                        image_path=entry["ref"].image_path,
                        retrieval_score=retrieval_score,
                        vlm_score=vlm_score,
                        final_score=final,
                        match_reason=reason or "matched on caption text",
                    )
                )
            else:
                results.append(
                    MultimodalResult(
                        unit_id=entry["ref"].chunk_id,
                        doc_id=entry["ref"].doc_id,
                        title=entry["ref"].title,
                        page=entry["ref"].page,
                        kind="text",
                        snippet=entry["text"][:220],
                        image_path=None,
                        retrieval_score=retrieval_score,
                        vlm_score=None,
                        final_score=retrieval_score,
                        match_reason="matched on extracted text",
                    )
                )

        results.sort(key=lambda r: r.final_score, reverse=True)
        results = results[:top_k]

        timings = {
            "query_understanding_ms": (t_understanding - t0) * 1000,
            "candidate_retrieval_ms": (t_retrieval - t_understanding) * 1000,
            "vlm_rerank_ms": (t_vlm - t_retrieval) * 1000,
            "total_ms": (t_vlm - t0) * 1000,
            "understanding_note": understanding_note,
            "num_candidates": len(candidate_idx),
            "num_image_candidates_reranked": len(image_candidates),
        }
        return results, timings


if __name__ == "__main__":
    from ingestion import ingest_corpus, _load_manifest

    chunks, images = ingest_corpus()
    manifest = _load_manifest()
    rag = MultimodalRAG(chunks, images, manifest)

    for q in ["What was the highest-revenue region in Q3?", "which service connects to the redis cache"]:
        results, timings = rag.search(text_query=q, top_k=3)
        print(f"\nQuery: {q}  (total {timings['total_ms']:.1f}ms, "
              f"retrieval {timings['candidate_retrieval_ms']:.1f}ms, "
              f"vlm {timings['vlm_rerank_ms']:.1f}ms)")
        for r in results:
            print(f"  [{r.final_score:.3f}|{r.kind}] {r.title} (p{r.page}): {r.snippet[:100]}")

# Task 1 — Multimodal Retrieval with Qwen-VL + Streamlit

## What this is

A retrieval prototype over a synthetic 12-document intranet corpus (10 PDFs
+ 2 standalone images), comparing:
- **`baseline.py`** — a genuinely text-only RAG pipeline (TF-IDF over
  extracted PDF text only, zero access to images).
- **`retrieval.py`** — a multimodal pipeline: TF-IDF candidate retrieval
  over text chunks *and* image captions, then a Qwen-VL reranking step
  over the image candidates.

Six of the twelve documents are built so a genuine fact lives **only** in
a chart, diagram, floor-plan, or scanned table image — not anywhere in
the document's extractable text — so the comparison between pipelines is
real, not contrived. See `data/manifest.json`'s `visual_only_facts` field
for the ground truth we used to build the corpus and the eval labels.

## Setup

```bash
cd task1_multimodal_retrieval
pip install -r requirements.txt
python data/generate_corpus.py     # regenerates the 12-doc corpus (already included, but reproducible)
```

Optional — for the real hosted Qwen-VL backend instead of the offline
mock:
```bash
export DASHSCOPE_API_KEY=sk-...
```

## Run

```bash
# quick CLI smoke tests
python ingestion.py       # prints extracted chunk/image counts
python baseline.py        # runs 2 sample queries through the text-only baseline
python retrieval.py       # runs 2 sample queries through the multimodal pipeline

# the actual deliverable
streamlit run app.py

# evaluation (writes data/eval_results.json + .csv, prints the summary table)
python evaluation.py

# tests
python -m pytest tests/ -v
```

## Corpus

`data/generate_corpus.py` builds:
- 5 PDFs with a chart/diagram/scanned-table embedded, where the specific
  numeric or structural answer is visual-only (revenue by region, network
  topology, incident timing, expense variances, roadmap durations).
- 5 pure-text PDFs (onboarding, benefits, security policy, remote-work
  policy, data-retention policy) — negative controls, so the pipelines
  aren't just "always find the doc with a picture."
- 2 standalone images with **no accompanying text at all** (a server-room
  floor plan, an org chart) — these can only ever be found by a
  multimodal pipeline; the text baseline is structurally unable to
  retrieve them.

`ingestion.py` re-derives this from the actual PDF bytes with real
`pdfplumber` text extraction and real embedded-image extraction (crop +
render each image's bounding box out of the rendered page) — it does not
read the manifest at query time; the manifest is only used to (a) build
the corpus and (b) label the ground truth for the offline mock VLM and
the eval set.

## Embedding strategy: hybrid (text-chunk embeddings + image-caption
embeddings), not a joint pixel embedding

See the top of `ingestion.py` and `multimodal.py` for the full reasoning.
Short version: every image is captioned (via the VLM) at ingestion time,
and that caption is embedded in the *same* TF-IDF space as the text
chunks, so one index serves both. This is simple, debuggable, and cheap,
but it means retrieval quality for images is bounded by caption quality
— a captioning miss is a retrieval miss. A production system with more
time would use true joint embeddings (CLIP-family or Qwen-VL's own
embedding variants) to keep a visual-similarity signal that captions
can't carry (e.g. matching by layout/color, not just by what's written).

## Qwen-VL: two different jobs, not one

- **Captioning** (ingestion time): describe each image so it's indexable
  as text. This is what lets a *text* query find an image.
- **Reranking** (query time): given a shortlist of candidate images
  already narrowed by TF-IDF, judge relevance by actually looking at
  each one. This is NOT an embedding step — Qwen-VL doesn't produce
  fixed-size vectors for large-scale ANN search here; TF-IDF does the
  narrowing, Qwen-VL does the judgment call on the shortlist. See
  `multimodal.py`'s docstring for why conflating these would be
  misleading.

## Local vs. hosted Qwen-VL (judgment call, explicit — and both are now implemented)

Three backends exist behind the same `MultimodalModelClient` interface,
selected via `QWEN_VL_BACKEND`:

| `QWEN_VL_BACKEND=` | Client | Needs | Verified in this sandbox? |
|---|---|---|---|
| `local` | `QwenVLLocalClient` (Qwen2-VL-2B/7B via HF `transformers`) | `pip install -r requirements-local-qwenvl.txt`, GPU optional (CPU works, slower) | **Yes, on a real machine** — not exercised in the original dev sandbox (no HF Hub network there), but run end-to-end against real weights during development on an actual Windows machine with real CPU inference. Three real bugs were found and fixed this way — see "Bugs found via real local-model testing" below. |
| `hosted` (or unset + `DASHSCOPE_API_KEY` set) | `QwenVLHostedClient` (`qwen-vl-max` via DashScope) | `DASHSCOPE_API_KEY` | **No** — no DashScope network access here |
| `mock` (or unset + no key) | `OfflineMockClient` | nothing | **Yes** — this is what actually ran for every result in this README |

### Running Qwen-VL locally

```bash
pip install -r requirements-local-qwenvl.txt
export QWEN_VL_BACKEND=local
# optional: use the larger, more accurate model if you have 16GB+ VRAM
# export QWEN_VL_LOCAL_MODEL=Qwen/Qwen2-VL-7B-Instruct
streamlit run app.py
```

First run downloads the model from Hugging Face Hub (needs network once;
fully offline after). Default model is `Qwen/Qwen2-VL-2B-Instruct` —
runs on a consumer GPU with ~6-8GB VRAM in fp16, or on CPU (slower, no
GPU required at all). See `multimodal.py`'s `QwenVLLocalClient`
docstring for the full hardware/quality trade-off notes.

### The underlying judgment call, unchanged by having both implemented

**Default recommendation is still hosted**, because:
- Local Qwen2-VL-7B needs ~16GB VRAM for good quality; the 2B model that
  fits more modest hardware trades off caption precision, especially on
  dense/small text in a busy diagram.
- Cost: pay-per-call suits a low-volume prototype; a GPU is a fixed cost
  that only pays off at sustained volume.
- Latency: hosted adds network round-trip (typically ~1-3s/image call for
  VLM APIs this size) vs. local inference on a warm GPU being faster
  per-call but needing that GPU always on.
- Privacy: hosted sends document images to a third party. **This is
  exactly the case where you'd flip to `QWEN_VL_BACKEND=local`** — a
  privacy-sensitive team keeps documents in-house at the cost of some
  caption quality and needing to own the GPU/ops burden.
- Reproducibility: hosted model versions can drift; local checkpoints
  are pinned.

Both `QwenVLHostedClient` and `QwenVLLocalClient` are implemented in
full — real API/inference code, not stubs. `QwenVLLocalClient` has since
been run end-to-end on a real machine (Windows, CPU-only, no GPU) against
real Qwen2-VL-2B-Instruct weights downloaded from Hugging Face Hub —
model loading, real image captioning, and real VLM reranking all
confirmed working, and three real bugs surfaced and were fixed in the
process (see below). `QwenVLHostedClient` (DashScope API) remains
unverified against the live API — implemented but not yet run against
real credentials. All quantitative numbers in the Evaluation section
below still come from `OfflineMockClient`, since a full 15-query
evaluation run against a local 2B model on CPU (every image reranked,
every query) takes tens of minutes rather than milliseconds — see
`MANUAL_TESTING_GUIDE.md` for spot-checking real local-model results
instead of a full automated re-run.

## The offline mock — what it is and isn't

`multimodal.OfflineMockClient` returns, as its "caption," the exact
ground-truth `visual_only_facts` we wrote when generating the synthetic
corpus (i.e. it behaves the way a *correct* Qwen-VL call would behave on
these specific images). Its "reranker" is simple query/caption token
overlap, not a model judgment.

This is an honest simplification to make the whole pipeline — ingestion,
indexing, retrieval, Streamlit UI, evaluation — runnable and gradable
without network access or API credentials, in the time available. It is
**not** evidence that captioning will generalize to real, unseen
documents; that can only be validated against the real API. The
evaluation numbers below should be read as a best-case (perfect
captioner) ceiling, not a general performance claim.

## Streamlit UI

`streamlit run app.py` — text or image query, mode toggle (Multimodal /
Text-only baseline), Top-K slider, corpus browser, and per-result: title,
doc ID, page, kind (text/image), relevance indicator (explicitly labelled
"not a calibrated probability"), match reason, and the actual image or
text snippet. Latency is broken into query-understanding / candidate-
retrieval / VLM-rerank stages for the multimodal mode. Verified to boot
cleanly (`streamlit run app.py --server.headless true` → HTTP 200, no
runtime errors) in this environment.

For a ready-made list of exact queries to try, expected results, and
Top-K settings, see [`MANUAL_TESTING_GUIDE.md`](MANUAL_TESTING_GUIDE.md).

## Evaluation — real numbers, run in this environment

15 labelled queries (`data/eval_queries.json`): 7 visual-favoring
(doc05 and doc08 have literally zero extractable text; doc01/03/06/07/10
have a chart/diagram/table holding the specific answer) + 8 text-only
control queries. Metrics are doc-level Precision@5 / Recall@5 /
Hit Rate@5 / MRR, plus per-query and average latency. Run with:

```bash
python evaluation.py
```

**Actual results from this run** (`data/eval_results.json`):

| Pipeline | Precision@5 | Recall@5 | Hit Rate@5 | MRR | Avg latency |
|---|---|---|---|---|---|
| Text-only RAG | 0.441 | 0.867 | 0.867 | 0.867 | 0.61 ms |
| Multimodal | 0.532 | 1.000 | 1.000 | 1.000 | 0.62 ms |

MRR on the 7 visual-favoring queries only: text_rag **0.714**,
multimodal **1.000**.

### Honest discussion — where multimodal won and where it didn't

Of the 7 visual-favoring queries, multimodal only *beat* text-only RAG on
**2**: `q05` (server-room floor plan) and `q06` (org chart) — the two
documents with **zero** extractable text, where the text baseline had
literally nothing to match against and scored 0. On the other 5
visual-favoring queries (`q01, q02, q03, q04, q07`), **both pipelines
found the correct document**, tied at MRR 1.0 — because the surrounding
PDF text still contains enough topical vocabulary ("Q3 Sales Report",
"network architecture", "incident postmortem") for TF-IDF to retrieve
the right *document* by topic, even though it can't recover the specific
*fact* (the actual revenue number, the actual service name) that only
exists in the image.

**This is the most important honest finding of this evaluation**: our
doc-level MRR/Recall metrics measure "did we find the right document,"
not "can we answer the question." A text-only RAG baseline can look
deceptively competitive on doc-level retrieval metrics for a chart-heavy
PDF as long as the page has *any* related prose, while still being
functionally unable to answer the question a user actually asked. A
production evaluation should also score **answer correctness** (e.g. via
an LLM-as-judge comparing the retrieved snippet/caption against the
labelled answer), not just document-level hit rate — flagged as a known
gap in this evaluation, and as the single highest-impact next step (see
`reflection.md`).

Where multimodal unambiguously and structurally wins: any document with
**no text at all** — the two standalone images. That's the case where
"text RAG can't even see it" isn't a subtle quality gap, it's a hard
zero.

### Failure modes / limitations observed or expected

- **Doc-level metrics hide answer-quality gaps** (see above) — the
  biggest limitation of this specific evaluation.
- **Offline mock is a best-case captioner.** Real Qwen-VL captioning can
  mis-transcribe numbers, hallucinate labels it can't clearly read, or
  miss small text in a busy diagram — none of that failure mode is
  present in the mock, so real-world scores would likely be lower,
  especially for the scanned-expense-table document (doc07), which is
  exactly the kind of image (dense numeric table) real OCR/VLM pipelines
  struggle with most.
- **Latency numbers here are not representative of a real Qwen-VL
  deployment.** TF-IDF search is sub-millisecond; the offline mock has no
  network delay. A real hosted Qwen-VL rerank call adds roughly 1-3
  seconds *per image candidate reranked*, which for `RERANK_CANDIDATE_POOL
  = 8` could mean several seconds of added latency per query if candidates
  are reranked serially — a production version should batch or
  parallelize these calls, and cap how many images get reranked more
  aggressively than we did here.
- **TF-IDF is lexical, not semantic** — a query using different words
  than the document (e.g. "who's Divya's manager" vs. the org chart
  captioned as "reports to") can miss even when a neural embedder
  would catch the paraphrase. Swapping in a sentence-transformer is the
  most direct fix (see reflection.md).
- **Caption-quality-bounded retrieval**: since images are only findable
  via their caption text, a caption that omits a detail makes that
  detail permanently unsearchable regardless of how good the reranker
  is downstream.

## Known limitations (repo-level)
- `QwenVLHostedClient` (DashScope API) is implemented but unverified
  against the live API — no DashScope credentials exercised yet.
- TF-IDF, not a neural sentence embedder, due to no HuggingFace network
  access in the original dev sandbox — documented as an environment
  constraint, not a preference (see `baseline.py`'s docstring).
- Evaluation is doc-level, not answer-level (see "Honest discussion"
  above).
- Every image in the corpus is reranked on every multimodal query, with
  no TF-IDF pre-filter (`MAX_IMAGES_TO_ALWAYS_INCLUDE` in `retrieval.py`
  — see the bugs below for why). Correct and necessary for this 7-image
  corpus; does **not** scale to a large one. A production version needs a
  real image embedding index (e.g. CLIP/SigLIP) for first-stage candidate
  narrowing instead of exhaustive reranking.
- The local 2B Qwen2-VL model occasionally returns garbled "reason" text
  in its rerank output (e.g. literal bounding-box-shaped coordinates
  instead of a sentence) — a known quirk of small VLMs trained heavily on
  visual grounding tasks. The rerank prompt explicitly forbids coordinate
  output, which helps but doesn't eliminate it. Honest small-model
  failure mode, not a code bug — exactly the kind of thing the
  Evaluation section's "obvious failure modes" discussion is meant to
  surface. Use `python debug_captions.py` (prints every image's actual
  generated caption under the active backend) to inspect this directly.

## Bugs found via real local-model testing (and fixed)

`QwenVLLocalClient` was written against the interface but only actually
exercised end-to-end once a real user ran it on their own Windows
machine. That surfaced three real bugs that offline-mock testing alone
could never have caught, since the mock's hand-written captions always
share vocabulary with the eval queries by construction — a real model's
captions don't have that guarantee. Documenting these here rather than
quietly fixing them, since the debugging process itself demonstrates the
kind of gap between "works against a mock" and "works against the real
thing" that this whole project's honesty-about-limitations stance is
about.

1. **Model loaded twice on every app start.** `get_multimodal_client()`
   constructed a fresh client on every call; `MultimodalRAG.__init__` and
   `caption_all_images` each called it once, so `QwenVLLocalClient.__init__`
   (which loads the full model into memory) ran twice per app start.
   Fixed with a process-wide singleton cache in `get_multimodal_client()`.

2. **The correct image could be silently excluded from candidates.**
   Candidate retrieval originally filtered every candidate (text and
   image) to TF-IDF `sims[i] > 0`. A real VLM caption can legitimately
   share zero query vocabulary even when the image answers the question
   (e.g. captioning a bar chart as "a chart with colored bars" instead of
   using the word "revenue") — confirmed live: the revenue chart never
   reached the reranker at all for "highest-revenue region." A first fix
   (guarantee 3 image slots) still wasn't enough — with real captions,
   an arbitrary tie-break excluded the one CORRECT image (the network
   diagram, for a "which service connects to redis" query) in favor of
   three unrelated ones. Since this corpus only has 7 images, the final
   fix reranks **every** image on every query, no TF-IDF pre-filter at
   all — confirmed the correct image then wins (0.657 final score).
   Locked in by
   `tests/test_retrieval.py::test_multimodal_includes_image_candidates_even_with_zero_lexical_overlap`,
   verified to fail against the old code and pass against the fix.

3. **UI crash on a legitimate zero score.** `app.py` computed the
   displayed score as `getattr(r, "final_score", None) or getattr(r,
   "score", None)`. Since `0.0` is falsy in Python, a candidate the VLM
   correctly scored as 0 relevance fell through to `None` (no `score`
   attribute exists on that result type), crashing `st.progress(None)`.
   Fixed to check `is None` explicitly instead of relying on truthiness.

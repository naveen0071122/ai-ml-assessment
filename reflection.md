# Task 3 — Written Reflection

## 1. If you had one more week, what is the single highest-impact thing you would add or change across both tasks, and why?

**Task 1: answer-level evaluation, not just document-level retrieval
metrics.** The evaluation run in this submission measures whether the
right *document* was retrieved (Precision@K, Recall@K, MRR), and the
honest result was that on 5 of 7 visual-favoring queries, the text-only
baseline tied the multimodal pipeline on document-level metrics — even
though it structurally cannot answer the question, because the answer
(a specific number, a specific service name) exists only in an image.
Doc-level metrics don't distinguish "found the right document" from "can
actually answer the question," which is the entire point a real user
cares about. With a week, I'd build a small answer-extraction/grading
step (an LLM-as-judge comparing each pipeline's retrieved
snippet-or-caption against the labelled ground-truth fact, scored
correct/partial/wrong) and report Answer Accuracy@K alongside the
retrieval metrics. This is the single change most likely to reveal a
different, more honest, more decision-useful story than the current
numbers tell — right now the metrics somewhat understate how much
multimodal retrieval actually matters for these documents.

Second priority, since done: the local Qwen2-VL backend has since
actually been run end-to-end against real model weights (the original
dev sandbox had no network egress to Hugging Face Hub; this was run
separately on a real machine). That surfaced three genuine bugs no
amount of offline-mock testing could have caught — see Q2. The hosted
DashScope API path remains untested against live credentials; that's
the one piece of "run it for real" still outstanding, and I'd expect it
to surface a different class of issue (network/rate-limit handling,
response format differences) than the local path did.

## 2. Name one thing in your submission you are NOT happy with and what the correct production approach would be.

**The offline mock made a whole category of bugs invisible, and I
initially over-trusted it.** `OfflineMockClient`'s "captions" are the
exact ground-truth facts I wrote by hand when generating the synthetic
corpus — so by construction, they always share vocabulary with the eval
queries. When I actually ran the real local Qwen2-VL model against real
images, I found three bugs the mock could never have exposed: (1) the
client-caching bug that loaded the model twice, (2) a candidate-filtering
bug where a real caption with zero lexical overlap with the query (e.g.
describing a bar chart without using the word "revenue") got silently
excluded before the reranker ever saw it — which meant the *correct*
image could lose to *unrelated* ones on an arbitrary tie-break, and (3)
a UI crash triggered only when the VLM legitimately scored something
exactly 0.0. None of my automated tests caught these, because the tests
were also written against the same mock. The honest lesson: a mock that
is *correct by construction* is good for testing plumbing (does data
flow end-to-end) but actively misleading for testing retrieval-quality
edge cases, because it can't produce the vocabulary mismatches,
uncertainty, or noise a real model produces. The correct production
approach is to never ship retrieval logic that's only been validated
against a mock — a small "golden set" run against the real model
(even just 5-10 images, even just once) before merging would have
caught all three bugs before a user did.

Closely related and still true: **TF-IDF as the text embedder** is a
defensible, standard *lexical* baseline, and it's honest about why it's
there (no HuggingFace network access in the original sandbox to download
a sentence-transformer), but it's genuinely weaker than what I'd ship —
it can't handle paraphrase or synonym queries (a query like "who manages
the ML team" would miss a caption that says "reports to," since there's
no lexical overlap on the load-bearing words). `TextOnlyRAG` and
`MultimodalRAG` were deliberately written so swapping in a neural
sentence embedder (e.g. `bge-small-en`) is a one-method change (`fit` /
`transform` semantics), not a redesign — but I haven't made that swap or
benchmarked it, so I can predict the direction of the improvement, not
claim it.

## 3. For Task 1, when is multimodal retrieval worth the extra cost/latency over text-only RAG, and when would you advise a team to skip it?

**Worth it when a meaningful fraction of your corpus's answer-bearing
content is genuinely only visual** — dashboards, architecture diagrams,
scanned forms/tables, floor plans, org charts, screenshots — content
where no amount of better text chunking or a smarter text embedder would
ever recover the fact, because the fact was never extracted as text in
the first place. Our own evaluation makes this concrete: for the two
standalone images with zero text (floor plan, org chart), multimodal
retrieval is not "a bit better," it's the *only* way those documents are
findable at all. If your corpus has documents like that in meaningful
volume, and users actually ask questions whose answers live in them,
multimodal retrieval isn't a nice-to-have — the text-only system is
silently failing a real slice of queries.

**Worth skipping when your corpus is mostly prose** — policy documents,
FAQs, onboarding guides, contracts — where any charts/images are
decorative or redundant with nearby text (a chart that restates a number
already written in the paragraph above it). In that regime, the extra
API cost, added latency (a hosted VLM rerank call is roughly 1-3 seconds
per image candidate — and confirmed even higher on local CPU inference,
which is minutes rather than seconds for a handful of images), and
engineering complexity (captioning pipeline, a second model dependency,
a reranking step to tune) buys you very little, and our own evaluation
shows exactly that pattern on 5 of the 7 "visual-favoring" queries — the
text-only baseline still found the right *document* because there was
enough surrounding prose to match on topic, even without the exact
figure.

The pragmatic middle path I'd actually recommend to a team deciding this:
run a quick audit like the one in this submission — sample your real
corpus, tag which documents have text-invisible visual content, and see
what fraction of realistic user queries would touch those documents.
If it's a small, identifiable subset (e.g. "just the ops/infra docs" or
"just the finance reports"), you don't need multimodal retrieval across
the whole system — route only that subset through the more expensive
pipeline, and keep the fast, cheap text-only path for everything else.

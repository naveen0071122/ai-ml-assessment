# Manual Testing Guide — Task 1 Streamlit UI

This is a ready-to-run checklist for manually exercising `app.py` in the
browser. It complements (does not replace) the automated evaluation in
`evaluation.py` / `data/eval_queries.json` — this guide is for a human
clicking through the UI; that one is the quantitative Precision/Recall/MRR
report. Same 12-document corpus, same underlying queries where relevant.

**Default settings unless a row says otherwise:** Retrieval mode =
Multimodal, Query type = Text, Top-K = 3.

---

## 1. Visual-favoring queries (the core claim of this project)

Answer exists ONLY in a chart/diagram/table/standalone image — not in any
extractable text. This is what should make multimodal retrieval win.

| # | Query | Expected top doc | What to check in the result |
|---|---|---|---|
| 1 | `What was the highest-revenue region in Q3?` | doc01 (kind: image) | Snippet/caption mentions APAC and ~$6.4M |
| 2 | `Which production service connects directly to the redis cache?` | doc03 (kind: image) | Mentions `auth-prod-3` |
| 3 | `How long did the payment latency spike last during the incident?` | doc06 (kind: image) | Mentions ~10 minutes |
| 4 | `Which expense category had the largest budget overage in Q3?` | doc07 (kind: image) | Mentions Contractor Fees / +$6,900 — **hardest case, a small local VLM may fail here; that's an expected, documented limitation, not a bug** |
| 5 | `Which rack is physically isolated near the cooling unit?` | doc05 (kind: image) | Mentions Rack C7 |
| 6 | `Who does Divya Krishnan report to?` | doc08 (kind: image) | Mentions Karthik Iyer |
| 7 | `How long does the Build phase take in the 2025 product roadmap?` | doc10 (kind: image) | Mentions 9 weeks |

## 2. Plain-text queries (negative control)

No image dependency — multimodal and text-only baseline should agree.

| # | Query | Expected top doc |
|---|---|---|
| 8 | `What should new engineers do in their first week?` | doc02 |
| 9 | `When does health insurance coverage start for a new employee?` | doc04 |

**Also do:** switch Retrieval mode to **Text-only baseline** and re-run
#8 and #9 — results should be the same doc (no image needed).

## 3. Image query

| # | Steps | Expected |
|---|---|---|
| 10 | Query type → **Image** → upload `data/images/doc05_floorplan.png` → Search | doc05 top result; "Image query captioned by VLM" note shows a real caption |
| 11 | Same, upload `data/images/doc08_org_chart.png` | doc08 top result |

## 4. Guardrail edge case

| # | Steps | Expected |
|---|---|---|
| 12 | Query type → **Image** + Retrieval mode → **Text-only baseline** → Search | Error message: baseline cannot accept image queries |

## 5. Mode comparison (same query, both modes) — Top-K = 5

| # | Query | Multimodal result | Text-only result |
|---|---|---|---|
| 13 | `What was the highest-revenue region in Q3?` | doc01, with the actual answer (APAC $6.4M) in the snippet | doc01 may still be found (topic words match), but the specific number will be missing |

## 6. Top-K slider sanity check

| # | Query | Try Top-K = | Expect |
|---|---|---|---|
| 14 | `What was the highest-revenue region in Q3?` | 1, then 5, then 10 | Result count changes accordingly; top result should stay doc01 |

---

## Notes specific to running with `QWEN_VL_BACKEND=local`

- First query after app startup is slow (model load + first captions)
- Every image in the corpus gets reranked on every query by design (see
  `MAX_IMAGES_TO_ALWAYS_INCLUDE` in `retrieval.py`) — this is correct for
  a 7-image corpus but means each query takes noticeably longer than
  offline-mock mode. Expect **several minutes per query on CPU**, faster
  with a GPU.
- If a caption or rerank "reason" looks garbled (e.g. literal coordinate
  text instead of a sentence), that's a known small-2B-model quirk
  documented in `README.md` — not a bug to report.

## Recording results

For each row above, note: Pass / Fail / Partial and the actual top
result returned. This table — filled in — is itself useful evidence for
the interview walkthrough (in addition to `evaluation.py`'s automated
metrics).

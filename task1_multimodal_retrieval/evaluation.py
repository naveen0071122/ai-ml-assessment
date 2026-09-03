"""
evaluation.py
Runs the labelled query set (data/eval_queries.json) against BOTH
pipelines (baseline.TextOnlyRAG and retrieval.MultimodalRAG) and reports
Precision@K, Recall@K, Hit Rate@K, MRR, and per-query latency, plus a
per-query winner table. Metrics are computed at the DOCUMENT level (a hit
means the retrieved chunk/image's doc_id is in the query's
relevant_doc_ids), since our labels are doc-level, not chunk-level.

Run:
    python evaluation.py                 # prints table, writes results.json + results.csv
"""
from __future__ import annotations

import json
import os
import time

from ingestion import ingest_corpus, _load_manifest
from baseline import TextOnlyRAG
from retrieval import MultimodalRAG

HERE = os.path.dirname(os.path.abspath(__file__))
QUERIES_PATH = os.path.join(HERE, "data", "eval_queries.json")
TOP_K = 5


def precision_at_k(retrieved_doc_ids: list[str], relevant: set[str], k: int) -> float:
    top = retrieved_doc_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for d in top if d in relevant)
    return hits / len(top)


def recall_at_k(retrieved_doc_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(retrieved_doc_ids[:k])
    return len(top & relevant) / len(relevant)


def hit_rate_at_k(retrieved_doc_ids: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(retrieved_doc_ids[:k]) & relevant else 0.0


def reciprocal_rank(retrieved_doc_ids: list[str], relevant: set[str]) -> float:
    for i, d in enumerate(retrieved_doc_ids, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def run_evaluation():
    print("Ingesting corpus...")
    chunks, images = ingest_corpus()
    manifest = _load_manifest()

    print("Building text-only baseline index...")
    text_rag = TextOnlyRAG(chunks)

    print("Building multimodal index (captioning images via VLM client)...")
    mm_rag = MultimodalRAG(chunks, images, manifest)

    with open(QUERIES_PATH) as f:
        queries = json.load(f)

    per_query_rows = []
    text_metrics_acc = {"precision": [], "recall": [], "hit": [], "mrr": [], "latency_ms": []}
    mm_metrics_acc = {"precision": [], "recall": [], "hit": [], "mrr": [], "latency_ms": []}

    for q in queries:
        relevant = set(q["relevant_doc_ids"])

        t0 = time.time()
        text_results, _ = text_rag.search(q["query"], top_k=TOP_K)
        text_latency = (time.time() - t0) * 1000
        text_doc_ids = [r.doc_id for r in text_results]

        t0 = time.time()
        mm_results, mm_timings = mm_rag.search(text_query=q["query"], top_k=TOP_K)
        mm_latency = (time.time() - t0) * 1000
        mm_doc_ids = [r.doc_id for r in mm_results]

        text_p = precision_at_k(text_doc_ids, relevant, TOP_K)
        text_r = recall_at_k(text_doc_ids, relevant, TOP_K)
        text_h = hit_rate_at_k(text_doc_ids, relevant, TOP_K)
        text_mrr = reciprocal_rank(text_doc_ids, relevant)

        mm_p = precision_at_k(mm_doc_ids, relevant, TOP_K)
        mm_r = recall_at_k(mm_doc_ids, relevant, TOP_K)
        mm_h = hit_rate_at_k(mm_doc_ids, relevant, TOP_K)
        mm_mrr = reciprocal_rank(mm_doc_ids, relevant)

        for acc, p, r, h, mrr, lat in [
            (text_metrics_acc, text_p, text_r, text_h, text_mrr, text_latency),
            (mm_metrics_acc, mm_p, mm_r, mm_h, mm_mrr, mm_latency),
        ]:
            acc["precision"].append(p)
            acc["recall"].append(r)
            acc["hit"].append(h)
            acc["mrr"].append(mrr)
            acc["latency_ms"].append(lat)

        if mm_mrr > text_mrr:
            winner = "multimodal"
        elif text_mrr > mm_mrr:
            winner = "text_rag"
        else:
            winner = "tie"

        per_query_rows.append({
            "query_id": q["query_id"],
            "query": q["query"],
            "query_type": q["query_type"],
            "text_rag_mrr": round(text_mrr, 3),
            "multimodal_mrr": round(mm_mrr, 3),
            "text_rag_top_doc": text_doc_ids[0] if text_doc_ids else None,
            "multimodal_top_doc": mm_doc_ids[0] if mm_doc_ids else None,
            "winner": winner,
            "text_rag_latency_ms": round(text_latency, 2),
            "multimodal_latency_ms": round(mm_latency, 2),
        })

    def summarize(acc):
        n = len(acc["precision"])
        return {
            f"precision@{TOP_K}": round(sum(acc["precision"]) / n, 3),
            f"recall@{TOP_K}": round(sum(acc["recall"]) / n, 3),
            f"hit_rate@{TOP_K}": round(sum(acc["hit"]) / n, 3),
            "mrr": round(sum(acc["mrr"]) / n, 3),
            "avg_latency_ms": round(sum(acc["latency_ms"]) / n, 2),
        }

    summary = {"text_rag": summarize(text_metrics_acc), "multimodal": summarize(mm_metrics_acc)}

    visual_rows = [r for r in per_query_rows if
                   next(q["query_type"] for q in queries if q["query_id"] == r["query_id"]) == "visual_favoring"]
    visual_mm_mrr = sum(r["multimodal_mrr"] for r in visual_rows) / len(visual_rows)
    visual_text_mrr = sum(r["text_rag_mrr"] for r in visual_rows) / len(visual_rows)

    # ---- print report ----
    print("\n" + "=" * 100)
    print(f"{'query_id':8} {'type':16} {'text winner doc':16} {'mm winner doc':16} "
          f"{'text MRR':9} {'mm MRR':7} {'winner':10}")
    print("-" * 100)
    for r in per_query_rows:
        print(f"{r['query_id']:8} {r['query_type']:16} {str(r['text_rag_top_doc']):16} "
              f"{str(r['multimodal_top_doc']):16} {r['text_rag_mrr']:<9} {r['multimodal_mrr']:<7} {r['winner']:10}")
    print("=" * 100)
    print("\nAggregate metrics (all 15 queries):")
    print(json.dumps(summary, indent=2))
    print(f"\nAggregate MRR on the {len(visual_rows)} visual-favoring queries only:")
    print(f"  text_rag:   {visual_text_mrr:.3f}")
    print(f"  multimodal: {visual_mm_mrr:.3f}")

    out = {
        "top_k": TOP_K,
        "per_query": per_query_rows,
        "summary": summary,
        "visual_favoring_only": {"text_rag_mrr": round(visual_text_mrr, 3), "multimodal_mrr": round(visual_mm_mrr, 3)},
    }
    with open(os.path.join(HERE, "data", "eval_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    import csv
    with open(os.path.join(HERE, "data", "eval_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_query_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_query_rows)

    print("\nWrote data/eval_results.json and data/eval_results.csv")
    return out


if __name__ == "__main__":
    run_evaluation()

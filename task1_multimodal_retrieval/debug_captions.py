"""
debug_captions.py -- prints the ACTUAL caption text the active VLM backend
(local Qwen2-VL / hosted / offline mock) generates for every image in the
corpus, so you can see exactly why an image did or didn't get retrieved
for a given query (TF-IDF only finds an image if its caption shares
vocabulary with the query).

Run (with QWEN_VL_BACKEND=local already set in the same terminal):
    python debug_captions.py
"""
import os
from ingestion import ingest_corpus, _load_manifest
from multimodal import caption_all_images, RESOLVED_BACKEND

print(f"Active backend: {RESOLVED_BACKEND}\n")

_chunks, images = ingest_corpus()
manifest = _load_manifest()

# Use the SAME function retrieval.py actually calls, so this reflects
# real app behavior exactly (not a simplified direct client call).
captions = caption_all_images(images, manifest)

for img in images:
    print("=" * 80)
    print(f"{img.image_id}  (doc: {img.doc_id}, page: {img.page})")
    print(f"file: {img.image_path}")
    result = captions[img.image_id]
    print(f"\nCAPTION ({result.source}):\n{result.caption}\n")

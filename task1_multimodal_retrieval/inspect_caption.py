"""
inspect_caption.py -- quick CLI to see exactly what caption the currently
active VLM backend produces for one image, without going through the full
Streamlit app. Useful for debugging retrieval misses: if a query isn't
finding an image you expect it to, check its caption here first --
if the caption doesn't mention the words in your query, TF-IDF will
never surface it (see README "failure modes": small/busy text in an
image is a genuine local-VLM limitation, not a retrieval bug).

Usage:
    $env:QWEN_VL_BACKEND="local"
    python inspect_caption.py data/images/doc08_org_chart.png
    python inspect_caption.py data/extracted_images/doc01_p1_img0.png
"""
import sys
from ingestion import _load_manifest
from multimodal import get_multimodal_client

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_caption.py <path-to-image>")
        sys.exit(1)

    image_path = sys.argv[1]
    manifest = _load_manifest()
    client = get_multimodal_client(manifest)

    print(f"Backend: {type(client).__name__}")
    print(f"Captioning: {image_path}\n")
    result = client.caption_image(image_path)
    print(f"--- Caption (source: {result.source}) ---")
    print(result.caption)

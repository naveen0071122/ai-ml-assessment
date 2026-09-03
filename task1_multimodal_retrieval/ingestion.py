"""
ingestion.py
Reads the corpus (PDFs + standalone images) and produces two things every
downstream module shares:

  1. `TextChunk` records  -- for the text-only baseline AND the text half
     of the multimodal hybrid index.
  2. `ImageUnit` records   -- one per extracted/standalone image, used only
     by the multimodal pipeline.

Design note (embedding strategy, elaborated further in README):
We use a HYBRID strategy: text chunks are embedded as text; each image is
turned into a text caption (via the VLM -- see multimodal.py) and that
caption is *also* embedded as text, sharing the same TF-IDF vector space
as the text chunks. This lets both text and image content be searched
through one retrieval index while keeping the code simple enough for a
1-day build. It is NOT the same as a true joint text/image embedding
space (e.g. CLIP/Qwen-VL native embeddings), which would preserve visual
similarity (e.g. "find documents that *look* like this screenshot" based
on layout/color, not just on what the caption says). What we give up:
- Any query similarity signal that depends on visual appearance rather
  than caption content (e.g. "documents with this specific chart color
  scheme") is invisible to this pipeline.
- Retrieval quality for an image query is bounded by how good the
  caption is -- a captioning failure/hallucination directly becomes a
  retrieval failure. A true image embedding degrades more gracefully.
- We pay a small amount of information loss going pixels -> text ->
  vector instead of pixels -> vector directly, but we gain: one simple
  index, one embedder, code any teammate can read in five minutes, and a
  debuggable, human-readable intermediate artifact (the caption itself)
  in every failure-mode discussion.

Run:
    python ingestion.py            # prints a summary of extracted units
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict

import pdfplumber

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingestion")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
IMG_DIR = os.path.join(DATA_DIR, "images")
EXTRACTED_IMG_DIR = os.path.join(DATA_DIR, "extracted_images")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")

CHUNK_SIZE_CHARS = 500
CHUNK_OVERLAP_CHARS = 80


@dataclass
class TextChunk:
    chunk_id: str
    doc_id: str
    title: str
    source_filename: str
    page: int
    text: str


@dataclass
class ImageUnit:
    image_id: str
    doc_id: str
    title: str
    source_filename: str
    page: int | None       # None for standalone images
    image_path: str        # path to the extracted/standalone PNG
    bbox: dict | None = None  # {x0,x1,top,bottom} within the page, if extracted from a PDF


def _chunk_text(text: str, chunk_size=CHUNK_SIZE_CHARS, overlap=CHUNK_OVERLAP_CHARS) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def _load_manifest() -> list[dict]:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def extract_pdf(doc_id: str, title: str, pdf_path: str) -> tuple[list[TextChunk], list[ImageUnit]]:
    """Real pdfplumber extraction: text per page + embedded raster images,
    cropped out of the rendered page and saved as standalone PNGs."""
    text_chunks: list[TextChunk] = []
    image_units: list[ImageUnit] = []
    filename = os.path.basename(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            for i, chunk_text in enumerate(_chunk_text(page_text)):
                text_chunks.append(
                    TextChunk(
                        chunk_id=f"{doc_id}-p{page_idx}-c{i}",
                        doc_id=doc_id,
                        title=title,
                        source_filename=filename,
                        page=page_idx,
                        text=chunk_text,
                    )
                )

            for img_idx, im in enumerate(page.images):
                try:
                    cropped = page.crop((im["x0"], im["top"], im["x1"], im["bottom"]))
                    rendered = cropped.to_image(resolution=150)
                    out_name = f"{doc_id}_p{page_idx}_img{img_idx}.png"
                    out_path = os.path.join(EXTRACTED_IMG_DIR, out_name)
                    rendered.save(out_path)
                    image_units.append(
                        ImageUnit(
                            image_id=f"{doc_id}-p{page_idx}-img{img_idx}",
                            doc_id=doc_id,
                            title=title,
                            source_filename=filename,
                            page=page_idx,
                            image_path=out_path,
                            bbox={"x0": im["x0"], "x1": im["x1"], "top": im["top"], "bottom": im["bottom"]},
                        )
                    )
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning(f"Failed to extract image {img_idx} on {doc_id} p{page_idx}: {e}")

    return text_chunks, image_units


def load_standalone_image(doc_id: str, title: str, image_path: str) -> ImageUnit:
    return ImageUnit(
        image_id=f"{doc_id}-standalone",
        doc_id=doc_id,
        title=title,
        source_filename=os.path.basename(image_path),
        page=None,
        image_path=image_path,
        bbox=None,
    )


def ingest_corpus() -> tuple[list[TextChunk], list[ImageUnit]]:
    os.makedirs(EXTRACTED_IMG_DIR, exist_ok=True)
    manifest = _load_manifest()
    all_chunks: list[TextChunk] = []
    all_images: list[ImageUnit] = []

    t0 = time.time()
    for doc in manifest:
        doc_id, kind = doc["doc_id"], doc["kind"]
        title = doc["path"].split("/")[-1].replace("_", " ").replace(".pdf", "").replace(".png", "")
        full_path = os.path.join(DATA_DIR, doc["path"])
        if kind == "pdf":
            chunks, images = extract_pdf(doc_id, title, full_path)
            all_chunks.extend(chunks)
            all_images.extend(images)
            logger.info(f"{doc_id}: {len(chunks)} text chunks, {len(images)} images extracted from {full_path}")
        elif kind == "image":
            img_unit = load_standalone_image(doc_id, title, full_path)
            all_images.append(img_unit)
            logger.info(f"{doc_id}: standalone image {full_path}")
        else:
            logger.warning(f"Unknown doc kind '{kind}' for {doc_id}, skipping")
    elapsed = time.time() - t0
    logger.info(
        f"Ingestion complete in {elapsed:.2f}s -- {len(all_chunks)} text chunks, "
        f"{len(all_images)} image units across {len(manifest)} documents."
    )
    return all_chunks, all_images


if __name__ == "__main__":
    chunks, images = ingest_corpus()
    print(f"\n{len(chunks)} text chunks, {len(images)} image units.\n")
    print("Sample text chunk:")
    print(json.dumps(asdict(chunks[0]), indent=2)[:500])
    print("\nSample image unit:")
    print(json.dumps(asdict(images[0]), indent=2))

"""
tests/test_ingestion.py
Run:
    cd task1_multimodal_retrieval
    python -m pytest tests/ -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion import ingest_corpus, _chunk_text, TextChunk, ImageUnit  # noqa: E402


def test_chunking_short_text_returns_single_chunk():
    chunks = _chunk_text("short text", chunk_size=500)
    assert chunks == ["short text"]


def test_chunking_empty_text_returns_empty_list():
    assert _chunk_text("") == []
    assert _chunk_text("   ") == []


def test_chunking_long_text_splits_with_overlap():
    long_text = "word " * 300  # ~1500 chars
    chunks = _chunk_text(long_text, chunk_size=500, overlap=80)
    assert len(chunks) > 1
    # every chunk should respect the size bound
    assert all(len(c) <= 500 for c in chunks)


def test_ingest_corpus_produces_chunks_and_images():
    chunks, images = ingest_corpus()
    assert len(chunks) > 0
    assert len(images) > 0
    assert all(isinstance(c, TextChunk) for c in chunks)
    assert all(isinstance(i, ImageUnit) for i in images)


def test_ingest_corpus_covers_all_12_documents():
    chunks, images = ingest_corpus()
    doc_ids = {c.doc_id for c in chunks} | {i.doc_id for i in images}
    assert len(doc_ids) == 12


def test_standalone_images_have_no_page_number():
    _chunks, images = ingest_corpus()
    standalone = [i for i in images if i.doc_id in ("doc05", "doc08")]
    assert len(standalone) == 2
    assert all(i.page is None for i in standalone)


def test_extracted_pdf_images_have_page_and_bbox():
    _chunks, images = ingest_corpus()
    extracted = [i for i in images if i.doc_id == "doc01"]
    assert len(extracted) == 1
    assert extracted[0].page == 1
    assert extracted[0].bbox is not None
    assert os.path.exists(extracted[0].image_path)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

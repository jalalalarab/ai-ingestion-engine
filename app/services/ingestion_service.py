"""
Ingestion service — orchestrates the pipeline.

Given a PDF's bytes + filename, runs:
  extract (PDF parser) → chunk → embed → upsert to Qdrant

The `_ingest_texts` helper is source-agnostic: it takes a list of (text, page_number)
tuples and does the shared "chunk + embed + store" work. Phase 5's video extractor
will feed into this same helper, so the engine is only written once.

Returns an IngestionReport summarizing what happened, which the API returns as JSON.
"""
from dataclasses import dataclass, asdict
from uuid import uuid4

from app.parsers.pdf_parser import extract_pdf_pages
from app.chunking.simple_chunker import chunk_text
from app.embeddings.embedding_client import embed_batch
from app.vector_store.qdrant_store import ensure_collection, upsert_chunks


@dataclass
class IngestionReport:
    file_id: str
    file_name: str
    source_type: str
    pages_processed: int
    chunks_created: int
    ocr_pages_count: int


def ingest_pdf(pdf_bytes: bytes, file_name: str) -> IngestionReport:
    """
    Full PDF ingestion pipeline.

    Steps:
      1. Extract text page-by-page.
      2. Chunk each page (attaching page number to every chunk).
      3. Embed all chunks in one batch.
      4. Upsert to Qdrant with metadata.
    """
    ensure_collection()  # cheap safety check every ingest

    file_id = str(uuid4())  # unique per file, used as the Qdrant point-ID seed

    # Step 1: extract pages
    pages = extract_pdf_pages(pdf_bytes)

    # Count how many pages required OCR fallback — useful signal for the report
    # and for spotting scanned documents in the audit trail.
    ocr_pages_count = sum(1 for _, _, method in pages if method == "ocr")

    # Step 2: chunk each page, tagging every chunk with its origin page number.
    # The `method` is captured above but not yet threaded into chunk payloads —
    # that upgrade is planned as a follow-up (extraction_method on every chunk).
    all_chunks: list[str] = []
    all_page_numbers: list[int | None] = []
    for page_number, page_text, _method in pages:
        for chunk in chunk_text(page_text):
            all_chunks.append(chunk)
            all_page_numbers.append(page_number)

    # If the PDF was image-only or empty, nothing to embed — return early.
    if not all_chunks:
        return IngestionReport(
            file_id=file_id,
            file_name=file_name,
            source_type="pdf",
            pages_processed=len(pages),
            chunks_created=0,
            ocr_pages_count=ocr_pages_count,
        )

    # Step 3 + 4: embed and upsert (shared with video in Phase 5)
    n = _ingest_texts(
        file_id=file_id,
        file_name=file_name,
        source_type="pdf",
        chunks=all_chunks,
        page_numbers=all_page_numbers,
    )

    return IngestionReport(
        file_id=file_id,
        file_name=file_name,
        source_type="pdf",
        pages_processed=len(pages),
        chunks_created=n,
        ocr_pages_count=ocr_pages_count,
    )


def _ingest_texts(
    file_id: str,
    file_name: str,
    source_type: str,
    chunks: list[str],
    page_numbers: list[int | None],
) -> int:
    """
    Shared 'embed + store' step. Source-agnostic.

    This is the seam where PDF ingestion and (later) video ingestion converge.
    Both hand off a list of chunks + metadata; this function embeds and stores.
    Returns the number of chunks upserted.
    """
    vectors = embed_batch(chunks)
    return upsert_chunks(
        file_id=file_id,
        file_name=file_name,
        source_type=source_type,
        chunks=chunks,
        vectors=vectors,
        page_numbers=page_numbers,
    )
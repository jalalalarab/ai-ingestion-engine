"""
Ingestion service — orchestrates the pipeline.

Given a file's bytes/path + filename, runs:
  extract (PDF parser OR video parser) -> chunk -> embed -> upsert to Qdrant

The `_ingest_texts` helper is source-agnostic: it takes a list of chunks plus
metadata and does the shared "embed + store" work. Both PDF and video ingestion
feed into it, so the engine is only written once.

Returns a report summarizing what happened, which the API returns as JSON.
"""
from dataclasses import dataclass

from uuid import uuid4

from app.parsers.pdf_parser import extract_pdf_pages
from app.parsers.video_parser import extract_video_frames
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


@dataclass
class VideoIngestionReport:
    file_id: str
    file_name: str
    source_type: str
    frames_ingested: int      # frames that survived OCR + de-duplication
    chunks_created: int


def ingest_pdf(pdf_bytes: bytes, file_name: str) -> IngestionReport:
    """
    Full PDF ingestion pipeline.

    Steps:
      1. Extract text page-by-page (OCR fallback for scanned pages).
      2. Chunk each page (attaching page number to every chunk).
      3. Embed all chunks in one batch.
      4. Upsert to Qdrant with metadata.
    """
    ensure_collection()  # cheap safety check every ingest

    file_id = str(uuid4())  # unique per file, used as the Qdrant point-ID seed

    # Step 1: extract pages -> list of (page_number, text, method)
    pages = extract_pdf_pages(pdf_bytes)

    # Count how many pages required OCR fallback — useful signal for the report
    # and for spotting scanned documents in the audit trail.
    ocr_pages_count = sum(1 for _, _, method in pages if method == "ocr")

    # Step 2: chunk each page, tagging every chunk with its origin page number.
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

    # Step 3 + 4: embed and upsert (shared seam)
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


def ingest_video(video_path: str, file_name: str) -> VideoIngestionReport:
    """
    Full video ingestion pipeline.

    Steps:
      1. Sample frames every couple of seconds, OCR each, drop blanks/duplicates.
      2. Chunk each frame's text (attaching timestamp + frame number).
      3. Embed all chunks in one batch.
      4. Upsert to Qdrant with video metadata.
    """
    ensure_collection()

    file_id = str(uuid4())

    # Step 1: extract frames -> list of (timestamp_seconds, frame_number, text)
    frames = extract_video_frames(video_path)

    # Step 2: chunk each frame, tagging every chunk with timestamp + frame number.
    all_chunks: list[str] = []
    all_timestamps: list[int | None] = []
    all_frame_numbers: list[int | None] = []
    for timestamp_seconds, frame_number, frame_text in frames:
        for chunk in chunk_text(frame_text):
            all_chunks.append(chunk)
            all_timestamps.append(timestamp_seconds)
            all_frame_numbers.append(frame_number)

    # No readable frames -> nothing to embed.
    if not all_chunks:
        return VideoIngestionReport(
            file_id=file_id,
            file_name=file_name,
            source_type="video",
            frames_ingested=len(frames),
            chunks_created=0,
        )

    # Step 3 + 4: embed and upsert through the same shared seam as PDF.
    n = _ingest_texts(
        file_id=file_id,
        file_name=file_name,
        source_type="video",
        chunks=all_chunks,
        page_numbers=[None] * len(all_chunks),  # video has no pages
        timestamps=all_timestamps,
        frame_numbers=all_frame_numbers,
    )

    return VideoIngestionReport(
        file_id=file_id,
        file_name=file_name,
        source_type="video",
        frames_ingested=len(frames),
        chunks_created=n,
    )


def _ingest_texts(
    file_id: str,
    file_name: str,
    source_type: str,
    chunks: list[str],
    page_numbers: list[int | None],
    timestamps: list[int | None] | None = None,
    frame_numbers: list[int | None] | None = None,
) -> int:
    """
    Shared 'embed + store' step. Source-agnostic.

    This is the seam where PDF ingestion and video ingestion converge.
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
        timestamps=timestamps,
        frame_numbers=frame_numbers,
    )

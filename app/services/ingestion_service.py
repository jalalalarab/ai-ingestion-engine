"""
Ingestion service - orchestrates the pipeline.

Given a file's bytes/path + filename, runs:
  extract (PDF parser OR video parser) -> chunk -> embed -> upsert to Qdrant

The `_ingest_texts` helper is source-agnostic: it takes a list of chunks plus
metadata and does the shared "embed + store" work. Both PDF and video ingestion
feed into it, so the engine is only written once.

Returns a report summarizing what happened, which the API returns as JSON.

Phase 7 change (content-hash dedup):
  file_id is DERIVED FROM THE FILE'S CONTENT instead of a random uuid4().
  Same bytes in -> same file_id out. Because Qdrant point IDs are seeded from
  file_id, re-ingesting an identical file reuses the same IDs (upsert overwrites
  in place) instead of writing a second random copy. We keep the ID in UUID form
  (uuid5 of the SHA-256 digest) so it stays a valid drop-in for str(uuid4()).

Phase 7 change (logging):
  Each stage logs at INFO so ingestion is visible in the console. The file_id is
  logged on every run - re-ingesting an identical file prints the SAME id, which
  is the live proof that dedup works (it overwrites in place).

Chunking strategy toggle:
  `_chunk` picks the chunker based on settings.CHUNKING_STRATEGY:
  "semantic" (embedding-similarity splits) or "simple" (fixed-size window).
  Both PDF and video route through `_chunk`, so the choice is made in one place.
"""
from dataclasses import dataclass

import hashlib
import logging
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

from app.config import settings
from app.parsers.pdf_parser import extract_pdf_pages
from app.parsers.video_parser import extract_video_frames
from app.parsers.audio_extractor import extract_audio
from app.transcription.transcription_client import transcribe_audio
from app.chunking.simple_chunker import chunk_text as _simple_chunk
from app.chunking.semantic_chunker import semantic_chunk_text as _semantic_chunk
from app.embeddings.embedding_client import embed_batch
from app.vector_store.qdrant_store import ensure_collection, upsert_chunks


logger = logging.getLogger(__name__)


# Fixed namespace for turning a content hash into a stable UUID. Any constant
# UUID works; NAMESPACE_URL is a standard, well-known one. Keeping it fixed is
# what guarantees "same file -> same file_id" across runs and machines.
_HASH_NAMESPACE = NAMESPACE_URL


def _chunk(text: str) -> list[str]:
    """
    Dispatch to the configured chunker. Defaults to semantic for any value
    other than the explicit "simple", so a typo in .env fails safe (better
    chunking, never a crash).
    """
    if settings.CHUNKING_STRATEGY == "simple":
        return _simple_chunk(text)
    return _semantic_chunk(text)


def _file_id_from_bytes(data: bytes) -> str:
    """
    Deterministic file_id from raw file bytes.

    SHA-256 the content, then fold that digest into a UUID (uuid5). Identical
    content always yields the same UUID; any change to the bytes changes the
    hash and therefore the id, so an edited file is correctly treated as new.
    Returns a str so it's an exact drop-in for the old str(uuid4()).
    """
    digest = hashlib.sha256(data).hexdigest()
    return str(uuid5(_HASH_NAMESPACE, digest))


def _file_id_from_path(path: str) -> str:
    """
    Same as _file_id_from_bytes, but streams the file from disk in 1 MB blocks
    so we don't load an entire video into memory just to hash it.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return str(uuid5(_HASH_NAMESPACE, hasher.hexdigest()))


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
    transcript_segments: int = 0   # spoken segments captured via transcription
    chunks_created: int = 0


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

    # Content-hash id: same PDF -> same id -> upsert overwrites instead of
    # duplicating. Replaces the old random str(uuid4()).
    file_id = _file_id_from_bytes(pdf_bytes)
    logger.info("PDF ingest start: '%s' [file_id=%s] [chunker=%s]",
                file_name, file_id[:8], settings.CHUNKING_STRATEGY)

    # Step 1: extract pages -> list of (page_number, text, method)
    pages = extract_pdf_pages(pdf_bytes)

    # Count how many pages required OCR fallback - useful signal for the report
    # and for spotting scanned documents in the audit trail.
    ocr_pages_count = sum(1 for _, _, method in pages if method == "ocr")
    logger.info("Extracted %d pages (%d via OCR)", len(pages), ocr_pages_count)

    # Step 2: chunk each page, tagging every chunk with its origin page number.
    all_chunks: list[str] = []
    all_page_numbers: list[int | None] = []
    for page_number, page_text, _method in pages:
        for chunk in _chunk(page_text):
            all_chunks.append(chunk)
            all_page_numbers.append(page_number)

    # If the PDF was image-only or empty, nothing to embed - return early.
    if not all_chunks:
        logger.warning("No text extracted from '%s' - nothing to embed", file_name)
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

    logger.info("PDF ingest done: '%s' -> %d chunks stored [file_id=%s]",
                file_name, n, file_id[:8])
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

    # Content-hash id from the file on disk (streamed, so we don't load the
    # whole video into memory). Replaces the old random str(uuid4()).
    file_id = _file_id_from_path(video_path)
    logger.info("Video ingest start: '%s' [file_id=%s] [chunker=%s]",
                file_name, file_id[:8], settings.CHUNKING_STRATEGY)

    # Step 1: extract frames -> list of (timestamp_seconds, frame_number, text)
    frames = extract_video_frames(video_path)
    logger.info("Extracted %d usable frames from '%s'", len(frames), file_name)

    # Step 2: chunk each frame, tagging every chunk with timestamp + frame number.
    all_chunks: list[str] = []
    all_timestamps: list[int | None] = []
    all_frame_numbers: list[int | None] = []
    for timestamp_seconds, frame_number, frame_text in frames:
        for chunk in _chunk(frame_text):
            all_chunks.append(chunk)
            all_timestamps.append(timestamp_seconds)
            all_frame_numbers.append(frame_number)

    # Step 2b: transcribe the spoken audio (if enabled and the video has audio).
    # Each transcript segment becomes a chunk tagged with its start timestamp,
    # exactly like a frame chunk — so spoken content is searchable and citable.
    transcript_segment_count = 0
    if settings.TRANSCRIBE_VIDEO and settings.OPENAI_API_KEY:
        try:
            audio_path = extract_audio(video_path)
            if audio_path is None:
                logger.info("No audio track in '%s' - skipping transcription", file_name)
            else:
                try:
                    segments = transcribe_audio(audio_path)
                    transcript_segment_count = len(segments)
                    logger.info("Transcribed %d segments from '%s'",
                                len(segments), file_name)
                    for start_seconds, seg_text in segments:
                        for chunk in _chunk(seg_text):
                            all_chunks.append(chunk)
                            all_timestamps.append(start_seconds)
                            all_frame_numbers.append(None)  # transcript has no frame
                finally:
                    Path(audio_path).unlink(missing_ok=True)  # clean up temp mp3
        except Exception as exc:  # transcription is best-effort; never fail ingest
            logger.warning("Transcription failed for '%s': %s", file_name, exc)
    elif settings.TRANSCRIBE_VIDEO and not settings.OPENAI_API_KEY:
        logger.info("TRANSCRIBE_VIDEO is on but no OPENAI_API_KEY - skipping transcription")

    # No readable frames -> nothing to embed.
    if not all_chunks:
        logger.warning("No readable text in '%s' - nothing to embed", file_name)
        return VideoIngestionReport(
            file_id=file_id,
            file_name=file_name,
            source_type="video",
            frames_ingested=len(frames),
            transcript_segments=transcript_segment_count,
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

    logger.info("Video ingest done: '%s' -> %d chunks stored [file_id=%s]",
                file_name, n, file_id[:8])
    return VideoIngestionReport(
        file_id=file_id,
        file_name=file_name,
        source_type="video",
        frames_ingested=len(frames),
        transcript_segments=transcript_segment_count,
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
    logger.debug("Embedding %d chunks (%s) then upserting to Qdrant",
                 len(chunks), source_type)
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

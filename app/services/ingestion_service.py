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

Video ingestion (synthesis):
  `ingest_video` assembles the frames + transcript into ONE grounded Markdown
  document (the "long document"), then chunks/embeds THAT document. Raw frame and
  transcript chunks are not stored — the synthesized document is the sole index.
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
from app.synthesis.document_assembler import assemble_document


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
    Video ingestion via multi-agent synthesis.

    Instead of storing raw frame + transcript chunks (the old approach, which
    left a "moment" split across un-aligned chunks), this:
      1. Extracts frames (vision descriptions) + transcript (Whisper).
      2. Assembles them into ONE grounded Markdown document — the "long
         document" narrating the whole video (windows -> synthesize -> stitch).
      3. Chunks THAT document and embeds/stores it via the shared seam.

    The synthesized document is the only thing indexed; raw frame/transcript
    chunks are not stored. Document chunks carry no per-frame timestamp
    (narrative prose), so video results cite the file, not a moment.

    Rate-limit safety lives upstream: VIDEO_SAMPLE_SECONDS controls how many
    frames are described, and LLM_CALL_DELAY_SECONDS paces the vision + synthesis
    calls so a long video doesn't exceed OpenAI's tokens-per-minute limit.
    """
    ensure_collection()

    file_id = _file_id_from_path(video_path)
    logger.info("Video ingest (synthesis) start: '%s' [file_id=%s]",
                file_name, file_id[:8])

    # Step 1: frames -> (timestamp_seconds, frame_number, description)
    frames = extract_video_frames(video_path)
    logger.info("Extracted %d usable frames from '%s'", len(frames), file_name)

    # Step 1b: transcript -> (start_seconds, text)
    transcript: list[tuple[int, str]] = []
    if settings.TRANSCRIBE_VIDEO and settings.OPENAI_API_KEY:
        try:
            audio_path = extract_audio(video_path)
            if audio_path is None:
                logger.info("No audio track in '%s' - skipping transcription", file_name)
            else:
                try:
                    transcript = transcribe_audio(audio_path)
                    logger.info("Transcribed %d segments from '%s'",
                                len(transcript), file_name)
                finally:
                    Path(audio_path).unlink(missing_ok=True)
        except Exception as exc:  # best-effort; never fail ingest on transcription
            logger.warning("Transcription failed for '%s': %s", file_name, exc)
    elif settings.TRANSCRIBE_VIDEO and not settings.OPENAI_API_KEY:
        logger.info("TRANSCRIBE_VIDEO is on but no OPENAI_API_KEY - skipping transcription")

    # Step 2: assemble the long document (windows -> synthesize -> stitch)
    assembly = assemble_document(
        file_name=file_name,
        frames=frames,
        transcript=transcript,
        max_chars=settings.SYNTHESIS_WINDOW_CHARS,
    )
    document = assembly["document"]
    logger.info(
        "Synthesized document for '%s': %d/%d windows, %d chars",
        file_name, assembly["windows_written"], assembly["windows_total"],
        assembly["char_count"],
    )

    # Step 3: chunk the DOCUMENT (not raw frames/transcript)
    doc_chunks = _chunk(document)

    if not doc_chunks:
        logger.warning("Synthesized document for '%s' is empty - nothing to embed", file_name)
        return VideoIngestionReport(
            file_id=file_id,
            file_name=file_name,
            source_type="video",
            frames_ingested=len(frames),
            transcript_segments=len(transcript),
            chunks_created=0,
        )

    # Step 4: embed + store via the same shared seam as PDF.
    # Document chunks have no per-frame timestamp or frame number (narrative prose).
    n = _ingest_texts(
        file_id=file_id,
        file_name=file_name,
        source_type="video",
        chunks=doc_chunks,
        page_numbers=[None] * len(doc_chunks),
        timestamps=[None] * len(doc_chunks),
        frame_numbers=[None] * len(doc_chunks),
    )

    logger.info("Video ingest (synthesis) done: '%s' -> %d document chunks stored [file_id=%s]",
                file_name, n, file_id[:8])
    return VideoIngestionReport(
        file_id=file_id,
        file_name=file_name,
        source_type="video",
        frames_ingested=len(frames),
        transcript_segments=len(transcript),
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

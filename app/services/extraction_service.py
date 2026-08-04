"""
Extraction service — coordinates entity/relationship extraction for a file.

Pulls a file's content chunks back out of Qdrant, then runs the entity extractor
over them to produce triples.

This is the Phase B seam: content in Qdrant -> triples out. Later phases load
these triples into a graph database (Neo4j) and traverse them (GraphRAG).

Handles BOTH video ingestion modes:
  - RAW mode: the file has separate transcript chunks (timestamp set, no frame
    number) and frame chunks. We extract from the transcript segments — the
    spoken meeting content.
  - SYNTHESIS mode: the file is one synthesized document, stored as narrative
    chunks with NO timestamp and NO frame number. We extract from those document
    chunks directly (they ARE the content).
"""
import logging

from app.vector_store.qdrant_store import get_chunks_by_file_id
from app.extraction.entity_extractor import extract_triples

logger = logging.getLogger(__name__)


def extract_from_file(file_id: str) -> dict:
    """
    Extract triples from an ingested file's content.

    Selects the right chunks depending on how the video was ingested:
      - If there are transcript chunks (timestamp set, no frame number), use those
        (RAW mode — the spoken meeting content, ordered by time).
      - Otherwise, fall back to the document/narrative chunks (SYNTHESIS mode —
        the synthesized long document, which has no timestamps).

    Returns the extractor's result dict plus file_id/file_name for the caller.

    Raises:
        RuntimeError: if the file has no usable content chunks at all.
    """
    chunks = get_chunks_by_file_id(file_id, source_type="video")

    if not chunks:
        raise RuntimeError(
            f"No chunks found for file_id '{file_id}'. "
            f"Extraction needs an ingested video."
        )

    # RAW-mode transcript segments: spoken audio (timestamp present, no frame_number).
    transcript = [
        c for c in chunks
        if c.get("timestamp_seconds") is not None and c.get("frame_number") is None
    ]

    if transcript:
        # RAW mode: order the spoken segments by time.
        transcript.sort(key=lambda c: c.get("timestamp_seconds") or 0)
        source_chunks = transcript
        mode = "raw transcript"
    else:
        # SYNTHESIS mode: no timestamped transcript. Extract from the document
        # chunks — the synthesized narrative. Exclude any raw frame chunks
        # (frame_number set) just in case, and skip empty text.
        source_chunks = [
            c for c in chunks
            if c.get("frame_number") is None and (c.get("text") or "").strip()
        ]
        mode = "synthesis document"

    if not source_chunks:
        raise RuntimeError(
            f"No usable content found for file_id '{file_id}'. "
            f"Extraction needs an ingested video with a transcript or synthesized document."
        )

    file_name = source_chunks[0].get("file_name")
    segments = [c.get("text", "") for c in source_chunks]

    logger.info(
        "Extracting entities from '%s' (%d chunks, %s)",
        file_name, len(segments), mode,
    )
    result = extract_triples(segments)

    return {
        "file_id": file_id,
        "file_name": file_name,
        **result,
    }

"""
Extraction service — coordinates entity/relationship extraction for a file.

Pulls a file's transcript chunks back out of Qdrant (same scroll the minutes
service uses), then runs the entity extractor over them to produce triples.

This is the Phase B seam: transcript in Qdrant -> triples out. Later phases load
these triples into a graph database (Neo4j) and traverse them (GraphRAG).
"""
import logging

from app.vector_store.qdrant_store import get_chunks_by_file_id
from app.extraction.entity_extractor import extract_triples

logger = logging.getLogger(__name__)


def extract_from_file(file_id: str) -> dict:
    """
    Extract triples from an ingested file's transcript.

    Pulls the file's chunks from Qdrant, keeps the spoken-audio transcript
    segments (timestamp set, no frame number — the actual meeting speech), and
    runs extraction over them.

    Returns the extractor's result dict plus file_id/file_name for the caller.

    Raises:
        RuntimeError: if the file has no transcript chunks (nothing to extract).
    """
    chunks = get_chunks_by_file_id(file_id, source_type="video")

    # Transcript segments = spoken audio (timestamp present, no frame_number).
    # These carry the meeting's actual content; frame/vision chunks are separate.
    transcript = [
        c for c in chunks
        if c.get("timestamp_seconds") is not None and c.get("frame_number") is None
    ]
    transcript.sort(key=lambda c: c.get("timestamp_seconds") or 0)

    if not transcript:
        raise RuntimeError(
            f"No transcript found for file_id '{file_id}'. "
            f"Extraction needs an ingested video with a transcript."
        )

    file_name = transcript[0].get("file_name")
    segments = [c.get("text", "") for c in transcript]

    logger.info("Extracting entities from '%s' (%d transcript segments)", file_name, len(segments))
    result = extract_triples(segments)

    return {
        "file_id": file_id,
        "file_name": file_name,
        **result,
    }

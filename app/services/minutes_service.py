"""
Minutes-of-Meeting service.

Given an ingested video's file_id, pull its full transcript back out of Qdrant
and produce structured meeting minutes (Overview, Attendees, Key Points,
Decisions, Action Items) using the LLM.

The context-window problem:
  A long meeting's transcript can exceed what the LLM can read in one call.
  So this uses MAP-REDUCE:
    - If the transcript fits in one call, summarize it directly (fast path).
    - If it's too long, split it into batches, summarize EACH batch ("map"),
      then combine those partial summaries into the final minutes ("reduce").
  This way the feature works for a 2-minute stand-up or a 2-hour meeting.

Token budgeting is approximate: we estimate ~4 characters per token and keep
each LLM call's input under a conservative character budget.
"""
import logging

from app.config import settings
from app.llm.llm_client import generate_answer
from app.vector_store.qdrant_store import get_chunks_by_file_id

logger = logging.getLogger(__name__)

# Conservative input budget per LLM call, in characters (~4 chars/token).
# Kept well under the model's real context window to leave room for the prompt
# and the generated output. Tune via .env if needed.
_CHARS_PER_BATCH = settings.MOM_BATCH_CHARS


_MINUTES_SYSTEM_PROMPT = (
    "You are an assistant that writes clear, professional Minutes of Meeting from "
    "a meeting transcript. Produce well-structured minutes with these sections:\n"
    "1. Overview — a 1-2 sentence summary of the meeting.\n"
    "2. Attendees — names mentioned, if any (otherwise 'Not specified').\n"
    "3. Key Points — the main topics discussed, as bullet points.\n"
    "4. Decisions — any decisions made.\n"
    "5. Action Items — tasks assigned, with who is responsible if stated.\n"
    "Base the minutes ONLY on the transcript provided. Do not invent details. "
    "If a section has no content, write 'None noted.'"
)

_REDUCE_SYSTEM_PROMPT = (
    "You are combining several partial meeting-minutes summaries (each covering a "
    "part of the same meeting, in order) into ONE final set of Minutes of Meeting. "
    "Merge them into a single coherent document with these sections: Overview, "
    "Attendees, Key Points, Decisions, Action Items. Remove duplication, keep it "
    "consistent and professional. Do not invent details beyond the partial summaries."
)


def _batch_transcript(chunks: list[dict]) -> list[str]:
    """
    Group ordered transcript chunks into batches that each fit the char budget.

    Returns a list of transcript-text batches (each a big string).
    """
    batches: list[str] = []
    current: list[str] = []
    current_len = 0
    for c in chunks:
        text = c.get("text", "").strip()
        if not text:
            continue
        if current_len + len(text) > _CHARS_PER_BATCH and current:
            batches.append("\n".join(current))
            current = []
            current_len = 0
        current.append(text)
        current_len += len(text)
    if current:
        batches.append("\n".join(current))
    return batches


def generate_minutes(file_id: str) -> dict:
    """
    Produce Minutes of Meeting for an ingested video.

    Returns a dict: {file_id, file_name, minutes, batches_used, method}.
    Raises RuntimeError if the file has no transcript chunks.
    """
    # Pull the whole transcript (video chunks) for this file, in order.
    chunks = get_chunks_by_file_id(file_id, source_type="video")
    # Keep only chunks that came from speech (transcript), i.e. have a timestamp
    # but no frame_number — frame OCR chunks have a frame_number.
    transcript_chunks = [
        c for c in chunks
        if c.get("timestamp_seconds") is not None and c.get("frame_number") is None
    ]

    if not transcript_chunks:
        raise RuntimeError(
            f"No transcript found for file_id={file_id}. "
            "Was this an audio-bearing video ingested with transcription enabled?"
        )

    file_name = transcript_chunks[0].get("file_name") or "unknown"
    batches = _batch_transcript(transcript_chunks)
    logger.info("MoM for '%s': %d transcript chunks -> %d batch(es)",
                file_name, len(transcript_chunks), len(batches))

    # FAST PATH — everything fits in one call.
    if len(batches) == 1:
        minutes = generate_answer(
            system_prompt=_MINUTES_SYSTEM_PROMPT,
            user_prompt=f"Meeting transcript:\n\n{batches[0]}\n\nWrite the Minutes of Meeting.",
        )
        return {
            "file_id": file_id,
            "file_name": file_name,
            "minutes": minutes.strip(),
            "batches_used": 1,
            "method": "single-pass",
        }

    # MAP-REDUCE PATH — transcript too long for one call.
    # MAP: summarize each batch into partial minutes.
    partials: list[str] = []
    for i, batch in enumerate(batches, start=1):
        logger.info("MoM map step %d/%d for '%s'", i, len(batches), file_name)
        partial = generate_answer(
            system_prompt=_MINUTES_SYSTEM_PROMPT,
            user_prompt=(
                f"This is part {i} of {len(batches)} of a meeting transcript.\n\n"
                f"{batch}\n\nWrite partial Minutes of Meeting for THIS part only."
            ),
        )
        partials.append(f"--- Partial minutes {i} ---\n{partial.strip()}")

    # REDUCE: combine the partial minutes into one final document.
    logger.info("MoM reduce step for '%s' (%d partials)", file_name, len(partials))
    combined = "\n\n".join(partials)
    final_minutes = generate_answer(
        system_prompt=_REDUCE_SYSTEM_PROMPT,
        user_prompt=(
            f"Combine these {len(partials)} partial meeting-minutes summaries into "
            f"one final Minutes of Meeting:\n\n{combined}"
        ),
    )

    return {
        "file_id": file_id,
        "file_name": file_name,
        "minutes": final_minutes.strip(),
        "batches_used": len(batches),
        "method": "map-reduce",
    }

"""
Document assembler — Step 3 of the video-synthesis pipeline.

Ties the whole thing together. Given a video's raw streams (frames + transcript),
it:
  1. splits them into time-ordered windows        (Step 1: window_splitter)
  2. synthesizes each window into grounded prose   (Step 2: synthesis_agent)
  3. stitches the narratives into ONE Markdown document describing the whole video

The output document is the "long document" — a clean, human-readable narrative of
the entire video, which downstream steps chunk, embed, and graph (replacing the
old raw-chunk approach).

Windows are synthesized SEQUENTIALLY in this version: simpler to debug, and it
respects OpenAI rate limits (the real video hit 429s under bursty load).
Parallelism is a later optimization if speed demands it.

The LLM client is injected so the whole assembly is unit-testable with a fake.
"""
from __future__ import annotations

import logging

from app.synthesis.window_splitter import split_into_windows
from app.synthesis.synthesis_agent import synthesize_window

logger = logging.getLogger(__name__)


def assemble_document(
    file_name: str,
    frames: list[tuple[int, int, str]],
    transcript: list[tuple[int, str]],
    max_chars: int = 2500,
    client=None,
) -> dict:
    """
    Build one Markdown document narrating a whole video.

    Args:
        file_name:  the video's name, used in the document title.
        frames:     (timestamp_seconds, frame_number, description) tuples.
        transcript: (start_seconds, text) tuples.
        max_chars:  window size budget (see window_splitter). ~2500 keeps each
                    synthesis call focused, which limits hallucination.
        client:     OpenAI-compatible client (injected for tests). If None, the
                    synthesis agent creates a real one per call.

    Returns:
        {
          "file_name": str,
          "document": str,          # the full Markdown document
          "windows_total": int,
          "windows_written": int,   # windows that produced non-empty narrative
          "char_count": int,        # length of the document
        }

    An empty video (no usable content) yields a document with just the title and
    a note — never fabricated content.
    """
    windows = split_into_windows(frames, transcript, max_chars=max_chars)

    title = f"# {_clean_title(file_name)}\n"

    if not windows:
        logger.info("Assembler: no windows for '%s' — empty document.", file_name)
        document = title + "\n_No transcript or on-screen content was available for this video._\n"
        return {
            "file_name": file_name,
            "document": document,
            "windows_total": 0,
            "windows_written": 0,
            "char_count": len(document),
        }

    sections: list[str] = [title]
    written = 0

    for window in windows:
        narrative = synthesize_window(window, client=client)
        if not narrative.strip():
            # A window with nothing to say contributes nothing — we do NOT invent
            # filler to keep sections uniform.
            continue
        sections.append(narrative.strip())
        written += 1

    logger.info(
        "Assembler: '%s' -> %d/%d windows written.",
        file_name, written, len(windows),
    )

    # Join sections with blank lines so the Markdown reads as flowing paragraphs.
    document = "\n\n".join(sections) + "\n"

    return {
        "file_name": file_name,
        "document": document,
        "windows_total": len(windows),
        "windows_written": written,
        "char_count": len(document),
    }


def _clean_title(file_name: str) -> str:
    """Turn a filename into a readable title: strip extension, de-underscore, title-case."""
    name = file_name.rsplit(".", 1)[0]           # drop extension
    name = name.replace("_", " ").replace("-", " ").strip()
    return name.title() if name else "Video"

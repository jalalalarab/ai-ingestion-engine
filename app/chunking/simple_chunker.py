"""
Simple V1 chunker: split text into overlapping fixed-size chunks.

Rules (from build plan §5):
- Target size ~500-900 tokens.
- Overlap ~80-150 tokens between adjacent chunks.
- Approximates tokens as characters * 0.25 (1 token ≈ 4 chars in English).
- Doesn't try to split cleanly on sentences yet — that's V2.

This module doesn't know about PDFs, videos, or Qdrant. It takes text, returns chunks.
"""
from app.config import settings

# 1 token ≈ 4 characters in English — good enough for V1.
CHARS_PER_TOKEN = 4


def chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping chunks.

    Returns:
        A list of chunk strings, in order.
        Empty input returns [].
        Text shorter than one chunk returns [text].
    """
    # Skip empty / whitespace-only input — happens for blank pages, image-only pages.
    if not text or not text.strip():
        return []

    # Convert token sizes from settings into character sizes for this V1.
    chunk_chars = settings.CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN     # e.g. 700 * 4 = 2800
    overlap_chars = settings.CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN  # e.g. 100 * 4 = 400
    step = chunk_chars - overlap_chars  # how far we advance per chunk

    # If the whole text fits in one chunk, return it as-is (no overlap needed).
    if len(text) <= chunk_chars:
        return [text.strip()]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_chars
        chunk = text[start:end].strip()
        if chunk:  # don't add empty chunks (can happen at the very end)
            chunks.append(chunk)
        start += step

    return chunks
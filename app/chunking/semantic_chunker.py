"""
Semantic chunker (V3): split text where the *meaning* shifts, not at a fixed
character count.

How it works
------------
1. Split the text into sentences.
2. Embed every sentence (reusing the same embedding model as the rest of the
   pipeline, via embed_batch).
3. Walk down the sentences and measure the cosine DISTANCE between each
   sentence and the next. Similar neighbours (same topic) have small distance;
   a topic change shows up as a spike in distance.
4. Call the biggest spikes "breakpoints" and split there. We define "biggest"
   relative to THIS document using a percentile of the distances, so we adapt
   to each document instead of relying on a magic fixed threshold that only
   works for one embedding model.
5. Enforce a max chunk size so a long single-topic stretch can't produce one
   giant chunk (which would hurt retrieval).

Design note: the embedding step (impure, calls Ollama) is kept separate from
the grouping step (pure, testable). Graceful fallbacks: very short text, or too
few sentences to find meaningful structure, fall back to simple size-based
chunking so ingestion never breaks.

Interface matches simple_chunker.chunk_text: text in, list[str] of chunks out.
"""
import logging
import math
import re

from app.config import settings
from app.embeddings.embedding_client import embed_batch
from app.chunking.simple_chunker import chunk_text as _size_chunk


logger = logging.getLogger(__name__)

# 1 token ~= 4 chars in English (same rule the simple chunker uses).
CHARS_PER_TOKEN = 4

# A "breakpoint" is a distance in the top (100 - PERCENTILE)% of gaps in this
# document. 90 means: split at roughly the largest ~10% of topic jumps.
# Higher -> fewer, larger chunks. Lower -> more, smaller chunks.
# (Kept here as a constant for now; easy to promote to .env later.)
BREAKPOINT_PERCENTILE = 90

# Below this many sentences there isn't enough signal to find topic boundaries
# reliably, so we just size-chunk instead.
MIN_SENTENCES_FOR_SEMANTIC = 4

# Above this many sentences on a single page, embedding every sentence gets
# expensive (one embedding call each). Fall back to size-based chunking so a
# monster page can't trigger hundreds of embedding requests.
MAX_SENTENCES_FOR_SEMANTIC = 400


def _split_sentences(text: str) -> list[str]:
    """
    Naive sentence splitter: break after ., !, or ? followed by whitespace.

    Good enough for chunking. It will occasionally mis-split on abbreviations
    ("Dr. Smith"), but a slightly-off boundary only nudges a chunk edge; it
    doesn't corrupt anything. A real NLP sentence tokenizer (nltk/spacy) would
    be the upgrade, at the cost of a dependency.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors (pure Python, no numpy needed)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _percentile(values: list[float], p: float) -> float:
    """
    The p-th percentile of `values` using linear interpolation (like numpy's
    default). p is 0-100. Used to pick an adaptive breakpoint threshold.
    """
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def _breakpoint_indices(embeddings: list[list[float]], percentile: float) -> set[int]:
    """
    Given per-sentence embeddings, return the set of indices i meaning
    "start a new chunk BEFORE sentence i+1" (i.e. split between i and i+1).

    distance[i] = 1 - cosine(sentence_i, sentence_i+1). Large distance = topic
    shift. We split wherever the distance is at/above the chosen percentile.
    """
    if len(embeddings) < 2:
        return set()
    distances = [
        1.0 - _cosine(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]
    threshold = _percentile(distances, percentile)
    # >= so that when many distances tie at the threshold we still split.
    return {i for i, d in enumerate(distances) if d >= threshold}


def _assemble_chunks(
    sentences: list[str],
    breakpoints: set[int],
    max_chars: int,
) -> list[str]:
    """
    Group sentences into chunks, cutting after any index in `breakpoints`, and
    also cutting when the running chunk would exceed max_chars (hard size cap).

    Pure function: no embeddings, fully testable with hand-made inputs.
    """
    chunks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if not current:
            return
        text = " ".join(current).strip()
        if len(text) <= max_chars:
            chunks.append(text)
        else:
            # A single-topic run longer than the cap: fall back to size-based
            # splitting for just this piece so we never emit an oversized chunk.
            chunks.extend(_size_chunk(text))
        current.clear()

    for i, sentence in enumerate(sentences):
        # Would adding this sentence blow the size cap? Flush first.
        projected = len(" ".join(current + [sentence]))
        if current and projected > max_chars:
            flush()
        current.append(sentence)
        # Semantic boundary right after this sentence?
        if i in breakpoints:
            flush()

    flush()
    return chunks


def semantic_chunk_text(text: str) -> list[str]:
    """
    Split `text` into semantically-coherent chunks.

    Falls back to simple size-based chunking when the input is too short or has
    too few sentences to find meaningful topic boundaries.
    """
    if not text or not text.strip():
        return []

    sentences = _split_sentences(text)

    # Not enough sentences to reason about topic structure -> size-chunk.
    if len(sentences) < MIN_SENTENCES_FOR_SEMANTIC:
        return _size_chunk(text)

    # Too many sentences -> embedding each one is too costly; size-chunk instead.
    if len(sentences) > MAX_SENTENCES_FOR_SEMANTIC:
        logger.info("Page has %d sentences (> %d cap); using size-based chunking",
                    len(sentences), MAX_SENTENCES_FOR_SEMANTIC)
        return _size_chunk(text)

    max_chars = settings.CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN

    try:
        embeddings = embed_batch(sentences)
    except Exception as exc:  # embedding backend down -> don't break ingestion
        logger.warning("Semantic chunking failed to embed (%s); "
                       "falling back to size-based chunking", exc)
        return _size_chunk(text)

    breakpoints = _breakpoint_indices(embeddings, BREAKPOINT_PERCENTILE)
    chunks = _assemble_chunks(sentences, breakpoints, max_chars)

    logger.debug("Semantic chunking: %d sentences -> %d chunks (%d breakpoints)",
                 len(sentences), len(chunks), len(breakpoints))
    return chunks

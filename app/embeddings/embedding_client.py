"""
Embedding client - wraps the Ollama HTTP API.
One job: given text, return a list of floats (the embedding vector).
Configuration comes from settings (OLLAMA_BASE_URL, EMBEDDING_MODEL, EMBEDDING_DIM).
The dimension pin from Decision 2 is enforced here: every returned vector is checked
to be exactly EMBEDDING_DIM floats long. If Ollama ever returns the wrong size, we
fail fast instead of silently corrupting the Qdrant collection.

Throughput note:
  Ollama's /api/embeddings takes ONE text per call. Embedding many texts one after
  another (the old serial loop) meant N blocking round-trips - painfully slow for
  semantic chunking, which embeds every sentence. embed_batch now sends those
  requests CONCURRENTLY over a shared connection pool, so the network waits overlap
  and a few hundred embeddings take seconds instead of minutes. Same endpoint, same
  model - just no longer waiting for each call to finish before starting the next.
"""
from concurrent.futures import ThreadPoolExecutor

import httpx

from app.config import settings


# How many embedding requests to have in flight at once. Ollama serves them off a
# single model, but overlapping the request/response waits is where the speedup
# comes from. Modest so we don't overwhelm a CPU-only Ollama.
_MAX_CONCURRENCY = 8


def _embed_one(client: httpx.Client, text: str) -> list[float]:
    """Embed a single text using an existing (shared) httpx client."""
    url = f"{settings.OLLAMA_BASE_URL}/api/embeddings"
    payload = {"model": settings.EMBEDDING_MODEL, "prompt": text}
    response = client.post(url, json=payload)
    response.raise_for_status()  # raises on 4xx / 5xx
    data = response.json()

    vector = data.get("embedding")
    if vector is None:
        raise RuntimeError(f"Ollama response missing 'embedding' field: {data}")
    # Enforce the dimension pin - the whole system depends on this being 768.
    if len(vector) != settings.EMBEDDING_DIM:
        raise RuntimeError(
            f"Embedding dimension mismatch: got {len(vector)}, "
            f"expected {settings.EMBEDDING_DIM}. "
            f"Model={settings.EMBEDDING_MODEL} may not match EMBEDDING_DIM in .env."
        )
    return vector


def embed_text(text: str) -> list[float]:
    """
    Get a single embedding for a piece of text.

    Returns:
        A list of floats of length settings.EMBEDDING_DIM (768 for nomic-embed-text).
    Raises:
        RuntimeError: if Ollama returns a vector with the wrong dimension.
        httpx.HTTPError: if the Ollama call itself fails (network, 500, etc.).
    """
    # timeout=60 because on CPU-only, the first call to a fresh model can be slow
    # (Ollama loads the model on first use). Subsequent calls are fast.
    with httpx.Client(timeout=60.0) as client:
        return _embed_one(client, text)


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts CONCURRENTLY and return vectors in the SAME order.

    Ollama's /api/embeddings still takes one text per call, but we fire several
    calls at once over a shared connection pool instead of strictly one-after-
    another. Results are reassembled in input order, so callers see no
    difference except speed.
    """
    if not texts:
        return []

    # A shared client = one connection pool reused across all requests (no
    # per-call setup/teardown). Slightly higher timeout for batches under load.
    with httpx.Client(timeout=120.0) as client:
        workers = min(_MAX_CONCURRENCY, len(texts))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # executor.map preserves input order in its results.
            return list(pool.map(lambda t: _embed_one(client, t), texts))

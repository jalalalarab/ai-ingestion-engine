"""
Embedding client — wraps the Ollama HTTP API.

One job: given text, return a list of floats (the embedding vector).

Configuration comes from settings (OLLAMA_BASE_URL, EMBEDDING_MODEL, EMBEDDING_DIM).
The dimension pin from Decision 2 is enforced here: every returned vector is checked
to be exactly EMBEDDING_DIM floats long. If Ollama ever returns the wrong size, we
fail fast instead of silently corrupting the Qdrant collection.
"""
import httpx
from app.config import settings


def embed_text(text: str) -> list[float]:
    """
    Get a single embedding for a piece of text.

    Returns:
        A list of floats of length settings.EMBEDDING_DIM (768 for nomic-embed-text).

    Raises:
        RuntimeError: if Ollama returns a vector with the wrong dimension.
        httpx.HTTPError: if the Ollama call itself fails (network, 500, etc.).
    """
    url = f"{settings.OLLAMA_BASE_URL}/api/embeddings"
    payload = {
        "model": settings.EMBEDDING_MODEL,
        "prompt": text,
    }

    # timeout=60 because on CPU-only, the first call to a fresh model can be slow
    # (Ollama loads the model on first use). Subsequent calls are fast.
    with httpx.Client(timeout=60.0) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()  # raises on 4xx / 5xx
        data = response.json()

    vector = data.get("embedding")
    if vector is None:
        raise RuntimeError(f"Ollama response missing 'embedding' field: {data}")

    # Enforce the dimension pin — the whole system depends on this being 768.
    if len(vector) != settings.EMBEDDING_DIM:
        raise RuntimeError(
            f"Embedding dimension mismatch: got {len(vector)}, "
            f"expected {settings.EMBEDDING_DIM}. "
            f"Model={settings.EMBEDDING_MODEL} may not match EMBEDDING_DIM in .env."
        )

    return vector


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts one at a time.

    Ollama's /api/embeddings doesn't accept batches — one call per text.
    Kept as a simple loop for now. If throughput matters later, we swap this
    for an async version or a bulk backend without touching the caller.
    """
    return [embed_text(t) for t in texts]

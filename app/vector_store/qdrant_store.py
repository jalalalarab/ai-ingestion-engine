"""
Qdrant vector store — the only module that talks to Qdrant.

Provides:
- ensure_collection(): verifies the collection exists with the pinned dimension.
- upsert_chunks(): stores a batch of (id, vector, payload) points.
- count_points(): diagnostic, returns total points in the collection.
- delete_by_file_id(): remove all chunks belonging to a specific file.
"""
from uuid import uuid5, NAMESPACE_URL
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from app.config import settings


# Single shared client — lazy-initialized on first use.
_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Return a shared Qdrant client, creating it on first call."""
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.QDRANT_URL)
    return _client


def ensure_collection() -> None:
    """
    Verify the collection exists with the expected dimension and distance.

    Fails loudly if the collection dim doesn't match settings.EMBEDDING_DIM.
    That's the Decision 2 pin enforced against the actual database — the last
    place a mismatch could sneak in.
    """
    client = get_client()
    name = settings.QDRANT_COLLECTION

    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        # Auto-create if missing (idempotent). Matches the manual PUT we did
        # in Phase 0 through the dashboard console.
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=settings.EMBEDDING_DIM, distance=Distance.COSINE),
        )
        return

    # Collection exists — verify its dimension matches.
    info = client.get_collection(name)
    actual_dim = info.config.params.vectors.size
    if actual_dim != settings.EMBEDDING_DIM:
        raise RuntimeError(
            f"Qdrant collection '{name}' has dimension {actual_dim}, "
            f"but EMBEDDING_DIM in .env is {settings.EMBEDDING_DIM}. "
            f"Either recreate the collection or fix .env."
        )


def make_point_id(file_id: str, chunk_index: int) -> str:
    """
    Deterministic UUID from (file_id, chunk_index).

    Re-ingesting the same file produces the same IDs, so Qdrant treats it as
    an update rather than duplicate inserts. Idempotent by design.
    """
    return str(uuid5(NAMESPACE_URL, f"{file_id}:{chunk_index}"))


def upsert_chunks(
    file_id: str,
    file_name: str,
    source_type: str,      # "pdf" or "video"
    chunks: list[str],
    vectors: list[list[float]],
    page_numbers: list[int | None],  # None for video chunks
) -> int:
    """
    Insert or update chunks in the collection.

    All list args must be the same length. Returns the number of points upserted.
    """
    if not (len(chunks) == len(vectors) == len(page_numbers)):
        raise ValueError(
            f"Argument lengths must match: "
            f"chunks={len(chunks)}, vectors={len(vectors)}, pages={len(page_numbers)}"
        )

    client = get_client()
    points = []
    for i, (chunk_text, vector, page_num) in enumerate(zip(chunks, vectors, page_numbers)):
        payload = {
            "file_id": file_id,
            "file_name": file_name,
            "source_type": source_type,
            "chunk_index": i,
            "page_number": page_num,
            "text": chunk_text,  # store the text so we can build LLM prompts later
        }
        points.append(
            PointStruct(
                id=make_point_id(file_id, i),
                vector=vector,
                payload=payload,
            )
        )

    client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)
    return len(points)


def count_points() -> int:
    """Return the total number of points in the collection."""
    client = get_client()
    info = client.get_collection(settings.QDRANT_COLLECTION)
    return info.points_count


def delete_by_file_id(file_id: str) -> None:
    """Remove all chunks belonging to a specific file."""
    client = get_client()
    client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="file_id", match=MatchValue(value=file_id))]
        ),
    )
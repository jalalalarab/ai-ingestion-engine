"""
Search service — embed a query and find matching chunks.
"""
from app.embeddings.embedding_client import embed_text
from app.vector_store.qdrant_store import search


def search_chunks(
    query: str,
    top_k: int = 5,
    file_id: str | None = None,
    source_type: str | None = None,
) -> list[dict]:
    """
    Embed the query and return the top_k matching chunks from Qdrant.

    Empty query returns empty results.
    """
    if not query or not query.strip():
        return []

    query_vector = embed_text(query.strip())
    return search(
        query_vector=query_vector,
        top_k=top_k,
        file_id=file_id,
        source_type=source_type,
    )
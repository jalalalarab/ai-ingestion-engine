"""
One-off maintenance script: delete ALL video chunks from Qdrant.

We ingested the slideshow twice (each ingest generates a fresh random file_id,
so the second run didn't overwrite the first — it duplicated it). This wipes
every point with source_type == "video" so we can re-ingest one clean copy.

Run from the project root:  python clear_video_chunks.py
"""
from qdrant_client.models import Filter, FieldCondition, MatchValue
from app.config import settings
from app.vector_store.qdrant_store import get_client, count_points

client = get_client()

before = count_points()
print(f"Total points before: {before}")

client.delete(
    collection_name=settings.QDRANT_COLLECTION,
    points_selector=Filter(
        must=[FieldCondition(key="source_type", match=MatchValue(value="video"))]
    ),
)

after = count_points()
print(f"Total points after:  {after}")
print(f"Deleted {before - after} video chunk(s).")

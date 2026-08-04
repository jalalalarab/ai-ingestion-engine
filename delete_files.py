"""
Delete specific files from Qdrant by file_id, leaving everything else intact.

Unlike clear_video_chunks.py (which nukes ALL videos), this removes only the
file_ids you list — used to clean study-notes / unrelated docs out of the
knowledge base so search returns relevant content.

Edit FILE_IDS_TO_DELETE below, then run from project root:
    python delete_files.py
"""
from qdrant_client.models import Filter, FieldCondition, MatchValue
from app.config import settings
from app.vector_store.qdrant_store import get_client, count_points

# The file_ids to remove. Everything NOT in this list stays.
FILE_IDS_TO_DELETE = [
    "ab10ecf2-94ff-50cb-b902-01f31dd2d44d",  # Day1_SCANNED.pdf (study notes)
    "4db2b3f3-0da6-4a29-917c-239b6045054e",  # Day1_Study_Guide.pdf (study notes)
    "5d186011-13e4-4d39-86c5-bc6167b31204",  # Day1_Study_Guide.pdf (duplicate)
    "dc494977-c357-55e2-844b-8e736b326174",  # The_Ultimate_Tracker_Sheet_GUIDE.pdf (unrelated project)
]

client = get_client()

before = count_points()
print(f"Total points before: {before}")

for fid in FILE_IDS_TO_DELETE:
    client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="file_id", match=MatchValue(value=fid))]
        ),
    )
    print(f"  deleted chunks for file_id {fid[:8]}")

after = count_points()
print(f"Total points after:  {after}")
print(f"Deleted {before - after} chunk(s). Kept everything else.")

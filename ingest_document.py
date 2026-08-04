"""
Ingest an already-generated synthesis document (the .md) straight into Qdrant,
skipping frame extraction and synthesis entirely.

Why this exists:
  Full synthesis ingestion re-runs vision on every frame (182 calls) AND the
  synthesis agent (33 calls) — 200+ gpt-4o-mini calls that blow past the
  tokens-per-minute limit and 429-storm. But we already generated the document
  (noria_document.md). This script chunks/embeds THAT existing document, so we
  can validate that document-based retrieval + graph work — with ZERO OpenAI
  load for frames/synthesis.

  This is a TEST/UTILITY path. The real pipeline (ingest_video_synthesis) still
  regenerates the document from the video; this just lets us test the back half
  without paying the token cost every time.

Run from project root (venv active):
    python ingest_document.py noria_document.md sales_noria_erp.mp4
"""
import sys
from pathlib import Path

from app.services.ingestion_service import _chunk, _ingest_texts, _file_id_from_bytes
from app.vector_store.qdrant_store import ensure_collection


def ingest_document(doc_path: str, source_name: str) -> None:
    ensure_collection()

    text = Path(doc_path).read_text(encoding="utf-8")
    # Derive a stable file_id from the source video NAME so it groups as that video.
    # (Uses the document bytes so re-running the same doc overwrites in place.)
    file_id = _file_id_from_bytes(source_name.encode("utf-8"))

    chunks = _chunk(text)
    print(f"Document: {doc_path}")
    print(f"  {len(text)} chars -> {len(chunks)} chunks")

    if not chunks:
        print("  Nothing to ingest (empty document).")
        return

    n = _ingest_texts(
        file_id=file_id,
        file_name=source_name,
        source_type="video",
        chunks=chunks,
        page_numbers=[None] * len(chunks),
        timestamps=[None] * len(chunks),
        frame_numbers=[None] * len(chunks),
    )
    print(f"  Stored {n} document chunks [file_id={file_id[:8]}]")
    print("Done. The synthesized document is now indexed — try /search or /graphrag/ask.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ingest_document.py <document.md> <source_video_name>")
        print("Example: python ingest_document.py noria_document.md sales_noria_erp.mp4")
        sys.exit(1)
    ingest_document(sys.argv[1], sys.argv[2])

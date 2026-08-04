from app.vector_store.qdrant_store import get_chunks_by_file_id
from app.synthesis.document_assembler import assemble_document

FILE_ID = "b57db252-ecf7-5f3f-a550-37c8d381e98a"

chunks = get_chunks_by_file_id(FILE_ID, source_type="video")

frames = [
    (c["timestamp_seconds"], c.get("frame_number", 0), c["text"])
    for c in chunks
    if c.get("frame_number") is not None
]
transcript = [
    (c["timestamp_seconds"], c["text"])
    for c in chunks
    if c.get("timestamp_seconds") is not None and c.get("frame_number") is None
]

print(f"{len(frames)} frames, {len(transcript)} transcript segments")

result = assemble_document("sales_noria_erp.mp4", frames, transcript)

with open("noria_document.md", "w", encoding="utf-8") as f:
    f.write(result["document"])

print(f"{result['windows_written']}/{result['windows_total']} windows written, "
      f"{result['char_count']} chars -> saved to noria_document.md")
"""
Documents route — GET /documents.

Lists the files that have been ingested into the vector store, each with its
file_id, file_name, source_type, and chunk count. This lets a human (or the
n8n agent) look a file up by name and get its file_id — which is what the
minutes tool needs, instead of anyone having to know the raw UUID.

Thin route: it delegates to qdrant_store.list_documents().
"""
from pydantic import BaseModel
from fastapi import APIRouter

from app.vector_store.qdrant_store import list_documents

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentInfo(BaseModel):
    file_id: str
    file_name: str | None
    source_type: str | None
    chunk_count: int


class DocumentsResponse(BaseModel):
    count: int
    documents: list[DocumentInfo]


@router.get("", response_model=DocumentsResponse)
async def list_documents_endpoint(name: str | None = None) -> DocumentsResponse:
    """
    List ingested files (one entry per file, not per chunk).

    Optional query param:
        name: case-insensitive partial match on file_name. E.g. ?name=noria
              returns "sales_noria_erp.mp4". Omit it to list every file.

    Returns each file's id, name, type, and how many chunks it produced —
    enough for an agent to resolve a spoken name like "the Noria video" to
    the file_id it needs for /minutes/{file_id}.
    """
    docs = list_documents()

    # Optional case-insensitive substring filter on the file name.
    if name:
        needle = name.lower()
        docs = [
            d for d in docs
            if d.get("file_name") and needle in d["file_name"].lower()
        ]

    return DocumentsResponse(
        count=len(docs),
        documents=[DocumentInfo(**d) for d in docs],
    )

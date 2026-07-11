"""
Search API route — POST /search.
"""
from pydantic import BaseModel, Field
from fastapi import APIRouter
from app.services.search_service import search_chunks

router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural-language search query.")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to return.")
    file_id: str | None = Field(None, description="Restrict to a specific file_id.")
    source_type: str | None = Field(None, description="Restrict to 'pdf' or 'video'.")


class SearchHit(BaseModel):
    score: float
    text: str
    file_id: str | None
    file_name: str | None
    source_type: str | None
    page_number: int | None          # set for PDF chunks, None for video
    timestamp_seconds: int | None = None   # set for video chunks, None for PDF
    frame_number: int | None = None        # set for video chunks, None for PDF
    chunk_index: int | None


class SearchResponse(BaseModel):
    query: str
    count: int
    results: list[SearchHit]


@router.post("", response_model=SearchResponse)
async def search_endpoint(payload: SearchRequest) -> SearchResponse:
    """
    Run a semantic search over ingested chunks.

    Returns the top_k most similar chunks by cosine similarity, ordered best first.
    PDF hits carry page_number; video hits carry timestamp_seconds + frame_number.
    """
    hits = search_chunks(
        query=payload.query,
        top_k=payload.top_k,
        file_id=payload.file_id,
        source_type=payload.source_type,
    )
    return SearchResponse(
        query=payload.query,
        count=len(hits),
        results=[SearchHit(**h) for h in hits],
    )

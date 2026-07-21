"""
Entity-extraction route — POST /extract/{file_id}.

Given the file_id of an ingested video, extract the entities and relationships
(triples) from its transcript. Thin route: delegates to the extraction service.

This is the Phase B endpoint on the road to a knowledge graph. Its output — a
list of (subject, predicate, object) triples — is what later phases load into a
graph database.
"""
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, status

from app.services.extraction_service import extract_from_file

router = APIRouter(prefix="/extract", tags=["extract"])


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str


class ExtractResponse(BaseModel):
    file_id: str
    file_name: str | None
    triples: list[Triple]
    batches_used: int
    raw_count: int       # total triples before dedup (across batches)
    deduped_count: int   # unique triples after merge


@router.post("/{file_id}", response_model=ExtractResponse)
async def extract_endpoint(file_id: str) -> ExtractResponse:
    """
    Extract (subject, predicate, object) triples from a video's transcript.

    The video must have been ingested with transcription enabled. Batches the
    transcript, extracts per batch with gpt-4o-mini, then merges and dedupes.
    """
    try:
        result = extract_from_file(file_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    return ExtractResponse(**result)

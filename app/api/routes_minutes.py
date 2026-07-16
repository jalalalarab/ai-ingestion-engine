"""
Minutes-of-Meeting route — POST /minutes/{file_id}.

Given the file_id of an ingested video (with a transcript), generate structured
Minutes of Meeting. Thin route: it delegates to the minutes service, which
handles pulling the transcript and the map-reduce summarization.
"""
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, status

from app.services.minutes_service import generate_minutes

router = APIRouter(prefix="/minutes", tags=["minutes"])


class MinutesResponse(BaseModel):
    file_id: str
    file_name: str | None
    minutes: str
    batches_used: int
    method: str  # "single-pass" or "map-reduce"


@router.post("/{file_id}", response_model=MinutesResponse)
async def minutes_endpoint(file_id: str) -> MinutesResponse:
    """
    Generate Minutes of Meeting from an ingested video's transcript.

    The video must have been ingested with transcription enabled (so its spoken
    content is stored as transcript chunks). Uses map-reduce when the transcript
    is too long to summarize in a single LLM call.
    """
    try:
        result = generate_minutes(file_id)
    except RuntimeError as exc:
        # No transcript for this file_id — 404 is the right signal.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    return MinutesResponse(**result)

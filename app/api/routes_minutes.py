"""
Minutes-of-Meeting routes.

  POST /minutes/{file_id}   — generate minutes for a known file_id.
  POST /minutes/by-name     — resolve a video by (partial) name, then generate
                              its minutes in one call. Built so the n8n agent
                              needs only ONE tool (name in, minutes out) instead
                              of chaining a lookup and a minutes call — chaining
                              is where a small model fumbles, so we do the
                              resolution in deterministic code instead.

Thin routes: they delegate to the minutes service and the vector store.
"""
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, status

from app.services.minutes_service import generate_minutes
from app.vector_store.qdrant_store import list_documents

router = APIRouter(prefix="/minutes", tags=["minutes"])


class MinutesResponse(BaseModel):
    file_id: str
    file_name: str | None
    minutes: str
    batches_used: int
    method: str  # "single-pass" or "map-reduce"


# NOTE: /by-name is declared BEFORE /{file_id}. FastAPI matches routes in order,
# and a bare /{file_id} would happily swallow the literal path "by-name" as if it
# were an id. Declaring the specific path first avoids that collision.
@router.post("/by-name", response_model=MinutesResponse)
async def minutes_by_name_endpoint(name: str) -> MinutesResponse:
    """
    Generate Minutes of Meeting for a VIDEO identified by a partial name.

    Resolves the name to a single ingested video, then generates its minutes.
    This is the endpoint the agent uses: one call, name in, minutes out.

    Query param:
        name: case-insensitive partial match on file_name (e.g. "noria").

    Errors are deliberately clear so the agent can relay them to the user:
        404 — no file matches the name.
        400 — the match is a PDF (minutes are video-only), or the name is
              ambiguous (matches more than one video).
    """
    docs = list_documents()

    needle = (name or "").strip().lower()
    if not needle:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide a name to search for.",
        )

    matches = [
        d for d in docs
        if d.get("file_name") and needle in d["file_name"].lower()
    ]

    if not matches:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No ingested file matches the name '{name}'.",
        )

    videos = [d for d in matches if d.get("source_type") == "video"]

    if not videos:
        # Matched something, but it's not a video — minutes only apply to videos.
        names = ", ".join(sorted({d["file_name"] for d in matches}))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"'{name}' matches {names}, which is not a video. "
                f"Minutes of Meeting are only generated for videos."
            ),
        )

    # Collapse duplicate names (same file re-ingested) to distinct file_ids.
    distinct_ids = {d["file_id"] for d in videos}
    if len(distinct_ids) > 1:
        listing = "; ".join(f"{d['file_name']} (id {d['file_id'][:8]}…)" for d in videos)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"'{name}' matches more than one video: {listing}. "
                f"Please be more specific."
            ),
        )

    file_id = videos[0]["file_id"]

    try:
        result = generate_minutes(file_id)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    return MinutesResponse(**result)


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

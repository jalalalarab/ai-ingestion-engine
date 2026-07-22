"""
Community-summaries route — POST /graph/communities.

Detects the communities in the knowledge graph (GDS Louvain) and returns a
natural-language summary for each — the "long documents" that describe the
graph's themes. Thin route: delegates to the community service.
"""
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, status

from app.services.community_service import build_community_summaries

router = APIRouter(prefix="/graph", tags=["graph"])


class CommunitySummary(BaseModel):
    community: int
    entities: list[str]
    size: int
    summary: str


class CommunitiesResponse(BaseModel):
    community_count: int
    summaries: list[CommunitySummary]


@router.post("/communities", response_model=CommunitiesResponse)
async def communities_endpoint(min_size: int = 2) -> CommunitiesResponse:
    """
    Detect graph communities (Louvain) and summarize each into a long document.

    Query param:
        min_size: skip communities smaller than this (default 2).
    """
    try:
        result = build_community_summaries(min_size=min_size)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return CommunitiesResponse(**result)

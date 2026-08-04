"""
Entity-resolution route — POST /graph/resolve-entities.

Proposes (and optionally applies) merges of duplicate entity nodes.

Defaults to a DRY RUN: it returns the proposed merges without changing anything,
so you can review them first. Pass apply=true only after checking the list — a
wrong merge cannot be undone.
"""
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, status

from app.services.entity_resolution_service import resolve_entities

router = APIRouter(prefix="/graph", tags=["graph"])


class Merge(BaseModel):
    from_: str
    into: str
    reason: str

    class Config:
        fields = {"from_": "from"}


class GraphTotals(BaseModel):
    nodes: int
    relationships: int


class ResolveResponse(BaseModel):
    applied: bool
    entity_count_before: int
    merges: list[dict]
    graph_totals: GraphTotals


@router.post("/resolve-entities", response_model=ResolveResponse)
async def resolve_entities_endpoint(apply: bool = False) -> ResolveResponse:
    """
    Find duplicate entities in the graph.

    Query param:
        apply: false (default) = dry run, just propose. true = perform the merges.
    """
    try:
        result = resolve_entities(apply=apply)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return ResolveResponse(**result)

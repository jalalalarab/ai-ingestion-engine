"""
Graph routes — build and inspect the knowledge graph.

  POST /graph/build/{file_id}  — extract a file's triples and load them into Neo4j.
  GET  /graph/stats            — node and relationship counts for the whole graph.

Thin routes: they delegate to the graph service and store. This is the Phase C
endpoint — it turns the Phase B triples into an actual graph you can query in the
Neo4j browser.
"""
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, status

from app.services.graph_service import build_graph_for_file
from app.vector_store import graph_store

router = APIRouter(prefix="/graph", tags=["graph"])


class GraphTotals(BaseModel):
    nodes: int
    relationships: int


class ExtractionSummary(BaseModel):
    batches_used: int
    raw_count: int
    deduped_count: int


class GraphBuildResponse(BaseModel):
    file_id: str
    file_name: str | None
    triples_loaded: int
    graph_totals: GraphTotals
    extraction: ExtractionSummary


@router.post("/build/{file_id}", response_model=GraphBuildResponse)
async def build_graph_endpoint(file_id: str) -> GraphBuildResponse:
    """
    Build (or rebuild) the graph for one file: extract triples, load into Neo4j.

    Re-running is safe — the file's prior edges are cleared first, so you get a
    clean rebuild rather than stacked duplicates.
    """
    try:
        result = build_graph_for_file(file_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    return GraphBuildResponse(**result)


@router.get("/stats", response_model=GraphTotals)
async def graph_stats_endpoint() -> GraphTotals:
    """Return node and relationship counts for the whole graph."""
    return GraphTotals(**graph_store.count_graph())

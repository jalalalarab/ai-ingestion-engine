"""
Agent search route — POST /agent/search.

A thin, agent-friendly wrapper over the same `search_chunks` service that powers
/search. It exists specifically for the n8n AI Agent: instead of raw JSON tuned
for a program, it returns each retrieved chunk as clean, labeled text with a
citation (file name + page or timestamp + score), plus a single pre-formatted
`context` block the agent can drop straight into its reasoning.

Why a separate endpoint instead of reusing /search:
  - The logic is NOT duplicated — this calls the same search_chunks() service.
  - An LLM agent reads labeled text better than raw JSON, and having the sources
    pre-formatted makes it easy for the agent to cite them in its answer.
  - Keeping it separate leaves the existing /search contract untouched for any
    other caller.
"""
from pydantic import BaseModel, Field
from fastapi import APIRouter

from app.services.search_service import search_chunks

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentSearchRequest(BaseModel):
    query: str = Field(..., description="The question or search query from the agent.")
    top_k: int = Field(5, ge=1, le=20, description="How many chunks to retrieve.")
    file_id: str | None = Field(None, description="Optional: restrict to one file.")
    source_type: str | None = Field(None, description="Optional: 'pdf' or 'video'.")


class AgentSource(BaseModel):
    text: str
    citation: str          # e.g. "report.pdf — p.5" or "clip.mp4 — 04:32"
    file_id: str | None
    file_name: str | None
    source_type: str | None
    page_number: int | None
    timestamp_seconds: int | None
    frame_number: int | None
    score: float


class AgentSearchResponse(BaseModel):
    query: str
    count: int
    # A single ready-to-use context string, each chunk labeled with its citation.
    # The agent can paste this straight into its answer reasoning.
    context: str
    # The structured per-chunk data, so the execution log shows text + metadata.
    sources: list[AgentSource]


def _citation_for(hit: dict) -> str:
    """Build a human-readable citation: page for PDFs, MM:SS for videos."""
    name = hit.get("file_name") or "unknown"
    if hit.get("page_number") is not None:
        return f"{name} — p.{hit['page_number']}"
    ts = hit.get("timestamp_seconds")
    if ts is not None:
        m, s = divmod(int(ts), 60)
        h, m = divmod(m, 60)
        stamp = f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        return f"{name} — {stamp}"
    return name


@router.post("/search", response_model=AgentSearchResponse)
async def agent_search_endpoint(payload: AgentSearchRequest) -> AgentSearchResponse:
    """
    Retrieve chunks for an AI agent and return them as cited, agent-friendly text.

    Reuses the same semantic search as /search, then formats the results so an
    LLM agent can both READ the chunk text and CITE where each came from. The
    `context` field is a single labeled block; `sources` is the structured data
    (visible in the n8n execution log for full transparency).
    """
    hits = search_chunks(
        query=payload.query,
        top_k=payload.top_k,
        file_id=payload.file_id,
        source_type=payload.source_type,
    )

    sources: list[AgentSource] = []
    context_blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        citation = _citation_for(hit)
        # One labeled block per chunk, so the agent sees text + where it's from.
        context_blocks.append(f"[Source {i} — {citation}]\n{hit.get('text', '')}")
        sources.append(
            AgentSource(
                text=hit.get("text", ""),
                citation=citation,
                file_id=hit.get("file_id"),
                file_name=hit.get("file_name"),
                source_type=hit.get("source_type"),
                page_number=hit.get("page_number"),
                timestamp_seconds=hit.get("timestamp_seconds"),
                frame_number=hit.get("frame_number"),
                score=hit["score"],
            )
        )

    context = "\n\n".join(context_blocks) if context_blocks else "No relevant content found."

    return AgentSearchResponse(
        query=payload.query,
        count=len(sources),
        context=context,
        sources=sources,
    )

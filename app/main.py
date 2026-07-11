"""
AI Ingestion Engine — FastAPI application entry point.
"""
from fastapi import FastAPI

from app.api.routes_ingest import router as ingest_router
from app.api.routes_search import router as search_router


app = FastAPI(
    title="AI Ingestion Engine",
    description="Multimodal RAG pipeline for PDFs and videos.",
    version="0.2.0",
)


@app.get("/health", tags=["health"])
def health():
    """Liveness check."""
    return {"status": "ok"}


app.include_router(ingest_router)
app.include_router(search_router)
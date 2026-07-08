"""
AI Ingestion Engine — FastAPI application entry point.

Phase 0: only a /health route so we can verify the app runs and is reachable.
More routes get added in later phases (ingest, search, ask).
"""
from fastapi import FastAPI

app = FastAPI(
    title="AI Ingestion Engine",
    description="Multimodal RAG pipeline for PDFs and videos.",
    version="0.1.0",
)


@app.get("/health")
def health():
    """Liveness check. Returns ok if the app is running."""
    return {"status": "ok"}
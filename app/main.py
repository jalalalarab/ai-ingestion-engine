"""
AI Ingestion Engine — FastAPI application entry point.

Registers all API routers. Health check lives here; other endpoints live in api/.
"""
from fastapi import FastAPI

from app.api.routes_ingest import router as ingest_router


app = FastAPI(
    title="AI Ingestion Engine",
    description="Multimodal RAG pipeline for PDFs and videos.",
    version="0.1.0",
)


@app.get("/health", tags=["health"])
def health():
    """Liveness check. Returns ok if the app is running."""
    return {"status": "ok"}


# Register routers
app.include_router(ingest_router)
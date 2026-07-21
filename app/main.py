"""
AI Ingestion Engine - FastAPI application entry point.
"""
import logging

from fastapi import FastAPI

from app.logging_config import setup_logging
from app.api.routes_ingest import router as ingest_router
from app.api.routes_search import router as search_router
from app.api.routes_ask import router as ask_router
from app.api.routes_agent import router as agent_router
from app.api.routes_minutes import router as minutes_router
from app.api.routes_documents import router as documents_router

# Configure logging FIRST, before any router code runs or logs. Anything
# imported above/below can then use logging.getLogger(__name__) and it just
# works with our format.
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Ingestion Engine",
    description="Multimodal RAG pipeline for PDFs and videos.",
    version="0.3.0",
)

logger.info("AI Ingestion Engine starting (version %s)", app.version)


@app.get("/health", tags=["health"])
def health():
    """Liveness check."""
    return {"status": "ok"}


app.include_router(ingest_router)
app.include_router(search_router)
app.include_router(ask_router)
app.include_router(agent_router)
app.include_router(minutes_router)
app.include_router(documents_router)

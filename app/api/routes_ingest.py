"""
Ingestion API routes.

For now: POST /ingest/pdf.
Video ingestion arrives in Phase 5 as POST /ingest/video (async job).
"""
from dataclasses import asdict
from fastapi import APIRouter, UploadFile, File, HTTPException, status

from app.config import settings
from app.services.ingestion_service import ingest_pdf


router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/pdf", status_code=status.HTTP_201_CREATED)
async def ingest_pdf_endpoint(file: UploadFile = File(...)):
    """
    Upload a PDF and ingest it into the vector store.

    Returns a report with the generated file_id, pages processed, and chunks created.
    """
    # Basic content-type check (browsers set this; not bulletproof but a decent guard).
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Expected a PDF file, got content_type={file.content_type!r}",
        )

    # Filename sanity check.
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have a .pdf extension.",
        )

    # Read all bytes into memory. Fine for MVP-sized PDFs (Decision 7 caps at 50 MB).
    pdf_bytes = await file.read()

    # Enforce the size cap.
    max_bytes = settings.MAX_PDF_MB * 1024 * 1024
    if len(pdf_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"PDF is {len(pdf_bytes) // (1024*1024)} MB; limit is {settings.MAX_PDF_MB} MB.",
        )

    if len(pdf_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Run the pipeline. Errors propagate to FastAPI's default 500 handler for now.
    report = ingest_pdf(pdf_bytes, file_name=file.filename)
    return asdict(report)
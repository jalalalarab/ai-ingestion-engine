"""
Ingestion API routes.

  POST /ingest/pdf    — ingest a PDF (text layer + OCR fallback for scans).
  POST /ingest/video  — ingest a video (sampled frames -> OCR -> chunks).
"""
from pathlib import Path
from dataclasses import asdict

from fastapi import APIRouter, UploadFile, File, HTTPException, status

from app.config import settings
from app.services.ingestion_service import ingest_pdf, ingest_video

router = APIRouter(prefix="/ingest", tags=["ingest"])

# Video files are bigger than PDFs. Hardcoded cap for now; move to config
# alongside MAX_PDF_MB when convenient (keeps the "all config in one place" rule).
MAX_VIDEO_MB = 200

# Video containers we accept. OpenCV (via its bundled FFmpeg) reads all of these.
ALLOWED_VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm")


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


@router.post("/video", status_code=status.HTTP_201_CREATED)
async def ingest_video_endpoint(file: UploadFile = File(...)):
    """
    Upload a video and ingest it into the vector store.

    Frames are sampled every couple of seconds, OCR'd, de-duplicated, and stored
    as chunks with timestamp + frame-number metadata (instead of page numbers).
    Returns a report with the generated file_id, frames ingested, and chunks created.
    """
    # Extension check.
    if not file.filename or not file.filename.lower().endswith(ALLOWED_VIDEO_EXT):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File must be a video ({', '.join(ALLOWED_VIDEO_EXT)}).",
        )

    video_bytes = await file.read()

    # Size cap.
    max_bytes = MAX_VIDEO_MB * 1024 * 1024
    if len(video_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Video is {len(video_bytes) // (1024*1024)} MB; limit is {MAX_VIDEO_MB} MB.",
        )

    if len(video_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # OpenCV's VideoCapture reads from a file path, not raw bytes — so persist
    # the upload to storage/uploads first, then ingest from that path.
    uploads_dir = Path("storage/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = uploads_dir / file.filename
    dest.write_bytes(video_bytes)

    report = ingest_video(str(dest), file_name=file.filename)
    return asdict(report)

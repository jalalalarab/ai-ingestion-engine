"""
Central configuration.
Reads .env at import time and exposes typed constants everything else imports.
Rule: no other module reads env vars directly - all config comes through here.
"""
from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env from the project root (parent of app/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _require(name: str) -> str:
    """Fail loudly if a required env var is missing - better than silent bugs later."""
    value = os.getenv(name)
    if value is None or value == "":
        raise RuntimeError(f"Missing required env var: {name}")
    return value


class Settings:
    APP_ENV: str = os.getenv("APP_ENV", "local")

    # Qdrant
    QDRANT_URL: str = _require("QDRANT_URL")
    QDRANT_COLLECTION: str = _require("QDRANT_COLLECTION")

    # Embeddings
    OLLAMA_BASE_URL: str = _require("OLLAMA_BASE_URL")
    EMBEDDING_MODEL: str = _require("EMBEDDING_MODEL")
    EMBEDDING_DIM: int = int(_require("EMBEDDING_DIM"))

    # Chunking
    CHUNK_SIZE_TOKENS: int = int(os.getenv("CHUNK_SIZE_TOKENS", "700"))
    CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "100"))
    # Which chunker to use: "semantic" (embedding-similarity splits) or
    # "simple" (fixed-size sliding window). Semantic gives better retrieval;
    # simple is the fast, dependency-free fallback.
    CHUNKING_STRATEGY: str = os.getenv("CHUNKING_STRATEGY", "semantic").lower()

    # Upload limits
    MAX_PDF_MB: int = int(os.getenv("MAX_PDF_MB", "50"))

    # Video ingestion — sample one frame every N seconds. 5s suits meeting/slide
    # videos (slides change slowly); lower it for fast-changing content.
    VIDEO_SAMPLE_SECONDS: int = int(os.getenv("VIDEO_SAMPLE_SECONDS", "5"))

    # Transcription (OpenAI Whisper) — turn a video's spoken audio into text.
    # OPENAI_API_KEY is a secret: keep it in .env, never in .env.example or git.
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "whisper-1")
    # Master switch: if false (or no key), ingestion skips transcription entirely.
    TRANSCRIBE_VIDEO: bool = os.getenv("TRANSCRIBE_VIDEO", "true").lower() == "true"

    # Minutes of Meeting — max characters of transcript per LLM call before the
    # map-reduce path kicks in (~4 chars/token; 12000 chars ~= 3000 tokens input).
    MOM_BATCH_CHARS: int = int(os.getenv("MOM_BATCH_CHARS", "12000"))

    # Vision — describe video frames with a vision model (reads text + describes
    # visuals). Uses OpenAI gpt-4o-mini (reuses the Whisper key); Qwen on Ollama
    # Cloud needed a paid subscription and local Qwen was too large.
    # If DESCRIBE_FRAMES is false, ingestion uses OCR only (the old behavior).
    VISION_MODEL: str = os.getenv("VISION_MODEL", "gpt-4o-mini")
    VISION_TIMEOUT_SECONDS: int = int(os.getenv("VISION_TIMEOUT_SECONDS", "120"))
    DESCRIBE_FRAMES: bool = os.getenv("DESCRIBE_FRAMES", "true").lower() == "true"
    EXTRACTION_MODEL: str = os.getenv("EXTRACTION_MODEL", "gpt-4o-mini")
    EXTRACTION_TIMEOUT_SECONDS: int = int(os.getenv("EXTRACTION_TIMEOUT_SECONDS", "120"))
    EXTRACTION_BATCH_CHARS: int = int(os.getenv("EXTRACTION_BATCH_CHARS", "6000"))

    # LLM
    LLM_MODEL: str = _require("LLM_MODEL")
    LLM_TIMEOUT_SECONDS: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "120"))

    # OCR (Windows needs an explicit tesseract.exe path; on Linux/Mac PATH usually finds it)
    TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", "")

    NEO4J_URI: str = _require("NEO4J_URI")
    NEO4J_USER: str = _require("NEO4J_USER")
    NEO4J_PASSWORD: str = _require("NEO4J_PASSWORD")


settings = Settings()

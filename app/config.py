"""
Central configuration.

Reads .env at import time and exposes typed constants everything else imports.
Rule: no other module reads env vars directly — all config comes through here.
"""
from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env from the project root (parent of app/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _require(name: str) -> str:
    """Fail loudly if a required env var is missing — better than silent bugs later."""
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

    # Upload limits
    MAX_PDF_MB: int = int(os.getenv("MAX_PDF_MB", "50"))


settings = Settings()
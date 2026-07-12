"""
Logging configuration - call setup_logging() once at startup.

Configures the root logger so INFO-level logs from the app appear in the
uvicorn console with a consistent, readable format. Uses force=True so our
format wins even when uvicorn has already installed its own root handler.

Any module can then log with:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("something happened")
"""
import logging
import os


def setup_logging(level: str | None = None) -> None:
    """
    Initialize application logging.

    Level resolves in this order: explicit arg -> LOG_LEVEL env var -> "INFO".
    Call this once, at application startup, before anything else logs.

    force=True clears any handler uvicorn (or another import) already put on the
    root logger and installs ours, so the format below is what you actually see.
    """
    resolved = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()

    logging.basicConfig(
        level=resolved,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

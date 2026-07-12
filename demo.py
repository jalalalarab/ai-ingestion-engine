#!/usr/bin/env python3
"""
demo.py - end-to-end walkthrough of the AI Ingestion Engine.

Runs the whole pipeline against a running server, in order:

    health  ->  ingest a PDF  ->  search  ->  ask (RAG)

so anyone can see the system work in about a minute without remembering curl
commands.

Usage:
    python demo.py
    python demo.py --pdf storage/uploads/mydoc.pdf --query "..." --question "..."
    python demo.py --base-url http://localhost:8000 --top-k 3

Requires the API running in another terminal:
    uvicorn app.main:app --reload
"""
import argparse
import sys
from pathlib import Path

import httpx


DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_PDF = "storage/uploads/Day1_SCANNED.pdf"
DEFAULT_QUERY = "FastAPI endpoints and databases"
DEFAULT_QUESTION = "What does this document say about building APIs with FastAPI?"


def header(step: int, title: str) -> None:
    """Print a clear section banner so each stage is easy to spot in a demo."""
    print("\n" + "=" * 70)
    print(f"  STEP {step}: {title}")
    print("=" * 70)


def short(text: str, limit: int = 200) -> str:
    """Collapse whitespace and truncate long text so console output stays readable."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Ingestion Engine end-to-end demo (health -> ingest -> search -> ask)."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL.")
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="Path to a PDF to ingest.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Search query to run.")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Question for /ask.")
    parser.add_argument("--top-k", type=int, default=5, help="How many chunks to retrieve.")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    # Generous timeout: ingest (semantic chunking embeds every sentence) and
    # /ask (waits on the LLM) can each take a while on CPU-only setups.
    client = httpx.Client(timeout=180.0)

    # ---- STEP 1: health -------------------------------------------------
    header(1, "Health check")
    try:
        r = client.get(f"{base}/health")
        r.raise_for_status()
        print(f"  Server is up: {r.json()}")
    except httpx.ConnectError:
        print(f"  Could not reach the server at {base}")
        print("  Start it first, in another terminal:")
        print("      uvicorn app.main:app --reload")
        sys.exit(1)

    # ---- STEP 2: ingest -------------------------------------------------
    header(2, f"Ingest PDF  ({args.pdf})")
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"  PDF not found: {pdf_path}")
        print("  Pass a real file with:  --pdf <path>")
        sys.exit(1)
    with pdf_path.open("rb") as f:
        files = {"file": (pdf_path.name, f, "application/pdf")}
        r = client.post(f"{base}/ingest/pdf", files=files)
    r.raise_for_status()
    ing = r.json()
    print(f"  file_id:        {ing.get('file_id')}")
    print(f"  pages:          {ing.get('pages_processed')}")
    print(f"  chunks created: {ing.get('chunks_created')}")
    print(f"  OCR pages:      {ing.get('ocr_pages_count')}")

    # ---- STEP 3: search -------------------------------------------------
    header(3, f'Search  ("{args.query}")')
    r = client.post(f"{base}/search", json={"query": args.query, "top_k": args.top_k})
    r.raise_for_status()
    sr = r.json()
    print(f"  {sr.get('count', 0)} results:")
    for i, hit in enumerate(sr.get("results", []), 1):
        if hit.get("page_number") is not None:
            src = f"p.{hit['page_number']}"
        elif hit.get("timestamp_seconds") is not None:
            src = f"t={hit['timestamp_seconds']}s"
        else:
            src = "-"
        print(f"   {i}. score={hit['score']:.3f}  [{hit.get('file_name')} {src}]")
        print(f"      {short(hit['text'])}")

    # ---- STEP 4: ask (RAG) ----------------------------------------------
    header(4, f'Ask  ("{args.question}")')
    r = client.post(f"{base}/ask", json={"question": args.question, "top_k": args.top_k})
    r.raise_for_status()
    ar = r.json()
    print("  ANSWER:")
    print(f"    {short(ar.get('answer', ''), 600)}")
    sources = ar.get("sources", [])
    if sources:
        print("  SOURCES:")
        for s in sources:
            page = f"p.{s['page_number']}" if s.get("page_number") is not None else "-"
            print(f"    - {s.get('file_name')} {page} "
                  f"(score={s.get('score', 0):.3f}) {s.get('label', '')}")
    else:
        print("  SOURCES: none (confidence guard may have declined to answer)")

    print("\n" + "=" * 70)
    print("  Demo complete: health -> ingest -> search -> ask, all working.")
    print("=" * 70)


if __name__ == "__main__":
    main()

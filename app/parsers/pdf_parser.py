"""
PDF text extractor.

Given raw PDF bytes, return a list of (page_number, text) tuples — one per page.
Uses PyMuPDF (imported as `fitz`). No cleaning, no chunking here — that comes later.
This module is the EXTRACT stage of the pipeline for the PDF source type.
"""
import fitz  # PyMuPDF


def extract_pdf_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """
    Extract text page by page from a PDF given as bytes.

    Returns:
        A list of (page_number, page_text) tuples.
        page_number is 1-indexed (page 1 is the first page — matches how humans count).
        page_text may be empty for scanned/image pages (OCR is added in Phase 4).
    """
    # `stream=` lets fitz read from memory instead of a file path.
    # `filetype="pdf"` is a hint so it doesn't try to guess.
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text()  # PyMuPDF's built-in plain-text extraction
        pages.append((i, text))

    doc.close()
    return pages
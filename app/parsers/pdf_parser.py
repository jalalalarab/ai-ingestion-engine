"""
PDF text extractor.

Given raw PDF bytes, return a list of (page_number, text, method) tuples —
one per page. Uses PyMuPDF (imported as `fitz`). No cleaning, no chunking
here — that comes later. This module is the EXTRACT stage of the pipeline
for the PDF source type.

Two extraction methods:
  - 'pdf_text' — the page had a real text layer; PyMuPDF read it directly.
  - 'ocr'      — the page was scanned/image-only, so we rendered it to an
                 image and ran Tesseract on the pixels.
"""
import fitz  # PyMuPDF
from app.parsers.ocr_parser import ocr_pdf_page

# If a page's extracted text is shorter than this (after stripping whitespace),
# treat it as image-only and OCR it. Real prose pages return hundreds of chars.
# Fully scanned pages return 0. Even a page with just a heading returns 15+.
# 30 sits in the dead zone between "empty" and "real content" — nothing
# legitimate falls there.
OCR_TRIGGER_CHAR_COUNT = 30


def extract_pdf_pages(pdf_bytes: bytes) -> list[tuple[int, str, str]]:
    """
    Extract text page by page from a PDF given as bytes.

    Returns:
        A list of (page_number, page_text, extraction_method) tuples.
        page_number is 1-indexed (page 1 is the first page — matches how humans count).
        extraction_method is 'pdf_text' for pages with a real text layer,
        'ocr' for scanned pages that had to be recognized from pixels.
    """
    # `stream=` lets fitz read from memory instead of a file path.
    # `filetype="pdf"` is a hint so it doesn't try to guess.
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text()  # PyMuPDF's built-in plain-text extraction

        if len(text.strip()) < OCR_TRIGGER_CHAR_COUNT:
            # Scanned or image-only page — fall back to OCR.
            text = ocr_pdf_page(page)
            method = "ocr"
        else:
            method = "pdf_text"

        pages.append((i, text, method))

    doc.close()
    return pages
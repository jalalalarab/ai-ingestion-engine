"""
PDF text extractor.
Given raw PDF bytes, return a list of (page_number, text, method) tuples -
one per page. Uses PyMuPDF (imported as `fitz`). No cleaning, no chunking
here - that comes later. This module is the EXTRACT stage of the pipeline
for the PDF source type.

Three extraction outcomes:
  - 'pdf_text'      - the page had a real text layer; PyMuPDF read it directly.
  - 'pdf_text+ocr'  - a text page that ALSO had sizeable embedded images
                      (e.g. screenshots); we OCR'd those images and merged
                      their text in, so picture-text becomes searchable too.
  - 'ocr'           - the page was scanned/image-only, so we rendered the whole
                      page to an image and ran Tesseract on the pixels.
"""
import fitz  # PyMuPDF

from app.parsers.ocr_parser import ocr_pdf_page, ocr_page_images

# If a page's extracted text is shorter than this (after stripping whitespace),
# treat it as image-only and OCR the whole page. Real prose pages return
# hundreds of chars. Fully scanned pages return 0. Even a page with just a
# heading returns 15+. 30 sits in the dead zone between "empty" and "real
# content" - nothing legitimate falls there.
OCR_TRIGGER_CHAR_COUNT = 30


def extract_pdf_pages(pdf_bytes: bytes) -> list[tuple[int, str, str]]:
    """
    Extract text page by page from a PDF given as bytes.

    Returns:
        A list of (page_number, page_text, extraction_method) tuples.
        page_number is 1-indexed (page 1 is the first page - matches how humans count).
        extraction_method is 'pdf_text' for plain text pages, 'pdf_text+ocr' for
        text pages whose embedded images were also OCR'd, and 'ocr' for fully
        scanned pages recognized from pixels.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text()  # PyMuPDF's built-in plain-text extraction

        if len(text.strip()) < OCR_TRIGGER_CHAR_COUNT:
            # Whole page is image-only (a scan) - OCR the entire page.
            text = ocr_pdf_page(page)
            method = "ocr"
        else:
            # Page has a real text layer. But it may ALSO contain images
            # (screenshots, charts) whose text the text layer doesn't include.
            # OCR those embedded images and merge their text in.
            method = "pdf_text"
            image_text = ocr_page_images(page, doc)
            if image_text:
                text = f"{text}\n{image_text}"
                method = "pdf_text+ocr"

        pages.append((i, text, method))
    doc.close()
    return pages

"""
OCR helper. Renders a PyMuPDF page as an image and runs Tesseract on it.
Only used when the normal PDF text extractor finds a page with no selectable
text - scanned pages, screenshot-PDFs, image-only pages. Without this module,
those pages would ingest as empty and be invisible to search.

Also provides ocr_page_images(): OCR the images EMBEDDED inside an otherwise-text
page (e.g. spreadsheet screenshots), so text that only lives inside a picture on
a text page becomes searchable too.
"""
import io

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from app.config import settings

# pytesseract is just a wrapper - it needs to find the real tesseract.exe binary.
# On Windows there's no auto-discovery, so we set the path from .env.
# On Linux/Mac where tesseract is on PATH, TESSERACT_CMD can be blank.
if settings.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD

# 200 DPI is the sweet spot for OCR: high enough for accurate recognition on
# normal-sized print, low enough that a 10-page scan doesn't take a minute.
# Below 150 DPI accuracy drops fast; above 300 you burn CPU with no gain.
DEFAULT_DPI = 200

# Embedded images smaller than this (in pixels, width or height) are almost
# always icons, logos, or bullet decorations - OCR-ing them wastes time and
# produces garbage. Only images at least this big get OCR'd. A real screenshot
# is hundreds of pixels wide; an icon is ~16-48.
MIN_EMBEDDED_IMAGE_PX = 200


def ocr_pdf_page(page: fitz.Page, dpi: int = DEFAULT_DPI) -> str:
    """
    Render one PDF page as a PNG image at the given DPI, then OCR it.
    Returns the recognized text (may be empty if the page has no readable content).
    """
    # PyMuPDF's default rendering is 72 DPI. Scale up by dpi/72 to hit our target.
    scale = dpi / 72
    matrix = fitz.Matrix(scale, scale)
    pixmap = page.get_pixmap(matrix=matrix)
    # tobytes("png") gives us in-memory PNG bytes - no temp files on disk.
    img_bytes = pixmap.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))
    return pytesseract.image_to_string(img)


def ocr_page_images(page: fitz.Page, doc: fitz.Document) -> str:
    """
    OCR the images EMBEDDED inside a page (not the page itself).

    For each image on the page: extract its raw bytes, skip anything too small
    to be real content (icons/logos), OCR the rest, and return all recognized
    text joined together. Returns "" if the page has no sizeable images or none
    yield readable text.

    Used on text pages that also contain images (e.g. screenshots), so the text
    living inside those pictures becomes searchable. Note: OCR on small embedded
    screenshots is imperfect - expect useful-but-noisy results, not clean text.
    """
    pieces: list[str] = []

    # page.get_images(full=True) lists every image on the page; each entry's
    # first element (xref) is how we pull the actual image bytes from the doc.
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            base = doc.extract_image(xref)  # {'image': bytes, 'width', 'height', ...}
        except Exception:
            continue  # unreadable image - skip it, don't crash ingestion

        # Skip images too small to hold real text (icons, bullets, logos).
        width = base.get("width", 0)
        height = base.get("height", 0)
        if width < MIN_EMBEDDED_IMAGE_PX and height < MIN_EMBEDDED_IMAGE_PX:
            continue

        try:
            img = Image.open(io.BytesIO(base["image"]))
            text = pytesseract.image_to_string(img).strip()
        except Exception:
            continue  # a bad/unsupported image shouldn't break the page

        if text:
            pieces.append(text)

    return "\n".join(pieces)

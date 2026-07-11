"""
OCR helper. Renders a PyMuPDF page as an image and runs Tesseract on it.

Only used when the normal PDF text extractor finds a page with no selectable
text — scanned pages, screenshot-PDFs, image-only pages. Without this module,
those pages would ingest as empty and be invisible to search.
"""
import io
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from app.config import settings

# pytesseract is just a wrapper — it needs to find the real tesseract.exe binary.
# On Windows there's no auto-discovery, so we set the path from .env.
# On Linux/Mac where tesseract is on PATH, TESSERACT_CMD can be blank.
if settings.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD

# 200 DPI is the sweet spot for OCR: high enough for accurate recognition on
# normal-sized print, low enough that a 10-page scan doesn't take a minute.
# Below 150 DPI accuracy drops fast; above 300 you burn CPU with no gain.
DEFAULT_DPI = 200


def ocr_pdf_page(page: fitz.Page, dpi: int = DEFAULT_DPI) -> str:
    """
    Render one PDF page as a PNG image at the given DPI, then OCR it.
    Returns the recognized text (may be empty if the page has no readable content).
    """
    # PyMuPDF's default rendering is 72 DPI. Scale up by dpi/72 to hit our target.
    scale = dpi / 72
    matrix = fitz.Matrix(scale, scale)
    pixmap = page.get_pixmap(matrix=matrix)

    # tobytes("png") gives us in-memory PNG bytes — no temp files on disk.
    img_bytes = pixmap.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))

    return pytesseract.image_to_string(img)
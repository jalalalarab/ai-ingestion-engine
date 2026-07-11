"""
Create a synthetic 'scanned' PDF from an existing PDF.

Takes a real PDF and re-renders every page as a raster image, then
re-wraps those images in a new PDF. The output looks identical but has
NO text layer — so it forces the OCR path in our parser to fire.

This is how we test Phase 4 without needing a real scanner.
"""
import sys
from pathlib import Path
import fitz  # PyMuPDF


def rasterize_pdf(source_path: Path, output_path: Path, dpi: int = 100) -> None:
    src = fitz.open(source_path)
    dst = fitz.open()  # empty new PDF

    scale = dpi / 72
    matrix = fitz.Matrix(scale, scale)

    for page in src:
        # Render source page to a raster image
        pixmap = page.get_pixmap(matrix=matrix)
        img_bytes = pixmap.tobytes("png")

        # Create a new page in the destination at the same size,
        # and insert the image as the full page content.
        new_page = dst.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(new_page.rect, stream=img_bytes)

    dst.save(output_path)
    dst.close()
    src.close()
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    source = Path(sys.argv[1])
    output = Path(sys.argv[2])
    rasterize_pdf(source, output)
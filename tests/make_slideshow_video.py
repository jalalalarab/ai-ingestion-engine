"""
Create a synthetic slideshow video from an existing PDF.

Each page becomes one 'slide' held on screen for a fixed number of seconds.
Encoded as MP4 using ffmpeg (bundled via imageio-ffmpeg — no external install).

This is how we test Phase 5 (video ingestion) without needing a real video.
Same trick as tests/make_scanned_pdf.py did for Phase 4: known-content
synthetic input so we can verify OCR extracted what we expect.
"""
import sys
from pathlib import Path
import fitz  # PyMuPDF
import imageio
import numpy as np


def make_slideshow(source_pdf: Path, output_mp4: Path,
                   seconds_per_slide: int = 3, dpi: int = 100, fps: int = 10) -> None:
    """
    Render each PDF page as a still frame, hold for N seconds, mux to MP4.

    dpi=100 keeps file size sane (200 DPI produces massive frames).
    fps=10 is enough for slides — nothing is moving. Lower fps => smaller file.
    seconds_per_slide=3 gives OCR a comfortable window without bloating the video.
    """
    src = fitz.open(source_pdf)
    frames_per_slide = seconds_per_slide * fps
    scale = dpi / 72
    matrix = fitz.Matrix(scale, scale)

    writer = imageio.get_writer(
        str(output_mp4),
        fps=fps,
        codec="libx264",
        quality=6,
        macro_block_size=1,
    )

    try:
        for i, page in enumerate(src, start=1):
            pixmap = page.get_pixmap(matrix=matrix)
            frame = np.frombuffer(pixmap.samples, dtype=np.uint8)
            frame = frame.reshape(pixmap.height, pixmap.width, pixmap.n)
            if pixmap.n == 4:
                frame = frame[:, :, :3]

            # H.264 requires even-numbered dimensions. Crop one pixel off
            # any odd side. Invisible to the eye, keeps ffmpeg happy.
            h, w = frame.shape[:2]
            frame = frame[: h - (h % 2), : w - (w % 2)]

            for _ in range(frames_per_slide):
                writer.append_data(frame)

            print(f"  slide {i}/{len(src)} written")
    finally:
        writer.close()
        src.close()

    print(f"Wrote: {output_mp4}")


if __name__ == "__main__":
    source = Path(sys.argv[1])
    output = Path(sys.argv[2])
    make_slideshow(source, output)

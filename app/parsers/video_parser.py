"""
Video frame extractor — the EXTRACT stage for the video source type.

Given a video file path, sample one frame every N seconds, OCR each sampled
frame (Tesseract, same engine as the Phase 4 PDF OCR path), and return
(timestamp_seconds, frame_number, text) tuples.

Near-duplicate consecutive frames (a slide held on screen for several seconds)
are skipped, so a slide shown for 10 seconds doesn't produce 10 identical chunks.

This is the video analogue of pdf_parser.extract_pdf_pages: it hands the shared
ingestion seam a list of (metadata, text) rows and knows nothing about chunking,
embedding, or Qdrant.
"""
from difflib import SequenceMatcher

import cv2
from PIL import Image
import pytesseract

from app.config import settings

# pytesseract is only a wrapper — it needs the real tesseract binary.
# On Windows we set the path from .env; on Linux/Mac PATH usually finds it.
if settings.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD

# Sample a frame every this many seconds, read from config (VIDEO_SAMPLE_SECONDS,
# default 5). 5s suits slide/meeting videos; lower it for fast-changing content.
# Near-duplicate frames are dropped below, so over-sampling won't create dupes.
SAMPLE_EVERY_SECONDS = settings.VIDEO_SAMPLE_SECONDS

# A sampled frame whose OCR text is shorter than this is treated as blank
# (transition, black frame, logo) and dropped. Same idea as the PDF 30-char gate.
OCR_MIN_CHARS = 15

# If a frame's OCR text is at least this similar to the last KEPT frame's text,
# we treat it as the same slide and skip it. 0.90 = 90% similar.
DUP_SIMILARITY_THRESHOLD = 0.90


def extract_video_frames(video_path: str) -> list[tuple[int, int, str]]:
    """
    Sample, OCR, and de-duplicate frames from a video.

    Returns:
        A list of (timestamp_seconds, frame_number, text) tuples, in order.
        timestamp_seconds is the whole-second offset of the frame in the video.
        frame_number is the absolute frame index (useful for tracing back to
        the exact still).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Frames per second. Some containers report 0/garbage — fall back to 25.
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    # How many raw frames to skip between samples.
    frame_interval = max(1, int(round(fps * SAMPLE_EVERY_SECONDS)))

    results: list[tuple[int, int, str]] = []
    last_kept_text = ""
    frame_idx = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break  # end of video
        frame_idx += 1

        # Only OCR every Nth frame.
        if frame_idx % frame_interval != 0:
            continue

        # OpenCV gives BGR; Tesseract/PIL want RGB.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        text = pytesseract.image_to_string(image).strip()

        # Drop blank / low-content frames.
        if len(text) < OCR_MIN_CHARS:
            continue

        # Drop near-duplicates of the previous kept frame (same slide).
        if last_kept_text:
            similarity = SequenceMatcher(None, last_kept_text, text).ratio()
            if similarity >= DUP_SIMILARITY_THRESHOLD:
                continue

        timestamp_seconds = int(frame_idx / fps)
        results.append((timestamp_seconds, frame_idx, text))
        last_kept_text = text

    cap.release()
    return results

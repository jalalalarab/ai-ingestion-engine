"""
Vision client — describes video frames using OpenAI's vision model (gpt-4o-mini).

One job: given a frame image, return a single text description that BOTH reads
any on-screen text AND describes what the frame shows (charts, diagrams, layout,
people). Richer than OCR alone — captures visual content Tesseract can't, which
is what the instructor asked for ("vision model instead of OpenCV, images with
description").

We use OpenAI here (reusing the same key set up for Whisper transcription):
Qwen on Ollama Cloud required a paid subscription, and a local Qwen vision model
was too large to run on this machine. gpt-4o-mini sees images, is cheap and fast,
and delivers the same capability.

Config: VISION_MODEL (e.g. "gpt-4o-mini") and OPENAI_API_KEY from settings.
The frame is sent as a base64 data URL, which is how the OpenAI vision API
accepts inline images.
"""
import base64
import logging

import cv2

from app.config import settings

logger = logging.getLogger(__name__)

_VISION_PROMPT = (
    "You are analyzing a single frame from a video. In 1-3 sentences: first, "
    "transcribe any text visible in the frame exactly; then briefly describe what "
    "the frame shows (charts, diagrams, layout, people, or scene). Be concise and "
    "factual. If the frame is blank or a plain transition, just say 'blank frame'."
)


def describe_frame(frame_bgr) -> str:
    """
    Describe a single video frame using OpenAI's vision model.

    Args:
        frame_bgr: an OpenCV BGR image (numpy array) — one video frame.

    Returns:
        A text description (on-screen text + visual description), or "" if the
        vision call fails or returns nothing (the caller falls back to OCR).

    Raises:
        RuntimeError: if the OpenAI API key is missing.
    """
    if not settings.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set — cannot run vision. "
            "Add it to .env, or set DESCRIBE_FRAMES=false to use OCR only."
        )

    # Encode the frame as JPEG, then base64, then a data URL for the vision API.
    ok, buffer = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        return ""
    image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{image_b64}"

    # Import here so the module still loads if openai isn't installed yet.
    from openai import OpenAI

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=settings.VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        timeout=settings.VISION_TIMEOUT_SECONDS,
    )

    content = (response.choices[0].message.content or "") if response.choices else ""
    return content.strip()

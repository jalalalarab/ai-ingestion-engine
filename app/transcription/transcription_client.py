"""
Transcription client — wraps OpenAI's Whisper API.

One job: given an audio file, return the spoken words as a list of timestamped
segments — (start_seconds, text) — so each piece of speech can become a chunk
that carries its own timestamp, exactly like the video frame chunks do.

Whisper's `verbose_json` response format gives per-segment start/end times,
which is what lets a spoken answer cite "clip.mp4 — 04:32" the same way a
frame OCR chunk does.

Config: OPENAI_API_KEY and WHISPER_MODEL come from settings. If there's no key,
transcription is skipped (the caller checks settings.TRANSCRIBE_VIDEO first).
"""
from openai import OpenAI

from app.config import settings


def transcribe_audio(audio_path: str) -> list[tuple[int, str]]:
    """
    Transcribe an audio file into timestamped segments.

    Returns:
        A list of (start_seconds, text) tuples, in order. start_seconds is the
        whole-second offset where that spoken segment begins. Empty list if the
        audio produced no usable text.

    Raises:
        RuntimeError: if the OpenAI API key is missing.
        openai.OpenAIError: if the Whisper call itself fails.
    """
    if not settings.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set — cannot transcribe. "
            "Add it to .env, or set TRANSCRIBE_VIDEO=false to skip transcription."
        )

    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    # verbose_json returns segments with start/end timestamps (plain json does not).
    with open(audio_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(
            model=settings.WHISPER_MODEL,
            file=audio_file,
            response_format="verbose_json",
        )

    segments = getattr(result, "segments", None) or []
    output: list[tuple[int, str]] = []
    for seg in segments:
        # seg may be an object or a dict depending on SDK version — handle both.
        text = (getattr(seg, "text", None) if not isinstance(seg, dict) else seg.get("text")) or ""
        start = (getattr(seg, "start", None) if not isinstance(seg, dict) else seg.get("start")) or 0
        text = text.strip()
        if text:
            output.append((int(start), text))

    # Fallback: if no segments came back but there's a top-level text, use it at t=0.
    if not output:
        whole = (getattr(result, "text", "") or "").strip()
        if whole:
            output.append((0, whole))

    return output

"""
Audio extractor — the first stage of video transcription.

A video file bundles a video track and an audio track together. To transcribe
the spoken words, we first need to pull out just the audio. This module uses
ffmpeg (shipped via the imageio-ffmpeg package, so there's no separate system
install) to extract the audio into a temporary .mp3 file and return its path.

It knows nothing about transcription or chunking — its one job is
video path -> audio file path.
"""
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg


def extract_audio(video_path: str) -> str | None:
    """
    Extract the audio track from a video into a temporary mp3 file.

    Returns:
        The path to the extracted .mp3 file, or None if the video has no audio
        track (a silent slideshow, for example) — the caller should treat None
        as "nothing to transcribe" rather than an error.

    Raises:
        RuntimeError: if ffmpeg fails for a reason other than "no audio".
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # A temp file we hand back to the caller; they delete it when done.
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    audio_path = tmp.name
    tmp.close()

    # -vn  = drop the video track (audio only)
    # -acodec libmp3lame = encode as mp3 (small, Whisper-friendly)
    # -ar 16000 = 16 kHz sample rate — plenty for speech, keeps the file small
    # -ac 1 = mono — speech doesn't need stereo, halves the size
    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # If the video simply has no audio stream, ffmpeg says so — treat as "no audio".
    if result.returncode != 0:
        stderr = result.stderr.lower()
        if "does not contain any stream" in stderr or "output file #0 does not contain" in stderr:
            Path(audio_path).unlink(missing_ok=True)
            return None
        Path(audio_path).unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr}")

    # Guard: a produced-but-empty file also means no real audio.
    if not Path(audio_path).exists() or Path(audio_path).stat().st_size == 0:
        Path(audio_path).unlink(missing_ok=True)
        return None

    return audio_path

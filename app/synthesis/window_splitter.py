"""
Window splitter — Step 1 of the video-synthesis pipeline.

Takes the two raw streams a video produces:
  - frames:     list of (timestamp_seconds, frame_number, description)
  - transcript: list of (start_seconds, text)

and merges them into time-ordered WINDOWS. Each window collects everything (speech
+ frame descriptions) that occurred within a span of the video, small enough to fit
comfortably in an LLM's context so a synthesis agent can attend to all of it.

This is pure, deterministic logic — no LLM, no I/O — so it can be tested with hard
assertions. The window-boundary policy is a PARAMETER (size budget), so whatever
window size is decided later is just an argument, not a rewrite.

Design notes:
- We interleave both streams by timestamp into a single ordered list of "events,"
  each tagged with its kind ("speech" or "frame"), then pack events into windows
  until the next event would exceed the size budget.
- Size is measured in characters (a cheap, stable proxy for tokens — ~4 chars/token).
  Using characters keeps this dependency-free and deterministic for testing.
- An oversized single event (e.g. one very long transcript segment) still gets its
  own window rather than being dropped or split mid-sentence.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WindowEvent:
    """One thing that happened at a point in the video."""
    timestamp: int            # whole-second offset in the video
    kind: str                 # "speech" or "frame"
    text: str                 # the transcript text or the frame description
    frame_number: int | None = None  # set only for frame events


@dataclass
class Window:
    """A time-bounded slice of the video: everything that happened within it."""
    index: int                       # 0-based window position
    start_seconds: int               # timestamp of the first event in the window
    end_seconds: int                 # timestamp of the last event in the window
    events: list[WindowEvent] = field(default_factory=list)

    @property
    def char_size(self) -> int:
        """Total characters of text in this window (the budget metric)."""
        return sum(len(e.text) for e in self.events)

    @property
    def speech_events(self) -> list[WindowEvent]:
        return [e for e in self.events if e.kind == "speech"]

    @property
    def frame_events(self) -> list[WindowEvent]:
        return [e for e in self.events if e.kind == "frame"]


def _merge_streams(
    frames: list[tuple[int, int, str]],
    transcript: list[tuple[int, str]],
) -> list[WindowEvent]:
    """
    Interleave frames and transcript into a single timestamp-ordered event list.

    Empty/whitespace-only texts are dropped. Ties (same timestamp) put speech
    before the frame, since the spoken content is usually the higher-value signal
    to lead with.
    """
    events: list[WindowEvent] = []

    for ts, text in transcript:
        text = (text or "").strip()
        if text:
            events.append(WindowEvent(timestamp=int(ts), kind="speech", text=text))

    for ts, frame_no, text in frames:
        text = (text or "").strip()
        if text and text.lower() != "blank frame":  # skip the vision model's blanks
            events.append(
                WindowEvent(timestamp=int(ts), kind="frame", text=text, frame_number=frame_no)
            )

    # Sort by timestamp; on a tie, speech (kind sorts 's' > 'f'? no) — force speech first.
    # We give speech a sort-priority of 0 and frame 1 at equal timestamps.
    events.sort(key=lambda e: (e.timestamp, 0 if e.kind == "speech" else 1))
    return events


def split_into_windows(
    frames: list[tuple[int, int, str]],
    transcript: list[tuple[int, str]],
    max_chars: int = 4000,
) -> list[Window]:
    """
    Merge the two streams and pack them into time-ordered windows under a size budget.

    Args:
        frames:      (timestamp_seconds, frame_number, description) tuples.
        transcript:  (start_seconds, text) tuples.
        max_chars:   soft cap on characters per window. A window is sealed when
                     adding the next event would exceed this (and the window is
                     non-empty). This is THE window-boundary policy — change this
                     number to change window size; no code change needed.

    Returns:
        A list of Window objects, in time order. Empty input -> empty list.

    Guarantees (these are what the tests lock down):
        - Every non-empty event appears in exactly one window (nothing dropped,
          nothing duplicated).
        - Events within and across windows are in non-decreasing timestamp order.
        - No window exceeds max_chars UNLESS it holds a single oversized event.
        - start_seconds/end_seconds bound each window's events.
    """
    events = _merge_streams(frames, transcript)
    if not events:
        return []

    windows: list[Window] = []
    current: list[WindowEvent] = []
    current_size = 0

    for ev in events:
        ev_len = len(ev.text)
        # Seal the current window if adding this event would overflow it —
        # but only if the window already has something (never seal an empty one,
        # which also lets a single oversized event occupy its own window).
        if current and current_size + ev_len > max_chars:
            windows.append(_finalize(len(windows), current))
            current = [ev]
            current_size = ev_len
        else:
            current.append(ev)
            current_size += ev_len

    if current:
        windows.append(_finalize(len(windows), current))

    return windows


def _finalize(index: int, events: list[WindowEvent]) -> Window:
    """Build a Window from an accumulated event list, computing its time bounds."""
    return Window(
        index=index,
        start_seconds=events[0].timestamp,
        end_seconds=events[-1].timestamp,
        events=list(events),
    )

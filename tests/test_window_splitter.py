"""
Tests for the window splitter — the deterministic Step 1 of video synthesis.

Every test locks down a guarantee the splitter promises, so if a future change
breaks one, we know immediately. Run with `pytest test_window_splitter.py -v`
or standalone with `python test_window_splitter.py`.
"""
from app.synthesis.window_splitter import split_into_windows, _merge_streams, WindowEvent, Window

# ---- helpers -------------------------------------------------------------

def _all_texts(windows):
    """Flatten every event's text across all windows, in order."""
    return [e.text for w in windows for e in w.events]


# ---- core correctness ----------------------------------------------------

def test_empty_inputs_give_no_windows():
    assert split_into_windows([], []) == []


def test_nothing_is_dropped_or_duplicated():
    frames = [(0, 0, "frame A"), (5, 150, "frame B")]
    transcript = [(1, "said one"), (6, "said two")]
    windows = split_into_windows(frames, transcript, max_chars=10000)  # all in one window
    texts = _all_texts(windows)
    # every input text present exactly once
    assert sorted(texts) == sorted(["frame A", "frame B", "said one", "said two"])
    assert len(texts) == 4


def test_events_are_in_time_order_across_windows():
    frames = [(10, 300, "f10"), (0, 0, "f0"), (20, 600, "f20")]
    transcript = [(15, "t15"), (5, "t5")]
    windows = split_into_windows(frames, transcript, max_chars=5)  # force many windows
    timestamps = [e.timestamp for w in windows for e in w.events]
    assert timestamps == sorted(timestamps), "events must be non-decreasing in time"


def test_speech_leads_on_timestamp_tie():
    # same timestamp: speech should come before frame
    frames = [(5, 150, "the frame")]
    transcript = [(5, "the speech")]
    events = _merge_streams(frames, transcript)
    assert events[0].kind == "speech"
    assert events[1].kind == "frame"


# ---- the size-budget policy ---------------------------------------------

def test_windows_respect_char_budget():
    # each text is 5 chars; budget 12 => at most 2 per window (10), 3rd overflows
    transcript = [(i, "aaaaa") for i in range(6)]  # 6 events, 5 chars each
    windows = split_into_windows([], transcript, max_chars=12)
    for w in windows:
        # no window exceeds budget (none here is a single oversized event)
        assert w.char_size <= 12, f"window {w.index} size {w.char_size} > 12"
    # 6 events * 5 chars, 2 per window => 3 windows
    assert len(windows) == 3


def test_single_oversized_event_gets_its_own_window():
    # one event bigger than the whole budget must still appear (not dropped/split)
    big = "x" * 100
    transcript = [(0, "small"), (1, big), (2, "small2")]
    windows = split_into_windows([], transcript, max_chars=20)
    all_text = "".join(_all_texts(windows))
    assert big in all_text, "oversized event must not be dropped"
    # the big one should be alone in its window
    big_windows = [w for w in windows if any(len(e.text) == 100 for e in w.events)]
    assert len(big_windows) == 1
    assert len(big_windows[0].events) == 1, "oversized event should occupy its window alone"


def test_window_size_is_just_a_parameter():
    # same data, different budgets => different window counts, same total content
    transcript = [(i, "hello") for i in range(10)]
    few = split_into_windows([], transcript, max_chars=100)   # big budget -> fewer windows
    many = split_into_windows([], transcript, max_chars=5)    # tiny budget -> more windows
    assert len(many) > len(few)
    assert _all_texts(few) == _all_texts(many), "content identical regardless of window size"


# ---- cleaning / edge cases ----------------------------------------------

def test_blank_and_whitespace_events_are_dropped():
    frames = [(0, 0, "blank frame"), (5, 150, "  "), (10, 300, "real frame")]
    transcript = [(1, ""), (2, "real speech"), (3, "   ")]
    windows = split_into_windows(frames, transcript, max_chars=10000)
    texts = _all_texts(windows)
    assert sorted(texts) == sorted(["real frame", "real speech"])


def test_time_bounds_are_correct():
    frames = [(0, 0, "f0"), (30, 900, "f30")]
    transcript = [(10, "t10"), (20, "t20")]
    windows = split_into_windows(frames, transcript, max_chars=10000)  # one window
    w = windows[0]
    assert w.start_seconds == 0
    assert w.end_seconds == 30


def test_frames_only_no_transcript():
    frames = [(0, 0, "f0"), (5, 150, "f5")]
    windows = split_into_windows(frames, [], max_chars=10000)
    assert len(windows) == 1
    assert [e.kind for e in windows[0].events] == ["frame", "frame"]


def test_transcript_only_no_frames():
    transcript = [(0, "t0"), (5, "t5")]
    windows = split_into_windows([], transcript, max_chars=10000)
    assert len(windows) == 1
    assert all(e.kind == "speech" for e in windows[0].events)


def test_window_indices_are_sequential():
    transcript = [(i, "chunk") for i in range(9)]
    windows = split_into_windows([], transcript, max_chars=5)
    assert [w.index for w in windows] == list(range(len(windows)))


# ---- standalone runner ---------------------------------------------------

if __name__ == "__main__":
    import sys
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    sys.exit(1 if failed else 0)

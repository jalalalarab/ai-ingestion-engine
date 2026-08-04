"""
Tests for the document assembler (Step 3).

Uses a fake OpenAI client, so no key/network/cost. We test the ORCHESTRATION:
that windows are assembled in order, empty windows are skipped (not padded), the
title is built correctly, and stats are accurate. Writing quality is judged on
real output separately.

Run: python -m tests.test_document_assembler
"""
from app.synthesis.document_assembler import assemble_document, _clean_title


# ---- fake client: returns a per-call reply so we can see ordering ----

class _Msg:
    def __init__(self, c): self.content = c
class _Choice:
    def __init__(self, c): self.message = _Msg(c)
class _Resp:
    def __init__(self, c): self.choices = [_Choice(c)]

class SequentialFakeClient:
    """Returns 'narrative-1', 'narrative-2', ... on successive calls."""
    def __init__(self, replies=None, empty_on=None):
        self.replies = replies
        self.empty_on = empty_on or set()   # call indices that return "" (empty)
        self.calls = 0
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls += 1
        if self.calls in self.empty_on:
            return _Resp("")
        if self.replies:
            return _Resp(self.replies[self.calls - 1])
        return _Resp(f"narrative-{self.calls}")


# ---- title handling ------------------------------------------------------

def test_clean_title_strips_extension_and_underscores():
    assert _clean_title("sales_noria_erp.mp4") == "Sales Noria Erp"
    assert _clean_title("my-video.mov") == "My Video"
    assert _clean_title("") == "Video"

def test_document_starts_with_title():
    frames = [(0, 0, "a frame")]
    transcript = [(1, "some speech")]
    fake = SequentialFakeClient()
    result = assemble_document("demo_clip.mp4", frames, transcript, client=fake)
    assert result["document"].startswith("# Demo Clip")


# ---- core assembly -------------------------------------------------------

def test_all_windows_are_synthesized_and_included():
    # small budget -> multiple windows -> multiple narratives, in order
    transcript = [(i, "sentence " + "x" * 50) for i in range(6)]  # forces several windows
    fake = SequentialFakeClient()
    result = assemble_document("v.mp4", [], transcript, max_chars=60, client=fake)
    assert result["windows_total"] > 1
    assert result["windows_written"] == result["windows_total"]
    # narratives appear in order in the document
    doc = result["document"]
    assert "narrative-1" in doc
    assert doc.index("narrative-1") < doc.index("narrative-2")

def test_empty_narrative_windows_are_skipped_not_padded():
    # 3 windows, but the 2nd returns empty -> only 2 written, no filler
    transcript = [(i, "s" * 50) for i in range(3)]
    fake = SequentialFakeClient(empty_on={2})  # 2nd call returns ""
    result = assemble_document("v.mp4", [], transcript, max_chars=55, client=fake)
    assert result["windows_total"] == 3
    assert result["windows_written"] == 2       # the empty one contributed nothing
    assert "narrative-2" not in result["document"]

def test_empty_video_gives_title_and_note_no_fabrication():
    fake = SequentialFakeClient()
    result = assemble_document("silent.mp4", [], [], client=fake)
    assert result["windows_total"] == 0
    assert result["windows_written"] == 0
    assert result["document"].startswith("# Silent")
    assert "No transcript" in result["document"]
    assert fake.calls == 0, "no LLM calls for an empty video"

def test_stats_are_accurate():
    transcript = [(0, "hello world")]
    frames = [(0, 0, "a screen")]
    fake = SequentialFakeClient(replies=["the whole narrative"])
    result = assemble_document("v.mp4", frames, transcript, max_chars=10000, client=fake)
    assert result["windows_total"] == 1
    assert result["windows_written"] == 1
    assert result["char_count"] == len(result["document"])
    assert "the whole narrative" in result["document"]

def test_both_streams_reach_synthesis():
    # sanity: the assembler passes real windows (with both kinds) to synthesis
    frames = [(0, 0, "login screen")]
    transcript = [(1, "we log in")]
    fake = SequentialFakeClient(replies=["combined narrative"])
    result = assemble_document("v.mp4", frames, transcript, max_chars=10000, client=fake)
    assert result["windows_written"] == 1
    assert "combined narrative" in result["document"]


# ---- standalone runner ---------------------------------------------------

if __name__ == "__main__":
    import sys
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}"); failed += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}"); failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    sys.exit(1 if failed else 0)

"""
Tests for the synthesis agent (Step 2).

Uses a FAKE OpenAI client injected into synthesize_window, so these run with no
API key, no network, no cost — and are fully deterministic. We test the agent's
LOGIC (what it sends, how it handles edge cases), not the LLM's writing quality
(which is judged by eye on real output).

Run: python -m tests.test_synthesis_agent
"""
from app.synthesis.window_splitter import split_into_windows, Window, WindowEvent
from app.synthesis.synthesis_agent import synthesize_window, _format_window_for_prompt


# ---- a fake client that records what it was asked and returns a canned reply ----

class _FakeMessage:
    def __init__(self, content): self.content = content

class _FakeChoice:
    def __init__(self, content): self.message = _FakeMessage(content)

class _FakeResponse:
    def __init__(self, content): self.choices = [_FakeChoice(content)]

class FakeClient:
    """Mimics the OpenAI client surface we use, capturing the call for assertions."""
    def __init__(self, reply="a grounded narrative."):
        self.reply = reply
        self.last_messages = None
        self.last_kwargs = None
        self.chat = self  # so client.chat.completions.create works
        self.completions = self

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        self.last_messages = kwargs.get("messages")
        return _FakeResponse(self.reply)


# ---- helpers -------------------------------------------------------------

def _window(events):
    return Window(index=0, start_seconds=events[0].timestamp,
                  end_seconds=events[-1].timestamp, events=events)


# ---- tests ---------------------------------------------------------------

def test_empty_window_returns_empty_no_llm_call():
    # An empty window must NOT call the LLM and must NOT fabricate anything.
    fake = FakeClient()
    empty = Window(index=0, start_seconds=0, end_seconds=0, events=[])
    out = synthesize_window(empty, client=fake)
    assert out == ""
    assert fake.last_messages is None, "LLM should not be called for an empty window"


def test_returns_the_models_text():
    fake = FakeClient(reply="The speaker opened the invoice form.")
    w = _window([WindowEvent(0, "speech", "opening the form")])
    out = synthesize_window(w, client=fake)
    assert out == "The speaker opened the invoice form."


def test_prompt_contains_both_speech_and_frames():
    fake = FakeClient()
    w = _window([
        WindowEvent(0, "speech", "we open the proforma"),
        WindowEvent(1, "frame", "Proforma form on screen with Tax field"),
    ])
    synthesize_window(w, client=fake)
    user_msg = fake.last_messages[1]["content"]  # [0]=system, [1]=user
    assert "we open the proforma" in user_msg
    assert "Proforma form on screen with Tax field" in user_msg
    assert "SPEECH:" in user_msg and "ON SCREEN:" in user_msg


def test_speech_only_window_notes_no_screen_content():
    fake = FakeClient()
    w = _window([WindowEvent(0, "speech", "just talking, no slides")])
    synthesize_window(w, client=fake)
    user_msg = fake.last_messages[1]["content"]
    assert "just talking, no slides" in user_msg
    assert "nothing readable on screen" in user_msg  # honest about absence


def test_frames_only_window_notes_no_speech():
    fake = FakeClient()
    w = _window([WindowEvent(0, "frame", "a chart of quarterly sales")])
    synthesize_window(w, client=fake)
    user_msg = fake.last_messages[1]["content"]
    assert "a chart of quarterly sales" in user_msg
    assert "no speech in this segment" in user_msg


def test_temperature_is_zero_for_determinism():
    # Low temperature is part of the anti-hallucination design; lock it in.
    fake = FakeClient()
    w = _window([WindowEvent(0, "speech", "hello")])
    synthesize_window(w, client=fake)
    assert fake.last_kwargs.get("temperature") == 0


def test_grounding_rules_present_in_system_prompt():
    # The system prompt must carry the "only what's here, never invent" discipline.
    fake = FakeClient()
    w = _window([WindowEvent(0, "speech", "hello")])
    synthesize_window(w, client=fake)
    system_msg = fake.last_messages[0]["content"]
    assert "ONLY" in system_msg
    assert "not invent" in system_msg.lower() or "never invent" in system_msg.lower() or "do not add" in system_msg.lower()


def test_format_helper_directly():
    # _format_window_for_prompt should produce labeled SPEECH/ON SCREEN blocks.
    w = _window([
        WindowEvent(0, "speech", "spoken line"),
        WindowEvent(1, "frame", "screen thing"),
    ])
    text = _format_window_for_prompt(w)
    assert text.index("SPEECH:") < text.index("ON SCREEN:")  # speech block first
    assert "- spoken line" in text
    assert "- screen thing" in text


def test_integration_with_real_splitter():
    # End-to-end wiring: splitter output feeds straight into the agent.
    frames = [(0, 0, "login screen"), (5, 150, "inventory module")]
    transcript = [(1, "we log in"), (6, "here is inventory out")]
    windows = split_into_windows(frames, transcript, max_chars=10000)
    assert len(windows) == 1
    fake = FakeClient(reply="narrative")
    out = synthesize_window(windows[0], client=fake)
    assert out == "narrative"
    # both streams reached the prompt
    user_msg = fake.last_messages[1]["content"]
    assert "we log in" in user_msg and "login screen" in user_msg


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

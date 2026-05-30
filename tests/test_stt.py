"""
tests/test_stt.py
=================
Unit tests for DeepgramSTTEngine.

What we test
------------
* Transcript callback receives correctly-shaped TranscriptResult.
* English / Hindi / Hinglish flow through the language tag.
* Low-confidence path triggers the on_low_confidence callback (NOT on_transcript).
* Empty / silent audio is handled gracefully without crashes.
* VAD-gated send_audio() drops audio when VAD is closed.
* send_audio() triggers a reconnect on connection errors.

Mocking strategy
----------------
* No real Deepgram WebSocket — we feed fake "result" objects directly into
  _on_transcript_event() to exercise the parsing + callback dispatch logic.
* AsyncMock stands in for the live WebSocket connection in send_audio tests.
"""

from unittest.mock import AsyncMock
import pytest

# Skip the entire module if deepgram-sdk isn't installed in this env.
pytest.importorskip("deepgram", reason="deepgram-sdk not installed in this env")

from src.stt_engine import DeepgramSTTEngine, TranscriptResult, TranscriptType


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_engine(
    on_transcript=None,
    on_end_of_turn=None,
    on_low_confidence=None,
    threshold: float = 0.80,
) -> DeepgramSTTEngine:
    """Construct an engine with no real network connection."""
    return DeepgramSTTEngine(
        api_key="fake-key-for-tests",
        confidence_threshold=threshold,
        on_transcript=on_transcript,
        on_end_of_turn=on_end_of_turn,
        on_low_confidence=on_low_confidence,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Transcript handling
# ─────────────────────────────────────────────────────────────────────────────

async def test_transcription_returns_text(dg_result_factory):
    """A final result is parsed into a non-empty TranscriptResult."""
    captured: list = []
    engine = _make_engine(on_transcript=lambda r: captured.append(r))

    dg_result = dg_result_factory(
        text="Hello, tell me about your plans",
        confidence=0.95,
        language="en",
        is_final=True,
    )
    await engine._on_transcript_event(None, dg_result)

    assert len(captured) == 1
    assert captured[0].text == "Hello, tell me about your plans"
    assert captured[0].confidence > 0.0
    assert isinstance(captured[0], TranscriptResult)


async def test_transcription_english_detected(dg_result_factory):
    """The language tag from Deepgram is surfaced unchanged."""
    captured: list = []
    engine = _make_engine(on_transcript=lambda r: captured.append(r))

    dg_result = dg_result_factory(
        text="Hello world",
        confidence=0.95,
        language="en",
        is_final=True,
    )
    await engine._on_transcript_event(None, dg_result)

    assert captured[0].language == "en"
    assert captured[0].confidence == 0.95


async def test_transcription_hindi_detected(dg_result_factory):
    """Hindi text + language tag are both preserved."""
    captured: list = []
    engine = _make_engine(on_transcript=lambda r: captured.append(r))

    hindi_text = "नमस्ते, मुझे प्लान के बारे में बताएं"
    dg_result = dg_result_factory(
        text=hindi_text,
        confidence=0.92,
        language="hi",
        is_final=True,
    )
    await engine._on_transcript_event(None, dg_result)

    assert captured[0].language == "hi"
    # At least one Devanagari character present
    assert any("ऀ" <= c <= "ॿ" for c in captured[0].text)


async def test_low_confidence_triggers_fallback(dg_result_factory):
    """Final result below threshold → on_low_confidence fires, on_transcript does not."""
    transcripts: list = []
    low_conf_flag = {"called": False}

    def on_low():
        low_conf_flag["called"] = True

    engine = _make_engine(
        on_transcript=lambda r: transcripts.append(r),
        on_low_confidence=on_low,
        threshold=0.80,
    )

    dg_result = dg_result_factory(
        text="kinda muffled",
        confidence=0.45,   # well below 0.80
        language="en",
        is_final=True,
    )
    await engine._on_transcript_event(None, dg_result)

    assert transcripts == []           # nothing forwarded to the pipeline
    assert low_conf_flag["called"] is True


async def test_empty_transcript_ignored(dg_result_factory):
    """Empty transcript text is silently dropped."""
    captured: list = []
    engine = _make_engine(on_transcript=lambda r: captured.append(r))

    dg_result = dg_result_factory(text="", confidence=0.99)
    await engine._on_transcript_event(None, dg_result)

    assert captured == []


# ─────────────────────────────────────────────────────────────────────────────
# Audio send path (VAD gate + reconnect)
# ─────────────────────────────────────────────────────────────────────────────

async def test_empty_audio_handled_gracefully():
    """send_audio(b'') must not raise."""
    engine = _make_engine()
    # No connection, VAD inactive — should silently return
    await engine.send_audio(b"")


async def test_silence_not_sent_to_deepgram():
    """When the VAD gate is closed, audio is dropped before reaching Deepgram."""
    engine = _make_engine()
    mock_conn = AsyncMock()
    engine._connection = mock_conn
    engine._connected = True
    engine.set_vad_active(False)   # VAD closed

    silent_audio = b"\x00" * 640   # 20 ms of silence at 16 kHz / 16-bit
    await engine.send_audio(silent_audio)

    mock_conn.send.assert_not_called()


async def test_audio_sent_when_vad_open():
    """When the VAD gate is open, audio is forwarded to Deepgram."""
    engine = _make_engine()
    mock_conn = AsyncMock()
    engine._connection = mock_conn
    engine._connected = True
    engine.set_vad_active(True)

    pcm = b"\x01\x02" * 320
    await engine.send_audio(pcm)

    mock_conn.send.assert_awaited_once_with(pcm)


async def test_reconnect_on_stream_drop():
    """A send failure triggers the reconnect path."""
    engine = _make_engine()
    mock_conn = AsyncMock()
    mock_conn.send.side_effect = ConnectionError("connection lost")
    engine._connection = mock_conn
    engine._connected = True
    engine.set_vad_active(True)

    # Stub out the real reconnect to avoid real network + sleep
    engine._reconnect = AsyncMock()

    await engine.send_audio(b"\x01" * 320)

    engine._reconnect.assert_awaited_once()


# ─────────────────────────────────────────────────────────────────────────────
# Hinglish / code-switch
# ─────────────────────────────────────────────────────────────────────────────

async def test_hinglish_passthrough(dg_result_factory):
    """
    Deepgram does not natively tag 'hinglish' — it returns 'en' or 'hi'.
    The STT engine surfaces whatever tag Deepgram gives; downstream
    LanguageDetector handles the Hinglish reclassification.
    """
    captured: list = []
    engine = _make_engine(on_transcript=lambda r: captured.append(r))

    text = "Tell me about your plans, mujhe details chahiye yaar"
    dg_result = dg_result_factory(text=text, confidence=0.88, language="en")
    await engine._on_transcript_event(None, dg_result)

    assert len(captured) == 1
    assert "mujhe" in captured[0].text.lower()
    # STT layer surfaces Deepgram's tag, hinglish reclassification is downstream
    assert captured[0].language == "en"


# ─────────────────────────────────────────────────────────────────────────────
# End-of-turn callback
# ─────────────────────────────────────────────────────────────────────────────

async def test_end_of_turn_callback_fires():
    """UtteranceEnd event triggers on_end_of_turn."""
    eot_flag = {"called": False}

    def on_eot():
        eot_flag["called"] = True

    engine = _make_engine(on_end_of_turn=on_eot)
    await engine._on_utterance_end_event(None, None)

    assert eot_flag["called"] is True
    # The VAD gate should be closed after UtteranceEnd
    assert engine._vad_active is False

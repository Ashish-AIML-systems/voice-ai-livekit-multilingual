"""
tests/test_language_switching.py
================================
Unit tests for LanguageDetector + integration with ConversationManager +
TTSEngine voice-id selection.

What we test
------------
* Deepgram tag wins when confidence ≥ threshold.
* Low Deepgram confidence → falls back to text-based detect_language.
* Both signals fail → defaults to "en" (safe default).
* Mid-conversation switches with 2-confirmation hysteresis (no flicker).
* Hinglish detection from Roman-script marker words.
* Conversation history is preserved across language switches.
* TTSEngine returns the correct voice_id for each language.

Mocking strategy
----------------
* No external API calls.
* LanguageDetector is exercised directly with raw text + Deepgram metadata.
* ConversationManager is constructed without a TTS engine for these tests.
"""

import pytest

from src.language_detector    import LanguageDetector
from src.conversation_manager import ConversationManager

# TTS depends on the elevenlabs SDK. If it isn't installed in the dev env,
# we skip the TTS-related tests rather than failing the whole module.
try:
    from src.tts_engine import TTSEngine
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False
    TTSEngine = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def detector() -> LanguageDetector:
    return LanguageDetector(confidence_threshold=0.80)


@pytest.fixture
def convo() -> ConversationManager:
    return ConversationManager()


# ─────────────────────────────────────────────────────────────────────────────
# Deepgram-driven detection
# ─────────────────────────────────────────────────────────────────────────────

def test_english_detected_from_deepgram_confidence(detector):
    result = detector.detect_from_deepgram(
        text="Hello",
        deepgram_language="en",
        deepgram_confidence=0.95,
    )
    assert result["language"] == "en"
    assert result["confidence"] == 0.95


def test_hindi_detected_from_deepgram_confidence(detector):
    result = detector.detect_from_deepgram(
        text="नमस्ते",
        deepgram_language="hi",
        deepgram_confidence=0.92,
    )
    assert result["language"] == "hi"
    assert result["confidence"] == 0.92


def test_low_confidence_falls_back_to_text_analysis(detector):
    """
    When Deepgram confidence is below the 0.80 threshold, detect_from_deepgram
    must use the text-based detector — which catches Devanagari instantly.
    """
    result = detector.detect_from_deepgram(
        text="नमस्ते कैसे हो",      # Devanagari → 'hi' from script analysis
        deepgram_language="en",   # wrong tag
        deepgram_confidence=0.60, # below threshold → ignored
    )
    assert result["language"] == "hi"


def test_both_fail_defaults_to_english(detector):
    """Ambiguous text + low Deepgram confidence → safe default 'en'."""
    result = detector.detect_from_deepgram(
        text="xyz qrs lmn",
        deepgram_language="en",
        deepgram_confidence=0.50,
    )
    assert result["language"] == "en"


# ─────────────────────────────────────────────────────────────────────────────
# Mid-conversation language switching (hysteresis)
# ─────────────────────────────────────────────────────────────────────────────

def test_language_switch_english_to_hindi(detector):
    """Two consecutive Hindi detections required before switching from en → hi."""
    pid = "user-1"

    # Start in English
    assert detector.update_session_language(pid, "Hello there friend") == "en"

    # First Hindi detection — pending, NOT switched yet
    assert detector.update_session_language(pid, "नमस्ते भाई कैसे हो") == "en"

    # Second Hindi detection — confirmation reached, switch happens
    assert detector.update_session_language(pid, "मुझे जानकारी चाहिए") == "hi"

    assert detector.get_current_language(pid) == "hi"


def test_language_switch_hindi_to_english(detector):
    """Same hysteresis applies in the reverse direction."""
    pid = "user-2"
    # Bootstrap to Hindi
    detector.update_session_language(pid, "नमस्ते")
    detector.update_session_language(pid, "मुझे जानकारी चाहिए")
    assert detector.get_current_language(pid) == "hi"

    # Two English detections to switch back
    detector.update_session_language(pid, "Hello there my friend")
    detector.update_session_language(pid, "Tell me more about the offers")
    assert detector.get_current_language(pid) == "en"


def test_consecutive_confirmation_before_switch(detector):
    """A single odd-language detection should NOT flip the active language."""
    pid = "user-4"
    # Default
    assert detector.get_current_language(pid) == "en"

    # ONE Hindi utterance — pending, no switch
    detector.update_session_language(pid, "नमस्ते")
    assert detector.get_current_language(pid) == "en"

    # SECOND Hindi utterance — confirmation reached
    detector.update_session_language(pid, "क्या हाल है दोस्त")
    assert detector.get_current_language(pid) == "hi"


# ─────────────────────────────────────────────────────────────────────────────
# Hinglish detection
# ─────────────────────────────────────────────────────────────────────────────

def test_hinglish_detected_via_markers(detector):
    """Roman-script Hindi marker words trigger Hinglish classification.

    Markers used here come from _HINGLISH_MARKERS in language_detector.py:
    yaar, bahut, achha, hai (4 hits — well above the 2-marker threshold).
    """
    text = "Tell me about pricing yaar, bahut achha hai"
    result = detector.detect_language(text)
    assert result["language"] == "hinglish"
    assert result["confidence"] >= 0.80


def test_hinglish_via_mixed_script(detector):
    """Devanagari + Latin in the same string → Hinglish."""
    text = "yeh product bahut अच्छा hai"
    result = detector.detect_language(text)
    assert result["language"] == "hinglish"


# ─────────────────────────────────────────────────────────────────────────────
# Conversation context preservation across switches
# ─────────────────────────────────────────────────────────────────────────────

def test_context_preserved_after_language_switch(convo):
    """update_language() must not drop existing history entries."""
    pid = "user-3"
    convo.start_conversation(pid)

    # 3 English turns
    convo.add_message(pid, "user",      "Hello")
    convo.add_message(pid, "assistant", "Hi there")
    convo.add_message(pid, "user",      "What products do you have?")
    convo.add_message(pid, "assistant", "We have many great offers")
    convo.add_message(pid, "user",      "Cool")
    convo.add_message(pid, "assistant", "Yes")

    history_before = list(convo.get_context(pid))

    # Switch language to Hindi
    convo.update_language(pid, "hi")

    history_after = convo.get_context(pid)

    # No messages dropped
    assert len(history_after) == len(history_before)
    # Specific English messages still present
    contents = [m["content"] for m in history_after]
    assert "Hello" in contents
    assert "Cool" in contents


def test_session_language_updates_on_switch(convo):
    pid = "user-5"
    convo.start_conversation(pid)
    assert convo.get_session(pid)["language"] == "en"

    convo.update_language(pid, "hi")
    assert convo.get_session(pid)["language"] == "hi"


# ─────────────────────────────────────────────────────────────────────────────
# TTS voice selection on language switch
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _TTS_AVAILABLE, reason="elevenlabs SDK not installed")
def test_tts_voice_changes_on_language_switch():
    """get_voice_id returns the correct ElevenLabs voice ID per language."""
    tts = TTSEngine(
        api_key="fake-key",
        voice_id_english="VOICE_EN_ID",
        voice_id_hindi="VOICE_HI_ID",
    )
    assert tts.get_voice_id("en") == "VOICE_EN_ID"
    assert tts.get_voice_id("hi") == "VOICE_HI_ID"
    # Hinglish reuses English voice (handles code-switch better)
    assert tts.get_voice_id("hinglish") == "VOICE_EN_ID"


@pytest.mark.skipif(not _TTS_AVAILABLE, reason="elevenlabs SDK not installed")
def test_tts_voice_settings_change_per_language():
    """Voice settings (stability, similarity) are tuned per language."""
    tts = TTSEngine(api_key="fake-key")

    en_settings = tts.get_voice_settings("en")
    hi_settings = tts.get_voice_settings("hi")

    # English vs Hindi have different tuning (per spec)
    assert en_settings.stability == 0.5
    assert hi_settings.stability == 0.55
    assert en_settings.similarity_boost == 0.75
    assert hi_settings.similarity_boost == 0.80


@pytest.mark.skipif(not _TTS_AVAILABLE, reason="elevenlabs SDK not installed")
def test_tts_falls_back_to_english_voice_on_unknown_language():
    """An unknown language code falls back to the English voice."""
    tts = TTSEngine(
        api_key="fake-key",
        voice_id_english="VOICE_EN_ID",
        voice_id_hindi="VOICE_HI_ID",
    )
    # 'fr' is not configured — should fall back to English
    assert tts.get_voice_id("fr") == "VOICE_EN_ID"


# ─────────────────────────────────────────────────────────────────────────────
# Per-participant isolation
# ─────────────────────────────────────────────────────────────────────────────

def test_sessions_are_per_participant(detector):
    """Different participants maintain independent language state."""
    a = "user-a"
    b = "user-b"

    # User A flips to Hindi
    detector.update_session_language(a, "नमस्ते")
    detector.update_session_language(a, "मुझे जानकारी चाहिए")
    assert detector.get_current_language(a) == "hi"

    # User B remains English (default)
    assert detector.get_current_language(b) == "en"

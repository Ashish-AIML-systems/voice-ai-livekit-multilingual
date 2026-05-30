"""
tests/test_llm.py
=================
Unit tests for LLMProcessor (Groq via OpenAI-compatible client).

What we test
------------
* English / Hindi response generation paths.
* Streaming yields complete sentences (not partial tokens).
* System prompt is always at messages[0].
* History trimming caps at MAX_HISTORY_TURNS pairs.
* History is preserved across language switches.
* Empty input does not crash.
* Token-budget respect (trim_history).
* Fallback string returned on OpenAI errors (RateLimit / Connection / Timeout).

Mocking strategy
----------------
* `llm._client.chat.completions.create` is replaced with an AsyncMock that
  returns an AsyncStream of fake chunks (or raises a fake openai error).
* The system prompt file is the REAL marketing_system_prompt.txt — keeps the
  test honest about the actual prompt content.
"""

from unittest.mock import AsyncMock, MagicMock
import pytest

# Skip the entire module if the openai SDK isn't installed in this env.
openai = pytest.importorskip("openai", reason="openai SDK not installed in this env")

from src.llm_processor import LLMProcessor


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def llm() -> LLMProcessor:
    """Fresh LLMProcessor — uses a fake key so no real network calls happen."""
    return LLMProcessor(api_key="fake-test-key")


@pytest.fixture
def patched_llm(llm, async_stream_factory):
    """LLMProcessor with a mocked chat.completions.create method."""
    mock_create = AsyncMock()
    llm._client.chat.completions.create = mock_create
    return llm, mock_create


# ─────────────────────────────────────────────────────────────────────────────
# Response generation — language paths
# ─────────────────────────────────────────────────────────────────────────────

async def test_response_generated_in_english(patched_llm, openai_chunk_factory, async_stream_factory):
    llm, mock_create = patched_llm
    chunks = [
        openai_chunk_factory("Hello! "),
        openai_chunk_factory("We have many "),
        openai_chunk_factory("great Diwali offers."),
    ]
    mock_create.return_value = async_stream_factory(chunks)

    response = await llm.generate_response(
        user_input="Tell me about your marketing plans",
        language="en",
        scope="marketing",
    )

    assert isinstance(response, str)
    assert response.strip() != ""
    assert "offers" in response.lower()


async def test_response_generated_in_hindi(patched_llm, openai_chunk_factory, async_stream_factory):
    llm, mock_create = patched_llm
    chunks = [
        openai_chunk_factory("नमस्ते! "),
        openai_chunk_factory("हमारे पास कई "),
        openai_chunk_factory("शानदार ऑफर्स हैं।"),
    ]
    mock_create.return_value = async_stream_factory(chunks)

    response = await llm.generate_response(
        user_input="मुझे आपके प्लान के बारे में बताएं",
        language="hi",
    )

    assert isinstance(response, str)
    assert response.strip() != ""
    # Contains at least one Devanagari character
    assert any("ऀ" <= c <= "ॿ" for c in response)


# ─────────────────────────────────────────────────────────────────────────────
# System prompt placement
# ─────────────────────────────────────────────────────────────────────────────

def test_system_prompt_always_at_index_0(llm):
    """System prompt sits at messages[0] with marketing-related content."""
    messages = llm._build_messages("Hello", "en", history=[])

    assert messages[0]["role"] == "system"
    sys_content = messages[0]["content"].lower()
    # The real marketing_system_prompt.txt mentions Priya / BrandVoice / marketing
    assert any(token in sys_content for token in ("marketing", "priya", "brand"))


def test_system_prompt_with_history(llm):
    """System still at [0] even when history is non-empty."""
    history = [
        {"role": "user",      "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    messages = llm._build_messages("More info please", "en", history)

    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Hi"}
    assert messages[2] == {"role": "assistant", "content": "Hello!"}


# ─────────────────────────────────────────────────────────────────────────────
# History sliding window
# ─────────────────────────────────────────────────────────────────────────────

def test_history_sliding_window_max_8_turns(llm):
    """trim_history(max_turns=8) keeps only the last 8 user+assistant pairs."""
    history = []
    for i in range(12):
        history.append({"role": "user",      "content": f"user msg {i}"})
        history.append({"role": "assistant", "content": f"bot msg {i}"})

    trimmed = llm.trim_history(history, max_turns=8)

    assert len(trimmed) == 16              # 8 pairs × 2 messages
    assert trimmed[0]["content"]  == "user msg 4"
    assert trimmed[-1]["content"] == "bot msg 11"


def test_history_preserved_on_language_switch(llm):
    """Switching language must not drop existing history entries."""
    history = [
        {"role": "user",      "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user",      "content": "What's up?"},
        {"role": "assistant", "content": "Doing great"},
        {"role": "user",      "content": "Cool"},
        {"role": "assistant", "content": "Yes"},
    ]
    messages = llm._build_messages("नमस्ते", "hi", history)

    # system + 6 history messages + 1 new user message
    assert len(messages) == 1 + 6 + 1
    # First English turn still preserved
    assert messages[1]["content"] == "Hello"
    # The new user message carries the Hindi language tag
    assert "Hindi" in messages[-1]["content"]


def test_token_limit_respected(llm):
    """Long history is trimmed; system prompt is added separately by _build_messages."""
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 100}
        for i in range(100)
    ]
    trimmed = llm.trim_history(history, max_turns=10)

    assert len(trimmed) == 20                              # 10 turns × 2
    assert all(m["role"] in ("user", "assistant") for m in trimmed)


# ─────────────────────────────────────────────────────────────────────────────
# Streaming
# ─────────────────────────────────────────────────────────────────────────────

async def test_streaming_response_yields_sentences(patched_llm, openai_chunk_factory, async_stream_factory):
    """stream_sentences() yields a complete sentence at a time, not partial tokens."""
    llm, mock_create = patched_llm
    chunks = [
        openai_chunk_factory("First sentence. "),
        openai_chunk_factory("Second one is here. "),
        openai_chunk_factory("Third! "),
        openai_chunk_factory("Fourth? "),
        openai_chunk_factory("And finally fifth."),
    ]
    mock_create.return_value = async_stream_factory(chunks)

    sentences = []
    async for s in llm.stream_sentences("Test", "en"):
        sentences.append(s)

    assert len(sentences) == 5
    for s in sentences:
        assert isinstance(s, str)
        assert s.strip() != ""
    # Sentences come out in order
    assert sentences[0].startswith("First")
    assert sentences[-1].endswith("fifth.")


async def test_streaming_handles_hindi_danda(patched_llm, openai_chunk_factory, async_stream_factory):
    """Hindi danda (।) is treated as a sentence boundary."""
    llm, mock_create = patched_llm
    chunks = [
        openai_chunk_factory("पहला वाक्य। "),
        openai_chunk_factory("दूसरा वाक्य। "),
        openai_chunk_factory("तीसरा वाक्य।"),
    ]
    mock_create.return_value = async_stream_factory(chunks)

    sentences = []
    async for s in llm.stream_sentences("नमस्ते", "hi"):
        sentences.append(s)

    assert len(sentences) == 3
    for s in sentences:
        assert "।" in s or s.endswith("।")


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

async def test_empty_input_handled(patched_llm, openai_chunk_factory, async_stream_factory):
    """Empty user_input does not crash; LLM is still invoked normally."""
    llm, mock_create = patched_llm
    mock_create.return_value = async_stream_factory(
        [openai_chunk_factory("I'm here to help with marketing.")]
    )

    response = await llm.generate_response("", "en")
    assert isinstance(response, str)
    assert response.strip() != ""


def test_validate_response_rejects_junk(llm):
    assert llm.validate_response("This is a real response") is True
    assert llm.validate_response("")        is False
    assert llm.validate_response("a")       is False
    assert llm.validate_response("error")   is False


# ─────────────────────────────────────────────────────────────────────────────
# Error handling — fallback path
# ─────────────────────────────────────────────────────────────────────────────

async def test_fallback_on_openai_connection_error(patched_llm):
    """APIConnectionError → fallback string, no exception propagation."""
    llm, mock_create = patched_llm

    def _raise(*args, **kwargs):
        raise openai.APIConnectionError(request=MagicMock())

    mock_create.side_effect = _raise

    response = await llm.generate_response("hello", "en")
    assert isinstance(response, str)
    assert response.strip() != ""
    # The English fallback contains "trouble" — see _FALLBACKS in llm_processor.py
    assert "trouble" in response.lower()


async def test_fallback_in_hindi_on_error(patched_llm):
    """Hindi conversation → Hindi fallback string on error."""
    llm, mock_create = patched_llm

    def _raise(*args, **kwargs):
        raise openai.APIConnectionError(request=MagicMock())

    mock_create.side_effect = _raise

    response = await llm.generate_response("नमस्ते", "hi")
    # Hindi fallback contains Devanagari
    assert any("ऀ" <= c <= "ॿ" for c in response)

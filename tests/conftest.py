"""
tests/conftest.py — shared pytest fixtures and helpers.

Notes
-----
* Adds the project root to sys.path so `from src.* import …` works regardless
  of the cwd where pytest is invoked.
* Provides factory helpers for fake Deepgram results and OpenAI streaming
  chunks so individual test files stay focused on assertions.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest

# ── Project root on sys.path ─────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Deepgram result factory  (used by test_stt.py)
# ─────────────────────────────────────────────────────────────────────────────

def make_dg_result(
    text: str,
    confidence: float = 0.95,
    language: str = "en",
    is_final: bool = True,
) -> SimpleNamespace:
    """
    Build a SimpleNamespace that quacks like a Deepgram LiveTranscription
    result object so it can be fed directly into
    DeepgramSTTEngine._on_transcript_event.
    """
    alternative = SimpleNamespace(transcript=text, confidence=confidence)
    channel     = SimpleNamespace(alternatives=[alternative])
    return SimpleNamespace(
        channel=channel,
        language=language,
        is_final=is_final,
    )


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI / Groq streaming chunk factory  (used by test_llm.py)
# ─────────────────────────────────────────────────────────────────────────────

def make_openai_chunk(content: str) -> SimpleNamespace:
    """Build one streaming chunk shaped like openai's ChatCompletionChunk."""
    delta  = SimpleNamespace(content=content)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice])


class AsyncStream:
    """
    Wraps a list of chunks in an async iterator so it can stand in for
    the object returned by `await client.chat.completions.create(stream=True)`.
    """

    def __init__(self, chunks: List[Any]) -> None:
        self._chunks = list(chunks)
        self._idx    = 0

    def __aiter__(self) -> "AsyncStream":
        return self

    async def __anext__(self) -> Any:
        if self._idx >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._idx]
        self._idx += 1
        return chunk


# ─────────────────────────────────────────────────────────────────────────────
# Pytest fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def dg_result_factory():
    """Returns the make_dg_result helper as a fixture."""
    return make_dg_result


@pytest.fixture
def openai_chunk_factory():
    """Returns the make_openai_chunk helper as a fixture."""
    return make_openai_chunk


@pytest.fixture
def async_stream_factory():
    """Returns the AsyncStream class as a fixture."""
    return AsyncStream

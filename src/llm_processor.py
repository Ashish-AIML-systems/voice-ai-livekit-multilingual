import asyncio
import logging
import logging.handlers
import re
from pathlib import Path
from typing import AsyncGenerator, Optional

import openai
from openai import AsyncOpenAI

from config.settings import (
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
    LOG_DIR,
    LOG_LEVEL,
    LLM_MAX_RETRIES,
    MARKETING_PROMPT_FILE,
    MAX_HISTORY_TURNS,
    MAX_RESPONSE_TOKENS,
)

# ---------------------------------------------------------------------------
# Logging — writes to logs/llm.log in addition to the root logger
# ---------------------------------------------------------------------------

LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger(__name__)
if not logger.handlers:
    fh = logging.handlers.RotatingFileHandler(
        LOG_DIR / "llm.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(fh)
logger.setLevel(LOG_LEVEL)

# ---------------------------------------------------------------------------
# Sentence boundary detection — includes Hindi danda (।)
# ---------------------------------------------------------------------------

_SENTENCE_END_RE = re.compile(r'(?<=[.!?।])\s+')

# ---------------------------------------------------------------------------
# Canned fallback responses (when LLM is unreachable)
# ---------------------------------------------------------------------------

_FALLBACKS = {
    "en":       "I'm sorry, I'm having a bit of trouble right now. Please try again in a moment.",
    "hi":       "माफ़ करें, अभी कनेक्शन में समस्या है। कृपया एक पल बाद फिर कोशिश करें।",
    "hinglish": "Sorry yaar, abhi thodi connection problem hai. Ek second mein try karein.",
}


class LLMProcessor:
    """
    LLM processor using Groq (OpenAI-compatible API).

    Key features
    ------------
    * Streams response token-by-token; yields complete sentences for low-latency TTS.
    * Prepends the marketing system prompt on every call.
    * Caps conversation history to MAX_HISTORY_TURNS pairs.
    * Retries on RateLimitError with exponential back-off.
    * Returns language-appropriate canned fallback on connection/timeout errors.
    * Logs all errors to logs/llm.log.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = MAX_RESPONSE_TOKENS,
        prompt_file: Optional[str] = None,
        max_retries: int = LLM_MAX_RETRIES,
    ):
        self._client = AsyncOpenAI(
            api_key=api_key or GROQ_API_KEY,
            base_url=GROQ_BASE_URL,
        )
        self._model = model or GROQ_MODEL
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._system_prompt = self._load_system_prompt(prompt_file or MARKETING_PROMPT_FILE)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_response(
        self,
        user_input: str,
        language: str = "en",
        scope: str = "marketing",
        history: Optional[list] = None,
    ) -> str:
        """Generate a full LLM response as a single string."""
        parts = []
        async for sentence in self.stream_sentences(user_input, language, scope, history):
            parts.append(sentence)
        return " ".join(parts)

    async def stream_sentences(
        self,
        user_input: str,
        language: str = "en",
        scope: str = "marketing",
        history: Optional[list] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Yield complete sentences as they stream from the LLM.
        This allows the TTS engine to start speaking the first sentence
        before the full response has been generated.
        """
        messages = self._build_messages(user_input, language, history or [])
        buffer = ""
        attempt = 0

        while attempt <= self._max_retries:
            try:
                stream = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=self._max_tokens,
                    stream=True,
                    temperature=0.7,
                )

                async for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if not delta:
                        continue
                    buffer += delta

                    # Yield each complete sentence immediately
                    sentences = _SENTENCE_END_RE.split(buffer)
                    for sentence in sentences[:-1]:
                        sentence = sentence.strip()
                        if sentence:
                            yield sentence
                    buffer = sentences[-1]  # keep incomplete tail

                # yield any remaining text after the stream closes
                if buffer.strip():
                    yield buffer.strip()
                return

            except openai.RateLimitError as exc:
                attempt += 1
                wait = 2 ** attempt
                logger.error("RateLimitError (attempt %d/%d): %s", attempt, self._max_retries, exc)
                if attempt > self._max_retries:
                    yield self._fallback(language)
                    return
                await asyncio.sleep(wait)

            except openai.APIConnectionError as exc:
                logger.error("APIConnectionError: %s", exc)
                yield self._fallback(language)
                return

            except openai.APITimeoutError as exc:
                logger.error("APITimeoutError: %s", exc)
                yield self._fallback(language)
                return

            except Exception as exc:
                logger.error("Unexpected LLM error: %s", exc, exc_info=True)
                yield self._fallback(language)
                return

    def build_prompt(self, user_input: str, language: str, scope: str) -> str:
        """Build the user turn content string with language context tag."""
        lang_map = {"en": "English", "hi": "Hindi", "hinglish": "Hinglish"}
        lang_name = lang_map.get(language, "English")
        return f"[Language: {lang_name}]\n{user_input}"

    def validate_response(self, response: str) -> bool:
        """Sanity-check: non-empty, not suspiciously short, no error markers."""
        if not response or len(response.strip()) < 5:
            return False
        junk = {"error", "exception", "null", "undefined", "none", ""}
        return response.strip().lower() not in junk

    def trim_history(self, history: list, max_turns: int = MAX_HISTORY_TURNS) -> list:
        """
        Keep only the last max_turns user+assistant pairs to prevent token overflow.
        Each turn = 1 user message + 1 assistant message = 2 list entries.
        """
        cutoff = max_turns * 2
        return history[-cutoff:] if len(history) > cutoff else history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(self, user_input: str, language: str, history: list) -> list:
        lang_map = {"en": "English", "hi": "Hindi", "hinglish": "Hinglish"}
        lang_name = lang_map.get(language, "English")

        messages = [{"role": "system", "content": self._system_prompt}]
        messages.extend(history)
        messages.append({
            "role": "user",
            "content": f"[Language: {lang_name}]\n{user_input}",
        })
        return messages

    def _fallback(self, language: str) -> str:
        return _FALLBACKS.get(language, _FALLBACKS["en"])

    @staticmethod
    def _load_system_prompt(path: str) -> str:
        try:
            return Path(path).read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            logger.error("System prompt file not found: %s", path)
            return "You are a helpful marketing assistant for an Indian brand."

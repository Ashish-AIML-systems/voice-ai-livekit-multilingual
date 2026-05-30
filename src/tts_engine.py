import asyncio
import io
import logging
import logging.handlers
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import miniaudio
from gtts import gTTS

from config.settings import (
    LOG_DIR,
    LOG_LEVEL,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger(__name__)
if not logger.handlers:
    fh = logging.handlers.RotatingFileHandler(
        LOG_DIR / "tts.log", maxBytes=2_000_000, backupCount=2, encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(fh)
logger.setLevel(LOG_LEVEL)


# ---------------------------------------------------------------------------
# Language code map
# ---------------------------------------------------------------------------

# gTTS BCP-47 language codes per session language.
# Hinglish falls back to English — gTTS handles Romanised Hindi adequately.
LANG_CODE_MAP: dict[str, str] = {
    "en":       "en",
    "hi":       "hi",
    "hinglish": "en",
}


# ---------------------------------------------------------------------------
# Data classes  (unchanged — rest of pipeline depends on these)
# ---------------------------------------------------------------------------

@dataclass
class AudioChunk:
    """One streamed audio chunk from the TTS engine."""
    data: bytes
    is_final: bool = False


@dataclass
class AudioData:
    """Complete TTS result for non-streaming synthesis."""
    data: bytes
    language: str
    voice_id: str          # holds the lang code for gTTS (kept for API compat)
    character_count: int


# ---------------------------------------------------------------------------
# TTS Engine  (gTTS backend)
# ---------------------------------------------------------------------------

class TTSEngine:
    """
    Google Text-to-Speech (gTTS) engine with barge-in support.

    Capabilities
    ------------
    * Supports English, Hindi, and Hinglish (Hinglish routed via English).
    * gTTS is a blocking HTTP call — synthesis runs in a thread-pool executor
      so the asyncio event loop is never blocked.
    * Barge-in: stop_speaking() flips a flag checked after synthesis completes.
    * Returns empty bytes (silent) on synthesis failure — pipeline never crashes.
    * Character-count logging on every call for usage tracking.

    Integration
    -----------
    Wire stop_speaking() to LiveKitManager(on_barge_in=...) so the user
    interrupting the bot instantly halts TTS playback.
    """

    def __init__(self) -> None:
        self._lang_map = LANG_CODE_MAP

        # Barge-in state
        self._stop_flag = False
        self._is_speaking = False
        self._current_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lookup helper
    # ------------------------------------------------------------------

    def get_lang_code(self, language: str) -> str:
        """Return the gTTS BCP-47 language code for a session language."""
        return self._lang_map.get(language, "en")

    def is_speaking(self) -> bool:
        return self._is_speaking

    # ------------------------------------------------------------------
    # Blocking synthesis helper  (runs in executor)
    # ------------------------------------------------------------------

    def _gtts_synthesize(self, text: str, lang_code: str) -> bytes:
        """
        Synchronous gTTS call — always run via run_in_executor, never directly.

        Pipeline:
            gTTS HTTP  →  MP3 bytes  →  miniaudio.decode()
            →  16 kHz / 16-bit / mono raw PCM
            →  ready for LiveKit AudioFrame.

        Returns raw PCM bytes, or b"" on any failure.
        miniaudio is a self-contained C library (no ffmpeg needed).
        """
        try:
            # 1. Synthesise to MP3
            tts = gTTS(text=text, lang=lang_code, slow=False)
            mp3_buf = io.BytesIO()
            tts.write_to_fp(mp3_buf)
            mp3_bytes = mp3_buf.getvalue()

            # 2. Decode MP3 → 16 kHz / 16-bit signed / mono PCM
            decoded = miniaudio.decode(
                mp3_bytes,
                output_format=miniaudio.SampleFormat.SIGNED16,
                nchannels=1,
                sample_rate=16000,
            )
            return bytes(decoded.samples)

        except Exception as exc:
            logger.error("_gtts_synthesize failed: %s", exc, exc_info=True)
            return b""

    # ------------------------------------------------------------------
    # Synthesis — one-shot
    # ------------------------------------------------------------------

    async def synthesize_speech(self, text: str, language: str = "en") -> AudioData:
        """
        One-shot synthesis for short responses.
        Returns AudioData with the complete MP3 byte buffer plus metadata.
        """
        if not text.strip():
            return AudioData(data=b"", language=language,
                             voice_id="en", character_count=0)

        lang_code = self.get_lang_code(language)
        char_count = len(text)

        logger.info(
            "TTS one-shot — lang=%s code=%s chars=%d model=gtts",
            language, lang_code, char_count,
        )

        try:
            loop = asyncio.get_event_loop()
            audio_data = await loop.run_in_executor(
                None, self._gtts_synthesize, text, lang_code
            )
            return AudioData(
                data=audio_data,
                language=language,
                voice_id=lang_code,
                character_count=char_count,
            )

        except Exception as exc:
            logger.error("TTS one-shot failed: %s", exc, exc_info=True)
            return AudioData(
                data=b"",
                language=language,
                voice_id=lang_code,
                character_count=char_count,
            )

    # ------------------------------------------------------------------
    # Synthesis — streaming
    # ------------------------------------------------------------------

    async def stream_speech(
        self,
        text: str,
        language: str = "en",
    ) -> AsyncIterator[AudioChunk]:
        """
        Streaming-compatible synthesis.

        gTTS synthesises the full utterance in one HTTP round-trip, so we
        yield a single AudioChunk with is_final=True.  The executor call
        keeps the event loop free during the network request, and the
        stop_flag is checked both before and after synthesis for minimal
        barge-in latency.
        """
        if not text.strip():
            yield AudioChunk(data=b"", is_final=True)
            return

        if self._stop_flag:
            return

        lang_code = self.get_lang_code(language)
        char_count = len(text)

        self._stop_flag = False
        self._is_speaking = True

        logger.info(
            "TTS stream — lang=%s code=%s chars=%d model=gtts",
            language, lang_code, char_count,
        )

        try:
            loop = asyncio.get_event_loop()
            audio_data = await loop.run_in_executor(
                None, self._gtts_synthesize, text, lang_code
            )

            # Barge-in check after synthesis (user may have spoken during HTTP call)
            if self._stop_flag:
                logger.info("TTS stream halted by barge-in after synthesis (%d chars)", char_count)
                yield AudioChunk(data=b"", is_final=True)
                return

            if audio_data:
                yield AudioChunk(data=audio_data, is_final=True)
            else:
                yield AudioChunk(data=b"", is_final=True)

        except asyncio.CancelledError:
            logger.info("TTS stream cancelled")
            raise

        except Exception as exc:
            logger.error("TTS stream failed: %s", exc, exc_info=True)
            yield AudioChunk(data=b"", is_final=True)

        finally:
            self._is_speaking = False
            self._stop_flag = False

    # ------------------------------------------------------------------
    # Barge-in control
    # ------------------------------------------------------------------

    def stop_speaking(self) -> None:
        """
        Signal the running stream to halt at its next chunk boundary.
        Cancels the current TTS task if one is tracked.
        Safe to call when no synthesis is active (idempotent).
        """
        if not self._is_speaking:
            return
        self._stop_flag = True
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
        logger.debug("stop_speaking() invoked — barge-in flag set")

    def track_task(self, task: asyncio.Task) -> None:
        """
        Register the asyncio task running the stream so stop_speaking()
        can hard-cancel it if the user barges in.
        """
        self._current_task = task

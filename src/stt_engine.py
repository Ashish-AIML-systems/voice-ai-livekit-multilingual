import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveOptions,
    LiveTranscriptionEvents,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
CHANNELS = 1
CONFIDENCE_THRESHOLD = 0.80       # discard finals below this and ask user to repeat
UTTERANCE_END_MS = "1000"         # ms of silence before end-of-turn fires
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_BASE_DELAY = 1.0        # exponential back-off base (seconds)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class TranscriptType(Enum):
    INTERIM = "interim"
    FINAL = "final"
    END_OF_TURN = "end_of_turn"


@dataclass
class TranscriptResult:
    text: str
    confidence: float
    language: str           # e.g. "en", "hi" — detected by Deepgram
    is_final: bool
    type: TranscriptType
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# STT Engine
# ---------------------------------------------------------------------------

class DeepgramSTTEngine:
    """
    Streaming Speech-to-Text engine backed by Deepgram.

    Key capabilities
    ----------------
    * Accepts raw PCM (16 kHz / 16-bit / mono) or Opus audio from LiveKit.
    * Streams over WebSocket — no buffering, no REST round-trips.
    * Multi-language mode with automatic language detection (English + Hindi
      + Hinglish).  Deepgram returns Devanagari for Hindi by default.
    * VAD-gated: audio is only forwarded while speech is detected, saving
      credits and reducing latency.
    * End-of-turn detection via Deepgram UtteranceEnd event.
    * Confidence threshold with configurable low-confidence callback so the
      bot can ask the user to repeat instead of hallucinating.
    * Auto-reconnect with exponential back-off on network drops.

    Usage
    -----
    engine = DeepgramSTTEngine(
        on_transcript=handle_transcript,
        on_end_of_turn=handle_eot,
        on_low_confidence=ask_to_repeat,
    )
    await engine.connect()
    await engine.send_audio(pcm_bytes)   # call in your LiveKit audio loop
    await engine.disconnect()
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        on_transcript: Optional[Callable[[TranscriptResult], None]] = None,
        on_end_of_turn: Optional[Callable[[], None]] = None,
        on_low_confidence: Optional[Callable[[], None]] = None,
    ):
        self._api_key = api_key or os.environ["DEEPGRAM_API_KEY"]
        self._confidence_threshold = confidence_threshold

        # Callbacks wired by the caller (e.g. ConversationManager)
        self._on_transcript = on_transcript
        self._on_end_of_turn = on_end_of_turn
        self._on_low_confidence = on_low_confidence

        self._client: Optional[DeepgramClient] = None
        self._connection = None
        self._connected = False
        self._reconnect_attempts = 0

        # VAD gate — primary source is LiveKitManager.VADProcessor (webrtcvad).
        # Deepgram SpeechStarted/UtteranceEnd act as a second confirming layer.
        self._vad_active = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the Deepgram WebSocket.  Must be called before sending audio."""
        dg_options = DeepgramClientOptions(options={"keepalive": "true"})
        self._client = DeepgramClient(self._api_key, dg_options)
        await self._open_connection()

    async def disconnect(self) -> None:
        """Cleanly close the WebSocket."""
        if self._connection:
            await self._connection.finish()
            self._connected = False
            logger.info("Deepgram STT disconnected")

    def set_vad_active(self, active: bool) -> None:
        """
        Gate audio forwarding via external VAD (e.g. LiveKit built-in VAD).
        When False, audio chunks are silently dropped — no credits consumed.
        """
        self._vad_active = active

    async def send_audio(self, pcm_bytes: bytes) -> None:
        """
        Forward a raw PCM chunk (16 kHz / 16-bit / mono) to Deepgram.
        Chunks sent while VAD is inactive are dropped.
        """
        if not self._vad_active:
            return
        if not self._connected:
            logger.warning("STT not connected — dropping audio chunk")
            return
        try:
            await self._connection.send(pcm_bytes)
        except Exception as exc:
            logger.error("Audio send error: %s — triggering reconnect", exc)
            await self._reconnect()

    async def send_opus_audio(self, opus_bytes: bytes) -> None:
        """
        Decode an Opus frame from LiveKit then forward as PCM.
        LiveKit audio tracks arrive as Opus — this handles the conversion.
        """
        pcm = self._opus_to_pcm(opus_bytes)
        await self.send_audio(pcm)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _open_connection(self) -> None:
        live_options = LiveOptions(
            # nova-3 supports end-of-utterance detection and multi-language
            model="nova-3",
            # "multi" enables automatic per-utterance language detection
            # (detect_language is implicit when language="multi" in current SDK)
            language="multi",
            # Returns partial results while user is still speaking
            interim_results=True,
            # Fires UtteranceEnd after this many ms of silence
            utterance_end_ms=UTTERANCE_END_MS,
            # Deepgram VAD events (SpeechStarted / SpeechFinished)
            vad_events=True,
            punctuate=True,
            smart_format=True,
            # Raw PCM input format
            encoding="linear16",
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
        )

        self._connection = self._client.listen.asyncwebsocket.v("1")

        self._connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript_event)
        self._connection.on(LiveTranscriptionEvents.UtteranceEnd, self._on_utterance_end_event)
        self._connection.on(LiveTranscriptionEvents.SpeechStarted, self._on_speech_started_event)
        self._connection.on(LiveTranscriptionEvents.Error, self._on_error_event)
        self._connection.on(LiveTranscriptionEvents.Close, self._on_close_event)

        ok = await self._connection.start(live_options)
        if not ok:
            raise RuntimeError("Failed to open Deepgram WebSocket")

        self._connected = True
        self._reconnect_attempts = 0
        logger.info("Deepgram STT connected — model=nova-3, language=multi")

    async def _reconnect(self) -> None:
        if self._reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
            logger.error("Max reconnect attempts (%d) reached", MAX_RECONNECT_ATTEMPTS)
            return

        self._connected = False
        delay = RECONNECT_BASE_DELAY * (2 ** self._reconnect_attempts)
        self._reconnect_attempts += 1

        logger.info(
            "Reconnecting to Deepgram in %.1fs (attempt %d/%d)",
            delay, self._reconnect_attempts, MAX_RECONNECT_ATTEMPTS,
        )
        await asyncio.sleep(delay)
        try:
            await self._open_connection()
        except Exception as exc:
            logger.error("Reconnect failed: %s", exc)

    # ------------------------------------------------------------------
    # Deepgram event handlers
    # ------------------------------------------------------------------

    async def _on_transcript_event(self, *args, **kwargs) -> None:
        result = kwargs.get("result") or (args[1] if len(args) > 1 else None)
        if result is None:
            return

        try:
            alternatives = result.channel.alternatives
            if not alternatives:
                return

            best = alternatives[0]
            text: str = best.transcript.strip()
            confidence: float = float(best.confidence)
            # Deepgram populates result.language when detect_language=True
            language: str = getattr(result, "language", "en") or "en"
            is_final: bool = bool(result.is_final)

            if not text:
                return

            # Low-confidence guard — only applied to final results so we don't
            # discard useful interim context during partial speech
            if is_final and confidence < self._confidence_threshold:
                logger.debug(
                    "Low-confidence final (%.2f < %.2f): '%s'",
                    confidence, self._confidence_threshold, text,
                )
                if self._on_low_confidence:
                    try:
                        self._on_low_confidence()
                    except Exception as exc:
                        logger.error("on_low_confidence raised: %s", exc, exc_info=True)
                return

            transcript = TranscriptResult(
                text=text,
                confidence=confidence,
                language=language,
                is_final=is_final,
                type=TranscriptType.FINAL if is_final else TranscriptType.INTERIM,
            )

            if self._on_transcript:
                try:
                    self._on_transcript(transcript)
                except Exception as exc:
                    logger.error("on_transcript raised: %s", exc, exc_info=True)

        except Exception as exc:
            logger.error("Transcript handler error: %s", exc, exc_info=True)

    async def _on_utterance_end_event(self, *args, **kwargs) -> None:
        """
        Fires after UTTERANCE_END_MS of silence — reliable end-of-turn signal
        that avoids cutting the user off mid-sentence.

        Wrapped in try/except so a misbehaving on_end_of_turn callback can't
        kill the Deepgram WebSocket event loop.
        """
        logger.debug("UtteranceEnd: user finished speaking")
        self._vad_active = False
        if self._on_end_of_turn:
            try:
                self._on_end_of_turn()
            except Exception as exc:
                logger.error("on_end_of_turn callback raised: %s", exc, exc_info=True)

    async def _on_speech_started_event(self, *args, **kwargs) -> None:
        """Deepgram detected voice energy — open the VAD gate."""
        logger.debug("SpeechStarted: voice detected")
        self._vad_active = True

    async def _on_error_event(self, *args, **kwargs) -> None:
        error = kwargs.get("error") or (args[1] if len(args) > 1 else None)
        logger.error("Deepgram error event: %s", error)
        await self._reconnect()

    async def _on_close_event(self, *args, **kwargs) -> None:
        logger.warning("Deepgram connection closed unexpectedly")
        self._connected = False
        await self._reconnect()

    # ------------------------------------------------------------------
    # Audio conversion — Opus (LiveKit) → PCM (Deepgram)
    # ------------------------------------------------------------------

    @staticmethod
    def _opus_to_pcm(opus_bytes: bytes) -> bytes:
        """
        Decode Opus frames from LiveKit to 16 kHz / 16-bit / mono PCM.

        Requires 'opuslib' (pip install opuslib) and libopus native library.
        If opuslib is not installed the bytes pass through unchanged — this is
        fine when livekit-agents pre-decodes the audio track before handing it
        to the STT engine (the recommended integration pattern).
        """
        try:
            import opuslib  # type: ignore

            decoder = opuslib.Decoder(SAMPLE_RATE, CHANNELS)
            # 960 samples @ 16 kHz = 60 ms Opus frame
            return decoder.decode(opus_bytes, frame_size=960)
        except ImportError:
            logger.debug("opuslib not available — treating input as PCM")
            return opus_bytes
        except Exception as exc:
            logger.warning("Opus decode failed (%s) — passing raw bytes", exc)
            return opus_bytes


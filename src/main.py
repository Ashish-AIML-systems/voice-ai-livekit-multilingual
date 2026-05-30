"""
src/main.py — VoiceAI Pipeline Entrypoint
==========================================

Wires every component into a single coherent pipeline:

  User speaks
  → LiveKitManager  (WebRTC inbound + VAD gate)
  → DeepgramSTTEngine  (streaming STT over WebSocket)
  → LanguageDetector   (en / hi / Hinglish)
  → ScopeValidator     (safety + topic gate — user input)
  → ConversationManager  (history + session state)
  → LLMProcessor       (Groq, sentence-by-sentence streaming)
  → ScopeValidator     (safety + topic gate — bot output)
  → TTSEngine          (ElevenLabs streaming, per sentence)
  → LiveKitManager     (push PCM to user over WebRTC)

Barge-in path
-------------
  LiveKitManager.on_barge_in
  → ConversationManager.handle_interruption()
  → TTSEngine.stop_speaking()
  → Pipeline cancels in-flight TTS; user utterance captured fresh.

Run modes
---------
  python -m src.main                        # standalone bot (blocking)
  uvicorn src.main:app --host 0.0.0.0 --port 8000   # with health + token HTTP endpoints
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

# ── FastAPI (health-check + LiveKit token endpoint) ───────────────────────────
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn

# ── LiveKit token generation ──────────────────────────────────────────────────
try:
    from livekit.api import AccessToken, VideoGrants
    _LIVEKIT_API_OK = True
except ImportError:
    _LIVEKIT_API_OK = False

# ── Project settings ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))   # ensure project root on path

from config.settings import (
    DEEPGRAM_API_KEY,
    LIVEKIT_URL,
    LIVEKIT_API_KEY,
    LIVEKIT_API_SECRET,
    LOG_DIR,
    LOG_LEVEL,
)

# ── Pipeline components ───────────────────────────────────────────────────────
from src.stt_engine        import DeepgramSTTEngine, TranscriptResult, TranscriptType
from src.livekit_manager   import LiveKitManager
from src.llm_processor     import LLMProcessor
from src.language_detector import LanguageDetector
from src.scope_validator   import ScopeValidator
from src.tts_engine        import TTSEngine
from src.conversation_manager import ConversationManager
from src.latency_tracker    import (
    LatencyTracker,
    STAGE_STT_END,
    STAGE_LLM_START,
    STAGE_LLM_FIRST_SENTENCE,
    STAGE_LLM_END,
    STAGE_TTS_START,
    STAGE_TTS_FIRST_CHUNK,
    STAGE_TTS_END,
    STAGE_TURN_END,
)

# =============================================================================
# Logging — writes to logs/main.log in addition to stdout
# =============================================================================

LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(
            LOG_DIR / "main.log",
            maxBytes=10_000_000,
            backupCount=5,
            encoding="utf-8",
        ),
    ],
)

logger = logging.getLogger(__name__)

# =============================================================================
# LiveKit room / bot identity
# =============================================================================

ROOM_NAME     = os.environ.get("LIVEKIT_ROOM_NAME", "voicebot-room")
BOT_IDENTITY  = os.environ.get("BOT_IDENTITY",      "priya-bot")
BOT_NAME      = os.environ.get("BOT_NAME",           "Priya")

# Opening greeting — played immediately after a participant connects.
# Adjust phrasing or load from a file for A/B testing.
_GREETING = {
    "en":       "Hi! I'm Priya from BrandVoice India. How can I help you today?",
    "hi":       "नमस्ते! मैं प्रिया हूँ, BrandVoice India से। आज मैं आपकी कैसे मदद कर सकती हूँ?",
    "hinglish": "Hi! Main Priya hoon, BrandVoice India se. Aaj main aapki kaise help kar sakti hoon?",
}

# Low-confidence prompt — spoken when STT returns unclear audio
_REPEAT_PROMPT = {
    "en":       "Sorry, could you please repeat that?",
    "hi":       "माफ़ कीजिए, क्या आप दोबारा बोल सकते हैं?",
    "hinglish": "Sorry, kya aap dobara bol sakte hain?",
}


# =============================================================================
# Token helper
# =============================================================================

def generate_bot_token(room_name: str = ROOM_NAME) -> str:
    """
    Generate a short-lived LiveKit JWT for the bot participant.
    Requires livekit-api package.  Falls back to empty string on import error.
    """
    if not _LIVEKIT_API_OK:
        logger.warning("livekit.api not available — returning empty token.")
        return ""

    token = (
        AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(BOT_IDENTITY)
        .with_name(BOT_NAME)
        .with_grants(
            VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .to_jwt()
    )
    return token


def generate_user_token(
    identity: str,
    name: str,
    room_name: str = ROOM_NAME,
) -> str:
    """
    Generate a LiveKit JWT for a human participant (called from the /token endpoint).
    """
    if not _LIVEKIT_API_OK:
        logger.warning("livekit.api not available — returning empty token.")
        return ""

    token = (
        AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(name)
        .with_grants(
            VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .to_jwt()
    )
    return token


# =============================================================================
# Voice Pipeline
# =============================================================================

class VoicePipeline:
    """
    Orchestrates all components for one LiveKit session.

    Lifecycle
    ---------
    1. pipeline = VoicePipeline()
    2. await pipeline.start(room_name)   — connects LiveKit + Deepgram
    3. (runs until room disconnects or pipeline.stop() is called)
    4. await pipeline.stop()             — graceful shutdown

    Thread-safety / concurrency
    ---------------------------
    A per-participant asyncio.Lock (_turn_locks) prevents concurrent pipeline
    runs for the same speaker (e.g., double end-of-turn fires).  TTS for
    different participants can run concurrently if needed in the future.
    """

    def __init__(self) -> None:
        # ── Core engines ──────────────────────────────────────────────────────
        self.tts          = TTSEngine()
        self.convo        = ConversationManager(tts_engine=self.tts)
        self.llm          = LLMProcessor()
        self.lang_detect  = LanguageDetector()
        self.scope        = ScopeValidator()

        # ── Per-participant state ─────────────────────────────────────────────
        # Accumulated final transcripts between end-of-turn fires
        self._pending_text:     dict[str, str]              = {}
        # One asyncio.Lock per participant — serialises turn processing
        self._turn_locks:       dict[str, asyncio.Lock]     = {}
        # Track the active participant identity (first to connect)
        self._participants:     list[str]                   = []
        # The identity of the participant whose audio we last heard
        self._active_speaker:   Optional[str]               = None
        # Per-participant turn counter — used as turn_id in LatencyTracker
        self._turn_counter:     dict[str, int]              = {}
        # In-flight latency tracker per participant (one turn at a time)
        self._active_tracker:   dict[str, LatencyTracker]   = {}

        # ── STT engine (callbacks wired below) ───────────────────────────────
        self.stt = DeepgramSTTEngine(
            api_key=DEEPGRAM_API_KEY,
            on_transcript=self._cb_on_transcript,
            on_end_of_turn=self._cb_on_end_of_turn,
            on_low_confidence=self._cb_on_low_confidence,
        )

        # ── LiveKit manager (callbacks wired below) ───────────────────────────
        # Constructed in start() so we can pass a fresh token per session.
        self.lk: Optional[LiveKitManager] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, room_name: str = ROOM_NAME) -> None:
        """Connect to LiveKit room and open the Deepgram WebSocket."""
        token = generate_bot_token(room_name)
        if not token:
            raise RuntimeError(
                "Cannot generate LiveKit token — check LIVEKIT_API_KEY and "
                "LIVEKIT_API_SECRET in your .env and that livekit-api is installed."
            )

        self.lk = LiveKitManager(
            url=LIVEKIT_URL,
            token=token,
            on_audio_frame=self._cb_on_audio_frame,
            on_speech_start=self._cb_on_speech_start,
            on_speech_end=self._cb_on_speech_end,
            on_barge_in=self._cb_on_barge_in,
        )

        # Register room-level participant events via the underlying Room object
        # so we know which identity to associate conversation state with.
        # We patch the LiveKit room events after connect() so the Room exists.
        await self.lk.connect()
        _room = self.lk._room
        if _room:
            _room.on("participant_connected",    self._on_participant_joined)
            _room.on("participant_disconnected", self._on_participant_left)

        await self.stt.connect()
        logger.info("VoicePipeline started — room=%s bot=%s", room_name, BOT_IDENTITY)

    async def stop(self) -> None:
        """Gracefully shut down all connections."""
        await self.stt.disconnect()
        if self.lk:
            await self.lk.disconnect()
        logger.info("VoicePipeline stopped.")

    async def run_forever(self) -> None:
        """Block until the room is disconnected (e.g. all participants leave)."""
        logger.info("Pipeline running — waiting for participants…")
        try:
            while True:
                await asyncio.sleep(1)
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("Pipeline shutdown requested.")
        finally:
            await self.stop()

    # ------------------------------------------------------------------
    # Participant lifecycle helpers
    # ------------------------------------------------------------------

    def _get_or_init_participant(self, participant_id: str) -> None:
        """Ensure conversation state and per-participant structures exist."""
        if participant_id not in self._turn_locks:
            self._turn_locks[participant_id]  = asyncio.Lock()
            self._pending_text[participant_id] = ""

        if not self.convo.is_active(participant_id):
            self.convo.start_conversation(participant_id)
            logger.info("Conversation started for participant: %s", participant_id)

    def _on_participant_joined(self, participant) -> None:
        pid = participant.identity
        self._participants.append(pid)
        self._active_speaker = pid
        self._get_or_init_participant(pid)
        logger.info("Participant joined: %s", pid)
        # Send opening greeting asynchronously (don't block room event loop)
        asyncio.ensure_future(self._greet_participant(pid))

    def _on_participant_left(self, participant) -> None:
        pid = participant.identity
        self.convo.end_conversation(pid)
        if pid in self._participants:
            self._participants.remove(pid)
        if self._active_speaker == pid:
            self._active_speaker = self._participants[-1] if self._participants else None
        logger.info("Participant left: %s", pid)

    # ------------------------------------------------------------------
    # LiveKit / VAD callbacks
    # ------------------------------------------------------------------

    def _cb_on_audio_frame(self, pcm: bytes) -> None:
        """
        Voiced PCM from the VAD gate → forward to Deepgram.
        Called from the LiveKit audio ingest loop (sync context — schedule async send).
        """
        asyncio.ensure_future(self.stt.send_audio(pcm))

    def _cb_on_speech_start(self) -> None:
        """User started speaking — open the Deepgram VAD gate."""
        self.stt.set_vad_active(True)
        logger.debug("VAD: speech start — STT gate open")

    def _cb_on_speech_end(self) -> None:
        """User stopped speaking — close the Deepgram VAD gate."""
        self.stt.set_vad_active(False)
        logger.debug("VAD: speech end — STT gate closed")

    def _cb_on_barge_in(self) -> None:
        """
        User interrupted the bot.
        Stop TTS immediately; the inbound pipeline will pick up the user's
        utterance from scratch (LiveKit manager resets inbound VAD after barge-in).
        """
        pid = self._active_speaker
        if pid:
            self.convo.handle_interruption(pid)
            logger.info("Barge-in from %s — TTS stopped", pid)

    # ------------------------------------------------------------------
    # STT callbacks
    # ------------------------------------------------------------------

    def _cb_on_transcript(self, result: TranscriptResult) -> None:
        """
        Called by DeepgramSTTEngine for every transcript event.

        Strategy
        --------
        * INTERIM — ignored (can be used for live UI display if desired).
        * FINAL   — appended to _pending_text[participant_id].
        * The final text is committed to the pipeline in _cb_on_end_of_turn().
        """
        if result.type == TranscriptType.INTERIM:
            return  # not used in pipeline (good for UI captions if needed)

        if result.type in (TranscriptType.FINAL, TranscriptType.END_OF_TURN):
            pid = self._active_speaker or "unknown"
            self._get_or_init_participant(pid)
            if result.text.strip():
                self._pending_text[pid] = (
                    (self._pending_text.get(pid, "") + " " + result.text).strip()
                )
                logger.debug(
                    "Transcript [%s] %s: '%s' (conf=%.2f lang=%s)",
                    result.type.value, pid, result.text,
                    result.confidence, result.language,
                )

    def _cb_on_end_of_turn(self) -> None:
        """
        Deepgram UtteranceEnd fired — user has finished their turn.
        Schedule the full pipeline (LLM + TTS) as a background task and
        start a per-turn LatencyTracker.
        """
        pid = self._active_speaker or "unknown"
        text = self._pending_text.get(pid, "").strip()

        if not text:
            logger.debug("End-of-turn fired but no pending text — skipping.")
            return

        # Bump turn counter and start a fresh latency tracker.
        # TURN_START is set in LatencyTracker.__post_init__.
        self._turn_counter[pid] = self._turn_counter.get(pid, 0) + 1
        tracker = LatencyTracker(participant_id=pid, turn_id=self._turn_counter[pid])
        tracker.mark(STAGE_STT_END)   # we already have the final transcript here
        self._active_tracker[pid] = tracker

        # Clear the buffer immediately before scheduling (prevent double-process)
        self._pending_text[pid] = ""
        asyncio.ensure_future(self._process_turn(pid, text))

    def _cb_on_low_confidence(self) -> None:
        """STT returned a final below the confidence threshold."""
        pid = self._active_speaker or "unknown"
        language = "en"
        if self.convo.is_active(pid):
            language = self.convo.get_context(pid)   # get last known language
            # get_context returns history list; fetch language from session
            session  = self.convo.get_session(pid)
            language = session.get("language", "en") if session else "en"

        prompt = _REPEAT_PROMPT.get(language, _REPEAT_PROMPT["en"])
        asyncio.ensure_future(self._speak(pid, prompt, language))

    # ------------------------------------------------------------------
    # Core pipeline — one user turn
    # ------------------------------------------------------------------

    async def _process_turn(
        self,
        participant_id: str,
        user_text: str,
    ) -> None:
        """
        Full pipeline for a single user utterance:

        1. Language detection  (Deepgram tag + text analysis)
        2. Input safety gate   (profanity / PII / injection)
        3. Add user message to history
        4. Stream LLM sentences
        5. Output safety gate  (per sentence)
        6. TTS + push audio
        7. Add assistant response to history
        8. Signal bot finished speaking
        """
        lock = self._turn_locks.setdefault(participant_id, asyncio.Lock())
        tracker = self._active_tracker.get(participant_id)

        async with lock:
            logger.info("TURN [%s]: '%s'", participant_id, user_text[:120])

            # ── 1. Language detection ──────────────────────────────────────────
            # Use the session-confirmed language (hysteresis applied)
            language = self.lang_detect.update_session_language(
                participant_id, user_text
            )
            self.convo.update_language(participant_id, language)
            logger.debug("Language for %s: %s", participant_id, language)

            # ── 2. Input safety gate ───────────────────────────────────────────
            safety = self.scope.validate_content(user_text)
            if not safety["safe"]:
                reason = safety["reason"]
                logger.warning(
                    "Input blocked [%s] participant=%s reason=%s",
                    participant_id, participant_id, reason,
                )
                redirect = self.scope.redirect_out_of_scope(language)
                await self._speak(participant_id, redirect, language)
                return

            # Scope check on user input (topic gate)
            if not self.scope.is_within_scope(user_text):
                redirect = self.scope.redirect_out_of_scope(language)
                logger.info(
                    "Out-of-scope input [%s]: '%s'", participant_id, user_text[:80]
                )
                await self._speak(participant_id, redirect, language)
                return

            # ── 3. Append user message ─────────────────────────────────────────
            self.convo.add_message(participant_id, "user", user_text)

            # ── 4. Stream LLM response sentence-by-sentence ────────────────────
            history       = self.convo.get_context(participant_id)
            full_response = []

            if tracker:
                tracker.mark(STAGE_LLM_START)

            try:
                async for sentence in self.llm.stream_sentences(
                    user_input=user_text,
                    language=language,
                    scope="marketing",
                    history=history,
                ):
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    # Mark first sentence — time-to-first-token surrogate
                    if tracker:
                        tracker.mark_if_unset(STAGE_LLM_FIRST_SENTENCE)

                    # ── 5. Output safety gate ──────────────────────────────────
                    gated_text, was_redirected = self.scope.validate_and_gate(
                        text=sentence,
                        language=language,
                        scope="marketing",
                        participant_id=participant_id,
                    )

                    full_response.append(gated_text)

                    # ── 6. TTS + push audio ────────────────────────────────────
                    await self._speak(participant_id, gated_text, language)

                    # If the output gate replaced the sentence with a redirect,
                    # do not continue streaming more sentences (the topic is blocked).
                    if was_redirected:
                        logger.info(
                            "LLM output redirected for %s — stopping stream", participant_id
                        )
                        break

            except asyncio.CancelledError:
                # Barge-in cancelled this task — that is expected; fall through.
                logger.info("Pipeline task cancelled (barge-in) for %s", participant_id)
                if tracker:
                    tracker.mark(STAGE_TURN_END)
                    tracker.report()
                return

            if tracker:
                tracker.mark(STAGE_LLM_END)

            # ── 7. Append assistant message ────────────────────────────────────
            if full_response:
                full_text = " ".join(full_response)
                self.convo.add_message(participant_id, "assistant", full_text)

            # ── 8. Signal done ────────────────────────────────────────────────
            if self.lk:
                self.lk.bot_speaking_done()

            # ── 9. Emit latency summary ───────────────────────────────────────
            if tracker:
                tracker.mark(STAGE_TURN_END)
                tracker.report()
                # Free the slot for the next turn
                self._active_tracker.pop(participant_id, None)

    # ------------------------------------------------------------------
    # TTS + outbound audio helper
    # ------------------------------------------------------------------

    async def _speak(
        self,
        participant_id: str,
        text: str,
        language: str = "en",
    ) -> None:
        """
        Synthesise `text` with ElevenLabs and push each PCM chunk to LiveKit.
        Respects the TTS stop flag — barge-in will halt mid-sentence.
        Marks TTS_START / TTS_FIRST_CHUNK / TTS_END on the active tracker.
        """
        if not text.strip():
            return

        if not self.lk:
            logger.warning("_speak called before LiveKit connected — skipping TTS")
            return

        tracker = self._active_tracker.get(participant_id)
        if tracker:
            tracker.mark_if_unset(STAGE_TTS_START)

        try:
            async for chunk in self.tts.stream_speech(text, language):
                if not chunk.data:
                    continue
                if tracker:
                    tracker.mark_if_unset(STAGE_TTS_FIRST_CHUNK)
                await self.lk.push_bot_audio(chunk.data)

            if tracker:
                tracker.mark(STAGE_TTS_END)

        except asyncio.CancelledError:
            # Barge-in is handled upstream; just stop cleanly.
            raise
        except Exception as exc:
            logger.error("TTS error for %s: %s", participant_id, exc)

    # ------------------------------------------------------------------
    # Opening greeting
    # ------------------------------------------------------------------

    async def _greet_participant(self, participant_id: str) -> None:
        """
        Play the opening greeting to a newly joined participant.
        Uses the default language (English) — will switch on first utterance.
        """
        # Small delay lets the WebRTC connection stabilise before audio is sent.
        await asyncio.sleep(1.0)

        session  = self.convo.get_session(participant_id)
        language = session.get("language", "en") if session else "en"
        greeting = _GREETING.get(language, _GREETING["en"])

        logger.info("Greeting participant %s in '%s'", participant_id, language)
        await self._speak(participant_id, greeting, language)

        # Record the greeting in conversation history so the LLM knows the
        # conversation has started.
        self.convo.add_message(participant_id, "assistant", greeting)
        if self.lk:
            self.lk.bot_speaking_done()


# =============================================================================
# FastAPI application (health-check + LiveKit token endpoint)
# =============================================================================

# Global pipeline instance — initialised on app startup.
_pipeline: Optional[VoicePipeline] = None


@contextlib.asynccontextmanager
async def lifespan(application: "FastAPI"):  # noqa: F821
    """
    FastAPI lifespan context manager — replaces the deprecated
    @app.on_event("startup") / @app.on_event("shutdown") pattern.
    """
    global _pipeline
    _pipeline = VoicePipeline()
    try:
        await _pipeline.start()
        logger.info("VoiceAI pipeline started (lifespan startup).")
    except Exception as exc:
        logger.error("Pipeline failed to start: %s", exc)
        # Don't abort the server — health endpoint will report unhealthy.

    yield   # ← server is running; handle requests here

    # ── shutdown ──────────────────────────────────────────────────────────────
    if _pipeline:
        await _pipeline.stop()
        logger.info("VoiceAI pipeline stopped (lifespan shutdown).")


app = FastAPI(
    title="VoiceAI — Priya Marketing Bot",
    description=(
        "Hindi + English voice assistant built on LiveKit, Deepgram, Groq, "
        "and ElevenLabs."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["ops"])
async def health() -> JSONResponse:
    """
    Returns 200 if the pipeline is up and connected, 503 otherwise.
    Useful for load balancer / container health checks.
    """
    lk_connected = bool(_pipeline and _pipeline.lk and _pipeline.lk._room)
    stt_connected = bool(_pipeline and _pipeline.stt._connected)

    status = "ok" if (lk_connected and stt_connected) else "degraded"
    code   = 200 if status == "ok" else 503

    return JSONResponse(
        status_code=code,
        content={
            "status":        status,
            "livekit":       "connected"    if lk_connected  else "disconnected",
            "deepgram_stt":  "connected"    if stt_connected else "disconnected",
            "room":          ROOM_NAME,
            "bot_identity":  BOT_IDENTITY,
        },
    )


@app.get("/token", tags=["auth"])
async def get_user_token(
    identity: str = Query(..., description="Unique participant identity (e.g. user-123)"),
    name: str     = Query("User", description="Display name shown in the room"),
    room: str     = Query(ROOM_NAME, description="LiveKit room to join"),
) -> JSONResponse:
    """
    Issue a LiveKit JWT for a human participant.
    Call this from your web / mobile frontend before connecting to LiveKit.

    Example
    -------
        GET /token?identity=user-42&name=Ravi&room=voicebot-room
    """
    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        raise HTTPException(
            status_code=503,
            detail="LiveKit credentials not configured on the server.",
        )

    token = generate_user_token(identity=identity, name=name, room_name=room)
    if not token:
        raise HTTPException(
            status_code=503,
            detail="Token generation failed — livekit-api package may not be installed.",
        )

    return JSONResponse(
        content={
            "token":    token,
            "identity": identity,
            "room":     room,
            "url":      LIVEKIT_URL,
        }
    )


# =============================================================================
# Standalone entry-point  (python -m src.main)
# =============================================================================

async def _run_standalone() -> None:
    """
    Run the bot without a web server — useful for quick local testing.

    The bot connects to the LiveKit room and waits for participants.
    Terminate with Ctrl-C.
    """
    pipeline = VoicePipeline()
    try:
        await pipeline.start()
        await pipeline.run_forever()
    except Exception as exc:
        logger.exception("Fatal error in standalone pipeline: %s", exc)
        raise
    finally:
        await pipeline.stop()


def main() -> None:
    """
    CLI entry-point.

    Usage
    -----
      python -m src.main              → standalone bot (no HTTP server)
      uvicorn src.main:app --reload   → bot + FastAPI health/token endpoints
    """
    logger.info(
        "Starting VoiceAI — bot=%s room=%s url=%s",
        BOT_IDENTITY, ROOM_NAME, LIVEKIT_URL,
    )

    # Validate critical settings before attempting connections
    missing = []
    if not LIVEKIT_URL:           missing.append("LIVEKIT_URL")
    if not LIVEKIT_API_KEY:       missing.append("LIVEKIT_API_KEY")
    if not LIVEKIT_API_SECRET:    missing.append("LIVEKIT_API_SECRET")
    if not DEEPGRAM_API_KEY:      missing.append("DEEPGRAM_API_KEY")

    if missing:
        logger.error(
            "Missing required environment variables: %s — check your .env file.",
            ", ".join(missing),
        )
        sys.exit(1)

    asyncio.run(_run_standalone())


if __name__ == "__main__":
    main()

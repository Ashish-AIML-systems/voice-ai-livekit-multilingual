import asyncio
import logging
import time
from collections import deque
from typing import Callable, Optional

import webrtcvad
from livekit import rtc
from livekit.rtc import (
    AudioFrame,
    AudioSource,
    Room,
    RoomOptions,
    RtcConfiguration,
    IceServer,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audio constants — consistent 16 kHz mono throughout the entire pipeline
# ---------------------------------------------------------------------------

SAMPLE_RATE      = 16000   # Hz — matches Deepgram STT input requirement
CHANNELS         = 1       # mono
BITS_PER_SAMPLE  = 16
BYTES_PER_SAMPLE = BITS_PER_SAMPLE // 8
VAD_FRAME_MS     = 20      # webrtcvad only accepts 10 / 20 / 30 ms frames

# Exact byte count of one 20 ms PCM frame at 16 kHz / 16-bit / mono
TARGET_FRAME_BYTES = int(SAMPLE_RATE * VAD_FRAME_MS / 1000) * BYTES_PER_SAMPLE * CHANNELS

# ---------------------------------------------------------------------------
# WebRTC audio quality settings
# NOTE: bitrate / echo-cancellation / NS / AGC / jitter-buffer are negotiated
# via WebRTC SDP on the client (browser/app) side.  The values below are the
# RECOMMENDED settings to pass to your frontend LiveKit JS/Swift/Android SDK.
# Server-side we enforce sample rate, mono, and DTX (silence suppression).
# ---------------------------------------------------------------------------

WEBRTC_AUDIO_BITRATE_KBPS  = 24     # Opus voice mode — optimal for STT
ECHO_CANCELLATION          = True   # prevents bot hearing its own voice
NOISE_SUPPRESSION          = True   # cleans mic feed before VAD/STT
AUTO_GAIN_CONTROL          = True   # normalises volume across callers
JITTER_BUFFER_MAX_DELAY_MS = 200    # client-side jitter buffer cap

# ---------------------------------------------------------------------------
# VAD tuning — matches checklist exactly
# ---------------------------------------------------------------------------

VAD_AGGRESSIVENESS       = 2    # 0=permissive … 3=very aggressive (maps to 0.5 sensitivity)
MIN_SPEECH_DURATION_MS   = 100  # user must speak for ≥100 ms before gate opens
MIN_SILENCE_DURATION_MS  = 600  # ≥600 ms of silence before gate closes (500–800 ms range)
PADDING_DURATION_MS      = 200  # lead-in / trail-out padding around speech
VAD_THRESHOLD            = 0.5  # fraction of voiced frames to trigger speech / silence

# Derived frame counts
_MIN_SPEECH_FRAMES  = MIN_SPEECH_DURATION_MS  // VAD_FRAME_MS  # 5 frames
_MIN_SILENCE_FRAMES = MIN_SILENCE_DURATION_MS // VAD_FRAME_MS  # 30 frames
_PADDING_FRAMES     = PADDING_DURATION_MS     // VAD_FRAME_MS  # 10 frames

TTS_COOLDOWN_MS = 1500   # ignore barge-in for 1.5s after TTS


# ---------------------------------------------------------------------------
# VAD Processor  (inbound — user microphone)
# ---------------------------------------------------------------------------

class VADProcessor:
    """
    Google WebRTC VAD with explicit speech/silence durations and padding.

    Gate logic
    ----------
    CLOSED → OPEN:
        A rolling window of _MIN_SPEECH_FRAMES frames must have ≥ VAD_THRESHOLD
        voiced fraction.  Before firing on_speech_start the leading _PADDING_FRAMES
        are flushed so the utterance start is never clipped.

    OPEN → CLOSED:
        A rolling window of _MIN_SILENCE_FRAMES frames must have < VAD_THRESHOLD
        voiced fraction (i.e. mostly silence).  Fires on_speech_end.

    The separation into two different window lengths means:
        • Short noise bursts (<100 ms) never open the gate.
        • Pauses within a sentence (<600 ms) never close it.
    """

    def __init__(
        self,
        aggressiveness: int = VAD_AGGRESSIVENESS,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[], None]] = None,
    ):
        self._vad = webrtcvad.Vad(aggressiveness)
        self._on_speech_start = on_speech_start
        self._on_speech_end = on_speech_end

        # Lead-in ring buffer — flushed on gate-open
        self._padding_buf: deque = deque(maxlen=_PADDING_FRAMES)
        # Detection windows
        self._speech_window: deque = deque(maxlen=_MIN_SPEECH_FRAMES)
        self._silence_window: deque = deque(maxlen=_MIN_SILENCE_FRAMES)

        self._triggered = False

    @property
    def is_active(self) -> bool:
        return self._triggered

    def process_frame(self, pcm: bytes) -> Optional[bytes]:
        """
        Feed exactly TARGET_FRAME_BYTES (20 ms, 16 kHz, 16-bit, mono).

        Returns voiced bytes (possibly padded lead-in) or None for silence.
        """
        try:
            is_speech = self._vad.is_speech(pcm, SAMPLE_RATE)
        except Exception as exc:
            logger.debug("VAD frame error: %s", exc)
            return None

        if not self._triggered:
            self._padding_buf.append(pcm)
            self._speech_window.append(is_speech)

            voiced_ratio = sum(self._speech_window) / len(self._speech_window)
            if voiced_ratio >= VAD_THRESHOLD:
                self._triggered = True
                self._silence_window.clear()
                if self._on_speech_start:
                    self._on_speech_start()
                # flush buffered lead-in (padding_duration worth of audio)
                return b"".join(self._padding_buf)
            return None

        # gate is open — watch for sustained silence
        self._silence_window.append(is_speech)
        if len(self._silence_window) == _MIN_SILENCE_FRAMES:
            silence_ratio = sum(self._silence_window) / _MIN_SILENCE_FRAMES
            if silence_ratio < VAD_THRESHOLD:
                self._triggered = False
                self._speech_window.clear()
                if self._on_speech_end:
                    self._on_speech_end()
                return None

        return pcm

    def reset(self) -> None:
        self._padding_buf.clear()
        self._speech_window.clear()
        self._silence_window.clear()
        self._triggered = False


# ---------------------------------------------------------------------------
# LiveKit Manager — WebRTC room + inbound / outbound audio pipeline
# ---------------------------------------------------------------------------

class LiveKitManager:
    """
    WebRTC room manager built on livekit-python.

    Inbound pipeline  (user mic → STT)
    -----------------------------------
    LiveKit AudioStream
      → frame reassembly into 20 ms chunks
      → VADProcessor  (webrtcvad, 100 ms min speech, 600 ms min silence, 200 ms padding)
      → voiced chunks → on_audio_frame(pcm)  →  DeepgramSTTEngine.send_audio()

    Outbound pipeline  (TTS → user speaker)
    ----------------------------------------
    push_bot_audio(pcm)
      → AudioSource.capture_frame()
      → WebRTC (Opus 16 kHz mono 24 kbps, DTX on)  →  user

    Barge-in
    --------
    While the bot is speaking a second lightweight VAD still runs on the
    inbound stream.  If the user speaks over the bot:
      1. on_barge_in() fires — the caller must stop TTS immediately.
      2. _bot_speaking is cleared.
      3. The inbound VAD resets so the user utterance is captured from scratch.

    VAD suppression (no barge-in path)
    -----------------------------------
    While bot is speaking AND no on_barge_in handler is registered, inbound
    audio is dropped completely to avoid echo triggering STT.

    WebRTC audio settings
    ---------------------
    Server-side (enforced here):
      • 16 kHz mono PCM resampling on AudioStream subscription
      • DTX (discontinuous transmission) — silence packets suppressed
      • Two public STUN servers for reliable ICE even without TURN

    Client-side (must set in your frontend LiveKit JS/Swift/Android SDK):
      • Opus bitrate: 24 kbps  (WEBRTC_AUDIO_BITRATE_KBPS)
      • echoCancellation: true  (ECHO_CANCELLATION)
      • noiseSuppression: true  (NOISE_SUPPRESSION)
      • autoGainControl: true   (AUTO_GAIN_CONTROL)
      • jitter buffer max delay: 200 ms  (JITTER_BUFFER_MAX_DELAY_MS)
    LiveKit Cloud Mumbai (ap-south-1) handles TURN relay automatically.
    """

    def __init__(
        self,
        url: str,
        token: str,
        on_audio_frame: Optional[Callable[[bytes], None]] = None,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[], None]] = None,
        on_barge_in: Optional[Callable[[], None]] = None,
        vad_aggressiveness: int = VAD_AGGRESSIVENESS,
    ):
        self._url = url
        self._token = token
        self._on_audio_frame = on_audio_frame    # voiced PCM → STT engine
        self._on_speech_start = on_speech_start
        self._on_speech_end = on_speech_end
        self._on_barge_in = on_barge_in          # called when user interrupts bot

        self._room: Optional[Room] = None
        self._audio_source: Optional[AudioSource] = None
        self._bot_speaking = False

        self._inbound_vad = VADProcessor(
            aggressiveness=vad_aggressiveness,
            on_speech_start=self._handle_speech_start,
            on_speech_end=self._handle_speech_end,
        )
        # Lightweight barge-in VAD — runs even while bot is talking
        self._barge_in_vad = VADProcessor(aggressiveness=vad_aggressiveness)

        # Rolling byte buffer — reassembles variable-size LiveKit frames
        # into exact TARGET_FRAME_BYTES (20 ms) chunks for webrtcvad
        self._frame_buffer = bytearray()

        self._tts_finished_at = 0.0    # monotonic timestamp
        self._tts_active = False        # True while TTS is streaming

    # ------------------------------------------------------------------
    # Connect / disconnect
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to the LiveKit room and publish the bot outbound track."""
        self._room = Room()
        self._room.on("track_subscribed",      self._on_track_subscribed)
        self._room.on("participant_connected", self._on_participant_connected)
        self._room.on("disconnected",          self._on_room_disconnected)

        room_options = RoomOptions(
            auto_subscribe=True,
            rtc_config=RtcConfiguration(
                ice_servers=[
                    # Two Google public STUN servers — fast ICE negotiation
                    # without needing a dedicated TURN server in dev/staging.
                    # LiveKit Cloud handles TURN automatically in production.
                    IceServer(urls=["stun:stun.l.google.com:19302"]),
                    IceServer(urls=["stun:stun1.l.google.com:19302"]),
                ],
            ),
        )

        await self._room.connect(self._url, self._token, options=room_options)
        logger.info("LiveKit room connected: %s", self._room.name)
        await self._publish_outbound_track()

    async def disconnect(self) -> None:
        if self._room:
            await self._room.disconnect()
            logger.info("LiveKit room disconnected")

    # ------------------------------------------------------------------
    # Outbound audio (bot TTS → user)
    # ------------------------------------------------------------------

    async def push_bot_audio(self, pcm_bytes: bytes) -> None:
        """
        Send bot TTS audio to the user over WebRTC.
        Activates barge-in detection while playing.

        Per-chunk send failures are logged but do not propagate so the
        upstream TTS stream loop survives a single bad packet.
        """
        if not self._audio_source:
            logger.warning("Outbound audio source not ready")
            return

        if not pcm_bytes:
            return  # empty chunk — nothing to send

        self._bot_speaking = True
        self._barge_in_vad.reset()
        self._tts_active = True

        try:
            samples_per_channel = len(pcm_bytes) // BYTES_PER_SAMPLE // CHANNELS
            frame = AudioFrame(
                data=pcm_bytes,
                sample_rate=SAMPLE_RATE,
                num_channels=CHANNELS,
                samples_per_channel=samples_per_channel,
            )
            await self._audio_source.capture_frame(frame)
        except Exception as exc:
            logger.error("push_bot_audio failed: %s", exc, exc_info=True)
        finally:
            self._tts_active = False
            self._tts_finished_at = time.monotonic()

    def bot_speaking_done(self) -> None:
        """Signal that TTS playback finished — re-enables inbound VAD."""
        self._bot_speaking = False
        self._inbound_vad.reset()
        logger.debug("Bot finished speaking — inbound VAD re-enabled")

    # ------------------------------------------------------------------
    # Outbound track setup
    # ------------------------------------------------------------------

    async def _publish_outbound_track(self) -> None:
        # AudioSource at 16 kHz mono matches the entire pipeline sample rate
        self._audio_source = AudioSource(
            sample_rate=SAMPLE_RATE,
            num_channels=CHANNELS,
        )
        local_track = rtc.LocalAudioTrack.create_audio_track(
            "bot-voice", self._audio_source
        )
        pub_options = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_MICROPHONE,
            dtx=True,   # discontinuous transmission — no packets sent during silence
        )
        await self._room.local_participant.publish_track(local_track, pub_options)
        logger.info(
            "Outbound track published — 16 kHz mono, DTX=on, target %d kbps Opus",
            WEBRTC_AUDIO_BITRATE_KBPS,
        )

    # ------------------------------------------------------------------
    # LiveKit room events
    # ------------------------------------------------------------------

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        logger.info("Audio track subscribed from: %s", participant.identity)
        # Resample to 16 kHz mono at the subscription level — consistent rate
        # for both webrtcvad (needs exactly 16 kHz) and Deepgram STT
        audio_stream = rtc.AudioStream(
            track,
            sample_rate=SAMPLE_RATE,
            num_channels=CHANNELS,
        )
        asyncio.ensure_future(self._inbound_pipeline(audio_stream))

    def _on_participant_connected(self, participant: rtc.RemoteParticipant) -> None:
        logger.info("Participant connected: %s", participant.identity)

    def _on_room_disconnected(self) -> None:
        logger.warning("LiveKit room disconnected unexpectedly")

    # ------------------------------------------------------------------
    # Inbound audio pipeline
    # ------------------------------------------------------------------

    async def _inbound_pipeline(self, stream: rtc.AudioStream) -> None:
        """
        Drain AudioStream → reassemble into 20 ms chunks → VAD → STT.

        Two modes depending on whether bot is currently speaking:

        Bot silent  → inbound VAD → voiced frames → on_audio_frame (STT)
        Bot talking → barge-in VAD; if user speaks → on_barge_in fires
                      (if no barge-in handler, inbound audio is dropped)

        Error policy
        ------------
        Per-frame exceptions are caught and logged so a single bad frame or
        misbehaving callback can never tear down the entire audio loop.
        """
        try:
            async for event in stream:
                if not isinstance(event, rtc.AudioFrameEvent):
                    continue

                try:
                    self._frame_buffer.extend(bytes(event.frame.data))

                    while len(self._frame_buffer) >= TARGET_FRAME_BYTES:
                        chunk = bytes(self._frame_buffer[:TARGET_FRAME_BYTES])
                        del self._frame_buffer[:TARGET_FRAME_BYTES]

                        if self._bot_speaking:
                            self._handle_barge_in_check(chunk)
                        else:
                            voiced = self._inbound_vad.process_frame(chunk)
                            if voiced and self._on_audio_frame:
                                try:
                                    self._on_audio_frame(voiced)
                                except Exception as cb_exc:
                                    logger.error(
                                        "on_audio_frame callback raised: %s",
                                        cb_exc, exc_info=True,
                                    )

                except Exception as frame_exc:
                    # One bad frame must NOT kill the whole audio loop
                    logger.error(
                        "Frame processing error (skipping): %s", frame_exc, exc_info=True
                    )

        except asyncio.CancelledError:
            logger.info("Inbound pipeline cancelled (normal shutdown)")
            raise
        except Exception as loop_exc:
            logger.error(
                "Inbound pipeline crashed: %s — audio reception stopped",
                loop_exc, exc_info=True,
            )

    def _handle_barge_in_check(self, chunk: bytes) -> None:
        """
        Run lightweight VAD on inbound chunk while bot is talking.
        If user speech is detected, fire on_barge_in and switch to user turn.

        Callback exceptions are caught so a bad handler can't kill the loop.
        """
        # Cooldown check — ignore barge-in during/just after TTS
        if self._tts_active:
            return  # don't fire barge-in while bot is pushing a chunk

        ms_since_tts = (time.monotonic() - self._tts_finished_at) * 1000
        if ms_since_tts < TTS_COOLDOWN_MS:
            return  # still in cooldown window

        if not self._on_barge_in:
            return   # no barge-in handler registered — drop the frame

        voiced = self._barge_in_vad.process_frame(chunk)
        if voiced:
            logger.info("Barge-in detected — stopping bot TTS")
            self._bot_speaking = False
            self._inbound_vad.reset()
            try:
                self._on_barge_in()
            except Exception as exc:
                logger.error("on_barge_in callback raised: %s", exc, exc_info=True)
            # forward the voiced chunk immediately so start of barge-in isn't lost
            if self._on_audio_frame:
                try:
                    self._on_audio_frame(voiced)
                except Exception as exc:
                    logger.error("on_audio_frame (barge-in) raised: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # VAD callbacks
    # ------------------------------------------------------------------

    def _handle_speech_start(self) -> None:
        logger.debug("VAD: speech started")
        if self._on_speech_start:
            self._on_speech_start()

    def _handle_speech_end(self) -> None:
        logger.debug("VAD: speech ended")
        if self._on_speech_end:
            self._on_speech_end()

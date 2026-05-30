# Architecture

## Overview

The voicebot is a fully async Python pipeline that connects a user's browser microphone to an AI conversation loop and streams synthesised audio back in real time. All components talk through the asyncio event loop — no threads except for the gTTS blocking HTTP call, which runs in a thread-pool executor.

---

## Component Map

```
┌─────────────────────────────────────────────────────────────┐
│                        src/main.py                          │
│                       VoicePipeline                         │
│   Wires every component; owns participant lifecycle         │
└──┬──────────┬──────────┬──────────┬──────────┬─────────────┘
   │          │          │          │          │
   ▼          ▼          ▼          ▼          ▼
LiveKit   Deepgram    Groq LLM   gTTS      Scope /
Manager   STT Engine  Processor  Engine    Convo Mgr
```

### LiveKitManager (`src/livekit_manager.py`)

Owns the WebRTC room and both audio directions.

**Inbound pipeline (user mic → STT)**

```
LiveKit AudioStream
  → bytearray reassembly → 20 ms PCM chunks (TARGET_FRAME_BYTES = 640 bytes)
  → VADProcessor (webrtcvad aggressiveness=2)
      CLOSED→OPEN:  ≥50% voiced frames in a 100 ms window
      OPEN→CLOSED:  <50% voiced frames in a 600 ms window
      Lead-in pad:  200 ms buffered before gate-open (no clipped words)
  → voiced chunks → on_audio_frame(pcm) → DeepgramSTTEngine.send_audio()
```

**Outbound pipeline (TTS → user speaker)**

```
push_bot_audio(pcm_bytes)
  → sets _tts_active = True
  → AudioSource.capture_frame()
  → WebRTC Opus 16 kHz mono 24 kbps, DTX on → user speaker
  → finally: _tts_active = False, _tts_finished_at = time.monotonic()
```

**Barge-in detection**

While `_bot_speaking` is True a second `VADProcessor` (`_barge_in_vad`) runs on every inbound chunk. Before it can fire `on_barge_in()` two guards must pass:

1. `_tts_active` must be False (chunk is not being pushed right now).
2. `(time.monotonic() - _tts_finished_at) * 1000 >= TTS_COOLDOWN_MS` (1500 ms).

The 1500 ms cooldown absorbs echo: gTTS audio played through the user's speaker travels back over the microphone; without the cooldown the barge-in VAD would fire on the bot's own voice.

---

### DeepgramSTTEngine (`src/stt_engine.py`)

Streaming WebSocket connection to Deepgram Nova-2.

- Audio format: 16 kHz / 16-bit / mono raw PCM
- `utterance_end_ms = 1000` — fires `on_end_of_turn` after 1 second of silence
- Confidence threshold: 0.80 — finals below this trigger `on_low_confidence` instead
- Reconnect: exponential back-off, up to 5 attempts
- Language: `multi` mode — Deepgram returns `en` or `hi` per word

---

### LanguageDetector (`src/language_detector.py`)

Two-stage detection with session hysteresis:

1. Deepgram's per-word language tag (fast, reliable for clean speech).
2. `langdetect` on the transcript text (fallback for noisy Deepgram tags).

Session hysteresis: a language switch is only confirmed after 2 consecutive turns in the new language, preventing single-word code-switching from flipping the bot's persona.

---

### LLMProcessor (`src/llm_processor.py`)

Streams Groq responses sentence-by-sentence so TTS can start before the full reply is generated.

- Provider: Groq (OpenAI-compatible endpoint, `https://api.groq.com/openai/v1`)
- Model: `openai/gpt-oss-120b` (configurable via `GROQ_MODEL`)
- Max output tokens: 500 per turn
- History window: last 10 turns (sliding window in ConversationManager)
- Retries: 3 attempts on transient errors

---

### TTSEngine (`src/tts_engine.py`)

Google Text-to-Speech (gTTS) with miniaudio for format conversion.

```
gTTS HTTP call (blocking, run in executor)
  → MP3 bytes
  → miniaudio.decode(output_format=SIGNED16, nchannels=1, sample_rate=16000)
  → raw 16 kHz / 16-bit / mono PCM
  → yielded as AudioChunk(data=..., is_final=True)
```

Language routing:

| Session language | gTTS code |
| :--- | :--- |
| `en` | `en` |
| `hi` | `hi` |
| `hinglish` | `en` (Romanised Hindi works acceptably with the English model) |

Barge-in: `stop_speaking()` sets `_stop_flag = True`. The flag is checked after synthesis returns from the executor — if the user interrupted during the HTTP call, the chunk is discarded without being pushed to LiveKit.

---

### ScopeValidator (`src/scope_validator.py`)

Two-pass gate applied to both user input and LLM output:

1. **Safety check** — profanity, PII patterns, prompt-injection signatures.
2. **Topic gate** — compares against `config/scenarios/marketing_config.yaml` allow/deny lists.

Out-of-scope responses are replaced with a language-appropriate redirection phrase. The LLM stream is halted for that turn so no further sentences are synthesised.

---

### ConversationManager (`src/conversation_manager.py`)

Per-participant session state:

- Sliding window: last 8 turns sent to LLM (configurable via `CONVO_SLIDING_WINDOW_TURNS`).
- Stores: role, text, language, timestamp per message.
- `handle_interruption()`: called on barge-in — stops TTS via `TTSEngine.stop_speaking()`.

---

### LatencyTracker (`src/latency_tracker.py`)

Records a monotonic timestamp at each pipeline stage per turn:

```
TURN_START → STT_END → LLM_START → LLM_FIRST_SENTENCE → LLM_END
→ TTS_START → TTS_FIRST_CHUNK → TTS_END → TURN_END
```

Logs a `WARNING` if any stage exceeds its target. Full summary written to `logs/latency.log` after every turn.

---

## Audio Format — End-to-End

Every component in the pipeline uses **16 kHz / 16-bit signed / mono PCM**. There is no resampling between components.

| Boundary | Format |
| :--- | :--- |
| LiveKit inbound subscription | 16 kHz mono (resampled at subscription) |
| webrtcvad input | 16 kHz / 16-bit / mono, 20 ms frames (640 bytes) |
| Deepgram input | 16 kHz / 16-bit / mono raw PCM |
| gTTS output (after miniaudio decode) | 16 kHz / 16-bit signed / mono PCM |
| LiveKit AudioFrame | 16 kHz mono |

---

## WebRTC Settings

**Server-side (enforced in code)**

- Sample rate: 16 kHz mono
- DTX: enabled (no packets sent during silence)
- STUN: `stun.l.google.com:19302` + `stun1.l.google.com:19302`
- LiveKit Cloud Mumbai (ap-south-1) provides TURN automatically

**Client-side (set in your LiveKit JS/Swift/Android SDK)**

- `audioBitrate`: 24 kbps (Opus voice mode)
- `echoCancellation`: true
- `noiseSuppression`: true
- `autoGainControl`: true
- `jitterBufferMaxDelay`: 200 ms

# Multilingual LiveKit Voicebot

A production-grade **Hindi + English AI voice agent** built on LiveKit WebRTC. The bot (Priya, BrandVoice India) conducts natural conversations, detects language mid-sentence, enforces scenario-specific topic boundaries, and streams synthesised audio back in under 2.5 seconds end-to-end.

---

## Technology Stack

| Layer | Technology | Notes |
| :--- | :--- | :--- |
| **Voice Transport** | LiveKit WebRTC (Cloud, Mumbai ap-south-1) | DTX on, 24 kbps Opus |
| **Speech-to-Text** | Deepgram streaming WebSocket | 16 kHz mono, confidence ≥ 0.80 |
| **Language Detection** | Deepgram tag + langdetect | en / hi / Hinglish with hysteresis |
| **LLM** | Groq (OpenAI-compatible API) | `openai/gpt-oss-120b`, max 500 tokens |
| **Text-to-Speech** | gTTS (Google TTS) | Free, no API key — MP3 decoded to PCM via miniaudio |
| **VAD** | webrtcvad (Google WebRTC VAD) | 100 ms min speech, 600 ms min silence |
| **Backend** | Python 3.10+ / FastAPI + uvicorn | Fully async |

---

## Architecture

```
User mic  ──WebRTC──►  LiveKit Room  ──PCM──►  VAD Gate
                                                   │ voiced frames
                                                   ▼
                                          Deepgram STT (WebSocket)
                                                   │ final transcript
                                                   ▼
                                         Language Detector
                                                   │
                                                   ▼
                                         Scope Validator (input)
                                                   │ in-scope
                                                   ▼
                                      Conversation Manager (history)
                                                   │
                                                   ▼
                                       LLM — Groq stream (sentences)
                                                   │
                                                   ▼
                                         Scope Validator (output)
                                                   │
                                                   ▼
                                       gTTS → miniaudio → PCM
                                                   │ PCM chunks
                                                   ▼
                                       LiveKit push ──WebRTC──► User speaker
```

### Barge-in + Echo Suppression

While the bot is speaking, a second lightweight VAD monitors the inbound stream. When the user interrupts:

1. `on_barge_in()` fires → TTS stops immediately.
2. Inbound VAD resets and captures the new utterance from scratch.

**Echo / false barge-in fix (1.5 s cooldown):** gTTS audio playing through the user's speaker can echo back over the microphone and trigger the barge-in VAD. To prevent this, `_handle_barge_in_check()` ignores any barge-in signals for **1500 ms** after each TTS chunk is pushed (`TTS_COOLDOWN_MS`). Adjust in `src/livekit_manager.py` if needed.

---

## Project Structure

```
voice_ai/
├── src/
│   ├── main.py                 # Entry-point — wires pipeline, FastAPI app
│   ├── livekit_manager.py      # WebRTC room, VAD, inbound/outbound audio
│   ├── stt_engine.py           # Deepgram streaming STT
│   ├── tts_engine.py           # gTTS synthesis + barge-in stop flag
│   ├── llm_processor.py        # Groq sentence-streaming
│   ├── language_detector.py    # en / hi / Hinglish with session hysteresis
│   ├── scope_validator.py      # Topic gate + profanity / PII filter
│   ├── conversation_manager.py # Sliding-window history, session state
│   └── latency_tracker.py      # Per-turn stage timing + log warnings
├── config/
│   ├── settings.py             # All env-var bindings in one place
│   ├── prompts/                # System prompts per scenario
│   └── scenarios/              # YAML topic configs (presale / sales / marketing)
├── tests/                      # pytest unit tests
├── logs/                       # Rotating log files (auto-created)
├── requirements.txt
├── Dockerfile
└── .env                        # Your credentials (never commit)
```

---

## Setup

### Prerequisites

- Python 3.10+
- A LiveKit Cloud account (free tier works) — get URL + API key + secret
- Deepgram account — get API key
- Groq account — get API key
- **No ElevenLabs key needed** — TTS now uses free gTTS

### Install

```bash
git clone <repo-url>
cd voice_ai
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### Configure `.env`

```env
# LiveKit
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=APIxxxxxxxxxx
LIVEKIT_API_SECRET=your-api-secret

# Deepgram
DEEPGRAM_API_KEY=your-deepgram-key

# Groq
GROQ_API_KEY=your-groq-key
GROQ_MODEL=openai/gpt-oss-120b

# Optional tuning
MAX_RESPONSE_TOKENS=500
MAX_HISTORY_TURNS=10
LOG_LEVEL=INFO
```

### Run

```bash
# Standalone bot (no HTTP server)
python -m src.main

# Bot + health/token HTTP endpoints
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Wait for `Pipeline running — waiting for participants…` then join via [meet.livekit.io](https://meet.livekit.io) using a token from the `/token` endpoint.

---

## HTTP Endpoints

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/health` | Returns 200 (ok) or 503 (degraded) |
| `GET` | `/token?identity=X&name=Y&room=Z` | Issues a LiveKit JWT for a human participant |

---

## Latency Targets

| Stage | Target |
| :--- | :--- |
| STT (utterance → transcript) | < 500 ms |
| LLM time-to-first-sentence | < 1 s |
| TTS first chunk ready | < 300 ms |
| End-to-end (user stops → bot starts) | < 2.5 s |

Latency is logged per turn in `logs/latency.log`. Stages that exceed their target emit a `WARNING`.

---

## Scenarios

Three pre-built conversation contexts, each with its own system prompt and topic config:

| Scenario | Config | Prompt |
| :--- | :--- | :--- |
| Marketing | `config/scenarios/marketing_config.yaml` | `config/prompts/marketing_system_prompt.txt` |
| Presale | `config/scenarios/presale_config.yaml` | `config/prompts/presale_system_prompt.txt` |
| Sales | `config/scenarios/sales_config.yaml` | `config/prompts/sales_system_prompt.txt` |

The active scenario is `marketing` (hardcoded in `src/main.py` `_process_turn`). Switch by changing the `scope=` argument.

---

## Tests

```bash
pytest tests/ -v
```

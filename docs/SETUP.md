# Setup Guide

## Prerequisites

| Requirement | Version / Notes |
| :--- | :--- |
| Python | 3.10 or higher |
| pip | Latest recommended |
| LiveKit Cloud account | Free tier works — get URL, API Key, API Secret |
| Deepgram account | Free tier: 12,000 min/year — get API Key |
| Groq account | Free tier available — get API Key |
| Internet access | gTTS makes HTTP calls to Google TTS; Deepgram uses WebSocket |

**No ElevenLabs account needed.** TTS was migrated to gTTS (Google Text-to-Speech) which is free and requires no API key.

---

## 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd voice_ai

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

---

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

Key packages installed:

| Package | Purpose |
| :--- | :--- |
| `livekit`, `livekit-api` | WebRTC room management + JWT token generation |
| `deepgram-sdk>=3.11.0,<4.0.0` | Streaming STT |
| `websockets>=10.0,<14.0` | WebSocket transport for Deepgram |
| `openai` | Groq LLM via OpenAI-compatible endpoint |
| `gTTS` | Google Text-to-Speech (free, no key) |
| `miniaudio` | Decode gTTS MP3 → 16 kHz PCM (pure C, no ffmpeg) |
| `langdetect` | Fallback language detection on transcript text |
| `fastapi`, `uvicorn` | HTTP health + token endpoints |
| `python-dotenv` | Load `.env` file |

> **Note:** `webrtcvad` is used at runtime but commented out in `requirements.txt`. If you get an `ImportError` on `webrtcvad`, install it manually:
> ```bash
> pip install webrtcvad
> ```
> On Windows you may need Visual C++ build tools first.

---

## 3. Create your `.env` file

Copy from `.env.example` if it exists, or create `.env` manually in the project root:

```env
# ── LiveKit ───────────────────────────────────────────────────────────────────
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=APIxxxxxxxxxx
LIVEKIT_API_SECRET=your-api-secret

# Optional — override defaults
LIVEKIT_ROOM_NAME=voicebot-room
BOT_IDENTITY=priya-bot
BOT_NAME=Priya

# ── Deepgram ──────────────────────────────────────────────────────────────────
DEEPGRAM_API_KEY=your-deepgram-key

# ── Groq LLM ──────────────────────────────────────────────────────────────────
GROQ_API_KEY=your-groq-key
GROQ_MODEL=openai/gpt-oss-120b
GROQ_BASE_URL=https://api.groq.com/openai/v1

# ── TTS (gTTS — no key needed) ────────────────────────────────────────────────
# Nothing to configure here. Language is detected per-turn automatically.

# ── Tuning (optional) ─────────────────────────────────────────────────────────
MAX_RESPONSE_TOKENS=500
MAX_HISTORY_TURNS=10
LLM_MAX_RETRIES=3
STT_CONFIDENCE_THRESHOLD=0.80
STT_UTTERANCE_END_MS=1000
CONVO_SLIDING_WINDOW_TURNS=8
LOG_LEVEL=INFO
```

> **Never commit `.env` to git.** It is already listed in `.gitignore`.

---

## 4. Get your credentials

### LiveKit

1. Sign up at [livekit.io](https://livekit.io) and create a project.
2. Go to **Settings → Keys**.
3. Copy `API Key`, `API Secret`, and the `WebSocket URL` (starts with `wss://`).

### Deepgram

1. Sign up at [deepgram.com](https://deepgram.com).
2. Go to **API Keys** and create a new key with `Member` permissions.

### Groq

1. Sign up at [console.groq.com](https://console.groq.com).
2. Go to **API Keys** → **Create API Key**.

---

## 5. Run the bot

### Standalone (no HTTP server)

```bash
python -m src.main
```

Wait for:

```
Pipeline running — waiting for participants…
```

### With FastAPI (health + token endpoints)

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

---

## 6. Join the room

1. Get a participant token:
   ```
   GET http://localhost:8000/token?identity=user-1&name=Ravi&room=voicebot-room
   ```
   The response includes `token` and `url`.

2. Open [meet.livekit.io](https://meet.livekit.io) and enter the `url` and `token`.

3. Allow microphone access and start speaking. The bot (Priya) will greet you within ~1 second of joining.

---

## 6. Run tests

```bash
pytest tests/ -v
```

---

## Scenario selection

The active scenario is `marketing`. To switch, open `src/main.py` and change the `scope=` argument in `_process_turn()`:

```python
async for sentence in self.llm.stream_sentences(
    user_input=user_text,
    language=language,
    scope="marketing",   # ← change to "presale" or "sales"
    history=history,
):
```

# Troubleshooting

## Bot does not respond / no audio

**Check 1 — Missing env vars**

```bash
python -m src.main
```

If you see `Missing required environment variables: LIVEKIT_URL, ...`, fill in your `.env` file. See [SETUP.md](SETUP.md).

**Check 2 — LiveKit connection**

```
curl http://localhost:8000/health
# "livekit":"disconnected"  → check LIVEKIT_URL / API key / secret
# "deepgram_stt":"disconnected"  → check DEEPGRAM_API_KEY
```

**Check 3 — Room name mismatch**

The bot joins `LIVEKIT_ROOM_NAME` (default: `voicebot-room`). Make sure your participant token targets the same room:

```
GET /token?identity=user-1&name=Test&room=voicebot-room
```

---

## Bot hears its own voice / false barge-ins

**Symptom:** Bot cuts itself off mid-sentence. Logs show repeated `Barge-in detected` lines even when no one is speaking.

**Cause:** gTTS audio playing through the user's speaker echoes back over the microphone and triggers the barge-in VAD.

**Fix (already applied):** A 1500 ms cooldown (`TTS_COOLDOWN_MS`) in `_handle_barge_in_check()` suppresses barge-in signals for 1.5 seconds after each TTS chunk.

**Tuning:**

```python
# src/livekit_manager.py, line ~62
TTS_COOLDOWN_MS = 1500   # increase to 2000–2500 if still seeing false barge-ins
                          # decrease to 1000 if real barge-in feels sluggish
```

Also ensure your LiveKit frontend SDK has **echo cancellation enabled**:

```js
// LiveKit JS SDK
room.connect(url, token, {
  audioCaptureDefaults: {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
  },
})
```

---

## High latency (> 3 seconds end-to-end)

Check `logs/latency.log` for `WARNING` lines — they tell you which stage is slow.

| Slow stage | Likely cause | Fix |
| :--- | :--- | :--- |
| STT_END | Deepgram WebSocket reconnecting | Check DEEPGRAM_API_KEY; look for reconnect logs |
| LLM_FIRST_SENTENCE | Groq API slow / rate-limited | Check Groq dashboard; reduce `MAX_RESPONSE_TOKENS` |
| TTS_FIRST_CHUNK | gTTS HTTP call slow | Network issue to Google; retry logic is built-in |
| TURN_END | Barge-in cooldown too long | Reduce `TTS_COOLDOWN_MS` |

gTTS makes a blocking HTTP call to Google's servers. If you are in a region with high latency to Google, consider switching back to ElevenLabs (uncomment in `requirements.txt` and restore `src/tts_engine.py`).

---

## `ImportError: No module named 'webrtcvad'`

`webrtcvad` is commented out in `requirements.txt` (see note in that file). Install manually:

```bash
pip install webrtcvad
```

On Windows, if you get a compiler error:

1. Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
2. Then `pip install webrtcvad`.

On Linux:

```bash
sudo apt-get install python3-dev
pip install webrtcvad
```

---

## `ImportError: No module named 'miniaudio'`

```bash
pip install miniaudio
```

miniaudio is a pure-C library and ships pre-compiled wheels for Windows, macOS, and Linux — no ffmpeg or external binary required.

---

## gTTS synthesis fails / silent audio

**Symptom:** Bot joins the room but plays no audio. `logs/tts.log` shows `_gtts_synthesize failed`.

**Causes:**

- No internet access to `translate.google.com` — check firewall rules.
- Google temporarily rate-limiting your IP (rare). Wait a few minutes or use a VPN.
- `miniaudio` decode error — check that the MP3 bytes are non-empty in the log.

**Quick test:**

```python
from gtts import gTTS
import io
tts = gTTS("Hello", lang="en")
buf = io.BytesIO()
tts.write_to_fp(buf)
print(len(buf.getvalue()), "bytes")   # should be > 1000
```

---

## Deepgram returns no transcripts

**Check:** Is the VAD gate opening? Look for `VAD: speech started` in logs. If not:

- The webrtcvad aggressiveness may be too high for your microphone. Lower it:
  ```python
  # src/livekit_manager.py
  VAD_AGGRESSIVENESS = 1   # 0=permissive … 3=very aggressive
  ```
- Mic gain may be too low — the VAD sees only silence. Use `noiseSuppression: false` temporarily to diagnose.

**Check:** Is audio reaching Deepgram? Set `LOG_LEVEL=DEBUG` and look for `send_audio` calls in logs.

---

## Groq LLM errors

**`AuthenticationError`** — wrong or missing `GROQ_API_KEY`.

**`RateLimitError`** — free tier quota hit. Wait or upgrade your Groq plan.

**Model not found** — the model name in `GROQ_MODEL` is invalid. Check [Groq's model list](https://console.groq.com/docs/models) and update `.env`.

---

## Participant joins but bot does not greet

The greeting fires 1 second after `participant_connected` fires (a `asyncio.sleep(1.0)` to let WebRTC stabilise). If you still hear nothing:

1. Check `logs/main.log` for `Greeting participant` — if missing, the event did not fire.
2. Verify the user's participant token has `can_subscribe: true` so the bot's track is received.
3. Check `push_bot_audio failed` in logs — the `AudioSource` may not have been published yet.

---

## Tests fail

```bash
pytest tests/ -v --tb=short
```

Most tests mock external services. If you see `ModuleNotFoundError`, ensure the venv is active and `pip install -r requirements.txt` has been run. For async test failures, check that `pytest-asyncio` is installed and `pytest.ini` contains `asyncio_mode = auto`.

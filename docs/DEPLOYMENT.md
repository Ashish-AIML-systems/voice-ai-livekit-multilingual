# Deployment

## Docker

A `Dockerfile` is included. Build and run:

```bash
# Build
docker build -t voicebot .

# Run (pass .env file)
docker run --rm --env-file .env -p 8000:8000 voicebot
```

The container starts `uvicorn src.main:app` on port 8000.

Health check:

```bash
curl http://localhost:8000/health
# {"status":"ok","livekit":"connected","deepgram_stt":"connected",...}
```

---

## Environment variables at runtime

All configuration is via environment variables (see [SETUP.md](SETUP.md) for the full list). In production, inject them via your platform's secrets manager rather than a `.env` file:

- AWS: Secrets Manager + ECS task definition environment
- GCP: Secret Manager + Cloud Run environment variables
- Railway / Render / Fly.io: Dashboard environment section

---

## Required outbound network access

| Destination | Port | Purpose |
| :--- | :--- | :--- |
| `*.livekit.cloud` | 443 (WSS) | LiveKit room WebSocket |
| `stun.l.google.com` | 19302 (UDP) | STUN ICE negotiation |
| `stun1.l.google.com` | 19302 (UDP) | STUN ICE negotiation |
| `api.deepgram.com` | 443 (WSS) | Deepgram STT WebSocket |
| `api.groq.com` | 443 (HTTPS) | Groq LLM |
| `translate.google.com` | 443 (HTTPS) | gTTS synthesis |

If your deployment is behind a restrictive firewall, allow all of the above. LiveKit Cloud (Mumbai) handles TURN relay automatically — no self-hosted TURN server is required.

---

## Scaling

The current bot is single-room / single-participant. To serve multiple concurrent rooms:

1. Run one container (or process) per room — each `VoicePipeline` holds its own LiveKit room connection and Deepgram WebSocket.
2. Put a load balancer in front that routes `/token?room=X` requests and starts a worker for room X if one isn't running.
3. The FastAPI app exposes `/health` suitable for load-balancer health checks.

---

## Logging

Log files are written to `logs/` (auto-created):

| File | Content |
| :--- | :--- |
| `logs/main.log` | Pipeline events, participant joins/leaves, turn summaries |
| `logs/tts.log` | Per-synthesis language, char count, errors |
| `logs/latency.log` | Per-turn stage timings + target-exceeded warnings |
| `logs/llm.log` | LLM request/response metadata |
| `logs/scope.log` | Scope validation decisions |

All logs rotate at 2–10 MB with 2–5 backups. Set `LOG_LEVEL=DEBUG` in `.env` for verbose output during development.

---

## Production checklist

- [ ] `LOG_LEVEL=INFO` (not DEBUG)
- [ ] `.env` secrets injected via secrets manager, not committed to repo
- [ ] `/health` endpoint wired to load-balancer / container health check
- [ ] STUN ports (UDP 19302) open in security group / firewall
- [ ] Deepgram WebSocket URL whitelisted if egress filtering is in place
- [ ] gTTS internet access allowed (calls `translate.google.com`)
- [ ] Python 3.10+ in container base image
- [ ] `webrtcvad` installed (may need build tools on some Linux distros: `apt-get install python3-dev`)

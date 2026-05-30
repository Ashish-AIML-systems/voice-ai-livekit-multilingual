import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project root
BASE_DIR = Path(__file__).parent.parent

# ── Deepgram ──────────────────────────────────────────────────────────────────
DEEPGRAM_API_KEY: str = os.environ.get("DEEPGRAM_API_KEY", "")

# ── LiveKit ───────────────────────────────────────────────────────────────────
LIVEKIT_URL: str        = os.environ.get("LIVEKIT_URL", "")
LIVEKIT_API_KEY: str    = os.environ.get("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET: str = os.environ.get("LIVEKIT_API_SECRET", "")

# ── Groq LLM ──────────────────────────────────────────────────────────────────
GROQ_API_KEY: str  = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL: str    = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL: str = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

MAX_RESPONSE_TOKENS: int = int(os.environ.get("MAX_RESPONSE_TOKENS", "500"))
MAX_HISTORY_TURNS: int   = int(os.environ.get("MAX_HISTORY_TURNS", "10"))
LLM_MAX_RETRIES: int     = int(os.environ.get("LLM_MAX_RETRIES", "3"))

# ── ElevenLabs TTS ────────────────────────────────────────────────────────────
ELEVENLABS_API_KEY: str           = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID_ENGLISH: str  = os.environ.get("ELEVENLABS_VOICE_ID_ENGLISH", "")
ELEVENLABS_VOICE_ID_HINDI: str    = os.environ.get("ELEVENLABS_VOICE_ID_HINDI", "")
ELEVENLABS_MODEL: str             = os.environ.get("ELEVENLABS_MODEL", "eleven_turbo_v2_5")
ELEVENLABS_OUTPUT_FORMAT: str     = os.environ.get("ELEVENLABS_OUTPUT_FORMAT", "pcm_16000")

# ── STT tuning ────────────────────────────────────────────────────────────────
STT_CONFIDENCE_THRESHOLD: float = float(os.environ.get("STT_CONFIDENCE_THRESHOLD", "0.80"))
STT_SAMPLE_RATE: int            = int(os.environ.get("STT_SAMPLE_RATE", "16000"))
STT_UTTERANCE_END_MS: str       = os.environ.get("STT_UTTERANCE_END_MS", "1000")
STT_MAX_RECONNECT_ATTEMPTS: int = int(os.environ.get("STT_MAX_RECONNECT_ATTEMPTS", "5"))

# ── Language detection ────────────────────────────────────────────────────────
LANGUAGE_CONFIDENCE_THRESHOLD: float = float(
    os.environ.get("LANGUAGE_CONFIDENCE_THRESHOLD", "0.80")
)

# ── Conversation memory ──────────────────────────────────────────────────────
CONVO_SLIDING_WINDOW_TURNS: int = int(os.environ.get("CONVO_SLIDING_WINDOW_TURNS", "8"))

# ── Prompt / config paths ─────────────────────────────────────────────────────
PROMPT_DIR:   Path = BASE_DIR / "config" / "prompts"
SCENARIO_DIR: Path = BASE_DIR / "config" / "scenarios"

MARKETING_PROMPT_FILE: str = str(PROMPT_DIR   / "marketing_system_prompt.txt")
MARKETING_CONFIG_FILE: str = str(SCENARIO_DIR / "marketing_config.yaml")

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR: Path  = BASE_DIR / "logs"
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")

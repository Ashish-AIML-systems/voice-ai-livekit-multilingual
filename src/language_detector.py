import logging
import logging.handlers
import re
from collections import defaultdict
from typing import Optional

from config.settings import LANGUAGE_CONFIDENCE_THRESHOLD, LOG_DIR, LOG_LEVEL

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger(__name__)
if not logger.handlers:
    fh = logging.handlers.RotatingFileHandler(
        LOG_DIR / "language.log", maxBytes=2_000_000, backupCount=2, encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(fh)
logger.setLevel(LOG_LEVEL)

# ---------------------------------------------------------------------------
# Optional langdetect import
# ---------------------------------------------------------------------------

try:
    from langdetect import detect_langs, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
    DetectorFactory.seed = 0   # make results reproducible
    _LANGDETECT_OK = True
except ImportError:
    _LANGDETECT_OK = False
    logger.warning("langdetect not installed — falling back to script analysis only")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES = {"en", "hi", "hinglish"}

# Unicode range for Devanagari script (Hindi, Marathi, etc.)
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_LATIN_RE      = re.compile(r"[a-zA-Z]")

# Common Hindi words written in Roman script — strong Hinglish indicators
_HINGLISH_MARKERS = {
    "kya", "hai", "nahi", "nahin", "aap", "main", "hum", "tum", "yeh",
    "woh", "karo", "bolo", "samajh", "theek", "achha", "bas", "bilkul",
    "bahut", "thoda", "zyada", "matlab", "bhai", "yaar", "accha", "shukriya",
    "dhanyawad", "namaste", "arrey", "arre", "oye", "chal", "abhi", "sirf",
    "lekin", "aur", "toh", "kyunki", "isliye", "suniye", "boliye", "batao",
}

# Minimum distinct Hinglish marker words needed to classify as Hinglish
_HINGLISH_MIN_MARKERS = 2

# How many consecutive new-language detections before switching (hysteresis)
_SWITCH_CONFIRMATION_COUNT = 2


class LanguageDetector:
    """
    Detects English, Hindi, and Hinglish from text.

    Detection hierarchy (in order)
    --------------------------------
    1. Script analysis  — Devanagari presence is a near-certain Hindi/Hinglish signal.
    2. Hinglish markers — Roman-script Hindi words in an otherwise Latin text.
    3. Deepgram tag     — used when its confidence ≥ threshold (fast path for STT).
    4. langdetect lib   — NLP-based fallback for ambiguous Latin-only text.
    5. Default "en"     — returned when all above methods fail.

    Session tracking
    ----------------
    update_session_language() maintains per-participant language state with
    hysteresis: requires _SWITCH_CONFIRMATION_COUNT consecutive detections of
    a new language before officially switching, preventing single-word
    mis-detections from flipping the response language.
    """

    def __init__(self, confidence_threshold: float = LANGUAGE_CONFIDENCE_THRESHOLD):
        self._threshold = confidence_threshold
        # {participant_id: {"lang": str, "pending": str|None, "count": int}}
        self._sessions: dict = defaultdict(
            lambda: {"lang": "en", "pending": None, "count": 0}
        )

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def detect_language(self, text: str) -> dict:
        """
        Returns {"language": "en"|"hi"|"hinglish", "confidence": float}
        """
        if not text or not text.strip():
            return {"language": "en", "confidence": 1.0}

        has_devanagari = bool(_DEVANAGARI_RE.search(text))
        has_latin      = bool(_LATIN_RE.search(text))

        # Mixed scripts → Hinglish (e.g. "yeh product bahut अच्छा hai")
        if has_devanagari and has_latin:
            return {"language": "hinglish", "confidence": 0.92}

        # Pure Devanagari → Hindi
        if has_devanagari:
            return {"language": "hi", "confidence": 0.95}

        # Roman-script Hinglish marker words
        words = set(text.lower().split())
        marker_hits = words & _HINGLISH_MARKERS
        if len(marker_hits) >= _HINGLISH_MIN_MARKERS:
            logger.debug("Hinglish markers found: %s", marker_hits)
            return {"language": "hinglish", "confidence": 0.82}

        # langdetect NLP library
        if _LANGDETECT_OK:
            try:
                langs = detect_langs(text)
                if langs:
                    top = langs[0]
                    lang_code  = top.lang
                    confidence = float(top.prob)
                    logger.debug(
                        "langdetect result: %s (%.2f)", lang_code, confidence
                    )
                    if lang_code in ("hi",):
                        return {"language": "hi", "confidence": confidence}
                    if lang_code in ("en",):
                        return {"language": "en", "confidence": confidence}
            except LangDetectException as exc:
                logger.debug("langdetect exception: %s", exc)

        # Default
        return {"language": "en", "confidence": 0.5}

    def detect_from_deepgram(
        self,
        text: str,
        deepgram_language: Optional[str] = None,
        deepgram_confidence: float = 0.0,
    ) -> dict:
        """
        Use Deepgram's per-utterance language tag when its STT confidence
        meets the threshold; otherwise fall back to detect_language(text).
        This combines the two signals for maximum accuracy.
        """
        if deepgram_language and deepgram_confidence >= self._threshold:
            lang = self._normalise_deepgram_lang(deepgram_language)
            logger.debug(
                "Deepgram language tag used: %s → %s (conf %.2f)",
                deepgram_language, lang, deepgram_confidence,
            )
            return {"language": lang, "confidence": deepgram_confidence}

        # Deepgram confidence too low — use text-based detection
        return self.detect_language(text)

    def is_language_supported(self, lang_code: str) -> bool:
        return lang_code in SUPPORTED_LANGUAGES

    def get_primary_language(self, text_samples: list) -> str:
        """Detect language across multiple utterances and return the majority."""
        counts: dict = defaultdict(int)
        for text in text_samples:
            result = self.detect_language(text)
            counts[result["language"]] += 1
        return max(counts, key=counts.get, default="en")

    # ------------------------------------------------------------------
    # Session tracking (hysteresis)
    # ------------------------------------------------------------------

    def update_session_language(self, participant_id: str, text: str) -> str:
        """
        Detect language for this utterance and update the session tracker.
        Switches the active language only after _SWITCH_CONFIRMATION_COUNT
        consecutive detections of a new language — prevents flicker.
        Returns the currently confirmed language for this participant.
        """
        detected = self.detect_language(text)["language"]
        session  = self._sessions[participant_id]
        current  = session["lang"]

        if detected == current:
            # Confirmed existing language — reset pending
            session["pending"] = None
            session["count"]   = 0
            return current

        # New language detected — accumulate confirmations
        if session["pending"] == detected:
            session["count"] += 1
        else:
            session["pending"] = detected
            session["count"]   = 1

        if session["count"] >= _SWITCH_CONFIRMATION_COUNT:
            logger.info(
                "Participant %s: language switched %s → %s",
                participant_id, current, detected,
            )
            session["lang"]    = detected
            session["pending"] = None
            session["count"]   = 0

        return session["lang"]

    def get_current_language(self, participant_id: str) -> str:
        return self._sessions[participant_id]["lang"]

    def reset_session(self, participant_id: str) -> None:
        if participant_id in self._sessions:
            del self._sessions[participant_id]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_deepgram_lang(lang: str) -> str:
        mapping = {
            "en": "en", "en-US": "en", "en-IN": "en", "en-GB": "en",
            "hi": "hi", "hi-IN": "hi",
            "multi": "en",   # "multi" means Deepgram detected mixed — treat as en for now
        }
        return mapping.get(lang, "en")

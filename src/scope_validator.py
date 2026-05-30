import logging
import logging.handlers
import re
from pathlib import Path
from typing import Optional

import yaml

from config.settings import LOG_DIR, LOG_LEVEL, MARKETING_CONFIG_FILE

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger(__name__)
if not logger.handlers:
    fh = logging.handlers.RotatingFileHandler(
        LOG_DIR / "scope.log", maxBytes=2_000_000, backupCount=2, encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(fh)
logger.setLevel(LOG_LEVEL)

# ---------------------------------------------------------------------------
# PII patterns — Indian context
# ---------------------------------------------------------------------------

_PII_PATTERNS = [
    (re.compile(r"\b[6-9]\d{9}\b"),                                        "mobile_number"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),                               "pan_card"),
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),                            "aadhaar"),
    (re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"), "email"),
    (re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z\d]\b"),             "gstin"),
]

# ---------------------------------------------------------------------------
# Prompt injection patterns
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(previous|above|all|prior)\s+instructions?", re.IGNORECASE),
    re.compile(r"(system\s+prompt|you\s+are\s+now|pretend\s+(to\s+be|you\s+are))", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all|your\s+instructions?|your\s+role)", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"\bDAN\s+(mode|prompt|persona)\b", re.IGNORECASE),
    re.compile(r"act\s+as\s+(a\s+)?(different|unrestricted|evil|hacker)", re.IGNORECASE),
    re.compile(r"disregard\s+(your|all|any)\s+(previous\s+)?(instructions?|rules?)", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Profanity blocklist (English + Hindi Roman-script)
# ---------------------------------------------------------------------------

_PROFANITY = {
    # English
    "fuck", "fucking", "shit", "bitch", "asshole", "bastard", "cunt",
    # Hindi (Roman)
    "madarchod", "bhenchod", "chutiya", "randi", "gandu", "harami",
    "saala", "kamina",
}


class ScopeValidator:
    """
    Two-stage content gate used in the conversation pipeline.

    Stage 1 — Safety filter  (runs on ALL content: user input + LLM output)
        • PII detection     → blocks phone numbers, Aadhaar, PAN, email, GSTIN
        • Prompt injection  → blocks jailbreak / system-prompt manipulation attempts
        • Profanity         → blocks abusive language

    Stage 2 — Scope filter  (runs on user input + LLM output)
        • Keyword match against blocked_topics from marketing_config.yaml
        • Keyword match against allowed_topics (if blocked match → redirect)
        • No match on either list → allow (benefit of doubt)

    Redirect responses are loaded from marketing_config.yaml in all three
    languages (en / hi / hinglish).

    Integration points
    ------------------
    • Call validate_content(user_input) BEFORE sending to LLM.
    • Call validate_and_gate(llm_output, language) AFTER LLM responds, BEFORE TTS.
    • Log all violations with participant_id for audit.
    """

    def __init__(self, config_file: Optional[str] = None):
        self._config   = self._load_config(config_file or MARKETING_CONFIG_FILE)
        self._allowed  = [t.lower() for t in self._config.get("allowed_topics", [])]
        self._blocked  = [t.lower() for t in self._config.get("blocked_topics", [])]
        self._fallbacks = {
            "en":       self._config.get("fallback_response_en", ""),
            "hi":       self._config.get("fallback_response_hi", ""),
            "hinglish": self._config.get("fallback_response_hinglish", ""),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_within_scope(self, text: str, scope: str = "marketing") -> bool:
        """True if no blocked topic keyword is found in text."""
        text_lower = text.lower()
        for topic in self._blocked:
            if topic in text_lower:
                logger.info("Blocked topic matched: '%s' in '%.80s'", topic, text)
                return False
        return True

    def redirect_out_of_scope(self, language: str = "en") -> str:
        """Return the language-appropriate redirect message."""
        return (
            self._fallbacks.get(language)
            or self._fallbacks.get("en")
            or "I can only help with marketing topics. Can I tell you about our latest offers?"
        )

    def validate_content(self, text: str) -> dict:
        """
        Safety filter: profanity → PII → prompt injection.
        Returns {"safe": bool, "reason": str}
        """
        words = set(text.lower().split())

        # Profanity
        hits = words & _PROFANITY
        if hits:
            logger.warning("Profanity detected: %s", hits)
            return {"safe": False, "reason": f"profanity:{','.join(hits)}"}

        # PII
        for pattern, label in _PII_PATTERNS:
            if pattern.search(text):
                logger.warning("PII detected: %s", label)
                return {"safe": False, "reason": f"pii:{label}"}

        # Prompt injection
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(text):
                logger.warning("Prompt injection attempt detected: %.80s", text)
                return {"safe": False, "reason": "prompt_injection"}

        return {"safe": True, "reason": "ok"}

    def validate_and_gate(
        self,
        text: str,
        language: str = "en",
        scope: str = "marketing",
        participant_id: Optional[str] = None,
    ) -> tuple:
        """
        Full two-stage gate.

        Returns (final_text: str, was_redirected: bool)

        If safe and in-scope  → returns (text, False)
        If unsafe / out-scope → returns (redirect_message, True)
        """
        pid = participant_id or "unknown"

        # Stage 1: safety
        safety = self.validate_content(text)
        if not safety["safe"]:
            logger.warning(
                "[%s] Content blocked (reason=%s): %.80s", pid, safety["reason"], text
            )
            return self.redirect_out_of_scope(language), True

        # Stage 2: scope
        if not self.is_within_scope(text, scope):
            logger.info("[%s] Out-of-scope text: %.80s", pid, text)
            return self.redirect_out_of_scope(language), True

        return text, False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(path: str) -> dict:
        try:
            content = Path(path).read_text(encoding="utf-8")
            return yaml.safe_load(content) or {}
        except FileNotFoundError:
            logger.error("Config file not found: %s", path)
            return {}
        except yaml.YAMLError as exc:
            logger.error("YAML parse error in %s: %s", path, exc)
            return {}

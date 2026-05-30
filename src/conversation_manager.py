import logging
import logging.handlers
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from config.settings import (
    CONVO_SLIDING_WINDOW_TURNS,
    LOG_DIR,
    LOG_LEVEL,
    MARKETING_PROMPT_FILE,
)

if TYPE_CHECKING:
    from src.tts_engine import TTSEngine

# ---------------------------------------------------------------------------
# Logging — dedicated conversation log for audit / analytics
# ---------------------------------------------------------------------------

LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger(__name__)
if not logger.handlers:
    fh = logging.handlers.RotatingFileHandler(
        LOG_DIR / "conversation.log",
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(fh)
logger.setLevel(LOG_LEVEL)


# ---------------------------------------------------------------------------
# Trim window — keep system + first user turn + last N turns
# A "turn" = one user message + one assistant message = 2 list entries
# ---------------------------------------------------------------------------

_KEEP_LAST_MESSAGES = CONVO_SLIDING_WINDOW_TURNS * 2  # 8 turns → 16 messages
_TRIM_THRESHOLD     = 1 + 1 + _KEEP_LAST_MESSAGES     # system + first_user + 16


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class ConversationState:
    participant_id: str
    history:    list = field(default_factory=list)          # short-term: full message list
    session:    dict = field(default_factory=dict)          # mid-term: state + counters
    language:   str  = "en"
    started_at: datetime = field(default_factory=datetime.now)
    is_active:  bool = True


# ---------------------------------------------------------------------------
# Conversation Manager
# ---------------------------------------------------------------------------

class ConversationManager:
    """
    Per-participant conversation state with two memory tiers.

    TYPE 1 — Conversation history (short term)
    -------------------------------------------
    Format: list of {"role": ..., "content": ...} OpenAI-style dicts.
    Index 0 = system prompt (always preserved).
    Index 1 = first user turn (always preserved — gives the LLM intro context).
    Tail   = last CONVO_SLIDING_WINDOW_TURNS turns (16 messages).
    Anything between is dropped silently by trim_history().

    TYPE 2 — Session state (mid term)
    ---------------------------------
    Per-participant dict tracking:
        language, detected_topics, last_intent, turn_count,
        conversation_start, barge_in_count, is_active

    Storage
    -------
    Lives in RAM only — self.conversations[participant_id] = ConversationState.
    Cleared on end_conversation().  No DB / Redis / disk — demo-grade.

    Integration
    -----------
    * llm_processor calls get_context() to fetch the trimmed message list.
    * language_detector calls update_language() each turn.
    * livekit_manager (on_barge_in callback) calls handle_interruption().
    * stt → add_message(participant_id, "user", text)
      llm → add_message(participant_id, "assistant", text)
    """

    def __init__(
        self,
        system_prompt_file: str = MARKETING_PROMPT_FILE,
        tts_engine: Optional["TTSEngine"] = None,
    ):
        self.conversations: dict = {}    # {participant_id: ConversationState}
        self._tts_engine = tts_engine
        self._system_prompt = self._load_system_prompt(system_prompt_file)

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def start_conversation(self, participant_id: str) -> ConversationState:
        """Initialise a fresh conversation for a participant."""
        state = ConversationState(
            participant_id=participant_id,
            history=[{"role": "system", "content": self._system_prompt}],
            session={
                "participant_id":     participant_id,
                "language":           "en",
                "detected_topics":    [],
                "last_intent":        None,
                "turn_count":         0,
                "conversation_start": datetime.now(),
                "barge_in_count":     0,
                "is_active":          True,
            },
        )
        self.conversations[participant_id] = state
        logger.info(
            "CONVO_START participant=%s at=%s",
            participant_id, state.started_at.isoformat(),
        )
        return state

    def end_conversation(self, participant_id: str) -> None:
        """Tear down a conversation — logs summary, frees memory."""
        state = self.conversations.get(participant_id)
        if not state:
            return

        ended_at = datetime.now()
        duration = (ended_at - state.started_at).total_seconds()
        turn_count = state.session.get("turn_count", 0)
        barge_ins = state.session.get("barge_in_count", 0)

        logger.info(
            "CONVO_END participant=%s duration=%.1fs turns=%d barge_ins=%d",
            participant_id, duration, turn_count, barge_ins,
        )

        state.is_active = False
        state.session["is_active"] = False
        # Free memory — no need to retain after the call ends
        del self.conversations[participant_id]

    def is_active(self, participant_id: str) -> bool:
        state = self.conversations.get(participant_id)
        return bool(state and state.is_active)

    # ==================================================================
    # History (short-term memory)
    # ==================================================================

    def add_message(self, participant_id: str, role: str, content: str) -> None:
        """Append a message, trim history, bump turn counter."""
        state = self.conversations.get(participant_id)
        if not state:
            logger.warning("add_message on unknown participant: %s", participant_id)
            return

        state.history.append({"role": role, "content": content})
        self.trim_history(participant_id)

        # Count an assistant message as completing a "turn"
        if role == "assistant":
            state.session["turn_count"] = state.session.get("turn_count", 0) + 1

        logger.info(
            "TURN participant=%s role=%s lang=%s chars=%d turn=%d",
            participant_id, role, state.language, len(content),
            state.session.get("turn_count", 0),
        )

    def get_context(self, participant_id: str) -> list:
        """
        Return the trimmed sliding-window message list ready to feed the LLM.
        Guarantees system prompt at [0] and first user turn at [1] are present.
        """
        state = self.conversations.get(participant_id)
        if not state:
            return [{"role": "system", "content": self._system_prompt}]
        return list(state.history)

    def trim_history(self, participant_id: str) -> None:
        """
        Compact history to: system + first_user_turn + last 16 messages.
        Drops everything between silently when the threshold is exceeded.
        """
        state = self.conversations.get(participant_id)
        if not state:
            return

        history = state.history
        if len(history) <= _TRIM_THRESHOLD:
            return  # under the cap — nothing to do

        system     = history[:1]
        first_user = history[1:2] if history[1:2] and history[1]["role"] == "user" else []
        tail       = history[-_KEEP_LAST_MESSAGES:]

        new_history = system + first_user + tail
        dropped = len(history) - len(new_history)
        state.history = new_history

        logger.debug(
            "TRIM participant=%s dropped=%d kept=%d",
            participant_id, dropped, len(new_history),
        )

    # ==================================================================
    # Session state (mid-term memory)
    # ==================================================================

    def update_language(self, participant_id: str, language: str) -> None:
        """Called by language_detector on every turn."""
        state = self.conversations.get(participant_id)
        if not state:
            return

        if state.language == language:
            return  # no change

        previous   = state.language
        turn_count = state.session.get("turn_count", 0)
        state.language = language
        state.session["language"] = language

        logger.info(
            "LANG_SWITCH participant=%s %s -> %s at_turn=%d",
            participant_id, previous, language, turn_count,
        )

    def update_session(self, participant_id: str, key: str, value: Any) -> None:
        """Generic session updater — used for topics, intent, etc."""
        state = self.conversations.get(participant_id)
        if not state:
            return
        state.session[key] = value

    def get_session(self, participant_id: str) -> dict:
        """Full session dict (for context-aware prompting)."""
        state = self.conversations.get(participant_id)
        return dict(state.session) if state else {}

    # ==================================================================
    # Barge-in handling
    # ==================================================================

    def handle_interruption(self, participant_id: str) -> None:
        """
        Called by LiveKitManager(on_barge_in=...) when the user starts
        speaking while TTS is playing.  Stops TTS immediately and bumps
        the barge_in counter.
        """
        state = self.conversations.get(participant_id)
        if not state:
            return

        state.session["barge_in_count"] = state.session.get("barge_in_count", 0) + 1
        logger.info(
            "BARGE_IN participant=%s count=%d at=%s",
            participant_id,
            state.session["barge_in_count"],
            datetime.now().isoformat(),
        )

        if self._tts_engine is not None:
            self._tts_engine.stop_speaking()

    # ==================================================================
    # Internal
    # ==================================================================

    @staticmethod
    def _load_system_prompt(path: str) -> str:
        try:
            from pathlib import Path
            return Path(path).read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            logger.error("System prompt not found: %s — using minimal fallback", path)
            return "You are a helpful marketing assistant."

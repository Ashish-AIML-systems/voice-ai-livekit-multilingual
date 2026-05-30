"""
src/latency_tracker.py
======================

Per-turn latency instrumentation for the voicebot pipeline.

Why
----
The end-to-end "user stops talking → bot starts talking" latency is THE single
most important UX metric for a voice agent.  We want every turn measured so
regressions show up the moment they happen, not when a user complains.

Stages tracked
--------------
* TURN_START         — Deepgram fired UtteranceEnd (user done speaking)
* STT_END            — last final transcript handed to the pipeline
* LLM_FIRST_SENTENCE — first sentence streamed back from Groq
* LLM_END            — streaming finished
* TTS_FIRST_CHUNK    — first PCM chunk ready from ElevenLabs
* TTS_END            — last chunk pushed to LiveKit
* TURN_END           — bot finished speaking

Targets  (per Phase 5 checklist)
--------------------------------
* STT total            < 500 ms
* LLM time-to-first    < 1 s
* TTS time-to-first    < 300 ms
* End-to-end total     < 2.5 s

Each stage logs a WARNING when it exceeds its target so the developer sees
regressions in `logs/latency.log` immediately.

Usage
-----
    tracker = LatencyTracker(participant_id, turn_id)
    tracker.mark("STT_END")
    ...
    tracker.mark("LLM_FIRST_SENTENCE")
    ...
    tracker.mark("TURN_END")
    tracker.report()           # writes summary line to logs/latency.log
"""

from __future__ import annotations

import logging
import logging.handlers
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from config.settings import LOG_DIR, LOG_LEVEL

# ---------------------------------------------------------------------------
# Dedicated latency log — one line per turn, easy to grep / pipe to dashboard
# ---------------------------------------------------------------------------

LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger(__name__)
if not logger.handlers:
    fh = logging.handlers.RotatingFileHandler(
        LOG_DIR / "latency.log",
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(fh)
logger.setLevel(LOG_LEVEL)

# ---------------------------------------------------------------------------
# Stage targets (ms) — sourced directly from the Phase 5 checklist
# ---------------------------------------------------------------------------

TARGET_STT_MS         = 500    # "If STT > 500ms → check audio chunk size"
TARGET_LLM_FIRST_MS   = 1000   # "If LLM > 1s → check if streaming is on"
TARGET_TTS_FIRST_MS   = 300    # "If TTS > 300ms → verify eleven_turbo_v2_5"
TARGET_TOTAL_MS       = 2500   # "Target < 2.5s total"

# Canonical stage names — keep in one place to avoid typos
STAGE_TURN_START         = "TURN_START"
STAGE_STT_END            = "STT_END"
STAGE_LLM_START          = "LLM_START"
STAGE_LLM_FIRST_SENTENCE = "LLM_FIRST_SENTENCE"
STAGE_LLM_END            = "LLM_END"
STAGE_TTS_START          = "TTS_START"
STAGE_TTS_FIRST_CHUNK    = "TTS_FIRST_CHUNK"
STAGE_TTS_END            = "TTS_END"
STAGE_TURN_END           = "TURN_END"


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

@dataclass
class LatencyTracker:
    """
    Records timestamps for one user turn and emits a summary line on report().

    Notes
    -----
    * time.monotonic() is used so wall-clock drift / NTP adjustments don't
      poison the delta calculations.
    * Marking the same stage twice overwrites — last write wins (useful
      when re-using the tracker for repeated end-of-turn fires).
    * report() never raises — even with missing stages it produces a partial
      line so the pipeline cannot be broken by instrumentation failures.
    """

    participant_id: str
    turn_id:        int          = 0
    marks:          Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # TURN_START is implicit on construction so callers don't forget it.
        self.marks[STAGE_TURN_START] = time.monotonic()

    # ------------------------------------------------------------------
    # Marking
    # ------------------------------------------------------------------

    def mark(self, stage: str) -> None:
        """Record the current monotonic time against `stage`."""
        try:
            self.marks[stage] = time.monotonic()
        except Exception as exc:
            # Instrumentation must NEVER raise into the pipeline
            logger.debug("LatencyTracker.mark failed for %s: %s", stage, exc)

    def mark_if_unset(self, stage: str) -> None:
        """Record `stage` only if it wasn't already recorded.

        Useful for first-chunk markers that fire in a loop — only the
        very first call should set the timestamp.
        """
        if stage not in self.marks:
            self.mark(stage)

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def _delta_ms(self, start: str, end: str) -> Optional[float]:
        """Return (end - start) in milliseconds, or None if either is missing."""
        s = self.marks.get(start)
        e = self.marks.get(end)
        if s is None or e is None:
            return None
        return (e - s) * 1000.0

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> dict:
        """
        Emit a summary log line and return a dict with all measured stage
        durations.  Never raises.

        Returns
        -------
        {
            "stt_ms":         float | None,
            "llm_first_ms":   float | None,
            "llm_total_ms":   float | None,
            "tts_first_ms":   float | None,
            "tts_total_ms":   float | None,
            "total_ms":       float | None,
        }
        """
        try:
            stt_ms       = self._delta_ms(STAGE_TURN_START,         STAGE_STT_END)
            llm_first_ms = self._delta_ms(STAGE_LLM_START,          STAGE_LLM_FIRST_SENTENCE)
            llm_total_ms = self._delta_ms(STAGE_LLM_START,          STAGE_LLM_END)
            tts_first_ms = self._delta_ms(STAGE_TTS_START,          STAGE_TTS_FIRST_CHUNK)
            tts_total_ms = self._delta_ms(STAGE_TTS_START,          STAGE_TTS_END)
            total_ms     = self._delta_ms(STAGE_TURN_START,         STAGE_TURN_END)

            summary = {
                "stt_ms":       stt_ms,
                "llm_first_ms": llm_first_ms,
                "llm_total_ms": llm_total_ms,
                "tts_first_ms": tts_first_ms,
                "tts_total_ms": tts_total_ms,
                "total_ms":     total_ms,
            }

            self._log_summary(summary)
            self._check_targets(summary)
            return summary

        except Exception as exc:
            # Instrumentation must never crash the pipeline
            logger.error("LatencyTracker.report failed: %s", exc, exc_info=True)
            return {}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log_summary(self, summary: dict) -> None:
        """One INFO line per turn — easy to grep / dashboard later."""
        parts = [f"participant={self.participant_id}", f"turn={self.turn_id}"]
        for key, val in summary.items():
            parts.append(f"{key}={'%.0f' % val if val is not None else 'NA'}")
        logger.info("LATENCY " + " ".join(parts))

    def _check_targets(self, summary: dict) -> None:
        """WARN log whenever a stage misses its target."""
        stt = summary.get("stt_ms")
        if stt is not None and stt > TARGET_STT_MS:
            logger.warning(
                "STT slow: %.0f ms > %d ms target — check audio chunk size",
                stt, TARGET_STT_MS,
            )

        llm_first = summary.get("llm_first_ms")
        if llm_first is not None and llm_first > TARGET_LLM_FIRST_MS:
            logger.warning(
                "LLM first-sentence slow: %.0f ms > %d ms target — check streaming",
                llm_first, TARGET_LLM_FIRST_MS,
            )

        tts_first = summary.get("tts_first_ms")
        if tts_first is not None and tts_first > TARGET_TTS_FIRST_MS:
            logger.warning(
                "TTS first-chunk slow: %.0f ms > %d ms target — verify eleven_turbo_v2_5",
                tts_first, TARGET_TTS_FIRST_MS,
            )

        total = summary.get("total_ms")
        if total is not None and total > TARGET_TOTAL_MS:
            logger.warning(
                "End-to-end slow: %.0f ms > %d ms target — investigate slowest stage",
                total, TARGET_TOTAL_MS,
            )

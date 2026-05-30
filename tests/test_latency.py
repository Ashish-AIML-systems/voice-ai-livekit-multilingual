"""
tests/test_latency.py
=====================
Tests for LatencyTracker — the per-turn instrumentation utility.

What we test
------------
* TURN_START is set automatically on construction.
* mark() records monotonic timestamps.
* mark_if_unset() only sets a stage once (first-call wins).
* report() produces a dict with the expected keys.
* Missing stages produce None in the report (no exception).
* report() never raises — even with no stages marked.
* Slow stages trigger WARNING logs (target validation).
"""

import logging
import time

import pytest

from src.latency_tracker import (
    LatencyTracker,
    STAGE_TURN_START,
    STAGE_STT_END,
    STAGE_LLM_START,
    STAGE_LLM_FIRST_SENTENCE,
    STAGE_LLM_END,
    STAGE_TTS_START,
    STAGE_TTS_FIRST_CHUNK,
    STAGE_TTS_END,
    STAGE_TURN_END,
    TARGET_TOTAL_MS,
)


def test_turn_start_marked_on_construction():
    """TURN_START is implicit so callers never forget it."""
    tracker = LatencyTracker(participant_id="u1", turn_id=1)
    assert STAGE_TURN_START in tracker.marks
    assert isinstance(tracker.marks[STAGE_TURN_START], float)


def test_mark_records_timestamp():
    tracker = LatencyTracker(participant_id="u1", turn_id=1)
    before = time.monotonic()
    tracker.mark(STAGE_LLM_START)
    after = time.monotonic()

    assert STAGE_LLM_START in tracker.marks
    assert before <= tracker.marks[STAGE_LLM_START] <= after


def test_mark_if_unset_only_sets_once():
    tracker = LatencyTracker(participant_id="u1", turn_id=1)

    tracker.mark_if_unset(STAGE_LLM_FIRST_SENTENCE)
    first_ts = tracker.marks[STAGE_LLM_FIRST_SENTENCE]

    # Sleep a bit, then try to overwrite
    time.sleep(0.01)
    tracker.mark_if_unset(STAGE_LLM_FIRST_SENTENCE)

    # Timestamp must not have changed
    assert tracker.marks[STAGE_LLM_FIRST_SENTENCE] == first_ts


def test_mark_overwrites_existing_stage():
    """mark() (not _if_unset) DOES overwrite — useful for re-using a tracker."""
    tracker = LatencyTracker(participant_id="u1", turn_id=1)
    tracker.mark(STAGE_LLM_START)
    first_ts = tracker.marks[STAGE_LLM_START]

    time.sleep(0.01)
    tracker.mark(STAGE_LLM_START)
    second_ts = tracker.marks[STAGE_LLM_START]

    assert second_ts > first_ts


def test_full_pipeline_report_returns_all_stages():
    """A fully-marked tracker reports all six computed durations."""
    tracker = LatencyTracker(participant_id="u1", turn_id=1)

    # Simulate a full turn with very small delays
    time.sleep(0.005)
    tracker.mark(STAGE_STT_END)
    tracker.mark(STAGE_LLM_START)
    time.sleep(0.005)
    tracker.mark(STAGE_LLM_FIRST_SENTENCE)
    time.sleep(0.005)
    tracker.mark(STAGE_LLM_END)
    tracker.mark(STAGE_TTS_START)
    time.sleep(0.005)
    tracker.mark(STAGE_TTS_FIRST_CHUNK)
    time.sleep(0.005)
    tracker.mark(STAGE_TTS_END)
    tracker.mark(STAGE_TURN_END)

    summary = tracker.report()

    expected_keys = {
        "stt_ms", "llm_first_ms", "llm_total_ms",
        "tts_first_ms", "tts_total_ms", "total_ms",
    }
    assert set(summary.keys()) == expected_keys

    # All values should be positive floats
    for key, val in summary.items():
        assert val is not None, f"{key} should not be None"
        assert val >= 0.0, f"{key} should be non-negative"


def test_partial_report_returns_none_for_missing_stages():
    """Only TURN_START is set → all deltas are None except where unreachable."""
    tracker = LatencyTracker(participant_id="u1", turn_id=1)
    summary = tracker.report()

    # Nothing past TURN_START was marked → every delta is None
    assert summary["stt_ms"]       is None
    assert summary["llm_first_ms"] is None
    assert summary["llm_total_ms"] is None
    assert summary["tts_first_ms"] is None
    assert summary["tts_total_ms"] is None
    assert summary["total_ms"]     is None


def test_report_never_raises():
    """Even with corrupted state, report() returns a dict and does not raise."""
    tracker = LatencyTracker(participant_id="u1", turn_id=1)
    # Inject a non-numeric value — should be caught internally
    tracker.marks[STAGE_LLM_START] = "not-a-number"  # type: ignore

    # Must not raise
    summary = tracker.report()
    assert isinstance(summary, dict)


def test_slow_total_triggers_warning(caplog):
    """When total_ms > target, a WARNING is logged."""
    tracker = LatencyTracker(participant_id="u1", turn_id=1)

    # Fake a slow turn by rewinding TURN_START
    tracker.marks[STAGE_TURN_START] = time.monotonic() - (TARGET_TOTAL_MS / 1000) - 1.0
    tracker.mark(STAGE_TURN_END)

    with caplog.at_level(logging.WARNING, logger="src.latency_tracker"):
        tracker.report()

    assert any("End-to-end slow" in rec.message for rec in caplog.records)


def test_fast_turn_no_warnings(caplog):
    """A fast turn does NOT trigger any warnings."""
    tracker = LatencyTracker(participant_id="u1", turn_id=1)
    tracker.mark(STAGE_STT_END)
    tracker.mark(STAGE_LLM_START)
    tracker.mark(STAGE_LLM_FIRST_SENTENCE)
    tracker.mark(STAGE_LLM_END)
    tracker.mark(STAGE_TTS_START)
    tracker.mark(STAGE_TTS_FIRST_CHUNK)
    tracker.mark(STAGE_TTS_END)
    tracker.mark(STAGE_TURN_END)

    with caplog.at_level(logging.WARNING, logger="src.latency_tracker"):
        tracker.report()

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning_records == []

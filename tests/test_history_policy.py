from datetime import datetime, timezone

from coinlab.history_policy import (
    humanize_backtest_failure,
    normalize_backtest_window,
    standard_policy,
)


NOW = datetime(2026, 8, 12, 18, 30, tzinfo=timezone.utc)


def test_standard_15m_policy_is_90_days():
    p = standard_policy("15m", NOW)
    assert p["max_history_days"] == 90
    assert p["earliest_safe_start"].startswith("2026-05-")
    assert p["latest_completed_bar"] == "2026-08-12T18:15:00Z"


def test_fully_expired_request_is_reset_to_current_allowed_window():
    w = normalize_backtest_window(
        timeframe="15m",
        requested_start="2025-01-01T00:00:00Z",
        requested_end="2026-01-01T00:00:00Z",
        now=NOW,
    )
    assert w.adjusted is True
    assert w.adjustment_reason == "requested_range_fully_expired"
    assert w.used_start.startswith("2026-05-")
    assert w.used_end == "2026-08-12T18:15:00Z"
    assert "自動改用" in w.message


def test_valid_recent_request_is_not_changed():
    w = normalize_backtest_window(
        timeframe="15m",
        requested_start="2026-07-01T00:00:00Z",
        requested_end="2026-08-01T00:00:00Z",
        now=NOW,
    )
    assert w.adjusted is False
    assert w.used_start == "2026-07-01T00:00:00Z"
    assert w.used_end == "2026-08-01T00:00:00Z"


def test_humanized_coinglass_time_error_contains_no_traceback():
    raw = """Traceback (most recent call last):\n  File '/x/providers.py', line 76\nRuntimeError: CoinGlass error: {'code': '400', 'msg': 'Invalid time range: the earliest allowed start_time is 1778783086000, and end_time must be greater than start_time.'}\n"""
    info = humanize_backtest_failure(raw, "15m")
    assert info["code"] == "COINGLASS_HISTORY_LIMIT"
    assert "90" in info["message"]
    assert "providers.py" not in info["message"]
    assert "RuntimeError" not in info["message"]
    assert "Traceback" not in info["message"]


def test_unknown_failure_stays_user_friendly():
    info = humanize_backtest_failure("Traceback: strange internal failure", "15m")
    assert info["code"] == "BACKTEST_FAILED"
    combined = " ".join(info.values())
    assert "providers.py" not in combined
    assert "Traceback" not in combined

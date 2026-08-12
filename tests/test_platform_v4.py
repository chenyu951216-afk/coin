import json
from datetime import datetime, timezone

import pandas as pd

from coinlab.backtest import BacktestConfig, run_backtest
from coinlab.reporting import save_report
from coinlab.scanner import next_completed_bar_time, seconds_until_next_completed_bar
from coinlab.server_v4 import DASHBOARD
from coinlab.strategies import StrategySignal


def _always_long(row):
    return StrategySignal(1, stop_distance_atr=1.0, reward_r=2.0, name="test")


def _frame():
    idx = pd.date_range("2026-08-01", periods=8, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100, 100, 100, 100, 100, 100, 100, 100],
            "high": [100, 101, 103, 101, 101, 101, 101, 101],
            "low": [100, 99.5, 99, 99.5, 99.5, 99.5, 99.5, 99.5],
            "close": [100, 100.5, 102, 100, 100, 100, 100, 100],
            "atr14": [1.0] * 8,
        },
        index=idx,
    )


def test_detailed_trade_contains_risk_levels_and_cost_breakdown():
    trades, metrics = run_backtest(
        _frame(),
        _always_long,
        BacktestConfig(slippage_bps=2, fee_bps=6, max_holding_bars=2),
    )
    assert not trades.empty
    required = {
        "initial_stop", "stop_at_exit", "target", "entry_notional", "risk_budget_usdt",
        "slippage_cost", "fees", "funding_pnl", "net_pnl", "r_multiple", "reason_text",
        "equity_before", "equity_after",
    }
    assert required.issubset(trades.columns)
    assert trades.iloc[0].slippage_cost >= 0
    assert metrics["total_fees"] >= 0
    assert metrics["total_slippage_cost"] >= 0


def test_report_v2_embeds_each_trade_for_web_and_chatgpt(tmp_path):
    df = _frame()
    trades, metrics = run_backtest(df, _always_long, BacktestConfig(slippage_bps=0, fee_bps=0, max_holding_bars=2))
    report_path = save_report(
        str(tmp_path),
        metadata={"symbol": "ETHUSDT", "timeframe": "15m"},
        features=df,
        results={"test": (trades, metrics)},
        validation={},
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "coinlab.backtest.v2"
    assert payload["strategies"]["test"]["trades"]
    assert payload["all_trades"]
    assert "net_pnl" in payload["all_trades"][0]
    assert "initial_stop" in payload["all_trades"][0]
    assert "target" in payload["all_trades"][0]


def test_scanner_schedule_waits_for_next_completed_candle():
    now = datetime(2026, 8, 12, 18, 7, 0, tzinfo=timezone.utc)
    wait = seconds_until_next_completed_bar("15m", now=now, grace_seconds=8)
    assert 1 <= wait <= 15 * 60 + 8
    next_at = next_completed_bar_time("15m", now=now, grace_seconds=8)
    assert next_at.endswith("Z")


def test_dashboard_has_auto_dates_pause_resume_and_trade_details():
    assert "自動開始 UTC" in DASHBOARD
    assert "開始 / 繼續自動掃描" in DASHBOARD
    assert "暫停掃描" in DASHBOARD
    assert "逐筆回測交易" in DASHBOARD
    assert "初始 SL" in DASHBOARD
    assert "淨損益 U" in DASHBOARD
    assert '"log_tail"' not in DASHBOARD

import pandas as pd

from coinlab.backtest import BacktestConfig, run_backtest
from coinlab.strategies import StrategySignal


def _always_long(row):
    return StrategySignal(1, stop_distance_atr=1.0, reward_r=2.0, name="test")


def test_entry_is_next_bar_open_not_signal_close():
    idx = pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "open": [100, 105, 106, 107],
        "high": [101, 106, 107, 108],
        "low": [99, 104, 105, 106],
        "close": [100, 105, 106, 107],
        "atr14": [1, 1, 1, 1],
    }, index=idx)
    trades, _ = run_backtest(df, _always_long, BacktestConfig(slippage_bps=0, fee_bps=0, max_holding_bars=1))
    assert float(trades.iloc[0].entry) == 105.0
    assert trades.iloc[0].signal_time == str(idx[0])
    assert trades.iloc[0].entry_time == str(idx[1])


def test_ambiguous_bar_is_stop_first():
    idx = pd.date_range("2026-01-01", periods=3, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "open": [100, 100, 100],
        "high": [100, 103, 100],
        "low": [100, 98, 100],
        "close": [100, 100, 100],
        "atr14": [1, 1, 1],
    }, index=idx)
    trades, metrics = run_backtest(df, _always_long, BacktestConfig(slippage_bps=0, fee_bps=0, max_holding_bars=1))
    assert trades.iloc[0].reason == "ambiguous_stop_first"
    assert bool(trades.iloc[0].ambiguous_exit)
    assert metrics["ambiguous_exit_count"] == 1
    assert trades.iloc[0].net_pnl < 0


def test_no_performance_is_invented_when_no_trades():
    def none(row):
        return StrategySignal(0, name="none")
    idx = pd.date_range("2026-01-01", periods=3, freq="15min", tz="UTC")
    df = pd.DataFrame({"open": [1,1,1], "high": [1,1,1], "low": [1,1,1], "close": [1,1,1], "atr14": [1,1,1]}, index=idx)
    trades, metrics = run_backtest(df, none, BacktestConfig())
    assert trades.empty
    assert metrics["win_rate"] is None
    assert metrics["profit_factor"] is None


def test_stop_updates_only_after_bar_close():
    idx = pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "open": [100, 100, 100, 100],
        "high": [100, 101.5, 100.2, 100],
        "low": [100, 99.2, 98.8, 100],
        "close": [100, 100.4, 99.0, 100],
        "atr14": [1, 1, 1, 1],
    }, index=idx)
    trades, _ = run_backtest(df, _always_long, BacktestConfig(slippage_bps=0, fee_bps=0, max_holding_bars=2))
    assert trades.iloc[0].reason == "stop"
    assert not bool(trades.iloc[0].breakeven_activated)

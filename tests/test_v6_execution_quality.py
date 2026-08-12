import pandas as pd

from coinlab.backtest import BacktestConfig, run_backtest
from coinlab.signal_quality import estimated_round_trip_cost_r
from coinlab.strategies import StrategySignal


def _always_long(row):
    return StrategySignal(1, stop_distance_atr=0.5, reward_r=2.0, name="test")


def test_high_execution_cost_signal_is_rejected_before_trade():
    idx = pd.date_range("2026-08-01", periods=5, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "open": [100.0] * 5,
        "high": [100.2] * 5,
        "low": [99.8] * 5,
        "close": [100.0] * 5,
        "atr14": [0.2] * 5,
    }, index=idx)
    trades, metrics = run_backtest(
        df,
        _always_long,
        BacktestConfig(fee_bps=6, slippage_bps=2, max_estimated_cost_r=0.18),
    )
    assert trades.empty
    assert metrics["signals_seen"] > 0
    assert metrics["signals_rejected_cost"] == metrics["signals_seen"]


def test_signal_feature_snapshot_contains_only_signal_bar_diagnostics():
    idx = pd.date_range("2026-08-01", periods=5, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "open": [100.0] * 5,
        "high": [102.0] * 5,
        "low": [98.0] * 5,
        "close": [100.0] * 5,
        "atr14": [2.0] * 5,
        "ret_1": [0.01, 0.02, 0.03, 0.04, 0.05],
        "taker_imb_z": [1.1, 1.2, 1.3, 1.4, 1.5],
        "oi_chg_1": [0.01] * 5,
    }, index=idx)
    trades, _ = run_backtest(
        df,
        _always_long,
        BacktestConfig(fee_bps=0, slippage_bps=0, max_holding_bars=1),
    )
    assert not trades.empty
    snapshot = trades.iloc[0].signal_features
    assert snapshot["ret_1"] == 0.01
    assert snapshot["taker_imb_z"] == 1.1
    assert "future_close" not in snapshot


def test_cost_r_is_scale_free_and_known_pretrade():
    a = estimated_round_trip_cost_r(entry=100, stop=99, fee_bps=6, slippage_bps=2)
    b = estimated_round_trip_cost_r(entry=2000, stop=1980, fee_bps=6, slippage_bps=2)
    assert abs(a - b) < 1e-12
    assert 0 < a < 0.18

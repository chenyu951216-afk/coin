import pandas as pd

from coinlab.backtest import BacktestConfig, run_backtest
from coinlab.features import build_feature_frame
from coinlab.position_sizing import paper_notional_for_price
from coinlab.strategies import StrategySignal


def _long(_row):
    return StrategySignal(1, stop_distance_atr=1.0, reward_r=2.0, name="test")


def test_fixed_notional_boundary_is_price_based_not_symbol_based():
    assert paper_notional_for_price(50.00) == 2_000.0
    assert paper_notional_for_price(50.0001) == 20_000.0
    assert paper_notional_for_price(0.10) == 2_000.0
    assert paper_notional_for_price(4_000.0) == 20_000.0


def test_backtest_uses_entry_time_price_for_fixed_notional():
    idx = pd.date_range("2026-01-01", periods=3, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "open": [49.0, 51.0, 51.0],
        "high": [49.5, 51.2, 51.2],
        "low": [48.5, 50.8, 50.8],
        "close": [49.0, 51.0, 51.0],
        "atr14": [1.0, 1.0, 1.0],
    }, index=idx)
    trades, _ = run_backtest(df, _long, BacktestConfig(fee_bps=0, slippage_bps=0, max_holding_bars=1, max_estimated_cost_r=0))
    assert float(trades.iloc[0].entry_raw) == 51.0
    assert float(trades.iloc[0].planned_notional_usdt) == 20_000.0
    assert abs(float(trades.iloc[0].entry_notional) - 20_000.0) < 1e-8


def test_historical_24h_turnover_requires_only_completed_past_window():
    idx = pd.date_range("2026-01-01", periods=100, freq="15min", tz="UTC")
    price = pd.DataFrame({
        "open": [100.0] * 100,
        "high": [101.0] * 100,
        "low": [99.0] * 100,
        "close": [100.0] * 100,
        "volume_base": [1.0] * 100,
        "volume_quote": [1_000.0] * 100,
    }, index=idx)
    out = build_feature_frame(price, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    # 15m => 96 completed bars in trailing 24h. Before that the historical
    # liquidity filter must be unavailable, never backfilled from future bars.
    assert pd.isna(out.iloc[94]["quote_volume_24h"])
    assert float(out.iloc[95]["quote_volume_24h"]) == 96_000.0

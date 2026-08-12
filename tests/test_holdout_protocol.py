import pandas as pd

from coinlab.backtest import BacktestConfig
from coinlab.strategies import StrategySignal
from coinlab.validation import evaluate_oos


def _always_long(row):
    return StrategySignal(1, stop_distance_atr=1.0, reward_r=2.0, name="demo")


def test_walk_forward_does_not_use_locked_test_segment():
    idx = pd.date_range("2026-01-01", periods=1200, freq="15min", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.2, "atr14": 1.0}, index=idx)
    result = evaluate_oos(df, _always_long, BacktestConfig(max_holding_bars=12, fee_bps=0, slippage_bps=0))
    test_start = pd.Timestamp(result["split_windows"]["test"]["start"])
    fold_ends = [pd.Timestamp(x["end"]) for x in result["walk_forward"] if x["end"]]
    assert result["walk_forward_scope"] == "TRAIN_PLUS_VALIDATION_ONLY_TEST_EXCLUDED"
    assert fold_ends and max(fold_ends) < test_start
    assert result["anti_overfit_policy"]["outcome_based_sample_exclusion"] == "FORBIDDEN"

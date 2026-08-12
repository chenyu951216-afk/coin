from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, run_backtest
from .strategies import StrategySignal


@dataclass(frozen=True)
class SplitSpec:
    train: float = 0.60
    validation: float = 0.20
    test: float = 0.20
    walk_forward_folds: int = 5

    def __post_init__(self):
        if not np.isclose(self.train + self.validation + self.test, 1.0):
            raise ValueError("train+validation+test must equal 1")


def chronological_slices(df: pd.DataFrame, spec: SplitSpec, purge_bars: int = 0) -> dict[str, pd.DataFrame]:
    """Chronological train/validation/test split with optional boundary embargo.

    ``purge_bars`` removes observations around split boundaries so a strategy
    with a non-zero holding horizon is not judged on nearly identical adjacent
    observations across development and holdout segments.
    """
    n = len(df)
    a = int(n * spec.train)
    b = int(n * (spec.train + spec.validation))
    p = max(0, int(purge_bars))
    train_end = max(0, a - p)
    valid_start = min(n, a + p)
    valid_end = max(valid_start, b - p)
    test_start = min(n, b + p)
    return {
        "train": df.iloc[:train_end],
        "validation": df.iloc[valid_start:valid_end],
        "test": df.iloc[test_start:],
    }


def _funding_for_slice(funding_events: pd.DataFrame | None, part: pd.DataFrame) -> pd.DataFrame | None:
    if funding_events is None or funding_events.empty or part.empty:
        return funding_events
    return funding_events[(funding_events.index > part.index.min()) & (funding_events.index <= part.index.max())]


def _window(part: pd.DataFrame) -> dict[str, str | int | None]:
    return {
        "rows": int(len(part)),
        "start": str(part.index.min()) if len(part) else None,
        "end": str(part.index.max()) if len(part) else None,
    }


def evaluate_oos(
    df: pd.DataFrame,
    strategy: Callable[[pd.Series], StrategySignal],
    cfg: BacktestConfig,
    spec: SplitSpec = SplitSpec(),
    funding_events: pd.DataFrame | None = None,
) -> dict:
    # Use the maximum holding horizon as a conservative embargo around the two
    # chronological split boundaries. This is fixed before outcomes are seen.
    purge_bars = max(1, int(cfg.max_holding_bars))
    segments = chronological_slices(df, spec, purge_bars=purge_bars)
    segment_results = {}
    for name, part in segments.items():
        trades, metrics = run_backtest(part, strategy, cfg, funding_events=_funding_for_slice(funding_events, part))
        segment_results[name] = {"metrics": metrics, "trades": trades}

    # IMPORTANT: walk-forward research is development-only. The locked test
    # segment is deliberately excluded from all fold diagnostics used to tune
    # or redesign strategy logic.
    n = len(df)
    development_end = int(n * (spec.train + spec.validation))
    development = df.iloc[:max(0, development_end - purge_bars)]
    folds = []
    boundaries = np.linspace(0, len(development), spec.walk_forward_folds + 1, dtype=int)
    for k in range(spec.walk_forward_folds):
        part = development.iloc[boundaries[k]:boundaries[k + 1]]
        _, metrics = run_backtest(part, strategy, cfg, funding_events=_funding_for_slice(funding_events, part))
        folds.append({
            "fold": k + 1,
            "start": str(part.index.min()) if len(part) else None,
            "end": str(part.index.max()) if len(part) else None,
            "metrics": metrics,
        })

    train_m = segment_results["train"]["metrics"]
    valid_m = segment_results["validation"]["metrics"]
    test_m = segment_results["test"]["metrics"]
    usable_folds = [f for f in folds if f["metrics"]["trades"] >= 5]
    profitable_fold_ratio = sum(f["metrics"]["net_pnl"] > 0 for f in usable_folds) / len(usable_folds) if usable_folds else None
    positive_expectancy_fold_ratio = sum((f["metrics"]["expectancy_r"] or -999) > 0 for f in usable_folds) / len(usable_folds) if usable_folds else None

    # Research grade is the only grade that should influence strategy changes.
    if valid_m["trades"] < 20:
        research_grade = "INSUFFICIENT_VALIDATION_TRADES"
    elif (valid_m["profit_factor"] or 0) <= 1.0 or (valid_m["expectancy_r"] or -999) <= 0:
        research_grade = "REJECT_RESEARCH_CANDIDATE"
    elif (profitable_fold_ratio or 0) >= 0.60 and (positive_expectancy_fold_ratio or 0) >= 0.60:
        research_grade = "ROBUST_RESEARCH_CANDIDATE"
    else:
        research_grade = "NEEDS_MORE_DEVELOPMENT_VALIDATION"

    # Promotion grade is a one-way audit gate after a candidate has been frozen.
    # It must not be used to choose filters/parameters or delete bad examples.
    if test_m["trades"] < 30:
        grade = "INSUFFICIENT_OOS_TRADES"
    elif (test_m["profit_factor"] or 0) <= 1.0 or (test_m["expectancy_r"] or -999) <= 0:
        grade = "REJECT_OOS"
    elif research_grade == "ROBUST_RESEARCH_CANDIDATE":
        grade = "PROMISING_RESEARCH_CANDIDATE"
    else:
        grade = "NEEDS_MORE_VALIDATION"

    return {
        "grade": grade,
        "research_grade": research_grade,
        "split": {k: v["metrics"] for k, v in segment_results.items()},
        "split_windows": {k: _window(v) for k, v in segments.items()},
        "purge_embargo_bars": purge_bars,
        "walk_forward_scope": "TRAIN_PLUS_VALIDATION_ONLY_TEST_EXCLUDED",
        "walk_forward": folds,
        "profitable_fold_ratio": profitable_fold_ratio,
        "positive_expectancy_fold_ratio": positive_expectancy_fold_ratio,
        "anti_overfit_policy": {
            "outcome_based_sample_exclusion": "FORBIDDEN",
            "single_trade_date_or_symbol_exceptions": "FORBIDDEN",
            "test_usage": "PROMOTION_ONLY_NOT_FOR_TUNING",
            "development_sources": ["train", "validation", "development_walk_forward"],
            "boundary_embargo": "max_holding_bars",
            "train_trade_count": train_m.get("trades"),
        },
    }

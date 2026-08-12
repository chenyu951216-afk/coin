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


def chronological_slices(df: pd.DataFrame, spec: SplitSpec) -> dict[str, pd.DataFrame]:
    n = len(df)
    a = int(n * spec.train)
    b = int(n * (spec.train + spec.validation))
    return {"train": df.iloc[:a], "validation": df.iloc[a:b], "test": df.iloc[b:]}


def _funding_for_slice(funding_events: pd.DataFrame | None, part: pd.DataFrame) -> pd.DataFrame | None:
    if funding_events is None or funding_events.empty or part.empty:
        return funding_events
    return funding_events[(funding_events.index > part.index.min()) & (funding_events.index <= part.index.max())]


def evaluate_oos(
    df: pd.DataFrame,
    strategy: Callable[[pd.Series], StrategySignal],
    cfg: BacktestConfig,
    spec: SplitSpec = SplitSpec(),
    funding_events: pd.DataFrame | None = None,
) -> dict:
    segments = chronological_slices(df, spec)
    segment_results = {}
    for name, part in segments.items():
        trades, metrics = run_backtest(part, strategy, cfg, funding_events=_funding_for_slice(funding_events, part))
        segment_results[name] = {"metrics": metrics, "trades": trades}

    folds = []
    boundaries = np.linspace(0, len(df), spec.walk_forward_folds + 1, dtype=int)
    for k in range(spec.walk_forward_folds):
        part = df.iloc[boundaries[k]:boundaries[k + 1]]
        _, metrics = run_backtest(part, strategy, cfg, funding_events=_funding_for_slice(funding_events, part))
        folds.append({
            "fold": k + 1,
            "start": str(part.index.min()) if len(part) else None,
            "end": str(part.index.max()) if len(part) else None,
            "metrics": metrics,
        })

    test_m = segment_results["test"]["metrics"]
    valid_m = segment_results["validation"]["metrics"]
    usable_folds = [f for f in folds if f["metrics"]["trades"] >= 5]
    profitable_fold_ratio = sum(f["metrics"]["net_pnl"] > 0 for f in usable_folds) / len(usable_folds) if usable_folds else None
    positive_expectancy_fold_ratio = sum((f["metrics"]["expectancy_r"] or -999) > 0 for f in usable_folds) / len(usable_folds) if usable_folds else None

    if test_m["trades"] < 30:
        grade = "INSUFFICIENT_OOS_TRADES"
    elif (test_m["profit_factor"] or 0) <= 1.0 or (test_m["expectancy_r"] or -999) <= 0:
        grade = "REJECT_OOS"
    elif valid_m["trades"] >= 20 and (valid_m["expectancy_r"] or -999) > 0 and (profitable_fold_ratio or 0) >= 0.60:
        grade = "PROMISING_RESEARCH_CANDIDATE"
    else:
        grade = "NEEDS_MORE_VALIDATION"

    return {
        "grade": grade,
        "split": {k: v["metrics"] for k, v in segment_results.items()},
        "walk_forward": folds,
        "profitable_fold_ratio": profitable_fold_ratio,
        "positive_expectancy_fold_ratio": positive_expectancy_fold_ratio,
    }

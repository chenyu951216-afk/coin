from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


SIGNAL_SNAPSHOT_COLUMNS = (
    "ret_1", "ret_4", "ret_8",
    "atr_pct", "ema20_slope_atr4", "ema50_slope_atr8", "trend_strength_atr",
    "price_ema20_atr", "efficiency_8", "close_location", "range_atr", "volume_quote_z",
    "breakout_up_20", "breakout_down_20",
    "oi_chg_1", "oi_chg_4", "oi_z",
    "fund_z", "taker_imb", "taker_imb_z",
    "long_liq_z", "short_liq_z", "ls_z", "book_imb", "book_imb_z",
)


def estimated_round_trip_cost_r(
    *,
    entry: float,
    stop: float,
    fee_bps: float,
    slippage_bps: float,
) -> float:
    """Conservative pre-trade estimate of fees + adverse entry/exit slippage in R.

    Uses only information known before the trade is opened. The estimate assumes
    roughly equal entry/exit notionals and therefore intentionally does not use
    the future exit price.
    """
    distance = abs(float(entry) - float(stop))
    if distance <= 0 or not math.isfinite(distance) or entry <= 0:
        return math.inf
    round_trip_rate = 2.0 * (max(0.0, fee_bps) + max(0.0, slippage_bps)) / 10_000.0
    return float(round_trip_rate * abs(float(entry)) / distance)


def cost_aware_breakeven_trigger(
    *,
    entry_fill: float,
    direction: int,
    fee_bps: float,
    slippage_bps: float,
) -> float:
    """Stop trigger that approximately covers exit slippage + both taker fees.

    Entry slippage is already embedded in ``entry_fill``. Funding is excluded
    because future funding settlements are unknown when the stop is moved.
    """
    fee = max(0.0, float(fee_bps)) / 10_000.0
    slip = max(0.0, float(slippage_bps)) / 10_000.0
    e = float(entry_fill)
    if direction > 0:
        exit_fill_needed = e * (1.0 + fee) / max(1e-12, 1.0 - fee)
        return exit_fill_needed / max(1e-12, 1.0 - slip)
    exit_fill_needed = e * (1.0 - fee) / max(1e-12, 1.0 + fee)
    return exit_fill_needed / (1.0 + slip)


def signal_snapshot(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in SIGNAL_SNAPSHOT_COLUMNS:
        if name not in row.index:
            continue
        value = row[name]
        if pd.isna(value):
            continue
        if isinstance(value, (np.bool_, bool)):
            out[name] = bool(value)
        elif isinstance(value, (np.integer, int)):
            out[name] = int(value)
        elif isinstance(value, (np.floating, float)):
            value = float(value)
            if math.isfinite(value):
                out[name] = value
        else:
            out[name] = str(value)
    return out

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class StrategySignal:
    direction: int
    stop_distance_atr: float = 1.8
    reward_r: float = 2.4
    name: str = ""


def _has(row: pd.Series, *cols: str) -> bool:
    return all(c in row.index and pd.notna(row[c]) for c in cols)


def _clean_long_trend(row: pd.Series) -> bool:
    return (
        row.ema20 > row.ema50
        and row.ema20_slope_atr4 > 0.08
        and row.trend_strength_atr > 0.20
        and row.efficiency_8 > 0.22
    )


def _clean_short_trend(row: pd.Series) -> bool:
    return (
        row.ema20 < row.ema50
        and row.ema20_slope_atr4 < -0.08
        and row.trend_strength_atr > 0.20
        and row.efficiency_8 > 0.22
    )


def oi_breakout(row: pd.Series) -> StrategySignal:
    """Actual price breakout confirmed by OI expansion and aggressive flow.

    The old version treated a four-bar return as a breakout and frequently
    entered late/noisy expansions. This version requires a close beyond the
    prior 20-bar range plus aligned trend/flow using only completed-bar data.
    """
    name = "oi_breakout"
    needed = (
        "ret_1", "oi_chg_4", "taker_imb_z", "fund_z", "ema20", "ema50",
        "ema20_slope_atr4", "trend_strength_atr", "efficiency_8",
        "price_ema20_atr", "breakout_up_20", "breakout_down_20", "close_location",
    )
    if not _has(row, *needed):
        return StrategySignal(0, name=name)
    if (
        row.breakout_up_20 > 0.5
        and row.ret_1 > 0
        and row.oi_chg_4 > 0.008
        and row.taker_imb_z > 1.0
        and row.fund_z < 2.0
        and _clean_long_trend(row)
        and row.price_ema20_atr < 2.0
        and row.close_location > 0.58
    ):
        return StrategySignal(1, 2.0, 2.6, name)
    if (
        row.breakout_down_20 > 0.5
        and row.ret_1 < 0
        and row.oi_chg_4 > 0.008
        and row.taker_imb_z < -1.0
        and row.fund_z > -2.0
        and _clean_short_trend(row)
        and row.price_ema20_atr > -2.0
        and row.close_location < 0.42
    ):
        return StrategySignal(-1, 2.0, 2.6, name)
    return StrategySignal(0, name=name)


def liquidation_reversal(row: pd.Series) -> StrategySignal:
    """Liquidation flush plus same-bar absorption/reversal confirmation."""
    name = "liquidation_reversal"
    needed = (
        "long_liq_z", "short_liq_z", "ret_1", "taker_imb_z", "atr_pct",
        "close_location", "range_atr",
    )
    if not _has(row, *needed):
        return StrategySignal(0, name=name)
    if (
        row.long_liq_z > 2.3
        and row.ret_1 < -0.004
        and row.taker_imb_z > 0.0
        and row.close_location > 0.55
        and row.range_atr > 0.8
    ):
        return StrategySignal(1, 2.2, 2.4, name)
    if (
        row.short_liq_z > 2.3
        and row.ret_1 > 0.004
        and row.taker_imb_z < 0.0
        and row.close_location < 0.45
        and row.range_atr > 0.8
    ):
        return StrategySignal(-1, 2.2, 2.4, name)
    return StrategySignal(0, name=name)


def funding_crowding(row: pd.Series) -> StrategySignal:
    name = "funding_crowding"
    if not _has(row, "fund_z", "ls_z", "ret_4", "ret_1", "taker_imb_z"):
        return StrategySignal(0, name=name)
    # Contrarian trade is allowed only after order-flow starts turning against
    # the crowded side; this avoids fading crowding merely because it is extreme.
    if row.fund_z > 2.2 and row.ls_z > 1.5 and row.ret_4 < 0.003 and row.ret_1 < 0 and row.taker_imb_z < -0.5:
        return StrategySignal(-1, 1.9, 2.2, name)
    if row.fund_z < -2.2 and row.ls_z < -1.5 and row.ret_4 > -0.003 and row.ret_1 > 0 and row.taker_imb_z > 0.5:
        return StrategySignal(1, 1.9, 2.2, name)
    return StrategySignal(0, name=name)


def taker_flow_momentum(row: pd.Series) -> StrategySignal:
    """Aggressive-flow momentum with price/trend confirmation and no chase."""
    name = "taker_flow_momentum"
    needed = (
        "taker_imb_z", "oi_chg_1", "ema20", "ema50", "fund_z", "ret_1",
        "ema20_slope_atr4", "trend_strength_atr", "efficiency_8",
        "price_ema20_atr", "close_location", "volume_quote_z",
    )
    if not _has(row, *needed):
        return StrategySignal(0, name=name)
    if (
        row.taker_imb_z > 1.9
        and row.oi_chg_1 > 0
        and row.ret_1 > 0
        and row.fund_z < 2.0
        and _clean_long_trend(row)
        and row.price_ema20_atr < 1.5
        and row.close_location > 0.55
        and row.volume_quote_z > -0.5
    ):
        return StrategySignal(1, 1.8, 2.4, name)
    if (
        row.taker_imb_z < -1.9
        and row.oi_chg_1 > 0
        and row.ret_1 < 0
        and row.fund_z > -2.0
        and _clean_short_trend(row)
        and row.price_ema20_atr > -1.5
        and row.close_location < 0.45
        and row.volume_quote_z > -0.5
    ):
        return StrategySignal(-1, 1.8, 2.4, name)
    return StrategySignal(0, name=name)


def orderbook_pressure(row: pd.Series) -> StrategySignal:
    """Orderbook imbalance is confirmation only, never sufficient on its own.

    Historical development data showed excessive cost/noise. Requiring executed
    taker flow in the same direction makes spoofable resting liquidity a
    secondary confirmation rather than the primary trigger.
    """
    name = "orderbook_pressure"
    needed = (
        "book_imb_z", "taker_imb_z", "ret_1", "ema20", "ema50", "oi_chg_1",
        "ema20_slope_atr4", "trend_strength_atr", "efficiency_8",
        "price_ema20_atr", "close_location",
    )
    if not _has(row, *needed):
        return StrategySignal(0, name=name)
    if (
        row.book_imb_z > 2.0
        and row.taker_imb_z > 0.75
        and row.oi_chg_1 >= 0
        and row.ret_1 > -0.002
        and _clean_long_trend(row)
        and row.price_ema20_atr < 1.3
        and row.close_location > 0.45
    ):
        return StrategySignal(1, 1.8, 2.2, name)
    if (
        row.book_imb_z < -2.0
        and row.taker_imb_z < -0.75
        and row.oi_chg_1 >= 0
        and row.ret_1 < 0.002
        and _clean_short_trend(row)
        and row.price_ema20_atr > -1.3
        and row.close_location < 0.55
    ):
        return StrategySignal(-1, 1.8, 2.2, name)
    return StrategySignal(0, name=name)


def oi_divergence(row: pd.Series) -> StrategySignal:
    """OI contraction reversal only after liquidation + price confirmation.

    The former rule tried to catch falling/rising knives directly from price/OI
    divergence. The revised form waits for a liquidation flush and a completed
    reversal candle with taker-flow confirmation before entering next bar.
    """
    name = "oi_divergence"
    needed = (
        "ret_1", "ret_4", "oi_chg_4", "taker_imb_z", "fund_z",
        "long_liq_z", "short_liq_z", "close_location", "trend_strength_atr",
    )
    if not _has(row, *needed):
        return StrategySignal(0, name=name)
    if (
        row.ret_4 < -0.008
        and row.oi_chg_4 < -0.012
        and row.long_liq_z > 1.5
        and row.ret_1 > 0
        and row.taker_imb_z > 0.75
        and row.close_location > 0.58
        and row.fund_z < 1.5
        and row.trend_strength_atr < 2.5
    ):
        return StrategySignal(1, 2.2, 2.4, name)
    if (
        row.ret_4 > 0.008
        and row.oi_chg_4 < -0.012
        and row.short_liq_z > 1.5
        and row.ret_1 < 0
        and row.taker_imb_z < -0.75
        and row.close_location < 0.42
        and row.fund_z > -1.5
        and row.trend_strength_atr < 2.5
    ):
        return StrategySignal(-1, 2.2, 2.4, name)
    return StrategySignal(0, name=name)


STRATEGIES = {
    "oi_breakout": oi_breakout,
    "liquidation_reversal": liquidation_reversal,
    "funding_crowding": funding_crowding,
    "taker_flow_momentum": taker_flow_momentum,
    "orderbook_pressure": orderbook_pressure,
    "oi_divergence": oi_divergence,
}

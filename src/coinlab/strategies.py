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


def oi_breakout(row: pd.Series) -> StrategySignal:
    name = "oi_breakout"
    if not _has(row, "ret_4", "oi_chg_4", "taker_imb", "fund_z", "ema20", "ema50"):
        return StrategySignal(0, name=name)
    if row.ret_4 > 0.004 and row.oi_chg_4 > 0.01 and row.taker_imb > 0.06 and row.fund_z < 2 and row.ema20 > row.ema50:
        return StrategySignal(1, 1.8, 2.6, name)
    if row.ret_4 < -0.004 and row.oi_chg_4 > 0.01 and row.taker_imb < -0.06 and row.fund_z > -2 and row.ema20 < row.ema50:
        return StrategySignal(-1, 1.8, 2.6, name)
    return StrategySignal(0, name=name)


def liquidation_reversal(row: pd.Series) -> StrategySignal:
    name = "liquidation_reversal"
    if not _has(row, "long_liq_z", "short_liq_z", "ret_1", "taker_imb", "atr_pct"):
        return StrategySignal(0, name=name)
    if row.long_liq_z > 2.5 and row.ret_1 < -0.006 and row.taker_imb > -0.05:
        return StrategySignal(1, 2.2, 2.2, name)
    if row.short_liq_z > 2.5 and row.ret_1 > 0.006 and row.taker_imb < 0.05:
        return StrategySignal(-1, 2.2, 2.2, name)
    return StrategySignal(0, name=name)


def funding_crowding(row: pd.Series) -> StrategySignal:
    name = "funding_crowding"
    if not _has(row, "fund_z", "ls_z", "ret_4", "taker_imb"):
        return StrategySignal(0, name=name)
    if row.fund_z > 2.2 and row.ls_z > 1.5 and row.ret_4 < 0.003 and row.taker_imb < 0:
        return StrategySignal(-1, 1.7, 2.0, name)
    if row.fund_z < -2.2 and row.ls_z < -1.5 and row.ret_4 > -0.003 and row.taker_imb > 0:
        return StrategySignal(1, 1.7, 2.0, name)
    return StrategySignal(0, name=name)


def taker_flow_momentum(row: pd.Series) -> StrategySignal:
    name = "taker_flow_momentum"
    if not _has(row, "taker_imb_z", "oi_chg_1", "ema20", "ema50", "fund_z"):
        return StrategySignal(0, name=name)
    if row.taker_imb_z > 1.8 and row.oi_chg_1 > 0 and row.ema20 > row.ema50 and row.fund_z < 2.5:
        return StrategySignal(1, 1.6, 2.3, name)
    if row.taker_imb_z < -1.8 and row.oi_chg_1 > 0 and row.ema20 < row.ema50 and row.fund_z > -2.5:
        return StrategySignal(-1, 1.6, 2.3, name)
    return StrategySignal(0, name=name)


def orderbook_pressure(row: pd.Series) -> StrategySignal:
    name = "orderbook_pressure"
    if not _has(row, "book_imb_z", "ret_1", "ema20", "ema50", "oi_chg_1"):
        return StrategySignal(0, name=name)
    if row.book_imb_z > 1.8 and row.ret_1 <= 0 and row.ema20 > row.ema50 and row.oi_chg_1 >= 0:
        return StrategySignal(1, 1.5, 2.0, name)
    if row.book_imb_z < -1.8 and row.ret_1 >= 0 and row.ema20 < row.ema50 and row.oi_chg_1 >= 0:
        return StrategySignal(-1, 1.5, 2.0, name)
    return StrategySignal(0, name=name)


def oi_divergence(row: pd.Series) -> StrategySignal:
    name = "oi_divergence"
    if not _has(row, "ret_4", "oi_chg_4", "taker_imb_z", "fund_z"):
        return StrategySignal(0, name=name)
    if row.ret_4 < -0.008 and row.oi_chg_4 < -0.012 and row.taker_imb_z > -0.5 and row.fund_z < 1.5:
        return StrategySignal(1, 2.0, 2.1, name)
    if row.ret_4 > 0.008 and row.oi_chg_4 < -0.012 and row.taker_imb_z < 0.5 and row.fund_z > -1.5:
        return StrategySignal(-1, 2.0, 2.1, name)
    return StrategySignal(0, name=name)


STRATEGIES = {
    "oi_breakout": oi_breakout,
    "liquidation_reversal": liquidation_reversal,
    "funding_crowding": funding_crowding,
    "taker_flow_momentum": taker_flow_momentum,
    "orderbook_pressure": orderbook_pressure,
    "oi_divergence": oi_divergence,
}

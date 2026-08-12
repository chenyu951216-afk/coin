from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable

import numpy as np
import pandas as pd

from .strategies import StrategySignal


@dataclass
class BacktestConfig:
    initial_equity: float = 10_000.0
    risk_per_trade: float = 0.01
    fee_bps: float = 6.0
    slippage_bps: float = 2.0
    max_holding_bars: int = 48
    ambiguous_policy: str = "stop_first"  # only conservative policy is accepted by default


@dataclass
class Trade:
    strategy: str
    signal_time: str
    entry_time: str
    exit_time: str
    direction: int
    entry: float
    exit: float
    stop: float
    target: float
    size: float
    gross_pnl: float
    fees: float
    funding_pnl: float
    net_pnl: float
    r_multiple: float
    reason: str
    ambiguous_exit: bool
    breakeven_activated: bool
    trailing_activated: bool
    mfe_r: float
    mae_r: float
    holding_bars: int


def _fill_entry(open_price: float, direction: int, slippage_bps: float) -> float:
    slip = slippage_bps / 10_000.0
    return open_price * (1 + slip * direction)


def _fill_exit(price: float, direction: int, slippage_bps: float) -> float:
    slip = slippage_bps / 10_000.0
    return price * (1 - slip * direction)


def run_backtest(df: pd.DataFrame, strategy: Callable[[pd.Series], StrategySignal], cfg: BacktestConfig, funding_events: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    required = {"open", "high", "low", "close", "atr14"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if cfg.ambiguous_policy != "stop_first":
        raise ValueError("Only stop_first is allowed without sub-bar data; this prevents optimistic same-bar fills.")

    equity = cfg.initial_equity
    trades: list[Trade] = []
    i = 0
    n = len(df)
    while i < n - 1:
        signal_row = df.iloc[i]
        signal = strategy(signal_row)
        if signal.direction == 0 or not np.isfinite(signal_row.get("atr14", np.nan)):
            i += 1
            continue

        # STRICT NO-LOOKAHEAD: decision at close[i], entry only at open[i+1].
        entry_i = i + 1
        entry_raw = float(df.iloc[entry_i]["open"])
        entry = _fill_entry(entry_raw, signal.direction, cfg.slippage_bps)
        stop_distance = float(signal_row["atr14"]) * signal.stop_distance_atr
        if stop_distance <= 0 or not math.isfinite(stop_distance):
            i += 1
            continue
        # Initial stop combines strategy ATR distance with recently confirmed market structure.
        lookback = df.iloc[max(0, i - 11): i + 1]
        atr = float(signal_row["atr14"])
        if signal.direction > 0:
            structure_stop = float(lookback["low"].min()) - 0.15 * atr
            atr_stop = entry - stop_distance
            raw_stop = min(atr_stop, structure_stop)
            # Prevent pathological position sizing from a distant historical wick.
            stop = max(raw_stop, entry - 3.5 * atr)
        else:
            structure_stop = float(lookback["high"].max()) + 0.15 * atr
            atr_stop = entry + stop_distance
            raw_stop = max(atr_stop, structure_stop)
            stop = min(raw_stop, entry + 3.5 * atr)
        stop_distance = abs(entry - stop)
        target = entry + signal.direction * stop_distance * signal.reward_r

        risk_budget = max(equity, 0) * cfg.risk_per_trade
        size = risk_budget / stop_distance
        exit_price = float(df.iloc[min(entry_i + cfg.max_holding_bars, n - 1)]["close"])
        exit_i = min(entry_i + cfg.max_holding_bars, n - 1)
        reason = "time_exit"
        ambiguous = False
        breakeven_activated = False
        trailing_activated = False
        max_favorable = 0.0
        max_adverse = 0.0
        current_stop = stop

        for j in range(entry_i, min(n, entry_i + cfg.max_holding_bars + 1)):
            bar = df.iloc[j]
            if signal.direction > 0:
                stop_hit = float(bar.low) <= current_stop
                target_hit = float(bar.high) >= target
                max_favorable = max(max_favorable, float(bar.high) - entry)
                max_adverse = max(max_adverse, entry - float(bar.low))
            else:
                stop_hit = float(bar.high) >= current_stop
                target_hit = float(bar.low) <= target
                max_favorable = max(max_favorable, entry - float(bar.low))
                max_adverse = max(max_adverse, float(bar.high) - entry)
            if stop_hit and target_hit:
                ambiguous = True
                exit_price, exit_i, reason = current_stop, j, "ambiguous_stop_first"
                break
            if stop_hit:
                exit_price, exit_i, reason = current_stop, j, "stop"
                break
            if target_hit:
                exit_price, exit_i, reason = target, j, "target"
                break

            # Stop updates happen only AFTER this bar closes, so they can only affect future bars.
            close_now = float(bar.close)
            atr_now = float(bar.atr14) if np.isfinite(bar.atr14) else atr
            favorable_close = (close_now - entry) * signal.direction
            if favorable_close >= stop_distance:
                be_stop = entry
                current_stop = max(current_stop, be_stop) if signal.direction > 0 else min(current_stop, be_stop)
                breakeven_activated = True
            if favorable_close >= 1.5 * stop_distance:
                trail = close_now - signal.direction * 1.2 * atr_now
                current_stop = max(current_stop, trail) if signal.direction > 0 else min(current_stop, trail)
                trailing_activated = True

        exit_fill = _fill_exit(float(exit_price), signal.direction, cfg.slippage_bps)
        gross = (exit_fill - entry) * signal.direction * size
        notional_in = abs(entry * size)
        notional_out = abs(exit_fill * size)
        fees = (notional_in + notional_out) * cfg.fee_bps / 10_000.0
        funding_pnl = 0.0
        if funding_events is not None and not funding_events.empty:
            entry_ts, exit_ts = df.index[entry_i], df.index[exit_i]
            events = funding_events[(funding_events.index > entry_ts) & (funding_events.index <= exit_ts)]
            for funding_ts, event in events.iterrows():
                # Use the last fully completed market candle strictly before settlement; never a post-settlement price.
                px_i = int(df.index.searchsorted(funding_ts, side="left")) - 1
                if px_i >= entry_i:
                    settlement_proxy = float(df.iloc[px_i]["close"])
                    funding_pnl += -signal.direction * abs(settlement_proxy * size) * float(event["funding_rate"])
        net = gross - fees + funding_pnl
        initial_risk = stop_distance * size
        r_multiple = net / initial_risk if initial_risk else np.nan
        equity += net
        trades.append(Trade(
            strategy=signal.name,
            signal_time=str(df.index[i]),
            entry_time=str(df.index[entry_i]),
            exit_time=str(df.index[exit_i]),
            direction=signal.direction,
            entry=entry,
            exit=exit_fill,
            stop=stop,
            target=target,
            size=size,
            gross_pnl=gross,
            fees=fees,
            funding_pnl=funding_pnl,
            net_pnl=net,
            r_multiple=r_multiple,
            reason=reason,
            ambiguous_exit=ambiguous,
            breakeven_activated=breakeven_activated,
            trailing_activated=trailing_activated,
            mfe_r=max_favorable / stop_distance if stop_distance else np.nan,
            mae_r=max_adverse / stop_distance if stop_distance else np.nan,
            holding_bars=int(exit_i - entry_i + 1),
        ))
        # One position per strategy: resume after exit, preventing overlapping duplicate trades.
        i = max(exit_i + 1, i + 1)

    trades_df = pd.DataFrame([asdict(t) for t in trades])
    metrics = compute_metrics(trades_df, cfg.initial_equity)
    metrics["final_equity"] = equity
    metrics["ambiguous_exit_count"] = int(trades_df["ambiguous_exit"].sum()) if not trades_df.empty else 0
    return trades_df, metrics


def compute_metrics(trades: pd.DataFrame, initial_equity: float) -> dict:
    if trades.empty:
        return {
            "trades": 0, "win_rate": None, "profit_factor": None, "expectancy_r": None,
            "net_pnl": 0.0, "return_pct": 0.0, "max_drawdown_pct": 0.0,
            "avg_win_r": None, "avg_loss_r": None, "max_consecutive_losses": 0,
        }
    pnl = trades["net_pnl"].astype(float)
    r = trades["r_multiple"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = wins.sum()
    gross_loss = -losses.sum()
    equity = initial_equity + pnl.cumsum()
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    streak = 0
    max_streak = 0
    for x in pnl:
        streak = streak + 1 if x < 0 else 0
        max_streak = max(max_streak, streak)
    return {
        "trades": int(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else None,
        "expectancy_r": float(r.mean()),
        "net_pnl": float(pnl.sum()),
        "return_pct": float(pnl.sum() / initial_equity),
        "max_drawdown_pct": float(dd.min()) if len(dd) else 0.0,
        "avg_win_r": float(r[r > 0].mean()) if (r > 0).any() else None,
        "avg_loss_r": float(r[r < 0].mean()) if (r < 0).any() else None,
        "max_consecutive_losses": int(max_streak),
        "avg_mfe_r": float(trades["mfe_r"].mean()) if "mfe_r" in trades else None,
        "avg_mae_r": float(trades["mae_r"].mean()) if "mae_r" in trades else None,
        "median_holding_bars": float(trades["holding_bars"].median()) if "holding_bars" in trades else None,
        "breakeven_activation_rate": float(trades["breakeven_activated"].mean()) if "breakeven_activated" in trades else None,
        "trailing_activation_rate": float(trades["trailing_activated"].mean()) if "trailing_activated" in trades else None,
    }

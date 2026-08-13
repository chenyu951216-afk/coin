from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from .position_sizing import (
    HIGH_PRICE_NOTIONAL_USDT,
    HIGH_PRICE_THRESHOLD_USDT,
    LOW_PRICE_NOTIONAL_USDT,
    paper_notional_for_price,
    sizing_tier_for_price,
)
from .signal_quality import cost_aware_breakeven_trigger, estimated_round_trip_cost_r, signal_snapshot
from .strategies import StrategySignal


@dataclass
class BacktestConfig:
    initial_equity: float = 10_000.0
    # Retained for API compatibility only. Simulation sizing is now fixed-notional.
    risk_per_trade: float = 0.01
    fee_bps: float = 6.0
    slippage_bps: float = 2.0
    max_holding_bars: int = 48
    ambiguous_policy: str = "stop_first"
    max_estimated_cost_r: float = 0.18
    paper_low_notional_usdt: float = LOW_PRICE_NOTIONAL_USDT
    paper_high_notional_usdt: float = HIGH_PRICE_NOTIONAL_USDT
    paper_high_price_threshold: float = HIGH_PRICE_THRESHOLD_USDT


@dataclass
class Trade:
    strategy: str
    signal_time: str
    entry_time: str
    exit_time: str
    direction: int
    direction_text: str
    entry_raw: float
    entry: float
    exit_trigger: float
    exit: float
    initial_stop: float
    stop_at_exit: float
    target: float
    stop_pct: float
    target_pct: float
    planned_reward_r: float
    estimated_cost_r: float
    sizing_tier: str
    planned_notional_usdt: float
    size: float
    entry_notional: float
    planned_stop_risk_usdt: float
    # Legacy-compatible alias. It now means planned stop risk, not equity-percent budget.
    risk_budget_usdt: float
    initial_risk_usdt: float
    equity_before: float
    equity_after: float
    gross_pnl_before_slippage: float
    slippage_cost: float
    gross_pnl: float
    fees: float
    funding_pnl: float
    net_pnl: float
    return_on_equity_pct: float
    r_multiple: float
    reason: str
    reason_text: str
    ambiguous_exit: bool
    breakeven_activated: bool
    trailing_activated: bool
    mfe_r: float
    mae_r: float
    holding_bars: int
    signal_features: dict[str, Any]


def _fill_entry(open_price: float, direction: int, slippage_bps: float) -> float:
    slip = slippage_bps / 10_000.0
    return open_price * (1 + slip * direction)


def _fill_exit(price: float, direction: int, slippage_bps: float) -> float:
    slip = slippage_bps / 10_000.0
    return price * (1 - slip * direction)


def _reason_text(reason: str) -> str:
    return {
        "stop": "止損／移動止損",
        "target": "止盈",
        "time_exit": "達持倉時間上限",
        "ambiguous_stop_first": "同一根 K 同時碰到止盈與止損，保守按止損先成交",
    }.get(reason, reason)


def run_backtest(
    df: pd.DataFrame,
    strategy: Callable[[pd.Series], StrategySignal],
    cfg: BacktestConfig,
    funding_events: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
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
    signals_seen = 0
    signals_rejected_cost = 0

    while i < n - 1:
        signal_row = df.iloc[i]
        signal = strategy(signal_row)
        if signal.direction == 0 or not np.isfinite(signal_row.get("atr14", np.nan)):
            i += 1
            continue
        signals_seen += 1

        # NO LOOKAHEAD: signal exists only after close[i]. Entry is open[i+1].
        # At the instant open[i+1] exists, that market price is legitimately known
        # and is therefore also the price used to select the fixed-notional tier.
        entry_i = i + 1
        entry_raw = float(df.iloc[entry_i]["open"])
        entry = _fill_entry(entry_raw, signal.direction, cfg.slippage_bps)
        atr = float(signal_row["atr14"])
        requested_stop_distance = atr * signal.stop_distance_atr
        if requested_stop_distance <= 0 or not math.isfinite(requested_stop_distance):
            i += 1
            continue

        # Stop structure uses only bars completed by the signal timestamp.
        lookback = df.iloc[max(0, i - 11): i + 1]
        if signal.direction > 0:
            structure_stop = float(lookback["low"].min()) - 0.15 * atr
            atr_stop = entry - requested_stop_distance
            raw_stop = min(atr_stop, structure_stop)
            initial_stop = max(raw_stop, entry - 3.5 * atr)
        else:
            structure_stop = float(lookback["high"].max()) + 0.15 * atr
            atr_stop = entry + requested_stop_distance
            raw_stop = max(atr_stop, structure_stop)
            initial_stop = min(raw_stop, entry + 3.5 * atr)

        stop_distance = abs(entry - initial_stop)
        if stop_distance <= 0 or not math.isfinite(stop_distance):
            i += 1
            continue

        estimated_cost_r = estimated_round_trip_cost_r(
            entry=entry,
            stop=initial_stop,
            fee_bps=cfg.fee_bps,
            slippage_bps=cfg.slippage_bps,
        )
        if cfg.max_estimated_cost_r > 0 and estimated_cost_r > cfg.max_estimated_cost_r:
            signals_rejected_cost += 1
            i += 1
            continue

        target = entry + signal.direction * stop_distance * signal.reward_r

        planned_notional = paper_notional_for_price(
            entry_raw,
            low_notional=cfg.paper_low_notional_usdt,
            high_notional=cfg.paper_high_notional_usdt,
            threshold=cfg.paper_high_price_threshold,
        )
        tier = sizing_tier_for_price(entry_raw, threshold=cfg.paper_high_price_threshold)
        # Size off the actual adverse-slippage entry fill so entry notional equals
        # the predeclared 2,000U / 20,000U target as closely as possible.
        size = planned_notional / entry
        if size <= 0 or not math.isfinite(size):
            i += 1
            continue

        planned_stop_risk = stop_distance * size
        equity_before = equity
        exit_i = min(entry_i + cfg.max_holding_bars, n - 1)
        exit_trigger = float(df.iloc[exit_i]["close"])
        reason = "time_exit"
        ambiguous = False
        breakeven_activated = False
        trailing_activated = False
        max_favorable = 0.0
        max_adverse = 0.0
        current_stop = initial_stop

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
                exit_trigger, exit_i, reason = current_stop, j, "ambiguous_stop_first"
                break
            if stop_hit:
                exit_trigger, exit_i, reason = current_stop, j, "stop"
                break
            if target_hit:
                exit_trigger, exit_i, reason = target, j, "target"
                break

            # Stop changes are computed only after this completed bar and can
            # affect later bars only. No intrabar hindsight is allowed.
            close_now = float(bar.close)
            atr_now = float(bar.atr14) if np.isfinite(bar.atr14) else atr
            favorable_close = (close_now - entry) * signal.direction
            if favorable_close >= stop_distance:
                be_trigger = cost_aware_breakeven_trigger(
                    entry_fill=entry,
                    direction=signal.direction,
                    fee_bps=cfg.fee_bps,
                    slippage_bps=cfg.slippage_bps,
                )
                current_stop = max(current_stop, be_trigger) if signal.direction > 0 else min(current_stop, be_trigger)
                breakeven_activated = True
            if favorable_close >= 1.5 * stop_distance:
                trail = close_now - signal.direction * 1.2 * atr_now
                current_stop = max(current_stop, trail) if signal.direction > 0 else min(current_stop, trail)
                trailing_activated = True

        exit_fill = _fill_exit(float(exit_trigger), signal.direction, cfg.slippage_bps)
        theoretical_gross = (float(exit_trigger) - entry_raw) * signal.direction * size
        gross = (exit_fill - entry) * signal.direction * size
        slippage_cost = max(0.0, theoretical_gross - gross)
        entry_notional = abs(entry * size)
        exit_notional = abs(exit_fill * size)
        fees = (entry_notional + exit_notional) * cfg.fee_bps / 10_000.0

        funding_pnl = 0.0
        if funding_events is not None and not funding_events.empty:
            entry_ts, exit_ts = df.index[entry_i], df.index[exit_i]
            events = funding_events[(funding_events.index > entry_ts) & (funding_events.index <= exit_ts)]
            for funding_ts, event in events.iterrows():
                # Strictly use the last completed candle BEFORE settlement.
                px_i = int(df.index.searchsorted(funding_ts, side="left")) - 1
                if px_i >= entry_i:
                    settlement_proxy = float(df.iloc[px_i]["close"])
                    funding_pnl += -signal.direction * abs(settlement_proxy * size) * float(event["funding_rate"])

        net = gross - fees + funding_pnl
        initial_risk = stop_distance * size
        r_multiple = net / initial_risk if initial_risk else np.nan
        equity = equity_before + net

        trades.append(Trade(
            strategy=signal.name,
            signal_time=str(df.index[i]),
            entry_time=str(df.index[entry_i]),
            exit_time=str(df.index[exit_i]),
            direction=signal.direction,
            direction_text="多" if signal.direction > 0 else "空",
            entry_raw=entry_raw,
            entry=entry,
            exit_trigger=float(exit_trigger),
            exit=exit_fill,
            initial_stop=initial_stop,
            stop_at_exit=current_stop,
            target=target,
            stop_pct=abs(entry - initial_stop) / entry if entry else np.nan,
            target_pct=abs(target - entry) / entry if entry else np.nan,
            planned_reward_r=signal.reward_r,
            estimated_cost_r=estimated_cost_r,
            sizing_tier=tier,
            planned_notional_usdt=planned_notional,
            size=size,
            entry_notional=entry_notional,
            planned_stop_risk_usdt=planned_stop_risk,
            risk_budget_usdt=planned_stop_risk,
            initial_risk_usdt=initial_risk,
            equity_before=equity_before,
            equity_after=equity,
            gross_pnl_before_slippage=theoretical_gross,
            slippage_cost=slippage_cost,
            gross_pnl=gross,
            fees=fees,
            funding_pnl=funding_pnl,
            net_pnl=net,
            return_on_equity_pct=(net / equity_before) if equity_before else np.nan,
            r_multiple=r_multiple,
            reason=reason,
            reason_text=_reason_text(reason),
            ambiguous_exit=ambiguous,
            breakeven_activated=breakeven_activated,
            trailing_activated=trailing_activated,
            mfe_r=max_favorable / stop_distance if stop_distance else np.nan,
            mae_r=max_adverse / stop_distance if stop_distance else np.nan,
            holding_bars=int(exit_i - entry_i + 1),
            signal_features=signal_snapshot(signal_row),
        ))

        # One open position per strategy instance. Resume only after it closes.
        i = max(exit_i + 1, i + 1)

    trades_df = pd.DataFrame([asdict(t) for t in trades])
    metrics = compute_metrics(trades_df, cfg.initial_equity)
    metrics["final_equity"] = float(equity)
    metrics["ambiguous_exit_count"] = int(trades_df["ambiguous_exit"].sum()) if not trades_df.empty else 0
    metrics["signals_seen"] = int(signals_seen)
    metrics["signals_rejected_cost"] = int(signals_rejected_cost)
    metrics["cost_rejection_rate"] = float(signals_rejected_cost / signals_seen) if signals_seen else None
    metrics["max_estimated_cost_r"] = float(cfg.max_estimated_cost_r)
    metrics["sizing_mode"] = "FIXED_NOTIONAL_BY_ENTRY_PRICE"
    metrics["paper_low_notional_usdt"] = float(cfg.paper_low_notional_usdt)
    metrics["paper_high_notional_usdt"] = float(cfg.paper_high_notional_usdt)
    metrics["paper_high_price_threshold"] = float(cfg.paper_high_price_threshold)
    return trades_df, metrics


def compute_metrics(trades: pd.DataFrame, initial_equity: float) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "profit_factor": None,
            "expectancy_r": None,
            "net_pnl": 0.0,
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_trade_pnl": None,
            "largest_win": None,
            "largest_loss": None,
            "total_fees": 0.0,
            "total_slippage_cost": 0.0,
            "total_funding_pnl": 0.0,
            "avg_win_r": None,
            "avg_loss_r": None,
            "max_consecutive_losses": 0,
            "avg_mfe_r": None,
            "avg_mae_r": None,
            "median_holding_bars": None,
            "breakeven_activation_rate": None,
            "trailing_activation_rate": None,
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
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else None,
        "expectancy_r": float(r.mean()),
        "net_pnl": float(pnl.sum()),
        "return_pct": float(pnl.sum() / initial_equity),
        "max_drawdown_pct": float(dd.min()) if len(dd) else 0.0,
        "avg_trade_pnl": float(pnl.mean()),
        "largest_win": float(pnl.max()),
        "largest_loss": float(pnl.min()),
        "total_fees": float(trades["fees"].sum()) if "fees" in trades else 0.0,
        "total_slippage_cost": float(trades["slippage_cost"].sum()) if "slippage_cost" in trades else 0.0,
        "total_funding_pnl": float(trades["funding_pnl"].sum()) if "funding_pnl" in trades else 0.0,
        "avg_win_r": float(r[r > 0].mean()) if (r > 0).any() else None,
        "avg_loss_r": float(r[r < 0].mean()) if (r < 0).any() else None,
        "max_consecutive_losses": int(max_streak),
        "avg_mfe_r": float(trades["mfe_r"].mean()) if "mfe_r" in trades else None,
        "avg_mae_r": float(trades["mae_r"].mean()) if "mae_r" in trades else None,
        "median_holding_bars": float(trades["holding_bars"].median()) if "holding_bars" in trades else None,
        "breakeven_activation_rate": float(trades["breakeven_activated"].mean()) if "breakeven_activated" in trades else None,
        "trailing_activation_rate": float(trades["trailing_activated"].mean()) if "trailing_activated" in trades else None,
    }

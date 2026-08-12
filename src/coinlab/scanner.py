from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import pandas as pd

from .exchange import BitgetV2Client
from .features import build_feature_frame
from .providers import BitgetPublicClient, CoinGlassClient
from .strategies import STRATEGIES, StrategySignal
from .universe import get_coinglass_exchange_pairs, resolve_coinglass_instrument


_INTERVAL_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
    "1d": 1440, "1w": 10080,
}


@dataclass(frozen=True)
class ScanConfig:
    timeframe: str = "15m"
    lookback_bars: int = 180
    min_aligned_rows: int = 110
    min_turnover_usdt: float = 1_000_000.0
    max_spread_pct: float = 0.50
    max_symbols: int = 0
    coinglass_exchange: str = "Bitget"


def _iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def seconds_until_next_completed_bar(timeframe: str, now: datetime | None = None, grace_seconds: int = 8) -> int:
    minutes = _INTERVAL_MINUTES[timeframe]
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    interval_seconds = minutes * 60
    epoch = int(current.timestamp())
    next_boundary = ((epoch // interval_seconds) + 1) * interval_seconds + grace_seconds
    return max(1, next_boundary - epoch)


def next_completed_bar_time(timeframe: str, now: datetime | None = None, grace_seconds: int = 8) -> str:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return _iso(current + timedelta(seconds=seconds_until_next_completed_bar(timeframe, current, grace_seconds)))


def _closed_window(timeframe: str, bars: int) -> tuple[str, str]:
    minutes = _INTERVAL_MINUTES[timeframe]
    now = datetime.now(timezone.utc)
    epoch_minute = int(now.timestamp() // 60)
    current_bucket = epoch_minute - (epoch_minute % minutes)
    last_closed_open = datetime.fromtimestamp((current_bucket - minutes) * 60, tz=timezone.utc)
    start = last_closed_open - timedelta(minutes=minutes * max(bars - 1, 1))
    return _iso(start), _iso(last_closed_open)


def derive_strategy_levels(df: pd.DataFrame, signal: StrategySignal, entry: float | None = None) -> dict[str, float]:
    """Mirror the current backtest's initial ATR + confirmed-structure exit geometry."""
    if df.empty:
        raise ValueError("empty feature frame")
    row = df.iloc[-1]
    atr = float(row["atr14"])
    entry_price = float(entry if entry is not None else row["close"])
    stop_distance = atr * signal.stop_distance_atr
    lookback = df.iloc[max(0, len(df) - 12):]
    if signal.direction > 0:
        structure_stop = float(lookback["low"].min()) - 0.15 * atr
        atr_stop = entry_price - stop_distance
        raw_stop = min(atr_stop, structure_stop)
        stop = max(raw_stop, entry_price - 3.5 * atr)
    else:
        structure_stop = float(lookback["high"].max()) + 0.15 * atr
        atr_stop = entry_price + stop_distance
        raw_stop = max(atr_stop, structure_stop)
        stop = min(raw_stop, entry_price + 3.5 * atr)
    actual_distance = abs(entry_price - stop)
    target = entry_price + signal.direction * actual_distance * signal.reward_r
    return {
        "entry": entry_price,
        "stop_loss": stop,
        "take_profit": target,
        "atr": atr,
        "stop_pct": actual_distance / entry_price if entry_price else 0.0,
        "take_profit_pct": abs(target - entry_price) / entry_price if entry_price else 0.0,
    }


def _public_prefilter(exchange: BitgetV2Client, cfg: ScanConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contracts = {c["symbol"]: c for c in exchange.get_contracts()}
    tickers = exchange.get_tickers()
    candidates: list[dict[str, Any]] = []
    for t in tickers:
        symbol = t["symbol"]
        if symbol not in contracts:
            continue
        turnover = float(t.get("volume_24h_usdt") or 0)
        spread = t.get("spread_pct")
        if turnover < cfg.min_turnover_usdt:
            continue
        if spread is not None and float(spread) > cfg.max_spread_pct:
            continue
        candidates.append({**contracts[symbol], **t})
    candidates.sort(key=lambda x: float(x.get("volume_24h_usdt") or 0), reverse=True)
    if cfg.max_symbols > 0:
        candidates = candidates[:cfg.max_symbols]
    return candidates, {
        "bitget_tradable_contracts": len(contracts),
        "bitget_public_prefilter_pass": len(candidates),
    }


def scan_market(
    *,
    coinglass_api_key: str,
    cfg: ScanConfig,
    progress: Callable[[dict[str, Any]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    if cfg.timeframe not in _INTERVAL_MINUTES:
        raise ValueError(f"unsupported scan timeframe: {cfg.timeframe}")

    bitget_exchange = BitgetV2Client()
    bitget_market = BitgetPublicClient()
    cg = CoinGlassClient(coinglass_api_key)
    pairs = get_coinglass_exchange_pairs(coinglass_api_key, cfg.coinglass_exchange)
    pair_bases = {p.base_asset for p in pairs}
    candidates, stats = _public_prefilter(bitget_exchange, cfg)
    candidates = [c for c in candidates if c.get("base_coin") in pair_bases]
    stats["coinglass_supported_after_prefilter"] = len(candidates)

    start, end = _closed_window(cfg.timeframe, cfg.lookback_bars)
    granularity = cfg.timeframe.replace("h", "H") if cfg.timeframe.endswith("h") else cfg.timeframe
    matches: list[dict[str, Any]] = []
    no_signal: list[str] = []
    skipped: list[dict[str, str]] = []
    paused = False

    for i, candidate in enumerate(candidates, 1):
        if should_stop and should_stop():
            paused = True
            break
        symbol = str(candidate["symbol"])
        if progress:
            progress({"current": i, "total": len(candidates), "symbol": symbol})
        try:
            cg_symbol = resolve_coinglass_instrument(
                coinglass_api_key, cfg.coinglass_exchange, symbol, pairs=pairs
            )
            common = dict(
                exchange=cfg.coinglass_exchange,
                symbol=cg_symbol,
                interval=cfg.timeframe,
                start=start,
                end=end,
            )
            price = bitget_market.candles(symbol, granularity, start, end)
            datasets = {
                "oi": cg.open_interest(**common),
                "funding": cg.funding(**common),
                "liq": cg.liquidations(**common),
                "ls": cg.long_short(**common),
                "taker": cg.taker_flow(**common),
                "orderbook": cg.orderbook(**common),
            }
            if price.empty or any(v.empty for v in datasets.values()):
                raise RuntimeError("one or more required history sources returned no rows")
            frame = build_feature_frame(
                price, datasets["oi"], datasets["funding"], datasets["liq"],
                datasets["ls"], datasets["taker"], datasets["orderbook"]
            ).dropna(subset=["atr14"])
            if len(frame) < cfg.min_aligned_rows:
                raise RuntimeError(f"only {len(frame)} aligned rows; need {cfg.min_aligned_rows}")

            latest = frame.iloc[-1]
            symbol_matched = False
            for strategy_name, fn in STRATEGIES.items():
                sig = fn(latest)
                if sig.direction == 0:
                    continue
                symbol_matched = True
                levels = derive_strategy_levels(frame, sig)
                direction = "long" if sig.direction > 0 else "short"
                signal_time = str(frame.index[-1])
                signal_key = f"{symbol}|{strategy_name}|{direction}|{signal_time}"
                matches.append({
                    "signal_key": signal_key,
                    "symbol": symbol,
                    "base_coin": candidate.get("base_coin"),
                    "strategy": strategy_name,
                    "direction": direction,
                    "direction_text": "多" if sig.direction > 0 else "空",
                    "signal_time": signal_time,
                    "reference_price": levels["entry"],
                    "stop_loss": levels["stop_loss"],
                    "take_profit": levels["take_profit"],
                    "stop_pct": levels["stop_pct"],
                    "take_profit_pct": levels["take_profit_pct"],
                    "reward_r": sig.reward_r,
                    "atr": levels["atr"],
                    "volume_24h_usdt": candidate.get("volume_24h_usdt"),
                    "spread_pct": candidate.get("spread_pct"),
                    "coinglass_exchange": cfg.coinglass_exchange,
                    "coinglass_instrument": cg_symbol,
                    "aligned_rows": len(frame),
                })
            if not symbol_matched:
                no_signal.append(symbol)
        except Exception as exc:
            skipped.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})

    matches.sort(key=lambda x: (x["symbol"], x["strategy"]))
    evaluated = max(0, (i if candidates else 0) - len(skipped)) if paused else len(candidates) - len(skipped)
    return {
        "status": "paused" if paused else "completed",
        "timeframe": cfg.timeframe,
        "window": {"start": start, "end": end},
        "stats": {
            **stats,
            "evaluated": evaluated,
            "matched_signals": len(matches),
            "matched_symbols": len({m["symbol"] for m in matches}),
            "no_signal_symbols": len(no_signal),
            "skipped": len(skipped),
        },
        "matches": matches,
        "skipped": skipped[:100],
    }

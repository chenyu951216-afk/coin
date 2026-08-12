from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import pandas as pd

from .exchange import BitgetV2Client
from .providers import BitgetPublicClient, CoinGlassClient
from .research import SOURCE_LABELS, build_strategy_frame, source_status, strategy_requirements
from .signal_quality import estimated_round_trip_cost_r
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
    fee_bps: float = 6.0
    slippage_bps: float = 2.0
    max_estimated_cost_r: float = 0.18


def _iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _closed_window(timeframe: str, bars: int) -> tuple[str, str]:
    minutes = _INTERVAL_MINUTES[timeframe]
    now = datetime.now(timezone.utc)
    epoch_minute = int(now.timestamp() // 60)
    current_bucket = epoch_minute - (epoch_minute % minutes)
    last_closed_open = datetime.fromtimestamp((current_bucket - minutes) * 60, tz=timezone.utc)
    start = last_closed_open - timedelta(minutes=minutes * max(bars - 1, 1))
    return _iso(start), _iso(last_closed_open)


def next_completed_bar_time(
    timeframe: str,
    now: datetime | None = None,
    grace_seconds: int = 8,
) -> str:
    minutes = _INTERVAL_MINUTES[timeframe]
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    minute_bucket = int(current.timestamp() // 60)
    next_bucket = minute_bucket - (minute_bucket % minutes) + minutes
    return _iso(datetime.fromtimestamp(next_bucket * 60 + max(0, grace_seconds), tz=timezone.utc))


def seconds_until_next_completed_bar(
    timeframe: str,
    now: datetime | None = None,
    grace_seconds: int = 8,
) -> float:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    target = pd.Timestamp(next_completed_bar_time(timeframe, now=current, grace_seconds=grace_seconds))
    current_ts = pd.Timestamp(current)
    return max(1.0, float((target - current_ts).total_seconds()))


def derive_strategy_levels(df: pd.DataFrame, signal: StrategySignal, entry: float | None = None) -> dict[str, float]:
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
        "stop_pct": actual_distance / entry_price if entry_price else 0.0,
        "take_profit_pct": abs(target - entry_price) / entry_price if entry_price else 0.0,
        "atr": atr,
    }


def _public_prefilter(exchange: BitgetV2Client, cfg: ScanConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contracts = {c["symbol"]: c for c in exchange.get_contracts()}
    candidates = []
    for ticker in exchange.get_tickers():
        symbol = ticker["symbol"]
        if symbol not in contracts:
            continue
        turnover = float(ticker.get("volume_24h_usdt") or 0)
        spread = ticker.get("spread_pct")
        if turnover < cfg.min_turnover_usdt:
            continue
        if spread is not None and float(spread) > cfg.max_spread_pct:
            continue
        candidates.append({**contracts[symbol], **ticker})
    candidates.sort(key=lambda x: float(x.get("volume_24h_usdt") or 0), reverse=True)
    if cfg.max_symbols > 0:
        candidates = candidates[: cfg.max_symbols]
    return candidates, {"bitget_tradable_contracts": len(contracts), "bitget_public_prefilter_pass": len(candidates)}


def _source_calls(cg: CoinGlassClient):
    return {
        "oi": cg.open_interest, "funding": cg.funding, "liq": cg.liquidations,
        "ls": cg.long_short, "taker": cg.taker_flow, "orderbook": cg.orderbook,
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
    skipped_symbols: list[dict[str, Any]] = []
    source_fail_counts = {name: 0 for name in SOURCE_LABELS}
    strategy_skip_counts = {name: 0 for name in STRATEGIES}
    cost_rejected_signals = 0

    for i, candidate in enumerate(candidates, 1):
        if should_stop and should_stop():
            return {
                "status": "paused", "timeframe": cfg.timeframe, "window": {"start": start, "end": end},
                "stats": {**stats, "evaluated": i - 1, "matched_signals": len(matches),
                          "matched_symbols": len({m["symbol"] for m in matches}),
                          "cost_rejected_signals": cost_rejected_signals,
                          "source_fail_counts": source_fail_counts, "strategy_skip_counts": strategy_skip_counts},
                "matches": matches, "skipped": skipped_symbols[:100],
            }
        symbol = str(candidate["symbol"])
        if progress:
            progress({"current": i, "total": len(candidates), "symbol": symbol})
        try:
            cg_symbol = resolve_coinglass_instrument(coinglass_api_key, cfg.coinglass_exchange, symbol, pairs=pairs)
            common = {"exchange": cfg.coinglass_exchange, "symbol": cg_symbol, "interval": cfg.timeframe, "start": start, "end": end}
            price = bitget_market.candles(symbol, granularity, start, end)
            if price.empty:
                skipped_symbols.append({"symbol": symbol, "reason": "Bitget 沒有回傳價格 K 線。"})
                continue

            datasets: dict[str, pd.DataFrame] = {}
            source_diags: dict[str, Any] = {}
            for source_name, method in _source_calls(cg).items():
                if should_stop and should_stop():
                    break
                try:
                    frame = method(**common)
                    datasets[source_name] = frame
                    source_diags[source_name] = source_status(frame)
                    if frame.empty:
                        source_fail_counts[source_name] += 1
                except Exception as exc:
                    datasets[source_name] = pd.DataFrame()
                    source_diags[source_name] = source_status(None, exc)
                    source_fail_counts[source_name] += 1
            if should_stop and should_stop():
                continue

            symbol_matched = False
            any_strategy_ready = False
            frame_cache: dict[tuple[str, ...], pd.DataFrame | None] = {}
            diag_cache: dict[tuple[str, ...], dict[str, Any]] = {}
            for strategy_name, fn in STRATEGIES.items():
                requirements = strategy_requirements(strategy_name)
                if requirements not in frame_cache:
                    frame, diagnostic = build_strategy_frame(
                        strategy_name=strategy_name,
                        price=price,
                        datasets=datasets,
                        min_coverage=0.90,
                        min_rows=cfg.min_aligned_rows,
                    )
                    frame_cache[requirements] = frame
                    diag_cache[requirements] = diagnostic
                frame = frame_cache[requirements]
                diagnostic = diag_cache[requirements]
                if frame is None:
                    strategy_skip_counts[strategy_name] += 1
                    continue
                any_strategy_ready = True
                usable = frame.dropna(subset=["atr14"])
                if usable.empty:
                    strategy_skip_counts[strategy_name] += 1
                    continue
                signal = fn(usable.iloc[-1])
                if signal.direction == 0:
                    continue
                levels = derive_strategy_levels(frame, signal)
                estimated_cost_r = estimated_round_trip_cost_r(
                    entry=levels["entry"],
                    stop=levels["stop_loss"],
                    fee_bps=cfg.fee_bps,
                    slippage_bps=cfg.slippage_bps,
                )
                if cfg.max_estimated_cost_r > 0 and estimated_cost_r > cfg.max_estimated_cost_r:
                    cost_rejected_signals += 1
                    continue
                symbol_matched = True
                signal_time = str(frame.index[-1])
                direction = "long" if signal.direction > 0 else "short"
                matches.append({
                    "signal_key": f"{symbol}|{strategy_name}|{direction}|{signal_time}",
                    "symbol": symbol, "base_coin": candidate.get("base_coin"), "strategy": strategy_name,
                    "direction": direction, "direction_text": "做多" if signal.direction > 0 else "做空",
                    "signal_time": signal_time, "reference_price": levels["entry"],
                    "stop_loss": levels["stop_loss"], "stop_pct": levels["stop_pct"],
                    "take_profit": levels["take_profit"], "take_profit_pct": levels["take_profit_pct"],
                    "reward_r": signal.reward_r, "estimated_cost_r": estimated_cost_r, "atr": levels["atr"],
                    "volume_24h_usdt": candidate.get("volume_24h_usdt"), "spread_pct": candidate.get("spread_pct"),
                    "coinglass_exchange": cfg.coinglass_exchange, "coinglass_instrument": cg_symbol,
                    "aligned_rows": len(frame), "data_window_start": diagnostic.get("common_start"),
                    "data_window_end": diagnostic.get("common_end"), "required_sources": list(requirements),
                })
            if not symbol_matched:
                no_signal.append(symbol)
            if not any_strategy_ready:
                failed_sources = [SOURCE_LABELS.get(name, name) for name, diag in source_diags.items() if diag.get("status") != "ready"]
                skipped_symbols.append({
                    "symbol": symbol,
                    "reason": "沒有任何策略具備完整必要資料。" + (f" 缺失／失敗來源：{'、'.join(failed_sources)}。" if failed_sources else ""),
                })
        except Exception as exc:
            skipped_symbols.append({"symbol": symbol, "reason": f"幣種資料準備失敗：{type(exc).__name__}: {exc}"})

    matches.sort(key=lambda x: (x["symbol"], x["strategy"]))
    return {
        "status": "completed", "timeframe": cfg.timeframe, "window": {"start": start, "end": end},
        "stats": {**stats, "evaluated": len(candidates), "matched_signals": len(matches),
                  "matched_symbols": len({m["symbol"] for m in matches}), "no_signal_symbols": len(no_signal),
                  "skipped_symbols": len(skipped_symbols), "cost_rejected_signals": cost_rejected_signals,
                  "source_fail_counts": source_fail_counts, "strategy_skip_counts": strategy_skip_counts},
        "matches": matches, "skipped": skipped_symbols[:100],
    }

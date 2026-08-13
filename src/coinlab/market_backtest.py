from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, compute_metrics, run_backtest
from .exchange import BitgetV2Client
from .history_policy import normalize_backtest_window
from .providers import BitgetPublicClient, CoinGlassClient
from .research import SOURCE_LABELS, build_strategy_frame, source_status, strategy_requirements
from .strategies import STRATEGIES, StrategySignal
from .universe import get_coinglass_exchange_pairs, resolve_coinglass_instrument


@dataclass(frozen=True)
class MarketBacktestConfig:
    timeframe: str = "15m"
    min_historical_24h_turnover_usdt: float = 1_000_000.0
    max_symbols: int = 0
    fee_bps: float = 6.0
    slippage_bps: float = 2.0
    max_estimated_cost_r: float = 0.18
    low_notional_usdt: float = 2_000.0
    high_notional_usdt: float = 20_000.0
    high_price_threshold_usdt: float = 50.0
    initial_equity: float = 10_000.0
    min_aligned_coverage: float = 0.90
    min_aligned_rows: int = 500
    coinglass_exchange: str = "Bitget"
    bitget_base_url: str = "https://api.bitget.com"
    cache_root: str = "artifacts/market_cache"


@dataclass(frozen=True)
class GlobalTimeSplit:
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    embargo_seconds: int


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(value))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _timeframe_seconds(timeframe: str) -> int:
    unit = timeframe[-1].lower()
    value = int(timeframe[:-1])
    return value * {"m": 60, "h": 3600, "d": 86400}[unit]


def _bars_24h(timeframe: str) -> int:
    return max(1, int(round(86400 / _timeframe_seconds(timeframe))))


def _granularity(timeframe: str) -> str:
    return timeframe.replace("h", "H") if timeframe.endswith("h") else timeframe


def _cache_file(root: Path, namespace: str, symbol: str, timeframe: str, start: str, end: str) -> Path:
    key = hashlib.sha256(f"{namespace}|{symbol}|{timeframe}|{start}|{end}".encode()).hexdigest()[:20]
    path = root / _safe_name(namespace) / _safe_name(timeframe)
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{_safe_name(symbol)}_{key}.pkl"


def _cached_frame(path: Path, loader: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    if path.exists():
        try:
            value = pd.read_pickle(path)
            if isinstance(value, pd.DataFrame):
                return value
        except Exception:
            pass
    value = loader()
    if isinstance(value, pd.DataFrame):
        try:
            value.to_pickle(path)
        except Exception:
            pass
    return value


def _source_calls(cg: CoinGlassClient) -> dict[str, Callable[..., pd.DataFrame]]:
    return {
        "oi": cg.open_interest,
        "funding": cg.funding,
        "liq": cg.liquidations,
        "ls": cg.long_short,
        "taker": cg.taker_flow,
        "orderbook": cg.orderbook,
    }


def _historical_liquidity_possible(price: pd.DataFrame, timeframe: str, minimum: float) -> tuple[bool, float]:
    if price.empty or "volume_quote" not in price.columns:
        return False, 0.0
    bars = _bars_24h(timeframe)
    volume = pd.to_numeric(price["volume_quote"], errors="coerce").clip(lower=0)
    rolling = volume.rolling(bars, min_periods=bars).sum()
    peak = float(rolling.max()) if rolling.notna().any() else 0.0
    return bool(peak >= minimum), peak


def _liquidity_wrapped_strategy(
    strategy: Callable[[pd.Series], StrategySignal],
    minimum: float,
) -> Callable[[pd.Series], StrategySignal]:
    def wrapped(row: pd.Series) -> StrategySignal:
        value = row.get("quote_volume_24h", np.nan)
        if not np.isfinite(value) or float(value) < minimum:
            return StrategySignal(0, name=getattr(strategy, "__name__", "liquidity_filtered"))
        return strategy(row)
    return wrapped


def _global_split(start: str, end: str, timeframe: str, max_holding_bars: int = 48) -> GlobalTimeSplit:
    a = pd.Timestamp(start)
    b = pd.Timestamp(end)
    total = b - a
    cut1 = a + total * 0.60
    cut2 = a + total * 0.80
    embargo = timedelta(seconds=_timeframe_seconds(timeframe) * max_holding_bars)
    train_end = cut1 - embargo
    valid_start = cut1 + embargo
    valid_end = cut2 - embargo
    test_start = cut2 + embargo
    return GlobalTimeSplit(
        train_start=a.isoformat(),
        train_end=train_end.isoformat(),
        validation_start=valid_start.isoformat(),
        validation_end=valid_end.isoformat(),
        test_start=test_start.isoformat(),
        test_end=b.isoformat(),
        embargo_seconds=int(embargo.total_seconds()),
    )


def _trade_slice(trades: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    ts = pd.to_datetime(trades["entry_time"], utc=True)
    return trades[(ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))].copy()


def _fold_metrics(trades: pd.DataFrame, start: str, end: str, folds: int, initial_equity: float) -> list[dict[str, Any]]:
    a = pd.Timestamp(start)
    b = pd.Timestamp(end)
    edges = [a + (b - a) * (i / folds) for i in range(folds + 1)]
    out = []
    for i in range(folds):
        part = _trade_slice(trades, edges[i].isoformat(), edges[i + 1].isoformat())
        out.append({
            "fold": i + 1,
            "start": edges[i].isoformat(),
            "end": edges[i + 1].isoformat(),
            "metrics": compute_metrics(part.sort_values("exit_time") if not part.empty else part, initial_equity),
        })
    return out


def _portfolio_exposure(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {"peak_open_tickets": 0, "peak_gross_notional_usdt": 0.0, "same_symbol_overlap_events": 0, "opposing_same_symbol_overlap_events": 0}
    events: list[tuple[pd.Timestamp, int, float]] = []
    overlap = 0
    opposing = 0
    for _, row in trades.iterrows():
        enter = pd.Timestamp(row["entry_time"])
        exit_ = pd.Timestamp(row["exit_time"])
        notional = float(row.get("planned_notional_usdt") or row.get("entry_notional") or 0.0)
        events.append((enter, +1, +notional))
        events.append((exit_, -1, -notional))
    # On identical timestamps, close before opening another ticket.
    events.sort(key=lambda x: (x[0], x[1]))
    open_count = 0
    gross = 0.0
    peak_count = 0
    peak_gross = 0.0
    for _, delta_count, delta_notional in events:
        open_count += delta_count
        gross += delta_notional
        peak_count = max(peak_count, open_count)
        peak_gross = max(peak_gross, gross)

    for symbol, group in trades.groupby("symbol"):
        rows = group.sort_values("entry_time").to_dict("records")
        active: list[dict[str, Any]] = []
        for row in rows:
            start = pd.Timestamp(row["entry_time"])
            active = [x for x in active if pd.Timestamp(x["exit_time"]) > start]
            for other in active:
                overlap += 1
                if int(other.get("direction", 0)) != int(row.get("direction", 0)):
                    opposing += 1
            active.append(row)
    return {
        "peak_open_tickets": int(peak_count),
        "peak_gross_notional_usdt": float(peak_gross),
        "same_symbol_overlap_events": int(overlap),
        "opposing_same_symbol_overlap_events": int(opposing),
    }


def _summary_group(trades: pd.DataFrame, key: str, initial_equity: float) -> list[dict[str, Any]]:
    if trades.empty or key not in trades.columns:
        return []
    out = []
    for value, group in trades.groupby(key):
        metrics = compute_metrics(group.sort_values("exit_time"), initial_equity)
        out.append({key: value, **metrics})
    return sorted(out, key=lambda x: str(x[key]))


def _write_packages(root: Path, report: dict[str, Any], trades: pd.DataFrame, split: GlobalTimeSplit) -> tuple[Path, Path]:
    research = root / f"COINLAB_MARKET_RESEARCH_{report['metadata']['timeframe']}_{root.name}.zip"
    audit = root / f"COINLAB_MARKET_AUDIT_{report['metadata']['timeframe']}_{root.name}.zip"
    dev = trades[pd.to_datetime(trades["entry_time"], utc=True) < pd.Timestamp(split.test_start)].copy() if not trades.empty else trades.copy()
    train = _trade_slice(trades, split.train_start, split.train_end)
    valid = _trade_slice(trades, split.validation_start, split.validation_end)
    research_payload = {
        "schema_version": "coinlab.market_research.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": report["metadata"],
        "universe": report["universe"],
        "data_integrity": report["data_integrity"],
        "sizing_policy": report["sizing_policy"],
        "global_split": asdict(split),
        "train_metrics": compute_metrics(train.sort_values("exit_time") if not train.empty else train, report["metadata"]["initial_equity"]),
        "validation_metrics": compute_metrics(valid.sort_values("exit_time") if not valid.empty else valid, report["metadata"]["initial_equity"]),
        "development_walk_forward": _fold_metrics(dev, split.train_start, split.validation_end, 5, report["metadata"]["initial_equity"]),
        "by_strategy_development": _summary_group(dev, "strategy", report["metadata"]["initial_equity"]),
        "by_symbol_development": _summary_group(dev, "symbol", report["metadata"]["initial_equity"]),
        "important": "Locked test trades/metrics are intentionally excluded from this research package. Do not remove dates/symbols/trades because of realized PnL.",
    }
    research_json = json.dumps(_json_safe(research_payload), ensure_ascii=False, indent=2, sort_keys=True).encode()
    dev_csv = dev.to_csv(index=False).encode("utf-8-sig") if not dev.empty else b""
    skipped_csv = pd.DataFrame(report.get("skipped_symbols", [])).to_csv(index=False).encode("utf-8-sig")
    with ZipFile(research, "w", ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("CHATGPT_MARKET_RESEARCH_INPUT.json", research_json)
        z.writestr("trades_development_all_symbols.csv", dev_csv)
        z.writestr("skipped_symbols.csv", skipped_csv)
        z.writestr("README_研究規則.txt", (
            "這是全市場 development 研究包。\n"
            "只能使用 train、validation、development walk-forward 與進場當下特徵研究策略。\n"
            "禁止依照某一天、某個幣或某筆已知盈虧新增例外。\n"
            "Locked test 不在此包中，避免反覆調參污染 OOS。\n"
        ).encode("utf-8"))
    report_path = root / "MARKET_BACKTEST_REPORT.json"
    trades_path = root / "all_market_trades.csv"
    with ZipFile(audit, "w", ZIP_DEFLATED, compresslevel=9) as z:
        z.write(report_path, report_path.name)
        z.write(trades_path, trades_path.name)
        for extra in (root / "symbol_strategy_summary.csv", root / "skipped_symbols.csv"):
            if extra.exists():
                z.write(extra, extra.name)
        z.writestr("README_完整稽核包.txt", "包含 locked test。只用於候選策略凍結後的最終稽核，不可拿 test 單筆盈虧反覆調參。\n")
    return research, audit


def run_market_backtest(
    *,
    coinglass_api_key: str,
    requested_start: str = "",
    requested_end: str = "",
    cfg: MarketBacktestConfig,
    outdir: str | Path,
    progress: Callable[[dict[str, Any]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    if not coinglass_api_key:
        raise RuntimeError("COINGLASS_API_KEY is required")
    window = normalize_backtest_window(timeframe=cfg.timeframe, requested_start=requested_start, requested_end=requested_end)
    start, end = window.used_start, window.used_end
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    cache = Path(cfg.cache_root)
    cache.mkdir(parents=True, exist_ok=True)

    exchange = BitgetV2Client(base_url=cfg.bitget_base_url)
    market = BitgetPublicClient(base_url=cfg.bitget_base_url)
    cg = CoinGlassClient(coinglass_api_key)
    pairs = get_coinglass_exchange_pairs(coinglass_api_key, cfg.coinglass_exchange)
    pair_bases = {p.base_asset for p in pairs}
    contracts = [c for c in exchange.get_contracts() if str(c.get("base_coin") or "").upper() in pair_bases]
    contracts.sort(key=lambda c: str(c.get("symbol") or ""))
    universe_total = len(contracts)
    if cfg.max_symbols > 0:
        contracts = contracts[: cfg.max_symbols]

    snapshot = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "symbols": [c.get("symbol") for c in contracts],
        "definition": "CURRENT_BITGET_TRADABLE_USDT_CONTRACTS_INTERSECT_CURRENT_COINGLASS_BITGET_SUPPORTED_PAIRS",
    }
    snap_dir = cache / "universe_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / f"bitget_{datetime.now(timezone.utc).date().isoformat()}.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    bt_cfg = BacktestConfig(
        initial_equity=cfg.initial_equity,
        fee_bps=cfg.fee_bps,
        slippage_bps=cfg.slippage_bps,
        max_estimated_cost_r=cfg.max_estimated_cost_r,
        paper_low_notional_usdt=cfg.low_notional_usdt,
        paper_high_notional_usdt=cfg.high_notional_usdt,
        paper_high_price_threshold=cfg.high_price_threshold_usdt,
    )
    source_methods = _source_calls(cg)
    all_trades: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    strategy_signal_stats: dict[str, dict[str, int]] = {name: {"signals_seen": 0, "signals_rejected_cost": 0, "trades": 0} for name in STRATEGIES}

    for idx, contract in enumerate(contracts, 1):
        if should_stop and should_stop():
            break
        symbol = str(contract.get("symbol") or "")
        base = str(contract.get("base_coin") or "")
        if progress:
            progress({"current": idx, "total": len(contracts), "symbol": symbol, "stage": "price"})
        try:
            price_cache = _cache_file(cache, "bitget_price", symbol, cfg.timeframe, start, end)
            price = _cached_frame(price_cache, lambda s=symbol: market.candles(s, _granularity(cfg.timeframe), start, end))
            if price.empty:
                skipped.append({"symbol": symbol, "reason": "BITGET_PRICE_EMPTY"})
                continue
            possible, peak_24h = _historical_liquidity_possible(price, cfg.timeframe, cfg.min_historical_24h_turnover_usdt)
            if not possible:
                skipped.append({"symbol": symbol, "reason": "NEVER_REACHED_HISTORICAL_24H_TURNOVER_MINIMUM", "peak_historical_24h_turnover_usdt": peak_24h})
                continue

            cg_symbol = resolve_coinglass_instrument(coinglass_api_key, cfg.coinglass_exchange, symbol, pairs=pairs)
            common = {"exchange": cfg.coinglass_exchange, "symbol": cg_symbol, "interval": cfg.timeframe, "start": start, "end": end}
            datasets: dict[str, pd.DataFrame] = {}
            source_diag: dict[str, Any] = {}
            for source_name, method in source_methods.items():
                if should_stop and should_stop():
                    break
                # CoinGlass official global account L/S endpoint does not list Bitget
                # as a supported exchange. Do not silently substitute another venue.
                if source_name == "ls" and cfg.coinglass_exchange.lower() == "bitget":
                    datasets[source_name] = pd.DataFrame()
                    source_diag[source_name] = {"status": "unsupported", "reason": "COINGLASS_GLOBAL_LONG_SHORT_DOES_NOT_SUPPORT_BITGET"}
                    continue
                if progress:
                    progress({"current": idx, "total": len(contracts), "symbol": symbol, "stage": source_name})
                path = _cache_file(cache, f"coinglass_{source_name}", cg_symbol, cfg.timeframe, start, end)
                try:
                    frame = _cached_frame(path, lambda m=method, kw=common: m(**kw))
                    datasets[source_name] = frame
                    source_diag[source_name] = source_status(frame)
                except Exception as exc:
                    datasets[source_name] = pd.DataFrame()
                    source_diag[source_name] = source_status(None, exc)

            funding_path = _cache_file(cache, "bitget_funding", symbol, cfg.timeframe, start, end)
            try:
                funding_events = _cached_frame(funding_path, lambda s=symbol: market.funding_history(s, start, end))
            except Exception as exc:
                funding_events = pd.DataFrame(columns=["funding_rate"])
                source_diag["bitget_funding_events"] = source_status(None, exc)
            else:
                source_diag["bitget_funding_events"] = source_status(funding_events)

            symbol_ready = False
            for strategy_name, fn in STRATEGIES.items():
                frame, diagnostic = build_strategy_frame(
                    strategy_name=strategy_name,
                    price=price,
                    datasets=datasets,
                    min_coverage=cfg.min_aligned_coverage,
                    min_rows=cfg.min_aligned_rows,
                )
                if frame is None:
                    diagnostics.append({"symbol": symbol, "strategy": strategy_name, **diagnostic})
                    continue
                symbol_ready = True
                wrapped = _liquidity_wrapped_strategy(fn, cfg.min_historical_24h_turnover_usdt)
                trades, metrics = run_backtest(frame, wrapped, bt_cfg, funding_events=funding_events)
                strategy_signal_stats[strategy_name]["signals_seen"] += int(metrics.get("signals_seen") or 0)
                strategy_signal_stats[strategy_name]["signals_rejected_cost"] += int(metrics.get("signals_rejected_cost") or 0)
                strategy_signal_stats[strategy_name]["trades"] += int(metrics.get("trades") or 0)
                diagnostics.append({
                    "symbol": symbol,
                    "strategy": strategy_name,
                    **diagnostic,
                    "metrics": metrics,
                    "historical_peak_24h_turnover_usdt": peak_24h,
                })
                if trades.empty:
                    continue
                trades.insert(0, "symbol", symbol)
                trades.insert(1, "base_coin", base)
                trades.insert(2, "timeframe", cfg.timeframe)
                trades.insert(3, "coinglass_instrument", cg_symbol)
                all_trades.append(trades)
            if not symbol_ready:
                skipped.append({"symbol": symbol, "reason": "NO_STRATEGY_HAD_COMPLETE_REQUIRED_DATA"})
        except Exception as exc:
            skipped.append({"symbol": symbol, "reason": "SYMBOL_PROCESSING_ERROR", "error_type": type(exc).__name__, "message": str(exc)})

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if not trades.empty:
        trades = trades.sort_values(["entry_time", "symbol", "strategy"]).reset_index(drop=True)
    trades_path = root / "all_market_trades.csv"
    trades.to_csv(trades_path, index=False)
    pd.DataFrame(skipped).to_csv(root / "skipped_symbols.csv", index=False)

    split = _global_split(start, end, cfg.timeframe, bt_cfg.max_holding_bars)
    train = _trade_slice(trades, split.train_start, split.train_end)
    valid = _trade_slice(trades, split.validation_start, split.validation_end)
    test = _trade_slice(trades, split.test_start, split.test_end)
    realized_order = trades.sort_values("exit_time") if not trades.empty else trades
    overall = compute_metrics(realized_order, cfg.initial_equity)
    exposure = _portfolio_exposure(trades)

    by_symbol_strategy = []
    if not trades.empty:
        for (symbol, strategy), group in trades.groupby(["symbol", "strategy"]):
            by_symbol_strategy.append({"symbol": symbol, "strategy": strategy, **compute_metrics(group.sort_values("exit_time"), cfg.initial_equity)})
    pd.DataFrame(by_symbol_strategy).to_csv(root / "symbol_strategy_summary.csv", index=False)

    report: dict[str, Any] = {
        "schema_version": "coinlab.market_backtest.v1",
        "status": "REAL_DATA_MARKET_BACKTEST",
        "metadata": {
            "timeframe": cfg.timeframe,
            "requested_start": requested_start,
            "requested_end": requested_end,
            "used_start": start,
            "used_end": end,
            "coinglass_standard_max_history_days": window.max_history_days,
            "initial_equity": cfg.initial_equity,
            "fee_bps": cfg.fee_bps,
            "slippage_bps": cfg.slippage_bps,
            "min_historical_24h_turnover_usdt": cfg.min_historical_24h_turnover_usdt,
        },
        "sizing_policy": {
            "mode": "FIXED_NOTIONAL_BY_ENTRY_MARKET_PRICE",
            "price_gt_50_usdt_notional": cfg.high_notional_usdt,
            "price_le_50_usdt_notional": cfg.low_notional_usdt,
            "threshold_usdt": cfg.high_price_threshold_usdt,
            "price_timestamp": "SIMULATED_ENTRY_TIME_OPEN_T_PLUS_1",
            "note": "The tier is chosen at simulated entry time, when that market price is actually knowable; no future price is used.",
        },
        "universe": {
            "candidate_contracts_before_debug_cap": universe_total,
            "candidate_contracts_processed": len(contracts),
            "definition": snapshot["definition"],
            "historical_trade_eligibility": "TRAILING_24H_BITGET_QUOTE_VOLUME_COMPUTED_FROM_COMPLETED_HISTORICAL_BARS_ONLY",
            "current_ticker_volume_used_for_past_trade_selection": False,
            "survivorship_bias_possible": True,
            "survivorship_bias_explanation": "For periods before locally archived universe snapshots existed, candidates come from contracts still tradable today; delisted historical contracts may therefore be absent. This limitation is reported rather than hidden.",
            "universe_snapshot_written": str(snap_dir),
        },
        "data_integrity": {
            "no_lookahead": "signal_after_close_t__entry_at_open_t_plus_1",
            "historical_liquidity_no_future": True,
            "same_bar_tp_sl": "STOP_FIRST_IF_ORDER_UNKNOWN",
            "stop_updates": "AFTER_COMPLETED_BAR_ONLY_AND_EFFECTIVE_ON_LATER_BARS",
            "missing_source_policy": "SKIP_ONLY_DEPENDENT_STRATEGY_NEVER_FILL_OR_INVENT_DATA",
            "long_short_bitget_policy": "UNSUPPORTED_SOURCE_IS_LEFT_MISSING_NOT_SILENTLY_REPLACED",
            "outcome_based_date_symbol_exclusion": "FORBIDDEN",
        },
        "global_split": asdict(split),
        "portfolio_summary": {
            **overall,
            **exposure,
            "train_metrics": compute_metrics(train.sort_values("exit_time") if not train.empty else train, cfg.initial_equity),
            "validation_metrics": compute_metrics(valid.sort_values("exit_time") if not valid.empty else valid, cfg.initial_equity),
            "locked_test_metrics": compute_metrics(test.sort_values("exit_time") if not test.empty else test, cfg.initial_equity),
            "ticket_model": "ALL_ELIGIBLE_STRATEGY_SIGNALS_ARE_SIMULATED_AS_INDEPENDENT_TICKETS",
            "one_way_live_equivalence_warning": "Opposing/overlapping tickets on the same symbol can differ from a future Bitget one-way account. Conflict counts are explicitly reported; live execution remains locked until an execution-arbitration policy is validated.",
        },
        "strategy_signal_stats": strategy_signal_stats,
        "by_strategy": _summary_group(trades, "strategy", cfg.initial_equity),
        "by_symbol": _summary_group(trades, "symbol", cfg.initial_equity),
        "by_symbol_strategy": by_symbol_strategy,
        "skipped_symbols": skipped,
        "strategy_diagnostics": diagnostics,
        "files": {
            "all_market_trades": trades_path.name,
            "all_market_trades_sha256": _sha256(trades_path),
            "symbol_strategy_summary": "symbol_strategy_summary.csv",
            "skipped_symbols": "skipped_symbols.csv",
        },
    }
    report_path = root / "MARKET_BACKTEST_REPORT.json"
    report_path.write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    research_zip, audit_zip = _write_packages(root, report, trades, split)
    report["files"]["research_package"] = research_zip.name
    report["files"]["audit_package"] = audit_zip.name
    report_path.write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report

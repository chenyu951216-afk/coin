from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import backtest as backtest_module
from . import features as features_module
from . import strategies as strategies_module
from .backtest import BacktestConfig, run_backtest
from .config import Settings
from .history_policy import normalize_backtest_window
from .providers import BitgetPublicClient, CoinGlassClient
from .reporting import save_report
from .research import SOURCE_LABELS, STRATEGY_SOURCE_REQUIREMENTS, build_strategy_frame, source_status
from .strategies import STRATEGIES
from .universe import resolve_coinglass_instrument
from .validation import evaluate_oos


def _module_sha256(module) -> str:
    return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


def _resolved_coinglass_symbol(s: Settings) -> str:
    configured = str(s.coinglass_symbol or "").strip()
    if configured and configured.upper() != "AUTO":
        return configured
    return resolve_coinglass_instrument(s.coinglass_api_key, s.coinglass_exchange, s.symbol)


def _source_calls(cg: CoinGlassClient):
    return {
        "oi": cg.open_interest,
        "funding": cg.funding,
        "liq": cg.liquidations,
        "ls": cg.long_short,
        "taker": cg.taker_flow,
        "orderbook": cg.orderbook,
    }


def fetch_research_inputs(s: Settings, *, start: str, end: str):
    cg_symbol = _resolved_coinglass_symbol(s)
    cg = CoinGlassClient(s.coinglass_api_key)
    bg = BitgetPublicClient(base_url=s.bitget_rest_base_url)
    common = {"exchange": s.coinglass_exchange, "symbol": cg_symbol, "interval": s.timeframe, "start": start, "end": end}
    granularity = s.timeframe.replace("h", "H") if s.timeframe.endswith("h") else s.timeframe

    print("COINLAB_STAGE:data:正在下載 Bitget 已完成 K 線。", flush=True)
    price = bg.candles(s.symbol, granularity, start, end)
    if price.empty:
        raise RuntimeError("Bitget returned no price candles; aborting rather than producing fake metrics.")

    datasets: dict[str, pd.DataFrame] = {}
    diagnostics: dict[str, Any] = {}
    for source_name, method in _source_calls(cg).items():
        label = SOURCE_LABELS.get(source_name, source_name)
        print(f"COINLAB_STAGE:data:正在取得 CoinGlass {label}。", flush=True)
        try:
            frame = method(**common)
            datasets[source_name] = frame
            diagnostics[source_name] = source_status(frame)
        except Exception as exc:
            datasets[source_name] = pd.DataFrame()
            diagnostics[source_name] = source_status(None, exc)

    print("COINLAB_STAGE:data:正在讀取 Bitget 真實 Funding 結算紀錄。", flush=True)
    try:
        funding_events = bg.funding_history(s.symbol, start, end)
        diagnostics["bitget_funding_events"] = {
            "status": "ready", "rows": int(len(funding_events)),
            "start": str(funding_events.index.min()) if len(funding_events) else None,
            "end": str(funding_events.index.max()) if len(funding_events) else None,
        }
    except Exception as exc:
        raise RuntimeError(f"Bitget funding history unavailable: {type(exc).__name__}: {exc}") from exc
    return price, funding_events, datasets, diagnostics, cg_symbol


def command_backtest(args):
    s = Settings()
    window = normalize_backtest_window(timeframe=s.timeframe, requested_start=s.start, requested_end=s.end)
    print(f"COINLAB_STAGE:window:{window.message}", flush=True)
    price, funding_events, datasets, source_diagnostics, cg_symbol = fetch_research_inputs(
        s, start=window.used_start, end=window.used_end
    )

    cfg = BacktestConfig(
        initial_equity=s.initial_equity,
        risk_per_trade=s.risk_per_trade,
        fee_bps=s.taker_fee_bps,
        slippage_bps=s.slippage_bps,
    )
    results: dict[str, tuple[pd.DataFrame, dict]] = {}
    validation: dict[str, Any] = {}
    features_by_strategy: dict[str, pd.DataFrame] = {}
    strategy_diagnostics: dict[str, dict[str, Any]] = {}

    print("COINLAB_STAGE:align:正在依每個策略真正需要的資料來源分別對時。", flush=True)
    for name, fn in STRATEGIES.items():
        frame, diagnostic = build_strategy_frame(
            strategy_name=name,
            price=price,
            datasets=datasets,
            min_coverage=s.min_aligned_coverage,
            min_rows=500,
        )
        strategy_diagnostics[name] = diagnostic
        if frame is None:
            print(f"COINLAB_STAGE:skip:{name} 暫不回測：{diagnostic.get('reason', '必要資料不完整。')}", flush=True)
            continue
        features_by_strategy[name] = frame
        print(f"COINLAB_STAGE:backtest:正在回測 {name}，有效對齊 {len(frame)} 根 K。", flush=True)
        results[name] = run_backtest(frame, fn, cfg, funding_events=funding_events)
        validation[name] = evaluate_oos(frame, fn, cfg, funding_events=funding_events)

    meta = {
        "symbol": s.symbol,
        "coinglass_symbol": cg_symbol,
        "coinglass_symbol_setting": s.coinglass_symbol,
        "coinglass_exchange": s.coinglass_exchange,
        "timeframe": s.timeframe,
        "requested_start": s.start,
        "requested_end": s.end,
        "used_start": window.used_start,
        "used_end": window.used_end,
        "history_window_adjusted": window.adjusted,
        "history_adjustment_reason": window.adjustment_reason,
        "coinglass_standard_max_history_days": window.max_history_days,
        "initial_equity": s.initial_equity,
        "risk_per_trade": s.risk_per_trade,
        "taker_fee_bps": s.taker_fee_bps,
        "slippage_bps": s.slippage_bps,
        "minimum_aligned_coverage": s.min_aligned_coverage,
        "bitget_price_rows": int(len(price)),
        "bitget_price_start": str(price.index.min()) if len(price) else None,
        "bitget_price_end": str(price.index.max()) if len(price) else None,
        "bitget_funding_events": int(len(funding_events)),
        "strategy_source_requirements": STRATEGY_SOURCE_REQUIREMENTS,
        "funding_model": "exact_published_rate_and_time_with_last_completed_market_candle_price_proxy",
        "code_fingerprints": {
            "strategies_py_sha256": _module_sha256(strategies_module),
            "backtest_py_sha256": _module_sha256(backtest_module),
            "features_py_sha256": _module_sha256(features_module),
        },
    }

    print("COINLAB_STAGE:report:正在產生逐筆交易與可複製回測報告。", flush=True)
    report = save_report(
        args.out,
        metadata=meta,
        results=results,
        validation=validation,
        features_by_strategy=features_by_strategy,
        strategy_diagnostics=strategy_diagnostics,
        source_diagnostics=source_diagnostics,
    )
    if not results:
        unavailable = [
            SOURCE_LABELS.get(name, name)
            for name, status in source_diagnostics.items()
            if isinstance(status, dict) and status.get("status") in {"error", "empty"}
        ]
        joined = "、".join(unavailable) or "必要 CoinGlass 資料"
        raise RuntimeError(
            "ALL_STRATEGIES_SKIPPED: 所有策略都因資料來源不可用或對齊不足而未執行；"
            f"主要缺少：{joined}。"
        )
    skipped = len(STRATEGIES) - len(results)
    print(f"COINLAB_STAGE:done:回測完成：{len(results)} 套策略成功，{skipped} 套因資料完整性不足跳過。", flush=True)
    print(report.read_text(encoding="utf-8"))


def command_validate(args):
    path = Path(args.report)
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "status", "metadata", "data_integrity", "strategies"}
    missing = required - payload.keys()
    if missing:
        raise SystemExit(f"invalid report; missing {sorted(missing)}")
    if payload["status"] != "REAL_DATA_BACKTEST":
        raise SystemExit("report is not marked as real-data backtest")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(prog="coinlab")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest", help="Fetch real CoinGlass + Bitget data and run all strategies")
    b.add_argument("--out", default="artifacts/latest")
    b.set_defaults(func=command_backtest)
    v = sub.add_parser("validate-report", help="Validate/copy a BACKTEST_REPORT.json before sharing")
    v.add_argument("report")
    v.set_defaults(func=command_validate)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

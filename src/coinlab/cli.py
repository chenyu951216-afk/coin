from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backtest import BacktestConfig, run_backtest
from .config import Settings
from .features import build_feature_frame
from .providers import BitgetPublicClient, CoinGlassClient
from .reporting import save_report
from .strategies import STRATEGIES
from .validation import evaluate_oos


def fetch_and_build(s: Settings):
    cg = CoinGlassClient(s.coinglass_api_key)
    bg = BitgetPublicClient()
    common = dict(exchange=s.coinglass_exchange, symbol=s.coinglass_symbol, interval=s.timeframe, start=s.start, end=s.end)
    price = bg.candles(s.symbol, s.timeframe.replace("h", "H") if s.timeframe.endswith("h") else s.timeframe, s.start, s.end)
    if price.empty:
        raise RuntimeError("Bitget returned no price candles; aborting rather than producing fake metrics.")
    datasets = {
        "oi": cg.open_interest(**common),
        "funding": cg.funding(**common),
        "liq": cg.liquidations(**common),
        "ls": cg.long_short(**common),
        "taker": cg.taker_flow(**common),
        "orderbook": cg.orderbook(**common),
    }
    missing = [k for k, v in datasets.items() if v.empty]
    if missing:
        raise RuntimeError(f"CoinGlass returned empty required datasets: {missing}. Backtest aborted for integrity.")
    funding_events = bg.funding_history(s.symbol, s.start, s.end)
    return build_feature_frame(price, datasets["oi"], datasets["funding"], datasets["liq"], datasets["ls"], datasets["taker"], datasets["orderbook"]), funding_events


def command_backtest(args):
    s = Settings()
    df, funding_events = fetch_and_build(s)
    if len(df) < 500:
        raise RuntimeError(f"Only {len(df)} aligned rows. Refusing to treat this as a meaningful backtest.")
    cfg = BacktestConfig(
        initial_equity=s.initial_equity,
        risk_per_trade=s.risk_per_trade,
        fee_bps=s.taker_fee_bps,
        slippage_bps=s.slippage_bps,
    )
    results = {name: run_backtest(df, fn, cfg, funding_events=funding_events) for name, fn in STRATEGIES.items()}
    validation = {name: evaluate_oos(df, fn, cfg, funding_events=funding_events) for name, fn in STRATEGIES.items()}
    meta = {
        "symbol": s.symbol,
        "coinglass_symbol": s.coinglass_symbol,
        "coinglass_exchange": s.coinglass_exchange,
        "timeframe": s.timeframe,
        "requested_start": s.start,
        "requested_end": s.end,
        "initial_equity": s.initial_equity,
        "risk_per_trade": s.risk_per_trade,
        "taker_fee_bps": s.taker_fee_bps,
        "slippage_bps": s.slippage_bps,
        "bitget_funding_events": int(len(funding_events)),
        "funding_model": "exact_rate_and_time_with_last_completed_market_candle_price_proxy",
    }
    report = save_report(args.out, metadata=meta, features=df, results=results, validation=validation)
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

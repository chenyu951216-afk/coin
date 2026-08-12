from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import backtest as backtest_module
from . import features as features_module
from . import strategies as strategies_module
from .backtest import BacktestConfig, run_backtest
from .config import Settings
from .features import build_feature_frame
from .history_policy import normalize_backtest_window
from .providers import BitgetPublicClient, CoinGlassClient
from .reporting import save_report
from .strategies import STRATEGIES
from .universe import resolve_coinglass_instrument
from .validation import evaluate_oos


def _stage(name: str, message: str) -> None:
    print(f"COINLAB_STAGE:{name}:{message}", flush=True)


def _module_sha256(module) -> str:
    return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


def _resolved_coinglass_symbol(s: Settings) -> str:
    configured = str(s.coinglass_symbol or "").strip()
    if configured and configured.upper() != "AUTO":
        return configured
    _stage("symbol", "正在確認 CoinGlass 與 Bitget 的商品對應")
    return resolve_coinglass_instrument(s.coinglass_api_key, s.coinglass_exchange, s.symbol)


def fetch_and_build(s: Settings, *, start: str, end: str):
    cg_symbol = _resolved_coinglass_symbol(s)
    cg = CoinGlassClient(s.coinglass_api_key)
    bg = BitgetPublicClient(base_url=s.bitget_rest_base_url)
    common = dict(
        exchange=s.coinglass_exchange,
        symbol=cg_symbol,
        interval=s.timeframe,
        start=start,
        end=end,
    )
    granularity = s.timeframe.replace("h", "H") if s.timeframe.endswith("h") else s.timeframe

    _stage("bitget", "正在下載 Bitget 已完成 K 線")
    price = bg.candles(s.symbol, granularity, start, end)
    if price.empty:
        raise RuntimeError("Bitget returned no price candles; aborting rather than producing fake metrics.")

    _stage("coinglass", "正在下載 CoinGlass OI / Funding / 清算 / 多空比 / Taker / Orderbook")
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

    _stage("align", "正在依時間戳精確對齊所有資料，缺資料不向未來補值")
    features = build_feature_frame(
        price,
        datasets["oi"],
        datasets["funding"],
        datasets["liq"],
        datasets["ls"],
        datasets["taker"],
        datasets["orderbook"],
    )
    stats = {
        "bitget_symbol": s.symbol,
        "resolved_coinglass_symbol": cg_symbol,
        "bitget_price_rows": int(len(price)),
        "coinglass_rows": {k: int(len(v)) for k, v in datasets.items()},
        "aligned_rows": int(len(features)),
        "aligned_coverage_vs_bitget": float(len(features) / len(price)) if len(price) else 0.0,
    }
    if stats["aligned_coverage_vs_bitget"] < s.min_aligned_coverage:
        raise RuntimeError(
            f"Aligned data coverage is only {stats['aligned_coverage_vs_bitget']:.2%}, below "
            f"MIN_ALIGNED_COVERAGE={s.min_aligned_coverage:.0%}. Raw rows={stats}. "
            "Aborting instead of backtesting an incomplete intersection; shorten/change the research range or source."
        )

    _stage("funding", "正在下載 Bitget 歷史資金費率並依結算時間加入損益")
    funding_events = bg.funding_history(s.symbol, start, end)
    return features, funding_events, stats


def command_backtest(args):
    s = Settings()
    _stage("prepare", "正在建立 CoinGlass Standard 可用的最大安全回測區間")
    window = normalize_backtest_window(
        timeframe=s.timeframe,
        requested_start=s.start or None,
        requested_end=s.end or None,
    )
    df, funding_events, source_stats = fetch_and_build(s, start=window.used_start, end=window.used_end)
    if len(df) < 500:
        raise RuntimeError(f"Only {len(df)} aligned rows. Refusing to treat this as a meaningful backtest.")

    cfg = BacktestConfig(
        initial_equity=s.initial_equity,
        risk_per_trade=s.risk_per_trade,
        fee_bps=s.taker_fee_bps,
        slippage_bps=s.slippage_bps,
    )

    _stage("backtest", "正在逐根 K 線回放六套策略；訊號收 K 後成立，下一根才允許成交")
    results = {
        name: run_backtest(df, fn, cfg, funding_events=funding_events)
        for name, fn in STRATEGIES.items()
    }

    _stage("validation", "正在執行 60/20/20 與 Walk-forward 樣本外驗證")
    validation = {
        name: evaluate_oos(df, fn, cfg, funding_events=funding_events)
        for name, fn in STRATEGIES.items()
    }

    meta = {
        "symbol": s.symbol,
        "coinglass_symbol": source_stats["resolved_coinglass_symbol"],
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
        "source_rows": source_stats,
        "bitget_funding_events": int(len(funding_events)),
        "funding_model": "exact_published_rate_and_time_with_last_completed_market_candle_price_proxy",
        "code_fingerprints": {
            "strategies_py_sha256": _module_sha256(strategies_module),
            "backtest_py_sha256": _module_sha256(backtest_module),
            "features_py_sha256": _module_sha256(features_module),
        },
    }

    _stage("report", "正在建立逐筆交易、策略統計與可複製報告")
    report = save_report(
        args.out,
        metadata=meta,
        features=df,
        results=results,
        validation=validation,
    )
    _stage("complete", f"回測完成，報告已建立：{report}")


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

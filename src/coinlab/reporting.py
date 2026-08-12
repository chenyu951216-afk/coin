from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
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


def _trade_rows(trades: pd.DataFrame) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    clean = trades.replace([np.inf, -np.inf], np.nan)
    return [_json_safe(row) for row in clean.to_dict(orient="records")]


def save_report(
    outdir: str,
    *,
    metadata: dict[str, Any],
    features: pd.DataFrame,
    results: dict[str, tuple[pd.DataFrame, dict]],
    validation: dict[str, Any] | None = None,
) -> Path:
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    features_path = root / "aligned_features.csv"
    features.to_csv(features_path, index=True)

    strategies: dict[str, Any] = {}
    all_trades: list[dict[str, Any]] = []
    for name, (trades, metrics) in results.items():
        trade_path = root / f"trades_{name}.csv"
        trades.to_csv(trade_path, index=False)
        rows = _trade_rows(trades)
        all_trades.extend(rows)
        strategies[name] = {
            "metrics": _json_safe(metrics),
            "trade_count": len(rows),
            "trades": rows,
            "trades_file": trade_path.name,
            "trades_sha256": _sha256_file(trade_path),
        }

    all_trades.sort(key=lambda r: (str(r.get("entry_time", "")), str(r.get("strategy", ""))))
    total_net = sum(float(r.get("net_pnl") or 0.0) for r in all_trades)
    total_fees = sum(float(r.get("fees") or 0.0) for r in all_trades)
    total_slippage = sum(float(r.get("slippage_cost") or 0.0) for r in all_trades)
    total_funding = sum(float(r.get("funding_pnl") or 0.0) for r in all_trades)

    report = {
        "schema_version": "coinlab.backtest.v2",
        "status": "REAL_DATA_BACKTEST",
        "metadata": _json_safe(metadata),
        "data_integrity": {
            "aligned_rows": int(len(features)),
            "start": str(features.index.min()) if len(features) else None,
            "end": str(features.index.max()) if len(features) else None,
            "features_sha256": _sha256_file(features_path),
            "no_lookahead": "signal_at_close_t_entry_at_open_t_plus_1",
            "same_bar_tp_sl_policy": "stop_first",
            "stop_update_policy": "stop_changes_only_after_completed_bar_and_affect_future_bars",
            "funding": "Bitget historical fundingRate/fundingTime; settlement notional uses last completed pre-settlement market candle close",
            "slippage": "adverse slippage applied to entry and exit fills; explicit slippage_cost is also reported per trade",
        },
        "portfolio_summary": {
            "strategies": len(strategies),
            "total_strategy_trades": len(all_trades),
            "sum_strategy_net_pnl": total_net,
            "sum_strategy_fees": total_fees,
            "sum_strategy_slippage_cost": total_slippage,
            "sum_strategy_funding_pnl": total_funding,
            "note": "Each strategy is backtested independently from the same initial equity. Sum-strategy figures are diagnostic totals, not a combined portfolio equity curve.",
        },
        "strategies": strategies,
        "all_trades": all_trades,
        "validation": _json_safe(validation or {}),
    }
    path = root / "BACKTEST_REPORT.json"
    path.write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path

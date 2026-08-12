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


def _empty_metrics() -> dict[str, Any]:
    return {
        "trades": 0, "win_rate": None, "profit_factor": None, "expectancy_r": None,
        "net_pnl": 0.0, "return_pct": 0.0, "max_drawdown_pct": 0.0,
        "avg_win_r": None, "avg_loss_r": None, "max_consecutive_losses": 0,
        "avg_mfe_r": None, "avg_mae_r": None, "median_holding_bars": None,
        "breakeven_activation_rate": None, "trailing_activation_rate": None,
        "total_fees": 0.0, "total_slippage_cost": 0.0, "total_funding_pnl": 0.0,
        "final_equity": None, "ambiguous_exit_count": 0,
    }


def save_report(
    outdir: str,
    *,
    metadata: dict[str, Any],
    results: dict[str, tuple[pd.DataFrame, dict]],
    validation: dict[str, Any] | None = None,
    features: pd.DataFrame | None = None,
    features_by_strategy: dict[str, pd.DataFrame] | None = None,
    strategy_diagnostics: dict[str, dict[str, Any]] | None = None,
    source_diagnostics: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Write a reproducible backtest report.

    ``features`` is retained for the original v2 API. New research runs use
    ``features_by_strategy`` so each strategy can be aligned only to the data
    sources it actually consumes. The JSON schema remains v2-compatible while
    adding optional diagnostics fields.
    """
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    features_by_strategy = dict(features_by_strategy or {})
    strategy_diagnostics = strategy_diagnostics or {}
    source_diagnostics = source_diagnostics or {}

    legacy_integrity: dict[str, Any] = {}
    if features is not None:
        legacy_path = root / "aligned_features.csv"
        features.to_csv(legacy_path, index=True)
        legacy_integrity = {
            "aligned_rows": int(len(features)),
            "start": str(features.index.min()) if len(features) else None,
            "end": str(features.index.max()) if len(features) else None,
            "features_sha256": _sha256_file(legacy_path),
        }
        # Preserve the original single/global-frame behavior for callers that
        # have not migrated to strategy-specific alignment yet.
        if not features_by_strategy:
            features_by_strategy = {name: features for name in results}

    strategies: dict[str, Any] = {}
    all_trades: list[dict[str, Any]] = []
    feature_integrity: dict[str, Any] = {}

    for name, frame in features_by_strategy.items():
        feature_path = root / f"aligned_features_{name}.csv"
        frame.to_csv(feature_path, index=True)
        feature_integrity[name] = {
            "rows": int(len(frame)),
            "start": str(frame.index.min()) if len(frame) else None,
            "end": str(frame.index.max()) if len(frame) else None,
            "sha256": _sha256_file(feature_path),
        }

    all_strategy_names = sorted(set(results) | set(strategy_diagnostics))
    for name in all_strategy_names:
        diagnostic = strategy_diagnostics.get(name, {})
        if name in results:
            trades, metrics = results[name]
            trade_path = root / f"trades_{name}.csv"
            trades.to_csv(trade_path, index=False)
            rows = _trade_rows(trades)
            all_trades.extend(rows)
            strategies[name] = {
                "status": "COMPLETED",
                "diagnostic": _json_safe(diagnostic),
                "metrics": _json_safe(metrics),
                "trade_count": len(rows),
                "trades": rows,
                "trades_file": trade_path.name,
                "trades_sha256": _sha256_file(trade_path),
                "data_integrity": feature_integrity.get(name, {}),
            }
        else:
            strategies[name] = {
                "status": "SKIPPED_DATA_UNAVAILABLE",
                "diagnostic": _json_safe(diagnostic),
                "metrics": _empty_metrics(),
                "trade_count": 0,
                "trades": [],
                "trades_file": None,
                "trades_sha256": None,
                "data_integrity": feature_integrity.get(name, {}),
            }

    all_trades.sort(key=lambda r: (str(r.get("entry_time", "")), str(r.get("strategy", ""))))
    total_net = sum(float(r.get("net_pnl") or 0.0) for r in all_trades)
    total_fees = sum(float(r.get("fees") or 0.0) for r in all_trades)
    total_slippage = sum(float(r.get("slippage_cost") or 0.0) for r in all_trades)
    total_funding = sum(float(r.get("funding_pnl") or 0.0) for r in all_trades)
    executed = [name for name, value in strategies.items() if value["status"] == "COMPLETED"]
    skipped = [name for name, value in strategies.items() if value["status"] != "COMPLETED"]

    report = {
        # Keep v2 because existing web/chat consumers already understand it;
        # resilience diagnostics are additive rather than a breaking schema change.
        "schema_version": "coinlab.backtest.v2",
        "status": "REAL_DATA_BACKTEST",
        "execution_status": "COMPLETED" if executed else "NO_STRATEGIES_EXECUTED",
        "metadata": _json_safe(metadata),
        "source_diagnostics": _json_safe(source_diagnostics),
        "data_integrity": {
            **legacy_integrity,
            "strategy_frames": feature_integrity,
            "no_lookahead": "signal_at_close_t_entry_at_open_t_plus_1",
            "same_bar_tp_sl_policy": "stop_first",
            "stop_update_policy": "stop_changes_only_after_completed_bar_and_affect_future_bars",
            "funding": "Bitget historical fundingRate/fundingTime; settlement notional uses last completed pre-settlement market candle close",
            "slippage": "adverse slippage applied to entry and exit fills; explicit slippage_cost reported per trade",
            "strategy_specific_alignment": "each strategy is aligned only against the CoinGlass sources it actually uses",
        },
        "portfolio_summary": {
            "strategies_total": len(strategies),
            "strategies_executed": len(executed),
            "strategies_skipped": len(skipped),
            "executed_strategy_names": executed,
            "skipped_strategy_names": skipped,
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

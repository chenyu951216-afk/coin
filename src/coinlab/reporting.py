from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_report(outdir: str, *, metadata: dict[str, Any], features: pd.DataFrame, results: dict[str, tuple[pd.DataFrame, dict]], validation: dict[str, Any] | None = None) -> Path:
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    features_path = root / "aligned_features.csv"
    features.to_csv(features_path, index=True)

    strategies = {}
    for name, (trades, metrics) in results.items():
        trade_path = root / f"trades_{name}.csv"
        trades.to_csv(trade_path, index=False)
        strategies[name] = {"metrics": metrics, "trades_file": trade_path.name, "trades_sha256": _sha256_file(trade_path)}

    report = {
        "schema_version": "coinlab.backtest.v1",
        "status": "REAL_DATA_BACKTEST",
        "metadata": metadata,
        "data_integrity": {
            "aligned_rows": int(len(features)),
            "start": str(features.index.min()) if len(features) else None,
            "end": str(features.index.max()) if len(features) else None,
            "features_sha256": _sha256_file(features_path),
            "no_lookahead": "signal_at_close_t_entry_at_open_t_plus_1",
            "same_bar_tp_sl_policy": "stop_first",
            "funding": "Bitget historical fundingRate/fundingTime; settlement notional uses last completed pre-settlement market candle close",
        },
        "strategies": strategies,
        "validation": validation or {},
    }
    path = root / "BACKTEST_REPORT.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path

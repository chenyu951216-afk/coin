from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile


RESEARCH_PROTOCOL: dict[str, Any] = {
    "version": "coinlab.research_protocol.v1",
    "goal": "Improve robust out-of-sample expectancy without cherry-picking historical outcomes.",
    "rules": [
        "Use every eligible timestamp in the predeclared data window; never delete a date, coin, or trade because its realized PnL was bad.",
        "Strategy changes may be justified only by train, validation, and development-only walk-forward evidence.",
        "The locked test/holdout segment is promotion-only. Individual holdout trades must not be used to tune entry, exit, filters, SL, or TP.",
        "Signals use completed bars only and enter no earlier than the next bar open; future prices must never be used to create a signal.",
        "Prefer parameter plateaus and repeated evidence across folds/regimes over a single best historical parameter point.",
        "A large winner or loser is an observation, not a reason to hard-code a date/symbol exception.",
        "Changes should improve or preserve validation expectancy, drawdown, and fold stability; a higher in-sample PnL alone is not sufficient.",
        "Before live promotion, a market-wide strategy also needs evidence across multiple symbols/regimes plus paper execution validation.",
    ],
    "forbidden_examples": [
        "Skip 2026-07-03 because AAVE lost money that day.",
        "Add a BTC-only rule because one known BTC trade made a large profit.",
        "Choose TP/SL using the future high/low of the same trade.",
        "Repeatedly change parameters until the locked test segment looks good.",
    ],
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _test_start(validation: dict[str, Any]) -> str | None:
    windows = validation.get("split_windows") or {}
    test = windows.get("test") or {}
    return test.get("start")


def _before(ts: Any, boundary: str | None) -> bool:
    if not boundary:
        return True
    return str(ts or "") < str(boundary)


def _development_view(report: dict[str, Any]) -> dict[str, Any]:
    strategies: dict[str, Any] = {}
    validation_map = report.get("validation") or {}
    for name, payload in (report.get("strategies") or {}).items():
        v = validation_map.get(name) or {}
        boundary = _test_start(v)
        dev_trades = [row for row in (payload.get("trades") or []) if _before(row.get("entry_time"), boundary)]
        split = v.get("split") or {}
        strategies[name] = {
            "status": payload.get("status"),
            "diagnostic": payload.get("diagnostic") or {},
            "data_integrity": payload.get("data_integrity") or {},
            "development_trade_count": len(dev_trades),
            "development_trades": dev_trades,
            "research_validation": {
                "research_grade": v.get("research_grade"),
                "train": split.get("train"),
                "validation": split.get("validation"),
                "development_walk_forward": v.get("walk_forward") or [],
                "profitable_fold_ratio": v.get("profitable_fold_ratio"),
                "positive_expectancy_fold_ratio": v.get("positive_expectancy_fold_ratio"),
                "test_start_hidden_from_tuning": boundary,
            },
        }
    return {
        "schema_version": "coinlab.chatgpt_research_input.v1",
        "generated_at": _iso_now(),
        "research_protocol": RESEARCH_PROTOCOL,
        "metadata": report.get("metadata") or {},
        "source_diagnostics": report.get("source_diagnostics") or {},
        "data_integrity": report.get("data_integrity") or {},
        "strategies": strategies,
        "important": "This development package intentionally excludes locked test metrics/trades from the tuning view. Do not infer or reconstruct them for strategy modification.",
    }


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        return b""
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    sio = io.StringIO()
    writer = csv.DictWriter(sio, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    return sio.getvalue().encode("utf-8-sig")


def _read_report(root: Path) -> tuple[Path, dict[str, Any]]:
    report_path = root / "BACKTEST_REPORT.json"
    if not report_path.exists():
        raise FileNotFoundError("BACKTEST_REPORT.json not found")
    return report_path, json.loads(report_path.read_text(encoding="utf-8"))


def build_research_package(outdir: str | Path) -> Path:
    root = Path(outdir)
    report_path, report = _read_report(root)
    research = _development_view(report)
    metadata = report.get("metadata") or {}
    symbol = str(metadata.get("symbol") or "symbol")
    timeframe = str(metadata.get("timeframe") or "tf")
    package = root / f"COINLAB_RESEARCH_{symbol}_{timeframe}_{root.name}.zip"

    entries: dict[str, bytes] = {}
    entries["CHATGPT_RESEARCH_INPUT.json"] = json.dumps(research, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    entries["RESEARCH_PROTOCOL.json"] = json.dumps(RESEARCH_PROTOCOL, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    readme = (
        "CoinLab 研究包\n\n"
        "用途：把這個 ZIP 直接傳給 ChatGPT，用來研究下一版策略。\n"
        "重要：此包刻意不提供 locked test 的績效/逐筆明細作為調參依據。\n"
        "策略修改只能根據 train、validation、development walk-forward、資料品質與交易型態。\n"
        "禁止因某個已知日期/幣種的輸贏而新增例外規則或刪除樣本。\n"
        "完整 test 只在策略候選凍結後用稽核包做 promotion gate。\n"
        f"完整原始報告 SHA256（伺服器留存）：{_sha256_file(report_path)}\n"
    )
    entries["README_研究規則.txt"] = readme.encode("utf-8")

    for name, payload in research["strategies"].items():
        rows = payload.get("development_trades") or []
        entries[f"trades_development/{name}.csv"] = _csv_bytes(rows)

    manifest = {
        "schema_version": "coinlab.package_manifest.v1",
        "package_type": "RESEARCH_DEVELOPMENT_ONLY",
        "generated_at": _iso_now(),
        "full_report_sha256": _sha256_file(report_path),
        "files": {},
    }
    for name, data in entries.items():
        manifest["files"][name] = {"bytes": len(data), "sha256": _sha256_bytes(data)}
    entries["PACKAGE_MANIFEST.json"] = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")

    with ZipFile(package, "w", compression=ZIP_DEFLATED, compresslevel=9) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return package


def build_audit_package(outdir: str | Path) -> Path:
    root = Path(outdir)
    report_path, report = _read_report(root)
    metadata = report.get("metadata") or {}
    symbol = str(metadata.get("symbol") or "symbol")
    timeframe = str(metadata.get("timeframe") or "tf")
    package = root / f"COINLAB_AUDIT_{symbol}_{timeframe}_{root.name}.zip"

    paths: list[Path] = [report_path]
    paths.extend(sorted(root.glob("trades_*.csv")))
    entries: dict[str, bytes] = {p.name: p.read_bytes() for p in paths if p.is_file()}
    entries["RESEARCH_PROTOCOL.json"] = json.dumps(RESEARCH_PROTOCOL, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    entries["README_完整稽核包.txt"] = (
        "這是完整稽核包，包含 locked test / holdout。\n"
        "只有在策略候選已凍結、準備做最終 promotion gate 時才應查看。\n"
        "不要根據這個包中的單筆 test 輸贏再修改策略，否則 test 會被污染並失去 OOS 意義。\n"
    ).encode("utf-8")

    manifest = {
        "schema_version": "coinlab.package_manifest.v1",
        "package_type": "FULL_AUDIT_WITH_LOCKED_HOLDOUT",
        "generated_at": _iso_now(),
        "files": {name: {"bytes": len(data), "sha256": _sha256_bytes(data)} for name, data in entries.items()},
    }
    entries["PACKAGE_MANIFEST.json"] = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")

    with ZipFile(package, "w", compression=ZIP_DEFLATED, compresslevel=9) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return package

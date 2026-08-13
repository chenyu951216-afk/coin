from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


def install_strict_market_packager() -> None:
    """Replace the market packager with a stricter development-only version.

    The research ZIP contains only the explicit train and validation windows.
    Purge/embargo gaps and locked test trades are excluded so they cannot leak
    into iterative strategy tuning.
    """
    from . import market_backtest as mb
    from .backtest import compute_metrics

    def strict_write_packages(root: Path, report: dict, trades: pd.DataFrame, split):
        research = root / f"COINLAB_MARKET_RESEARCH_{report['metadata']['timeframe']}_{root.name}.zip"
        audit = root / f"COINLAB_MARKET_AUDIT_{report['metadata']['timeframe']}_{root.name}.zip"
        train = mb._trade_slice(trades, split.train_start, split.train_end)
        valid = mb._trade_slice(trades, split.validation_start, split.validation_end)
        dev = pd.concat([train, valid], ignore_index=True) if (not train.empty or not valid.empty) else trades.iloc[:0].copy()
        if not dev.empty:
            dev = dev.sort_values(["entry_time", "symbol", "strategy"]).reset_index(drop=True)
        initial = report["metadata"]["initial_equity"]
        research_payload = {
            "schema_version": "coinlab.market_research.v2",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "metadata": report["metadata"],
            "universe": report["universe"],
            "data_integrity": report["data_integrity"],
            "sizing_policy": report["sizing_policy"],
            "global_split": {
                "train_start": split.train_start, "train_end": split.train_end,
                "validation_start": split.validation_start, "validation_end": split.validation_end,
                "locked_test_start": split.test_start, "locked_test_end": split.test_end,
                "embargo_seconds": split.embargo_seconds,
            },
            "train_metrics": compute_metrics(train.sort_values("exit_time") if not train.empty else train, initial),
            "validation_metrics": compute_metrics(valid.sort_values("exit_time") if not valid.empty else valid, initial),
            "development_walk_forward": mb._fold_metrics(dev, split.train_start, split.validation_end, 5, initial),
            "by_strategy_development": mb._summary_group(dev, "strategy", initial),
            "by_symbol_development": mb._summary_group(dev, "symbol", initial),
            "development_trade_count": int(len(dev)),
            "excluded_from_tuning": ["PURGE_EMBARGO_GAPS", "LOCKED_TEST_HOLDOUT"],
            "important": "Only explicit train+validation trades are included. Embargo gaps and locked test are excluded. Never delete a date/symbol/trade because its realized PnL was bad.",
        }
        with ZipFile(research, "w", ZIP_DEFLATED, compresslevel=9) as z:
            z.writestr("CHATGPT_MARKET_RESEARCH_INPUT.json", json.dumps(mb._json_safe(research_payload), ensure_ascii=False, indent=2, sort_keys=True).encode())
            z.writestr("trades_development_all_symbols.csv", dev.to_csv(index=False).encode("utf-8-sig") if not dev.empty else b"")
            z.writestr("skipped_symbols.csv", pd.DataFrame(report.get("skipped_symbols", [])).to_csv(index=False).encode("utf-8-sig"))
            z.writestr("README_研究規則.txt", (
                "全市場 development 研究包 v2。\n"
                "只包含 Train + Validation。Purged/embargo 邊界與 Locked Test 完全排除。\n"
                "禁止因特定日期、幣種、單筆盈虧新增例外或刪除樣本。\n"
                "策略修改後必須重新跑整個市場資料，不能拿舊交易結果直接篩選贏家。\n"
            ).encode("utf-8"))

        report_path = root / "MARKET_BACKTEST_REPORT.json"
        trades_path = root / "all_market_trades.csv"
        with ZipFile(audit, "w", ZIP_DEFLATED, compresslevel=9) as z:
            z.write(report_path, report_path.name)
            z.write(trades_path, trades_path.name)
            for extra in (root / "symbol_strategy_summary.csv", root / "skipped_symbols.csv"):
                if extra.exists():
                    z.write(extra, extra.name)
            z.writestr("README_完整稽核包.txt", "包含 Locked Test。只可用於候選策略凍結後的最終稽核，不得根據 test 單筆結果再調策略。\n")
        return research, audit

    mb._write_packages = strict_write_packages

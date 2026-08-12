import json
from zipfile import ZipFile

from coinlab.research_package import build_audit_package, build_research_package
from coinlab.server_v5 import DASHBOARD


def test_research_and_audit_packages(tmp_path):
    report = {
        "schema_version": "coinlab.backtest.v2",
        "status": "REAL_DATA_BACKTEST",
        "metadata": {"symbol": "BTCUSDT", "timeframe": "15m"},
        "source_diagnostics": {},
        "data_integrity": {},
        "strategies": {"demo": {"status": "COMPLETED", "diagnostic": {}, "data_integrity": {}, "trades": [
            {"strategy": "demo", "entry_time": "2026-08-01 00:00:00+00:00", "net_pnl": -1},
            {"strategy": "demo", "entry_time": "2026-08-20 00:00:00+00:00", "net_pnl": 2},
        ]}},
        "validation": {"demo": {"research_grade": "OK", "split": {"train": {}, "validation": {}, "test": {}},
            "split_windows": {"test": {"start": "2026-08-15 00:00:00+00:00"}}, "walk_forward": []}},
    }
    (tmp_path / "BACKTEST_REPORT.json").write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "trades_demo.csv").write_text("entry_time,net_pnl\n2026-08-01,-1\n2026-08-20,2\n", encoding="utf-8")

    research_zip = build_research_package(tmp_path)
    audit_zip = build_audit_package(tmp_path)
    with ZipFile(research_zip) as zf:
        research = json.loads(zf.read("CHATGPT_RESEARCH_INPUT.json"))
        assert research["strategies"]["demo"]["development_trade_count"] == 1
    with ZipFile(audit_zip) as zf:
        assert "BACKTEST_REPORT.json" in zf.namelist()


def test_dashboard_has_zip_download_buttons():
    assert "下載研究包 ZIP" in DASHBOARD
    assert "下載完整稽核包 ZIP" in DASHBOARD

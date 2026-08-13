from coinlab import market_backtest
from coinlab.market_research_package import install_strict_market_packager


def test_strict_market_packager_excludes_embargo_and_locked_test_by_installation():
    install_strict_market_packager()
    assert market_backtest._write_packages.__name__ == "strict_write_packages"


def test_v7_dashboard_replaces_single_symbol_backtest_card():
    install_strict_market_packager()
    from coinlab.server_v7 import DASHBOARD

    assert "Bitget 全市場歷史真實回測" in DASHBOARD
    assert "開始全市場完整回測" in DASHBOARD
    assert "單幣種真實資料回測" not in DASHBOARD
    assert "v0.7" in DASHBOARD

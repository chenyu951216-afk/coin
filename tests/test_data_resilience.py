import json

import pandas as pd

from coinlab.providers import CoinGlassAPIError, CoinGlassClient
from coinlab.reporting import save_report
from coinlab.research import build_strategy_frame


class FakeHistoryClient(CoinGlassClient):
    def __init__(self):
        super().__init__("test")
        self.calls = 0

    def _get(self, path, params):
        self.calls += 1
        if self.calls == 1:
            raise CoinGlassAPIError(
                path,
                "400",
                "Invalid time range: the earliest allowed start_time is 1767226500000, and end_time must be greater than start_time.",
            )
        start = int(params["start_time"])
        end = int(params["end_time"])
        step = 900_000
        rows = []
        t = start
        while t <= end and len(rows) < 20:
            rows.append({"time": t, "open": "1", "high": "1", "low": "1", "close": "1"})
            t += step
        return rows


def test_coinglass_history_moves_forward_to_api_reported_earliest_start():
    client = FakeHistoryClient()
    df = client.open_interest(
        exchange="Bitget",
        symbol="ETHUSDT_UMCBL",
        interval="15m",
        start="2026-01-01T00:00:00Z",
        end="2026-01-01T08:00:00Z",
    )
    assert client.calls >= 2
    assert not df.empty
    assert df.index.min() >= pd.Timestamp("2026-01-01T00:15:00Z")


def _frame(index, **cols):
    return pd.DataFrame(cols, index=index)


def test_strategy_specific_sources_do_not_require_unrelated_orderbook():
    idx = pd.date_range("2026-01-01", periods=140, freq="15min", tz="UTC")
    price = _frame(
        idx,
        open=[100.0] * len(idx), high=[101.0] * len(idx), low=[99.0] * len(idx),
        close=[100.0] * len(idx), volume_base=[1.0] * len(idx), volume_quote=[100.0] * len(idx),
    )
    oi = _frame(idx, open=[1000.0]*len(idx), high=[1001.0]*len(idx), low=[999.0]*len(idx), close=[1000.0]*len(idx))
    funding = _frame(idx, open=[0.001]*len(idx), high=[0.001]*len(idx), low=[0.001]*len(idx), close=[0.001]*len(idx))
    taker = _frame(idx, taker_buy_volume_usd=[60.0]*len(idx), taker_sell_volume_usd=[40.0]*len(idx))
    datasets = {"oi": oi, "funding": funding, "taker": taker, "liq": pd.DataFrame(), "ls": pd.DataFrame(), "orderbook": pd.DataFrame()}
    frame, diagnostic = build_strategy_frame(
        strategy_name="oi_breakout", price=price, datasets=datasets, min_coverage=0.90, min_rows=100,
    )
    assert frame is not None
    assert diagnostic["status"] == "ready"
    book_frame, book_diag = build_strategy_frame(
        strategy_name="orderbook_pressure", price=price, datasets=datasets, min_coverage=0.90, min_rows=100,
    )
    assert book_frame is None
    assert book_diag["code"] == "REQUIRED_SOURCE_UNAVAILABLE"
    assert "orderbook" in book_diag["missing_sources"]


def test_report_keeps_skipped_strategy_without_inventing_metrics(tmp_path):
    path = save_report(
        str(tmp_path), metadata={"symbol": "ETHUSDT"}, results={}, validation={}, features_by_strategy={},
        strategy_diagnostics={"orderbook_pressure": {"status": "skipped", "code": "REQUIRED_SOURCE_UNAVAILABLE", "reason": "缺少訂單簿資料"}},
        source_diagnostics={"orderbook": {"status": "error", "rows": 0}},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload["strategies"]["orderbook_pressure"]
    assert row["status"] == "SKIPPED_DATA_UNAVAILABLE"
    assert row["metrics"]["win_rate"] is None
    assert row["trade_count"] == 0
    assert payload["execution_status"] == "NO_STRATEGIES_EXECUTED"

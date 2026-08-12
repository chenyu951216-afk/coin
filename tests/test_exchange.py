import pytest

from coinlab.exchange import BitgetAPIError, BitgetV2Client


def _client_with_contract(*, min_size=0.001, step=0.001, min_notional=5.0, max_market=100.0, max_limit=200.0):
    client = BitgetV2Client()
    client._contracts = {
        "BTCUSDT": {
            "symbol": "BTCUSDT",
            "base_coin": "BTC",
            "quote_coin": "USDT",
            "tradable": True,
            "status": "normal",
            "symbol_type": "perpetual",
            "min_leverage": 1,
            "max_leverage": 50,
            "price_step": 0.1,
            "size_step": step,
            "min_size": min_size,
            "min_notional": min_notional,
            "max_order_qty": max_limit,
            "max_market_order_qty": max_market,
            "maker_fee_rate": 0.0,
            "taker_fee_rate": 0.0,
            "raw": {},
        }
    }
    return client


def test_symbol_normalization_for_search_and_execution():
    assert BitgetV2Client.normalize_symbol("btc_usdt") == "BTCUSDT"
    assert BitgetV2Client.normalize_symbol("BTC-USDT") == "BTCUSDT"
    assert BitgetV2Client.normalize_symbol("btc/usdt") == "BTCUSDT"


def test_risk_sizing_rounds_down_and_does_not_exceed_loss_budget():
    client = _client_with_contract(min_size=0.0001, step=0.0001, min_notional=1)
    result = client.calculate_size(
        symbol="BTCUSDT",
        direction="long",
        entry_price=100.0,
        stop_price=98.0,
        risk_per_trade=0.01,
        leverage=5,
        order_type="market",
        max_position_notional_equity_multiple=10,
        max_portfolio_notional_equity_multiple=10,
        available_margin_utilization_pct=1.0,
        positions=[],
        open_orders=[],
        account={"equity": 1000.0, "available": 1000.0},
        exchange_max_open=1000,
    )
    assert result.risk_budget_usdt == pytest.approx(10.0)
    assert result.estimated_stop_loss_usdt <= result.risk_budget_usdt + 1e-9
    assert float(result.quantity) == pytest.approx(5.0)


def test_risk_sized_below_exchange_minimum_is_rejected_not_rounded_up():
    client = _client_with_contract(min_size=1.0, step=1.0, min_notional=100)
    with pytest.raises(BitgetAPIError, match="below Bitget minimum"):
        client.calculate_size(
            symbol="BTCUSDT",
            direction="long",
            entry_price=100.0,
            stop_price=90.0,
            risk_per_trade=0.001,
            leverage=2,
            order_type="market",
            max_position_notional_equity_multiple=2,
            max_portfolio_notional_equity_multiple=5,
            available_margin_utilization_pct=0.7,
            positions=[],
            open_orders=[],
            account={"equity": 1000.0, "available": 1000.0},
            exchange_max_open=1000,
        )


def test_order_type_uses_correct_exchange_quantity_cap():
    client = _client_with_contract(min_size=0.1, step=0.1, min_notional=1, max_market=2, max_limit=9)
    common = dict(
        symbol="BTCUSDT", direction="long", entry_price=100.0, stop_price=99.0,
        risk_per_trade=0.05, leverage=10,
        max_position_notional_equity_multiple=100, max_portfolio_notional_equity_multiple=100,
        available_margin_utilization_pct=1, positions=[], open_orders=[],
        account={"equity": 10000.0, "available": 10000.0}, exchange_max_open=1000,
    )
    market = client.calculate_size(order_type="market", **common)
    limit = client.calculate_size(order_type="limit", **common)
    assert float(market.quantity) == pytest.approx(2.0)
    assert float(limit.quantity) == pytest.approx(9.0)


def test_portfolio_notional_ignores_reduce_only_orders():
    total = BitgetV2Client.portfolio_notional(
        [{"size": 2, "mark_price": 100}],
        [
            {"size": 3, "price": 100, "reduceOnly": "NO"},
            {"size": 99, "price": 100, "reduceOnly": "YES"},
        ],
    )
    assert total == pytest.approx(500.0)

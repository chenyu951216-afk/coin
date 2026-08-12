from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Any
from urllib.parse import urlencode

import httpx


class BitgetAPIError(RuntimeError):
    def __init__(self, message: str, *, code: str | None = None, endpoint: str | None = None):
        super().__init__(message)
        self.code, self.endpoint = code, endpoint


def _text(value: Any) -> str:
    d = Decimal(str(value))
    if not d.is_finite():
        raise ValueError(f"non-finite number: {value!r}")
    out = format(d, "f")
    return (out.rstrip("0").rstrip(".") if "." in out else out) or "0"


def _positive(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 and math.isfinite(number) else default


def _floor_step(value: float, step: float) -> Decimal:
    v, s = Decimal(str(value)), Decimal(str(step))
    return v if s <= 0 else (v / s).to_integral_value(rounding=ROUND_DOWN) * s


def _tick(raw: dict[str, Any]) -> float:
    try:
        return _positive(raw.get("priceEndStep"), 1.0) * 10 ** (-int(raw.get("pricePlace", 8)))
    except (TypeError, ValueError):
        return 1e-8


@dataclass(frozen=True)
class SizingResult:
    symbol: str
    quantity: str
    actual_notional: float
    risk_budget_usdt: float
    estimated_stop_loss_usdt: float
    account_equity: float
    available_margin: float
    entry_price: float
    stop_price: float
    stop_distance: float
    stop_pct: float
    leverage: int
    binding_cap: str
    caps: dict[str, float]


class BitgetV2Client:
    """Execution-only Bitget V2 adapter; it never derives or changes strategy levels."""

    def __init__(
        self, *, api_key: str = "", api_secret: str = "", passphrase: str = "",
        base_url: str = "https://api.bitget.com", product_type: str = "USDT-FUTURES",
        margin_coin: str = "USDT", timeout: float = 20.0, retries: int = 4,
        live_enabled: bool = False,
    ) -> None:
        self.api_key, self.api_secret, self.passphrase = api_key, api_secret, passphrase
        self.base_url, self.product_type, self.margin_coin = base_url.rstrip("/"), product_type.upper(), margin_coin.upper()
        self.timeout, self.retries, self.live_enabled = timeout, max(1, retries), live_enabled
        self._contracts: dict[str, dict[str, Any]] = {}

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        return str(symbol or "").upper().replace("-", "").replace("_", "").replace("/", "")

    def _headers(self, method: str, path: str, query: str, body: str) -> dict[str, str]:
        if not self.api_key or not self.api_secret or not self.passphrase:
            raise BitgetAPIError("Bitget API key/secret/passphrase are required", endpoint=path)
        ts = str(int(time.time() * 1000))
        prehash = ts + method.upper() + path + (("?" + query) if query else "") + body
        signature = base64.b64encode(hmac.new(self.api_secret.encode(), prehash.encode(), hashlib.sha256).digest()).decode()
        return {"ACCESS-KEY": self.api_key, "ACCESS-SIGN": signature, "ACCESS-PASSPHRASE": self.passphrase,
                "ACCESS-TIMESTAMP": ts, "Content-Type": "application/json", "locale": "en-US"}

    def _request(self, method: str, path: str, *, params: dict[str, Any] | None = None,
                 body: dict[str, Any] | None = None, auth: bool = False) -> Any:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        query = urlencode(params)
        body_text = "" if body is None else json.dumps(body, separators=(",", ":"), ensure_ascii=False)
        with httpx.Client(timeout=self.timeout) as client:
            for attempt in range(self.retries):
                headers = self._headers(method, path, query, body_text) if auth else {"locale": "en-US"}
                try:
                    r = client.request(method, self.base_url + path, params=params or None, headers=headers,
                                       content=body_text if body is not None else None)
                    if (r.status_code == 429 or r.status_code >= 500) and attempt + 1 < self.retries:
                        time.sleep(min(8.0, 0.5 * 2**attempt)); continue
                    r.raise_for_status(); payload = r.json()
                except (httpx.HTTPError, ValueError) as exc:
                    if attempt + 1 < self.retries:
                        time.sleep(min(8.0, 0.5 * 2**attempt)); continue
                    raise BitgetAPIError(f"Bitget request failed: {type(exc).__name__}: {exc}", endpoint=path) from exc
                if not isinstance(payload, dict) or str(payload.get("code")) != "00000":
                    code = str(payload.get("code")) if isinstance(payload, dict) else None
                    msg = payload.get("msg") if isinstance(payload, dict) else payload
                    raise BitgetAPIError(f"Bitget API error {code}: {msg}", code=code, endpoint=path)
                return payload.get("data")
        raise BitgetAPIError("Bitget request failed", endpoint=path)

    @staticmethod
    def _contract(raw: dict[str, Any]) -> dict[str, Any]:
        symbol = BitgetV2Client.normalize_symbol(raw.get("symbol", ""))
        margins = raw.get("supportMarginCoins")
        margin_ok = not isinstance(margins, list) or not margins or "USDT" in {str(v).upper() for v in margins}
        tradable = (str(raw.get("symbolStatus", "")).lower() == "normal" and
                    str(raw.get("quoteCoin", "")).upper() == "USDT" and
                    str(raw.get("symbolType", "perpetual")).lower() == "perpetual" and margin_ok)
        step = _positive(raw.get("sizeMultiplier"), _positive(raw.get("minTradeNum"), 1e-8))
        return {"symbol": symbol, "base_coin": str(raw.get("baseCoin", "")).upper(),
                "quote_coin": str(raw.get("quoteCoin", "")).upper(), "tradable": tradable,
                "price_step": _tick(raw), "size_step": step, "min_size": _positive(raw.get("minTradeNum"), step),
                "min_notional": _positive(raw.get("minTradeUSDT"), 0), "max_order_qty": _positive(raw.get("maxOrderQty"), 0),
                "max_market_order_qty": _positive(raw.get("maxMarketOrderQty"), 0),
                "min_leverage": max(1, int(_positive(raw.get("minLever"), 1))),
                "max_leverage": max(1, int(_positive(raw.get("maxLever"), 1))), "raw": raw}

    def get_contracts(self, *, include_untradable: bool = False, refresh: bool = False) -> list[dict[str, Any]]:
        if not self._contracts or refresh:
            data = self._request("GET", "/api/v2/mix/market/contracts", params={"productType": self.product_type})
            rows = [self._contract(dict(r)) for r in data if isinstance(r, dict)] if isinstance(data, list) else []
            self._contracts = {r["symbol"]: r for r in rows if r["symbol"]}
        rows = sorted(self._contracts.values(), key=lambda r: r["symbol"])
        return rows if include_untradable else [r for r in rows if r["tradable"]]

    def get_contract(self, symbol: str) -> dict[str, Any]:
        symbol = self.normalize_symbol(symbol)
        if symbol not in self._contracts: self.get_contracts(refresh=True)
        result = self._contracts.get(symbol)
        if not result or not result["tradable"]: raise BitgetAPIError(f"Bitget contract is not tradable: {symbol}")
        return result

    def get_tickers(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/api/v2/mix/market/tickers", params={"productType": self.product_type})
        out = []
        for r in data if isinstance(data, list) else []:
            if not isinstance(r, dict): continue
            bid, ask = _positive(r.get("bidPr")), _positive(r.get("askPr")); mid = (bid + ask) / 2 if bid and ask else 0
            out.append({"symbol": self.normalize_symbol(r.get("symbol", "")), "last": _positive(r.get("lastPr")),
                        "mark_price": _positive(r.get("markPrice")), "bid": bid, "ask": ask,
                        "spread_pct": ((ask - bid) / mid * 100) if mid and ask >= bid else None,
                        "volume_24h_usdt": _positive(r.get("usdtVolume") or r.get("quoteVolume")), "raw": r})
        return out

    def get_ticker(self, symbol: str) -> dict[str, Any]:
        symbol = self.normalize_symbol(symbol)
        data = self._request("GET", "/api/v2/mix/market/ticker", params={"symbol": symbol, "productType": self.product_type})
        r = data[0] if isinstance(data, list) and data else data if isinstance(data, dict) else {}
        return {"symbol": symbol, "last": _positive(r.get("lastPr")), "mark_price": _positive(r.get("markPrice")), "raw": r}

    def get_account(self) -> dict[str, Any]:
        data = self._request("GET", "/api/v2/mix/account/accounts", params={"productType": self.product_type}, auth=True)
        rows = data if isinstance(data, list) else []
        r = next((x for x in rows if str(x.get("marginCoin", "")).upper() == self.margin_coin), None)
        if not isinstance(r, dict): raise BitgetAPIError(f"No {self.margin_coin} futures account returned")
        return {"equity": _positive(r.get("accountEquity") or r.get("usdtEquity")),
                "available": _positive(r.get("available") or r.get("isolatedMaxAvailable")), "locked": _positive(r.get("locked")), "raw": r}

    def get_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        data = self._request("GET", "/api/v2/mix/position/all-position",
                             params={"productType": self.product_type, "marginCoin": self.margin_coin}, auth=True)
        wanted, out = self.normalize_symbol(symbol) if symbol else None, []
        for r in data if isinstance(data, list) else []:
            if not isinstance(r, dict): continue
            sym, size = self.normalize_symbol(r.get("symbol", "")), _positive(r.get("total"))
            if not size or (wanted and sym != wanted): continue
            hold = str(r.get("holdSide", "")).lower()
            out.append({"symbol": sym, "direction": "long" if hold in {"long", "buy"} else "short", "size": size,
                        "available": _positive(r.get("available")), "entry_price": _positive(r.get("openPriceAvg")),
                        "mark_price": _positive(r.get("markPrice")), "unrealized_pnl": float(r.get("unrealizedPL") or 0),
                        "leverage": int(_positive(r.get("leverage"), 1)), "margin_mode": str(r.get("marginMode", "")),
                        "position_mode": str(r.get("posMode", "")), "liquidation_price": _positive(r.get("liquidationPrice")),
                        "take_profit": _positive(r.get("takeProfit")), "stop_loss": _positive(r.get("stopLoss")), "raw": r})
        return out

    def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        p: dict[str, Any] = {"productType": self.product_type, "limit": 100}
        if symbol: p["symbol"] = self.normalize_symbol(symbol)
        data = self._request("GET", "/api/v2/mix/order/orders-pending", params=p, auth=True)
        return list(data.get("entrustedList", [])) if isinstance(data, dict) else []

    def get_plan_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        p: dict[str, Any] = {"productType": self.product_type, "planType": "profit_loss", "limit": 100}
        if symbol: p["symbol"] = self.normalize_symbol(symbol)
        data = self._request("GET", "/api/v2/mix/order/orders-plan-pending", params=p, auth=True)
        return list(data.get("entrustedList", [])) if isinstance(data, dict) else []

    def set_position_mode(self, mode: str) -> Any:
        pos = "one_way_mode" if mode.lower() in {"one_way_mode", "one-way", "single"} else "hedge_mode"
        return self._request("POST", "/api/v2/mix/account/set-position-mode", body={"productType": self.product_type, "posMode": pos}, auth=True)

    def set_margin_mode(self, symbol: str, mode: str) -> Any:
        margin = "isolated" if mode.lower() == "isolated" else "crossed"
        return self._request("POST", "/api/v2/mix/account/set-margin-mode",
                             body={"symbol": self.normalize_symbol(symbol), "productType": self.product_type,
                                   "marginCoin": self.margin_coin, "marginMode": margin}, auth=True)

    def set_leverage(self, symbol: str, leverage: int, *, hold_side: str | None = None) -> Any:
        body: dict[str, Any] = {"symbol": self.normalize_symbol(symbol), "productType": self.product_type,
                                "marginCoin": self.margin_coin, "leverage": str(max(1, int(leverage)))}
        if hold_side: body["holdSide"] = hold_side
        return self._request("POST", "/api/v2/mix/account/set-leverage", body=body, auth=True)

    def get_max_openable(self, symbol: str, direction: str, *, order_type: str, open_price: float | None = None) -> float:
        p: dict[str, Any] = {"symbol": self.normalize_symbol(symbol), "productType": self.product_type,
                             "marginCoin": self.margin_coin, "posSide": direction, "orderType": order_type}
        if order_type == "limit":
            if not open_price: raise BitgetAPIError("open_price is required for limit max-open query")
            p["openPrice"] = _text(open_price)
        data = self._request("GET", "/api/v2/mix/account/max-open", params=p, auth=True)
        return _positive(data.get("maxOpen")) if isinstance(data, dict) else 0.0

    @staticmethod
    def portfolio_notional(positions: list[dict[str, Any]], orders: list[dict[str, Any]]) -> float:
        total = sum(_positive(p.get("size")) * (_positive(p.get("mark_price")) or _positive(p.get("entry_price"))) for p in positions)
        for o in orders:
            if str(o.get("reduceOnly", "NO")).upper() == "YES" or str(o.get("tradeSide", "")).lower() == "close": continue
            total += _positive(o.get("size")) * (_positive(o.get("priceAvg")) or _positive(o.get("price")))
        return total

    def calculate_size(self, *, symbol: str, direction: str, entry_price: float, stop_price: float,
                       risk_per_trade: float, leverage: int, order_type: str = "market",
                       max_position_notional_equity_multiple: float, max_portfolio_notional_equity_multiple: float,
                       available_margin_utilization_pct: float, positions: list[dict[str, Any]] | None = None,
                       open_orders: list[dict[str, Any]] | None = None, account: dict[str, Any] | None = None,
                       exchange_max_open: float | None = None) -> SizingResult:
        contract = self.get_contract(symbol)
        if direction not in {"long", "short"} or order_type not in {"market", "limit"}: raise BitgetAPIError("invalid direction/order_type")
        if entry_price <= 0 or stop_price <= 0 or (direction == "long" and stop_price >= entry_price) or (direction == "short" and stop_price <= entry_price):
            raise BitgetAPIError("invalid strategy entry/stop geometry")
        acct = account or self.get_account(); equity, available = _positive(acct.get("equity")), _positive(acct.get("available"))
        if equity <= 0 or available <= 0: raise BitgetAPIError("positive Bitget equity and available margin are required")
        if not 0 < risk_per_trade <= 0.05: raise BitgetAPIError("risk_per_trade must be in (0, 0.05]")
        distance = abs(entry_price - stop_price); budget = equity * risk_per_trade; risk_qty = budget / distance
        open_notional = self.portfolio_notional(positions or [], open_orders or [])
        caps = {"risk_notional": risk_qty * entry_price,
                "position_cap": equity * max_position_notional_equity_multiple,
                "portfolio_remaining": max(0.0, equity * max_portfolio_notional_equity_multiple - open_notional),
                "margin_cap": available * max(1, leverage) * available_margin_utilization_pct}
        if exchange_max_open and exchange_max_open > 0: caps["exchange_max_open"] = exchange_max_open * entry_price
        binding, target = min(caps.items(), key=lambda item: item[1])
        if target <= 0: raise BitgetAPIError("portfolio/margin capacity is exhausted")
        raw_qty = target / entry_price
        exchange_qty_cap = (contract["max_market_order_qty"] if order_type == "market" else contract["max_order_qty"]) or contract["max_order_qty"] or contract["max_market_order_qty"]
        if exchange_qty_cap > 0: raw_qty = min(raw_qty, exchange_qty_cap)
        if exchange_max_open and exchange_max_open > 0: raw_qty = min(raw_qty, exchange_max_open)
        qty = _floor_step(raw_qty, contract["size_step"]); min_qty = Decimal(str(contract["min_size"] or 0))
        if qty <= 0 or (min_qty > 0 and qty < min_qty): raise BitgetAPIError("risk-sized quantity is below Bitget minimum; refusing to round risk upward")
        notional = float(qty) * entry_price
        if contract["min_notional"] > 0 and notional < contract["min_notional"]: raise BitgetAPIError("risk-sized notional is below Bitget minimum; refusing to round risk upward")
        loss = float(qty) * distance
        if loss > budget * 1.000001: raise BitgetAPIError("rounded quantity exceeds configured risk budget")
        return SizingResult(self.normalize_symbol(symbol), _text(qty), notional, budget, loss, equity, available,
                            entry_price, stop_price, distance, distance / entry_price, leverage, binding, caps)

    def _require_live(self) -> None:
        if not self.live_enabled: raise BitgetAPIError("LIVE_TRADING_ENABLED=false; live order submission is locked")
        if not self.api_key or not self.api_secret or not self.passphrase: raise BitgetAPIError("Bitget live credentials are incomplete")

    def place_tpsl(self, *, symbol: str, direction: str, size: str, trigger_price: float, kind: str,
                   position_mode: str = "one_way_mode", client_oid: str | None = None) -> dict[str, Any]:
        self._require_live()
        if kind not in {"stop", "take_profit"}: raise BitgetAPIError("kind must be stop or take_profit")
        hold = ("buy" if direction == "long" else "sell") if position_mode == "one_way_mode" else direction
        body = {"marginCoin": self.margin_coin, "productType": self.product_type, "symbol": self.normalize_symbol(symbol),
                "planType": "loss_plan" if kind == "stop" else "profit_plan", "triggerPrice": _text(trigger_price),
                "triggerType": "mark_price", "executePrice": "0", "holdSide": hold, "size": str(size),
                "clientOid": client_oid or f"coinlab-{kind}-{uuid.uuid4().hex[:20]}"}
        data = self._request("POST", "/api/v2/mix/order/place-tpsl-order", body=body, auth=True)
        return data if isinstance(data, dict) else {"raw": data}

    def execute_strategy_order(self, *, symbol: str, direction: str, strategy_entry: float, strategy_stop: float,
                               strategy_take_profit: float, risk_per_trade: float, leverage: int, margin_mode: str,
                               position_mode: str, order_type: str, max_position_notional_equity_multiple: float,
                               max_portfolio_notional_equity_multiple: float, available_margin_utilization_pct: float) -> dict[str, Any]:
        self._require_live(); symbol = self.normalize_symbol(symbol); contract = self.get_contract(symbol)
        ticker = self.get_ticker(symbol)
        if strategy_entry <= 0: strategy_entry = ticker.get("mark_price") or ticker.get("last") or 0
        valid = (direction == "long" and strategy_stop < strategy_entry < strategy_take_profit) or (direction == "short" and strategy_take_profit < strategy_entry < strategy_stop)
        if not valid: raise BitgetAPIError("strategy entry/SL/TP geometry is invalid")
        positions, orders = self.get_positions(), self.get_open_orders()
        if any(p["symbol"] == symbol for p in positions) or any(self.normalize_symbol(o.get("symbol", "")) == symbol for o in orders):
            raise BitgetAPIError(f"{symbol} already has a position or pending entry; refusing duplicate exposure")
        if not positions and not orders: self.set_position_mode(position_mode)
        self.set_margin_mode(symbol, margin_mode)
        safe_lev = max(contract["min_leverage"], min(int(leverage), contract["max_leverage"]))
        self.set_leverage(symbol, safe_lev, hold_side=direction if position_mode != "one_way_mode" and margin_mode == "isolated" else None)
        account = self.get_account(); max_open = self.get_max_openable(symbol, direction, order_type=order_type, open_price=strategy_entry if order_type == "limit" else None)
        sizing = self.calculate_size(symbol=symbol, direction=direction, entry_price=strategy_entry, stop_price=strategy_stop,
                                     risk_per_trade=risk_per_trade, leverage=safe_lev, order_type=order_type,
                                     max_position_notional_equity_multiple=max_position_notional_equity_multiple,
                                     max_portfolio_notional_equity_multiple=max_portfolio_notional_equity_multiple,
                                     available_margin_utilization_pct=available_margin_utilization_pct, positions=positions,
                                     open_orders=orders, account=account, exchange_max_open=max_open)
        client_oid = f"coinlab-{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"
        body: dict[str, Any] = {"symbol": symbol, "productType": self.product_type,
                                "marginMode": "isolated" if margin_mode == "isolated" else "crossed", "marginCoin": self.margin_coin,
                                "size": sizing.quantity, "side": "buy" if direction == "long" else "sell", "orderType": order_type,
                                "clientOid": client_oid, "reduceOnly": "NO",
                                "presetStopLossPrice": _text(strategy_stop), "presetStopSurplusPrice": _text(strategy_take_profit)}
        if position_mode != "one_way_mode": body["tradeSide"] = "open"
        if order_type == "limit": body.update(price=_text(strategy_entry), force="gtc")
        order = self._request("POST", "/api/v2/mix/order/place-order", body=body, auth=True)
        return {"accepted": True, "symbol": symbol, "direction": direction, "order": order, "client_oid": client_oid,
                "strategy_levels": {"entry": strategy_entry, "stop_loss": strategy_stop, "take_profit": strategy_take_profit},
                "exchange_settings": {"margin_mode": margin_mode, "position_mode": position_mode, "leverage": safe_lev},
                "sizing": asdict(sizing), "protection": "Bitget preset exchange-side SL/TP"}

    def close_position_market(self, symbol: str) -> dict[str, Any]:
        self._require_live(); positions = self.get_positions(symbol)
        if not positions: return {"closed": False, "reason": "no_position"}
        p = positions[0]
        body = {"symbol": p["symbol"], "productType": self.product_type,
                "marginMode": "isolated" if p["margin_mode"] == "isolated" else "crossed", "marginCoin": self.margin_coin,
                "size": _text(p["available"] or p["size"]), "side": "sell" if p["direction"] == "long" else "buy",
                "orderType": "market", "reduceOnly": "YES", "clientOid": f"coinlab-close-{uuid.uuid4().hex[:20]}"}
        if p["position_mode"] == "hedge_mode": body["tradeSide"] = "close"
        return {"closed": True, "order": self._request("POST", "/api/v2/mix/order/place-order", body=body, auth=True)}

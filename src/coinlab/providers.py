from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import httpx
import pandas as pd


def _utc_ms(value: str | pd.Timestamp) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def _interval_ms(interval: str) -> int:
    units = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
        "1h": 3_600_000, "4h": 14_400_000, "6h": 21_600_000, "8h": 28_800_000,
        "12h": 43_200_000, "1d": 86_400_000, "1w": 604_800_000,
    }
    if interval not in units:
        raise ValueError(f"Unsupported CoinGlass interval: {interval}")
    return units[interval]


@dataclass
class CoinGlassClient:
    api_key: str
    base_url: str = "https://open-api-v4.coinglass.com"
    timeout: float = 30.0
    _last_request_at: float = field(default=0.0, init=False, repr=False)
    _max_per_minute: int | None = field(default=None, init=False, repr=False)

    def _pace(self) -> None:
        if not self._max_per_minute:
            return
        minimum_gap = 60.0 / max(self._max_per_minute, 1) + 0.03
        remaining = minimum_gap - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)

    def _get(self, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError("COINGLASS_API_KEY is required. Refusing to invent backtest data.")
        with httpx.Client(timeout=self.timeout) as client:
            for attempt in range(4):
                self._pace()
                r = client.get(self.base_url + path, params=params, headers={"CG-API-KEY": self.api_key})
                self._last_request_at = time.monotonic()
                try:
                    header_limit = int(r.headers.get("API-KEY-MAX-LIMIT", "0"))
                    if header_limit > 0:
                        self._max_per_minute = header_limit
                except ValueError:
                    pass
                if r.status_code != 429:
                    break
                if attempt == 3:
                    r.raise_for_status()
                used = int(r.headers.get("API-KEY-USE-LIMIT", "0") or 0)
                wait = 61.0 if self._max_per_minute and used >= self._max_per_minute else max(2.0, 60.0 / max(self._max_per_minute or 30, 1))
                time.sleep(wait)
            r.raise_for_status()
            payload = r.json()
        if str(payload.get("code")) != "0":
            raise RuntimeError(f"CoinGlass error: {payload}")
        data = payload.get("data", [])
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected CoinGlass response shape for {path}")
        return data

    def history(
        self,
        path: str,
        *,
        exchange: str,
        symbol: str,
        interval: str,
        start: str,
        end: str,
        extra: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        start_ms, end_ms = _utc_ms(start), _utc_ms(end)
        if end_ms <= start_ms:
            raise ValueError("end must be after start")
        # Bounded chunks stay below CoinGlass' 1000-row maximum so long histories
        # cannot be silently truncated because of endpoint sort direction.
        bar_ms = _interval_ms(interval)
        max_bars_per_chunk = 900
        cursor = start_ms
        rows: list[dict[str, Any]] = []
        seen: set[int] = set()
        while cursor <= end_ms:
            chunk_end = min(end_ms, cursor + bar_ms * (max_bars_per_chunk - 1))
            params: dict[str, Any] = {
                "exchange": exchange, "symbol": symbol, "interval": interval, "limit": 1000,
                "start_time": cursor, "end_time": chunk_end,
            }
            if extra:
                params.update(extra)
            batch = self._get(path, params)
            if len(batch) >= 1000:
                raise RuntimeError(f"CoinGlass chunk unexpectedly hit 1000-row ceiling for {path}; aborting to avoid truncated backtest data.")
            for row in batch:
                t = int(row["time"])
                if cursor <= t <= chunk_end and t not in seen:
                    rows.append(row)
                    seen.add(t)
            cursor = chunk_end + 1
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).sort_values("time").drop_duplicates("time")
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        return df.set_index("time")

    def open_interest(self, **kwargs: Any) -> pd.DataFrame:
        return self.history("/api/futures/open-interest/history", **kwargs)

    def funding(self, **kwargs: Any) -> pd.DataFrame:
        return self.history("/api/futures/funding-rate/history", **kwargs)

    def liquidations(self, **kwargs: Any) -> pd.DataFrame:
        return self.history("/api/futures/liquidation/history", **kwargs)

    def long_short(self, **kwargs: Any) -> pd.DataFrame:
        return self.history("/api/futures/global-long-short-account-ratio/history", **kwargs)

    def taker_flow(self, **kwargs: Any) -> pd.DataFrame:
        return self.history("/api/futures/v2/taker-buy-sell-volume/history", **kwargs)

    def orderbook(self, **kwargs: Any) -> pd.DataFrame:
        kwargs["extra"] = {**kwargs.get("extra", {}), "range": 1}
        return self.history("/api/futures/orderbook/ask-bids-history", **kwargs)


@dataclass
class BitgetPublicClient:
    base_url: str = "https://api.bitget.com"
    timeout: float = 30.0

    def candles(self, symbol: str, granularity: str, start: str, end: str, product_type: str = "usdt-futures") -> pd.DataFrame:
        start_ms, end_ms = _utc_ms(start), _utc_ms(end)
        cursor = end_ms
        rows: dict[int, list[str]] = {}
        with httpx.Client(timeout=self.timeout) as client:
            while cursor > start_ms:
                params = {
                    "symbol": symbol,
                    "granularity": granularity,
                    "productType": product_type,
                    "endTime": cursor,
                    "limit": 200,
                }
                r = client.get(self.base_url + "/api/v2/mix/market/history-candles", params=params)
                r.raise_for_status()
                payload = r.json()
                if payload.get("code") != "00000":
                    raise RuntimeError(f"Bitget candle error: {payload}")
                batch = payload.get("data", [])
                if not batch:
                    break
                timestamps = []
                for item in batch:
                    t = int(item[0])
                    timestamps.append(t)
                    if start_ms <= t <= end_ms:
                        rows[t] = item
                oldest = min(timestamps)
                if oldest <= start_ms or oldest >= cursor:
                    break
                cursor = oldest - 1
                time.sleep(0.03)
        if not rows:
            return pd.DataFrame()
        out = pd.DataFrame.from_dict(rows, orient="index", columns=["time", "open", "high", "low", "close", "volume_base", "volume_quote"])
        out.index = pd.to_datetime(out.index.astype("int64"), unit="ms", utc=True)
        out = out.drop(columns=["time"]).sort_index()
        return out.astype(float)

    def funding_history(self, symbol: str, start: str, end: str, product_type: str = "usdt-futures") -> pd.DataFrame:
        start_ms, end_ms = _utc_ms(start), _utc_ms(end)
        rows: dict[int, float] = {}
        with httpx.Client(timeout=self.timeout) as client:
            for page_no in range(1, 101):
                params = {"symbol": symbol, "productType": product_type, "pageSize": 100, "pageNo": page_no}
                r = client.get(self.base_url + "/api/v2/mix/market/history-fund-rate", params=params)
                r.raise_for_status()
                payload = r.json()
                if payload.get("code") != "00000":
                    raise RuntimeError(f"Bitget funding history error: {payload}")
                batch = payload.get("data", [])
                if not batch:
                    break
                times = []
                for item in batch:
                    t = int(item["fundingTime"])
                    times.append(t)
                    if start_ms <= t <= end_ms:
                        rows[t] = float(item["fundingRate"])
                if min(times) <= start_ms:
                    break
        if not rows:
            return pd.DataFrame(columns=["funding_rate"])
        out = pd.DataFrame({"funding_rate": pd.Series(rows)})
        out.index = pd.to_datetime(out.index.astype("int64"), unit="ms", utc=True)
        return out.sort_index()


@dataclass
class BitgetTradeClient:
    api_key: str
    api_secret: str
    passphrase: str
    base_url: str = "https://api.bitget.com"
    timeout: float = 30.0
    enabled: bool = False

    def _headers(self, method: str, path: str, params: dict[str, Any] | None, body: dict[str, Any] | None) -> tuple[dict[str, str], str]:
        ts = str(int(time.time() * 1000))
        query = urlencode(params or {})
        body_text = json.dumps(body or {}, separators=(",", ":"), ensure_ascii=False) if body else ""
        prehash = ts + method.upper() + path + (("?" + query) if query else "") + body_text
        sig = base64.b64encode(hmac.new(self.api_secret.encode(), prehash.encode(), hashlib.sha256).digest()).decode()
        return {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": sig,
            "ACCESS-PASSPHRASE": self.passphrase,
            "ACCESS-TIMESTAMP": ts,
            "Content-Type": "application/json",
            "locale": "en-US",
        }, body_text

    def place_futures_order(self, *, symbol: str, side: str, size: str, margin_mode: str = "isolated", order_type: str = "market", client_oid: str) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("Live Bitget trading is disabled. Enable it explicitly only after OOS + paper validation.")
        path = "/api/v2/mix/order/place-order"
        body = {
            "symbol": symbol,
            "productType": "USDT-FUTURES",
            "marginMode": margin_mode,
            "marginCoin": "USDT",
            "size": size,
            "side": side,
            "orderType": order_type,
            "clientOid": client_oid,
        }
        headers, body_text = self._headers("POST", path, None, body)
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(self.base_url + path, headers=headers, content=body_text)
            r.raise_for_status()
            return r.json()

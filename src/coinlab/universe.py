from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class CoinGlassInstrument:
    exchange: str
    instrument_id: str
    base_asset: str
    quote_asset: str
    settlement_currency: str


def get_coinglass_exchange_pairs(
    api_key: str,
    exchange: str,
    *,
    base_url: str = "https://open-api-v4.coinglass.com",
    timeout: float = 30.0,
) -> list[CoinGlassInstrument]:
    if not api_key:
        raise RuntimeError("COINGLASS_API_KEY is required")
    with httpx.Client(timeout=timeout) as client:
        r = client.get(
            base_url.rstrip("/") + "/api/futures/supported-exchange-pairs",
            params={"exchange": exchange},
            headers={"CG-API-KEY": api_key},
        )
        r.raise_for_status()
        payload = r.json()
    if str(payload.get("code")) != "0":
        raise RuntimeError(f"CoinGlass supported pairs error: {payload}")
    data = payload.get("data", {})
    rows: list[dict[str, Any]] = []
    if isinstance(data, dict):
        value = data.get(exchange)
        if value is None:
            # CoinGlass exchange-key capitalization should be stable, but avoid
            # silently failing if the API varies the exact casing.
            value = next((v for k, v in data.items() if str(k).lower() == exchange.lower()), [])
        rows = value if isinstance(value, list) else []
    elif isinstance(data, list):
        rows = data
    out: list[CoinGlassInstrument] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        instrument_id = str(row.get("instrument_id") or "").upper()
        base = str(row.get("base_asset") or "").upper()
        quote = str(row.get("quote_asset") or "").upper()
        settle = str(row.get("settlement_currency") or "").upper()
        if instrument_id and base:
            out.append(CoinGlassInstrument(exchange, instrument_id, base, quote, settle))
    return out


def resolve_coinglass_instrument(
    api_key: str,
    exchange: str,
    bitget_symbol: str,
    *,
    pairs: list[CoinGlassInstrument] | None = None,
) -> str:
    symbol = str(bitget_symbol or "").upper().replace("-", "").replace("_", "").replace("/", "")
    if not symbol.endswith("USDT"):
        raise RuntimeError(f"Expected a USDT symbol, got {bitget_symbol!r}")
    base = symbol[:-4]
    available = pairs if pairs is not None else get_coinglass_exchange_pairs(api_key, exchange)
    candidates = [
        p for p in available
        if p.base_asset == base
        and (p.quote_asset in {"USDT", "USD", ""})
        and (p.settlement_currency in {"USDT", ""})
    ]
    if not candidates:
        raise RuntimeError(f"CoinGlass {exchange} has no supported USDT futures instrument for {symbol}")
    # Prefer the exact public symbol if CoinGlass exposes it directly; Bitget
    # commonly uses instrument IDs like BTCUSDT_UMCBL.
    exact = [p for p in candidates if p.instrument_id == symbol]
    if len(exact) == 1:
        return exact[0].instrument_id
    perpetualish = [
        p for p in candidates
        if any(tag in p.instrument_id for tag in ("PERP", "UMCBL", "USDT"))
    ]
    chosen = perpetualish or candidates
    # Avoid picking dated contracts when more than one unresolved candidate remains.
    if len(chosen) > 1:
        non_dated = [
            p for p in chosen
            if not any(part.isdigit() and len(part) >= 6 for part in p.instrument_id.split("_"))
        ]
        if len(non_dated) == 1:
            return non_dated[0].instrument_id
    return chosen[0].instrument_id

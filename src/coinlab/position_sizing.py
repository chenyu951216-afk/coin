from __future__ import annotations

import math


LOW_PRICE_NOTIONAL_USDT = 2_000.0
HIGH_PRICE_NOTIONAL_USDT = 20_000.0
HIGH_PRICE_THRESHOLD_USDT = 50.0


def paper_notional_for_price(
    market_price: float,
    *,
    low_notional: float = LOW_PRICE_NOTIONAL_USDT,
    high_notional: float = HIGH_PRICE_NOTIONAL_USDT,
    threshold: float = HIGH_PRICE_THRESHOLD_USDT,
) -> float:
    """Return the predeclared simulated-order notional from price at entry time.

    This is deliberately symbol-agnostic: any contract whose market price is
    strictly above 50 USDT uses 20,000 USDT notional; every other contract uses
    2,000 USDT. The backtester calls this at the simulated entry timestamp, not
    using any future price or realized PnL.
    """
    price = float(market_price)
    if not math.isfinite(price) or price <= 0:
        raise ValueError("market_price must be positive and finite")
    return float(high_notional if price > float(threshold) else low_notional)


def sizing_tier_for_price(market_price: float, *, threshold: float = HIGH_PRICE_THRESHOLD_USDT) -> str:
    return "PRICE_GT_50__20000U" if float(market_price) > float(threshold) else "PRICE_LE_50__2000U"

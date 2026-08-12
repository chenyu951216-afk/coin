from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    coinglass_api_key: str = os.getenv("COINGLASS_API_KEY", "")
    coinglass_exchange: str = os.getenv("COINGLASS_EXCHANGE", "Binance")
    coinglass_symbol: str = os.getenv("COINGLASS_SYMBOL", "ETHUSDT")
    symbol: str = os.getenv("SYMBOL", "ETHUSDT")
    timeframe: str = os.getenv("TIMEFRAME", "15m")
    start: str = os.getenv("START", "2025-01-01T00:00:00Z")
    end: str = os.getenv("END", "2026-01-01T00:00:00Z")
    initial_equity: float = float(os.getenv("INITIAL_EQUITY", "10000"))
    risk_per_trade: float = float(os.getenv("RISK_PER_TRADE", "0.01"))
    taker_fee_bps: float = float(os.getenv("TAKER_FEE_BPS", "6"))
    slippage_bps: float = float(os.getenv("SLIPPAGE_BPS", "2"))
    bitget_api_key: str = os.getenv("BITGET_API_KEY", "")
    bitget_api_secret: str = os.getenv("BITGET_API_SECRET", "")
    bitget_api_passphrase: str = os.getenv("BITGET_API_PASSPHRASE", "")
    bitget_product_type: str = os.getenv("BITGET_PRODUCT_TYPE", "USDT-FUTURES")
    bitget_margin_mode: str = os.getenv("BITGET_MARGIN_MODE", "isolated")
    live_trading_enabled: bool = os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true"

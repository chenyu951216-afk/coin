from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # CoinGlass research data. AUTO maps a Bitget public symbol to CoinGlass'
    # current instrument id for the selected exchange (for example *_UMCBL).
    coinglass_api_key: str = os.getenv("COINGLASS_API_KEY", "")
    coinglass_exchange: str = os.getenv("COINGLASS_EXCHANGE", "Bitget")
    coinglass_symbol: str = os.getenv("COINGLASS_SYMBOL", "AUTO")

    # Selected Bitget contract for a single-symbol backtest.
    symbol: str = os.getenv("SYMBOL", "ETHUSDT").upper()
    timeframe: str = os.getenv("TIMEFRAME", "15m")
    start: str = os.getenv("START", "2025-01-01T00:00:00Z")
    end: str = os.getenv("END", "2026-01-01T00:00:00Z")
    initial_equity: float = float(os.getenv("INITIAL_EQUITY", "10000"))
    risk_per_trade: float = float(os.getenv("RISK_PER_TRADE", "0.01"))
    taker_fee_bps: float = float(os.getenv("TAKER_FEE_BPS", "6"))
    slippage_bps: float = float(os.getenv("SLIPPAGE_BPS", "2"))
    min_aligned_coverage: float = float(os.getenv("MIN_ALIGNED_COVERAGE", "0.90"))

    # Bitget exchange layer. These values control execution mechanics only;
    # they never alter strategy entry/stop/take-profit levels.
    bitget_rest_base_url: str = os.getenv("BITGET_REST_BASE_URL", "https://api.bitget.com")
    bitget_api_key: str = os.getenv("BITGET_API_KEY", "")
    bitget_api_secret: str = os.getenv("BITGET_API_SECRET", "")
    bitget_api_passphrase: str = os.getenv("BITGET_API_PASSPHRASE", "")
    bitget_product_type: str = os.getenv("BITGET_PRODUCT_TYPE", "USDT-FUTURES")
    bitget_margin_coin: str = os.getenv("BITGET_MARGIN_COIN", "USDT")
    bitget_position_mode: str = os.getenv("BITGET_POSITION_MODE", "one_way_mode")
    bitget_margin_mode: str = os.getenv("BITGET_MARGIN_MODE", "isolated")
    bitget_leverage: int = int(os.getenv("BITGET_LEVERAGE", "5"))
    max_position_notional_equity_multiple: float = float(
        os.getenv("MAX_POSITION_NOTIONAL_EQUITY_MULTIPLE", "2.0")
    )
    max_portfolio_notional_equity_multiple: float = float(
        os.getenv("MAX_PORTFOLIO_NOTIONAL_EQUITY_MULTIPLE", "5.0")
    )
    available_margin_utilization_pct: float = float(
        os.getenv("AVAILABLE_MARGIN_UTILIZATION_PCT", "0.70")
    )
    live_trading_enabled: bool = _bool("LIVE_TRADING_ENABLED", "false")

    # Whole-market scanner. Bitget public liquidity is filtered first; only
    # survivors consume CoinGlass Standard history requests.
    scan_timeframe: str = os.getenv("SCAN_TIMEFRAME", "15m")
    scan_lookback_bars: int = int(os.getenv("SCAN_LOOKBACK_BARS", "180"))
    scan_min_aligned_rows: int = int(os.getenv("SCAN_MIN_ALIGNED_ROWS", "110"))
    scan_min_turnover_usdt: float = float(os.getenv("SCAN_MIN_TURNOVER_USDT", "1000000"))
    scan_max_spread_pct: float = float(os.getenv("SCAN_MAX_SPREAD_PCT", "0.50"))
    scan_max_symbols: int = int(os.getenv("SCAN_MAX_SYMBOLS", "0"))

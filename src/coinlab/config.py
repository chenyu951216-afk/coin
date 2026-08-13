from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # CoinGlass research data.
    coinglass_api_key: str = os.getenv("COINGLASS_API_KEY", "")
    coinglass_exchange: str = os.getenv("COINGLASS_EXCHANGE", "Bitget")
    coinglass_symbol: str = os.getenv("COINGLASS_SYMBOL", "AUTO")

    # Research defaults. Dates remain blank so CoinGlass Standard's maximum safe
    # window is selected automatically for the chosen timeframe.
    symbol: str = os.getenv("SYMBOL", "ETHUSDT").upper()
    timeframe: str = os.getenv("TIMEFRAME", "15m")
    start: str = os.getenv("START", "")
    end: str = os.getenv("END", "")
    initial_equity: float = float(os.getenv("INITIAL_EQUITY", "10000"))
    # Kept for backwards compatibility. Paper/backtest sizing is fixed-notional.
    risk_per_trade: float = float(os.getenv("RISK_PER_TRADE", "0.01"))
    taker_fee_bps: float = float(os.getenv("TAKER_FEE_BPS", "6"))
    slippage_bps: float = float(os.getenv("SLIPPAGE_BPS", "2"))
    max_estimated_cost_r: float = float(os.getenv("MAX_ESTIMATED_COST_R", "0.18"))
    min_aligned_coverage: float = float(os.getenv("MIN_ALIGNED_COVERAGE", "0.90"))

    # Fixed simulated-order notional policy requested by the user. The tier is
    # chosen from the market price at simulated entry time, never from future PnL.
    paper_low_notional_usdt: float = float(os.getenv("PAPER_LOW_NOTIONAL_USDT", "2000"))
    paper_high_notional_usdt: float = float(os.getenv("PAPER_HIGH_NOTIONAL_USDT", "20000"))
    paper_high_price_threshold: float = float(os.getenv("PAPER_HIGH_PRICE_THRESHOLD", "50"))

    # Whole-market historical backtest. 0 means process every eligible current
    # Bitget contract; a positive cap is for debugging only and is disclosed in reports.
    market_backtest_min_24h_turnover_usdt: float = float(os.getenv("MARKET_BACKTEST_MIN_24H_TURNOVER_USDT", "1000000"))
    market_backtest_max_symbols: int = int(os.getenv("MARKET_BACKTEST_MAX_SYMBOLS", "0"))
    market_backtest_min_aligned_rows: int = int(os.getenv("MARKET_BACKTEST_MIN_ALIGNED_ROWS", "500"))
    market_cache_root: str = os.getenv("MARKET_CACHE_ROOT", "artifacts/market_cache")

    # Bitget execution mechanics only. These values never derive strategy entry/SL/TP.
    bitget_rest_base_url: str = os.getenv("BITGET_REST_BASE_URL", "https://api.bitget.com")
    bitget_api_key: str = os.getenv("BITGET_API_KEY", "")
    bitget_api_secret: str = os.getenv("BITGET_API_SECRET", "")
    bitget_api_passphrase: str = os.getenv("BITGET_API_PASSPHRASE", "")
    bitget_product_type: str = os.getenv("BITGET_PRODUCT_TYPE", "USDT-FUTURES")
    bitget_margin_coin: str = os.getenv("BITGET_MARGIN_COIN", "USDT")
    bitget_position_mode: str = os.getenv("BITGET_POSITION_MODE", "one_way_mode")
    bitget_margin_mode: str = os.getenv("BITGET_MARGIN_MODE", "isolated")
    bitget_leverage: int = int(os.getenv("BITGET_LEVERAGE", "5"))
    max_position_notional_equity_multiple: float = float(os.getenv("MAX_POSITION_NOTIONAL_EQUITY_MULTIPLE", "2.0"))
    max_portfolio_notional_equity_multiple: float = float(os.getenv("MAX_PORTFOLIO_NOTIONAL_EQUITY_MULTIPLE", "5.0"))
    available_margin_utilization_pct: float = float(os.getenv("AVAILABLE_MARGIN_UTILIZATION_PCT", "0.70"))
    live_trading_enabled: bool = _bool("LIVE_TRADING_ENABLED", "false")

    # Continuous whole-market scanner. It runs once per newly completed scan
    # timeframe candle, not in a wasteful tight loop.
    scan_timeframe: str = os.getenv("SCAN_TIMEFRAME", "15m")
    scan_lookback_bars: int = int(os.getenv("SCAN_LOOKBACK_BARS", "180"))
    scan_min_aligned_rows: int = int(os.getenv("SCAN_MIN_ALIGNED_ROWS", "110"))
    scan_min_turnover_usdt: float = float(os.getenv("SCAN_MIN_TURNOVER_USDT", "1000000"))
    scan_max_spread_pct: float = float(os.getenv("SCAN_MAX_SPREAD_PCT", "0.50"))
    scan_max_symbols: int = int(os.getenv("SCAN_MAX_SYMBOLS", "0"))
    scan_auto_start: bool = _bool("SCAN_AUTO_START", "true")
    scan_grace_seconds: int = int(os.getenv("SCAN_GRACE_SECONDS", "8"))
    scan_error_retry_seconds: int = int(os.getenv("SCAN_ERROR_RETRY_SECONDS", "120"))

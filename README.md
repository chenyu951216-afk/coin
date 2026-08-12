# CoinGlass × Bitget Strategy Lab

Research-first crypto futures platform for **real-data backtests, multi-symbol Bitget discovery, whole-market strategy scans, and a separated Bitget execution layer**.

The project deliberately refuses to invent backtest metrics. Strategy research uses real CoinGlass derivatives history and Bitget finished futures candles, strict timestamp alignment, no-lookahead execution, fees/slippage/funding, chronological OOS validation, and reproducible reports.

## Architecture boundary

Two layers are intentionally separated:

- **Strategy layer** — `src/coinlab/strategies.py`, `features.py`, and `backtest.py`. These decide whether a strategy signals and how the current strategy derives initial stop/target behavior.
- **Exchange layer** — `src/coinlab/exchange.py`. This knows Bitget contracts, quantity/price precision, minimums, account/positions/orders, margin mode, leverage, risk-budget sizing, duplicate-exposure checks, live order submission and exchange-side protection.

The Bitget exchange adapter must **never improve, widen, tighten, replace, or optimize a strategy-provided entry, stop loss or take profit**. It may only reject an invalid/untradeable order, round quantity **down** to exchange precision, and cap size for account/exchange safety.

## Strategies (unchanged by the exchange upgrade)

1. `oi_breakout`
2. `liquidation_reversal`
3. `funding_crowding`
4. `taker_flow_momentum`
5. `orderbook_pressure`
6. `oi_divergence`

These remain research hypotheses, not claims of profitability.

## Web dashboard

Deploying the service opens a dashboard at `/` with:

- live Bitget USDT-futures contract search/dropdown;
- search-as-you-type symbol suggestions (for example `b` can surface BTC/BNB and other matching contracts);
- selected-symbol real-data backtest launcher;
- whole-Bitget-market strategy scanner;
- scanner results with strategy, direction, reference price, current strategy SL/TP and CoinGlass instrument mapping;
- protected Bitget account/position views;
- prominent live-trading lock state.

`/docs` remains available for API inspection.

## Whole-market scanner

The scanner does not blindly spend CoinGlass requests on every listed contract:

1. Read current tradable USDT perpetual contracts directly from Bitget.
2. Read Bitget public tickers.
3. Filter by minimum 24h USDT turnover and maximum spread.
4. Resolve only supported symbols to CoinGlass's Bitget instrument id.
5. Fetch CoinGlass OI, funding, liquidation, long/short, taker flow and orderbook history.
6. Build the same feature frame used by research.
7. Run the existing `STRATEGIES` functions on the latest completed-bar feature row.
8. Return only matching symbol × strategy signals.

The scan is a discovery process; it does not automatically place a live order.

## Backtest integrity

- Signal is computed only after bar `t` closes.
- Earliest simulated entry is bar `t+1` open.
- No future feature / negative shift is used to form a signal.
- CoinGlass and Bitget inputs are timestamp-aligned; missing derivative observations are not filled from the future.
- Fees, slippage and Bitget funding are included.
- Same-bar TP+SL ambiguity uses conservative stop-first handling unless lower-timeframe path data is available.
- Stop management only changes after a completed bar.
- Empty or materially incomplete data aborts the run instead of generating synthetic performance.
- Reports include source row counts and SHA-256 fingerprints.
- Every run includes chronological 60/20/20 train/validation/test plus ordered walk-forward metrics.

## Bitget execution safety

`src/coinlab/exchange.py` uses current Bitget contract metadata for price/size steps, minimum quantity/notional, maximum market/limit quantity and leverage range. Position sizing starts from the configured account-equity loss budget at the **strategy's supplied stop** and then applies position, portfolio, available-margin and Bitget max-open caps. Quantity is rounded down; an order below the exchange minimum is rejected rather than enlarged beyond the risk budget.

Entry orders send the exact strategy-provided SL/TP as Bitget exchange-side preset protection. The adapter also contains a separate TP/SL plan-order method for protection repair/management workflows without deriving new strategy prices.

Live submission has two independent controls:

1. protected API endpoints require `ADMIN_BEARER_TOKEN`;
2. real order submission additionally requires `LIVE_TRADING_ENABLED=true` and complete Bitget API credentials.

During research keep `LIVE_TRADING_ENABLED=false`.

## Environment

Copy `.env.example` and fill at least `COINGLASS_API_KEY` plus `ADMIN_BEARER_TOKEN` for web-controlled research. Bitget private credentials are unnecessary for public symbol search, scanning, or historical backtesting.

For private account reads add:

```env
BITGET_API_KEY=...
BITGET_API_SECRET=...
BITGET_API_PASSPHRASE=...
```

Do not commit secrets. If an older repository ever committed a real `.env`, rotate those credentials rather than reusing them.

## Zeabur

A deterministic `Dockerfile` is included. It installs the package and runs:

```text
python main.py
```

The application listens on `0.0.0.0:$PORT` (default 8080), which avoids relying on a platform guess about the entry file.

## Tests

```bash
pip install -e '.[dev]'
pytest -q
```

The exchange-layer tests verify normalization, risk-budget sizing, downward quantity rounding, rejection below Bitget minimums, correct market-vs-limit max quantity handling, and exclusion of reduce-only orders from portfolio exposure.

## Backtest report to send to ChatGPT

After a run, send `BACKTEST_REPORT.json`. If one strategy looks abnormal, also send its `trades_<strategy>.csv`. Do not optimize based only on the full-sample result; inspect OOS test and walk-forward consistency first.

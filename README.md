# CoinGlass × Bitget Strategy Lab

A research-first ETH futures strategy lab designed to **refuse fake backtest numbers**. It downloads real CoinGlass V4 derivatives history and finished Bitget futures candles, aligns them strictly by timestamp, runs multiple strategies without look-ahead, and exports a machine-readable report you can paste back into ChatGPT for review.

## Current strategy set

1. `oi_breakout` — price breakout + expanding OI + taker-flow confirmation.
2. `liquidation_reversal` — liquidation flush exhaustion / reversal.
3. `funding_crowding` — extreme funding + crowded long/short positioning mean reversion.
4. `taker_flow_momentum` — aggressive taker imbalance with OI/trend confirmation.
5. `orderbook_pressure` — historical ±1% bid/ask imbalance with trend/pullback filter.
6. `oi_divergence` — strong price move while OI contracts, looking for exhaustion.

These are research hypotheses, **not claimed profitable strategies**. The repo intentionally contains no made-up example PF/win-rate values.

## Backtest integrity rules

- Signal is computed from bar `t` only after it is closed.
- Earliest entry is bar `t+1` open.
- No `shift(-1)`/future feature is used to create a signal.
- CoinGlass and Bitget datasets are joined on exact timestamps; missing derivative bars are not forward-filled into the future.
- Market entries/exits include configurable slippage and both-side taker fees; positions crossing Bitget funding timestamps also include the published historical funding rate.
- Position size is based on equity risk and ATR stop distance, not arbitrary leverage.
- If TP and SL are both touched inside one OHLC bar and no lower-timeframe path is available, the engine uses **stop-first** and records `ambiguous_exit=true`.
- One strategy cannot stack overlapping positions in the same backtest.
- Empty/misaligned data aborts the run rather than producing synthetic statistics.
- The report includes SHA-256 hashes for aligned features and every trades CSV so a run can be reproduced and compared.

## Setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Put your CoinGlass Standard API key in `.env`:

```env
COINGLASS_API_KEY=your_key_here
```

Keep `LIVE_TRADING_ENABLED=false` during strategy research.

## Run a real backtest

```bash
coinlab backtest --out artifacts/eth_15m_run_001
```

Output:

- `BACKTEST_REPORT.json` — paste this whole file into ChatGPT when you want strategy changes.
- `aligned_features.csv` — exact aligned input used by the engine.
- `trades_<strategy>.csv` — complete trades, fills, fees, R, exit reason and ambiguous-fill flags.

The report is deliberately compact enough for iterative analysis while trade CSVs remain available for deeper debugging.

## What to send back to ChatGPT

The best artifact is `BACKTEST_REPORT.json`. If one strategy behaves strangely, also send its `trades_<strategy>.csv`.

Useful review request:

> Review this CoinLab BACKTEST_REPORT.json. Do not optimize on the full sample. Diagnose each strategy using trade count, PF, expectancy R, max drawdown, consecutive losses and ambiguous-exit count. Propose changes that can be validated out-of-sample and preserve the no-lookahead rules.

## Live Bitget execution

`BitgetTradeClient` already implements authenticated futures order signing and `POST /api/v2/mix/order/place-order`. Live trading is intentionally **not wired to strategy signals yet**. That should only be enabled after walk-forward/out-of-sample validation and paper trading.

Never commit API keys. Use environment variables/secrets on Zeabur or your server.

## Important next research steps

The first version prioritizes correctness and auditability. Before real money, add:

- parameter search that tunes only on train/validation and keeps the final test set untouched;
- 1m sub-bar execution data for resolving same-bar TP/SL instead of conservative stop-first;
- latency/spread model calibrated from live Bitget data;
- paper-trading shadow mode before any live-order switch.

## Out-of-sample / anti-overfit validation

Every run also produces chronological **60% train / 20% validation / 20% test** metrics plus five time-ordered walk-forward folds. The strategy rules are frozen before the test segment; the test segment is not used to tune thresholds in this version.

Each strategy receives a transparent research grade:

- `INSUFFICIENT_OOS_TRADES` — fewer than 30 test trades.
- `REJECT_OOS` — test PF <= 1 or test expectancy <= 0R.
- `PROMISING_RESEARCH_CANDIDATE` — positive validation expectancy, positive test edge and at least 60% profitable usable walk-forward folds.
- `NEEDS_MORE_VALIDATION` — not rejected, but consistency is not strong enough yet.

This grade is a research filter, not a promise of future profit.

### Adaptive exits

Initial risk uses the farther of strategy ATR risk and a confirmed 12-bar swing structure, capped at 3.5 ATR to avoid a distant wick creating tiny position size. Stop management is close-confirmed: after a bar **closes** at +1R the stop may move to breakeven for the next bar; after a close at +1.5R an ATR trail may tighten for later bars. The engine never retroactively moves a stop inside the same candle.

Trade CSVs also include funding PnL, MFE-R, MAE-R, holding bars, breakeven activation, trailing activation and ambiguous exits, which makes later strategy diagnosis much more useful than win rate alone.

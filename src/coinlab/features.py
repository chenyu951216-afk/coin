from __future__ import annotations

import numpy as np
import pandas as pd


def zscore(s: pd.Series, window: int = 96) -> pd.Series:
    mean = s.rolling(window, min_periods=window).mean()
    std = s.rolling(window, min_periods=window).std(ddof=0).replace(0, np.nan)
    return (s - mean) / std


def build_feature_frame(
    price: pd.DataFrame,
    oi: pd.DataFrame,
    funding: pd.DataFrame,
    liq: pd.DataFrame,
    ls: pd.DataFrame,
    taker: pd.DataFrame,
    orderbook: pd.DataFrame,
) -> pd.DataFrame:
    frames = [price.copy()]
    mappings = [
        (oi, {"open": "oi_open", "high": "oi_high", "low": "oi_low", "close": "oi_close"}),
        (funding, {"open": "fund_open", "high": "fund_high", "low": "fund_low", "close": "fund_close"}),
        (liq, {}),
        (ls, {}),
        (taker, {}),
        (orderbook, {}),
    ]
    for df, rename in mappings:
        if df is not None and not df.empty:
            x = df.rename(columns=rename).copy()
            for c in x.columns:
                x[c] = pd.to_numeric(x[c], errors="coerce")
            frames.append(x)
    # Strict timestamp intersection: no forward-filling future/late derivative observations.
    df = pd.concat(frames, axis=1, join="inner").sort_index()
    if df.empty:
        return df

    df["ret_1"] = df["close"].pct_change()
    df["ret_4"] = df["close"].pct_change(4)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14, min_periods=14).mean()
    df["atr_pct"] = df["atr14"] / df["close"]

    if "oi_close" in df:
        df["oi_chg_1"] = df["oi_close"].pct_change()
        df["oi_chg_4"] = df["oi_close"].pct_change(4)
        df["oi_z"] = zscore(df["oi_chg_1"])
    if "fund_close" in df:
        df["fund_z"] = zscore(df["fund_close"])
    if {"taker_buy_volume_usd", "taker_sell_volume_usd"}.issubset(df.columns):
        total = df["taker_buy_volume_usd"] + df["taker_sell_volume_usd"]
        df["taker_imb"] = (df["taker_buy_volume_usd"] - df["taker_sell_volume_usd"]) / total.replace(0, np.nan)
        df["taker_imb_z"] = zscore(df["taker_imb"])
    if {"long_liquidation_usd", "short_liquidation_usd"}.issubset(df.columns):
        df["long_liq_z"] = zscore(df["long_liquidation_usd"].clip(lower=0).map(np.log1p))
        df["short_liq_z"] = zscore(df["short_liquidation_usd"].clip(lower=0).map(np.log1p))
    if "global_account_long_short_ratio" in df:
        df["ls_z"] = zscore(df["global_account_long_short_ratio"])
    if {"bids_usd", "asks_usd"}.issubset(df.columns):
        total = df["bids_usd"] + df["asks_usd"]
        df["book_imb"] = (df["bids_usd"] - df["asks_usd"]) / total.replace(0, np.nan)
        df["book_imb_z"] = zscore(df["book_imb"])
    return df.replace([np.inf, -np.inf], np.nan)

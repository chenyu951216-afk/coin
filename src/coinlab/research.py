from __future__ import annotations

from typing import Any

import pandas as pd

from .features import build_feature_frame


SOURCE_LABELS: dict[str, str] = {
    "oi": "未平倉量 OI",
    "funding": "資金費率 Funding",
    "liq": "爆倉 Liquidation",
    "ls": "多空帳戶比 Long/Short",
    "taker": "主動買賣 Taker Flow",
    "orderbook": "訂單簿 Orderbook",
}

STRATEGY_SOURCE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "oi_breakout": ("oi", "funding", "taker"),
    "liquidation_reversal": ("liq", "taker"),
    "funding_crowding": ("funding", "ls", "taker"),
    "taker_flow_momentum": ("oi", "funding", "taker"),
    # Resting orderbook liquidity is noisy/spoofable, so executed taker flow is
    # now a required confirmation source rather than treating the book alone as edge.
    "orderbook_pressure": ("oi", "orderbook", "taker"),
    # OI contraction reversal now requires liquidation evidence before the
    # strategy is allowed to fade the prior move.
    "oi_divergence": ("oi", "funding", "taker", "liq"),
}


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame()


def strategy_requirements(strategy_name: str) -> tuple[str, ...]:
    return STRATEGY_SOURCE_REQUIREMENTS.get(strategy_name, tuple())


def source_status(df: pd.DataFrame | None, error: Exception | None = None) -> dict[str, Any]:
    if error is not None:
        return {
            "status": "error",
            "rows": 0,
            "start": None,
            "end": None,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
    if df is None or df.empty:
        return {"status": "empty", "rows": 0, "start": None, "end": None}
    return {
        "status": "ready",
        "rows": int(len(df)),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "api_adjusted_start_ms": df.attrs.get("api_adjusted_start_ms"),
    }


def build_strategy_frame(
    *,
    strategy_name: str,
    price: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
    min_coverage: float,
    min_rows: int,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    required = strategy_requirements(strategy_name)
    missing = [name for name in required if name not in datasets or datasets[name].empty]
    if price.empty:
        return None, {
            "status": "skipped",
            "code": "BITGET_PRICE_MISSING",
            "reason": "Bitget 價格 K 線沒有資料。",
            "required_sources": list(required),
            "missing_sources": ["price"],
        }
    if missing:
        labels = [SOURCE_LABELS.get(name, name) for name in missing]
        return None, {
            "status": "skipped",
            "code": "REQUIRED_SOURCE_UNAVAILABLE",
            "reason": "缺少此策略必要的 CoinGlass 資料：" + "、".join(labels),
            "required_sources": list(required),
            "missing_sources": missing,
        }

    starts = [price.index.min()] + [datasets[name].index.min() for name in required]
    ends = [price.index.max()] + [datasets[name].index.max() for name in required]
    common_start = max(starts)
    common_end = min(ends)
    if common_end <= common_start:
        return None, {
            "status": "skipped",
            "code": "NO_COMMON_WINDOW",
            "reason": "這個策略需要的資料來源沒有可共同對齊的時間區間。",
            "required_sources": list(required),
            "missing_sources": [],
        }

    p = price.loc[(price.index >= common_start) & (price.index <= common_end)].copy()
    sliced = {
        name: datasets[name].loc[
            (datasets[name].index >= common_start) & (datasets[name].index <= common_end)
        ].copy()
        for name in required
    }
    frame = build_feature_frame(
        p,
        sliced.get("oi", empty_frame()),
        sliced.get("funding", empty_frame()),
        sliced.get("liq", empty_frame()),
        sliced.get("ls", empty_frame()),
        sliced.get("taker", empty_frame()),
        sliced.get("orderbook", empty_frame()),
    )
    coverage = float(len(frame) / len(p)) if len(p) else 0.0
    diagnostic = {
        "status": "ready",
        "code": "READY",
        "required_sources": list(required),
        "common_start": str(common_start),
        "common_end": str(common_end),
        "price_rows_in_common_window": int(len(p)),
        "aligned_rows": int(len(frame)),
        "aligned_coverage": coverage,
        "source_rows_in_common_window": {name: int(len(sliced[name])) for name in required},
    }
    if coverage < min_coverage:
        diagnostic.update({
            "status": "skipped",
            "code": "DATA_ALIGNMENT_LOW",
            "reason": (
                f"必要資料精確時間對齊率只有 {coverage:.2%}，低於最低要求 {min_coverage:.0%}；"
                "系統拒絕用殘缺交集產生績效。"
            ),
        })
        return None, diagnostic
    if len(frame) < min_rows:
        diagnostic.update({
            "status": "skipped",
            "code": "TOO_FEW_ALIGNED_ROWS",
            "reason": f"共同可用資料只有 {len(frame)} 根 K，低於最低要求 {min_rows} 根。",
        })
        return None, diagnostic
    return frame, diagnostic

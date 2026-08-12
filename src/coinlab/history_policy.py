from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import re
from typing import Any


STANDARD_HISTORY_DAYS: dict[str, int | None] = {
    "1m": 6, "3m": 20, "5m": 30, "15m": 90, "30m": 180,
    "1h": 360, "2h": 360, "4h": 360, "6h": 360, "8h": 360, "12h": 360,
    "1d": None, "1w": None,
}

INTERVAL_SECONDS: dict[str, int] = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1_800,
    "1h": 3_600, "2h": 7_200, "4h": 14_400, "6h": 21_600, "8h": 28_800,
    "12h": 43_200, "1d": 86_400, "1w": 604_800,
}


@dataclass(frozen=True)
class BacktestWindow:
    timeframe: str
    max_history_days: int | None
    requested_start: str | None
    requested_end: str | None
    used_start: str
    used_end: str
    earliest_safe_start: str | None
    latest_completed_bar: str
    adjusted: bool
    adjustment_reason: str | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def latest_completed_bar(timeframe: str, now: datetime | None = None) -> datetime:
    seconds = INTERVAL_SECONDS.get(timeframe)
    if not seconds:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    epoch = int(current.timestamp())
    current_bar_open = epoch - (epoch % seconds)
    return datetime.fromtimestamp(current_bar_open - seconds, tz=timezone.utc)


def standard_policy(timeframe: str, now: datetime | None = None) -> dict[str, Any]:
    if timeframe not in STANDARD_HISTORY_DAYS:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    latest = latest_completed_bar(timeframe, current)
    max_days = STANDARD_HISTORY_DAYS[timeframe]
    earliest = None
    if max_days is not None:
        safety = timedelta(seconds=INTERVAL_SECONDS[timeframe] * 4)
        earliest = current - timedelta(days=max_days) + safety
        step = INTERVAL_SECONDS[timeframe]
        epoch = int(earliest.timestamp())
        aligned = epoch + ((step - epoch % step) % step)
        earliest = datetime.fromtimestamp(aligned, tz=timezone.utc)
    return {
        "plan": "Standard",
        "timeframe": timeframe,
        "max_history_days": max_days,
        "earliest_safe_start": _iso(earliest) if earliest else None,
        "latest_completed_bar": _iso(latest),
        "description": f"CoinGlass Standard 的 {timeframe} 歷史資料最多可查最近 {max_days} 天。" if max_days is not None else f"CoinGlass Standard 的 {timeframe} 可查完整歷史範圍。",
    }


def normalize_backtest_window(*, timeframe: str, requested_start: str | None, requested_end: str | None, now: datetime | None = None) -> BacktestWindow:
    policy = standard_policy(timeframe, now)
    latest = _parse(policy["latest_completed_bar"])
    earliest = _parse(policy["earliest_safe_start"])
    assert latest is not None
    try:
        req_start = _parse(requested_start)
        req_end = _parse(requested_end)
    except ValueError as exc:
        raise ValueError("日期格式不正確，請使用 YYYY-MM-DDTHH:MM:SSZ。") from exc

    adjusted = False
    reason: str | None = None
    if earliest is None:
        used_end = min(req_end or latest, latest)
        used_start = req_start or (used_end - timedelta(days=365))
        if req_end and req_end > latest:
            adjusted = True
            reason = "end_after_latest_completed_bar"
    else:
        if req_start is None and req_end is None:
            used_start, used_end = earliest, latest
        else:
            original_end = req_end or latest
            if original_end <= earliest:
                used_start, used_end = earliest, latest
                adjusted = True
                reason = "requested_range_fully_expired"
            else:
                used_end = min(original_end, latest)
                used_start = req_start or earliest
                if used_start < earliest:
                    used_start = earliest
                    adjusted = True
                    reason = "start_before_plan_limit"
                if original_end > latest:
                    adjusted = True
                    reason = reason or "end_after_latest_completed_bar"
                if used_end <= used_start:
                    used_start, used_end = earliest, latest
                    adjusted = True
                    reason = "invalid_or_empty_range_reset"

    if adjusted and reason == "requested_range_fully_expired":
        message = f"你選的整段日期已超出 CoinGlass Standard 的 {timeframe} 歷史範圍。系統已自動改用目前可取得的最近資料：{_iso(used_start)} ～ {_iso(used_end)}。"
    elif adjusted:
        message = f"你選的日期有一部分超出 CoinGlass Standard 的 {timeframe} 可用範圍，已自動調整為 {_iso(used_start)} ～ {_iso(used_end)}。"
    else:
        message = f"將回測 {_iso(used_start)} ～ {_iso(used_end)} 的真實資料。"

    return BacktestWindow(
        timeframe=timeframe, max_history_days=policy["max_history_days"], requested_start=requested_start,
        requested_end=requested_end, used_start=_iso(used_start), used_end=_iso(used_end),
        earliest_safe_start=policy["earliest_safe_start"], latest_completed_bar=policy["latest_completed_bar"],
        adjusted=adjusted, adjustment_reason=reason, message=message,
    )


def humanize_backtest_failure(log_text: str, timeframe: str | None = None) -> dict[str, str]:
    text = log_text or ""
    lower = text.lower()
    tf = timeframe or "目前週期"
    max_days = STANDARD_HISTORY_DAYS.get(timeframe or "")

    if "ALL_STRATEGIES_SKIPPED:" in text:
        detail = text.split("ALL_STRATEGIES_SKIPPED:", 1)[1].splitlines()[0].strip()
        return {
            "code": "ALL_STRATEGIES_SKIPPED",
            "title": "目前沒有策略能取得完整必要資料",
            "message": detail or "所有策略都因必要 CoinGlass 資料缺失或時間對齊不足而未執行。",
            "action": "系統會按策略分開檢查資料來源；不會再因單一端點失敗而把所有策略一起停掉。",
        }

    match = re.search(r"earliest allowed start_time is\s*(\d+)", text, flags=re.IGNORECASE)
    if "Invalid time range" in text or match:
        earliest = None
        if match:
            try:
                earliest = _iso(datetime.fromtimestamp(int(match.group(1)) / 1000, tz=timezone.utc))
            except (ValueError, OSError, OverflowError):
                earliest = None
        range_text = f"最近 {max_days} 天" if max_days else "方案允許的歷史範圍"
        return {
            "code": "COINGLASS_HISTORY_LIMIT",
            "title": "CoinGlass 某個資料端點的實際可用起點較晚",
            "message": f"CoinGlass Standard 的 {tf} 名義範圍是 {range_text}，但其中一個資料端點回報更晚的最早可用時間。" + (f" API 回報最早可從 {earliest} 開始。" if earliest else ""),
            "action": "最新版會依 API 實際回覆自動向後縮到可用起點，不需要你手動改日期。",
        }
    if "global-long-short-account-ratio" in lower:
        return {
            "code": "COINGLASS_LONG_SHORT_UNAVAILABLE",
            "title": "CoinGlass 多空帳戶比來源不可用",
            "message": "這個交易所／幣種目前沒有回傳 Long/Short 帳戶比資料。其他不依賴此資料的策略不應因此一起停止。",
            "action": "最新版只會跳過真正需要 Long/Short 的策略，其餘策略照常回測與掃描。",
        }
    if "orderbook/ask-bids-history" in lower:
        return {
            "code": "COINGLASS_ORDERBOOK_UNAVAILABLE",
            "title": "CoinGlass 訂單簿歷史來源不可用",
            "message": "這個交易所／幣種目前沒有完整回傳 Orderbook 歷史。其他策略不應因此一起停止。",
            "action": "最新版只會跳過依賴 Orderbook 的策略，其餘策略照常執行。",
        }
    if "429" in text or "rate limit" in lower or "too many requests" in lower:
        return {"code": "COINGLASS_RATE_LIMIT", "title": "CoinGlass API 暫時呼叫太快", "message": "這次回測碰到 CoinGlass 的 API 速率限制，資料沒有被亂補。", "action": "稍等一會再重試；系統會依 API 限額節流。"}
    if "returned empty required datasets" in lower or "empty required datasets" in lower:
        return {"code": "COINGLASS_DATA_MISSING", "title": "CoinGlass 缺少必要資料", "message": "至少有一種衍生品資料沒有回傳。最新版會改成只影響真正依賴該來源的策略。", "action": "不需要手動補假資料；讓可用策略先完成回測。"}
    if "aligned data coverage" in lower or "min_aligned_coverage" in lower:
        return {"code": "DATA_ALIGNMENT_LOW", "title": "資料時間對齊完整度不足", "message": "Bitget K 線與某套策略需要的 CoinGlass 資料能精確對上的比例太低，因此該策略不應產生績效。", "action": "最新版會按策略分開檢查，不會讓不相關策略一起失敗。"}
    if "returned no price candles" in lower or "no price candles" in lower:
        return {"code": "BITGET_PRICE_MISSING", "title": "Bitget 沒有取得價格 K 線", "message": "Bitget 沒有回傳這個幣種在所選期間的價格資料，所以無法可靠回測。", "action": "確認幣種仍在 Bitget USDT 永續合約中。"}
    if "coinglass_api_key" in lower and ("required" in lower or "configured" in lower):
        return {"code": "COINGLASS_KEY_MISSING", "title": "CoinGlass API 尚未設定", "message": "伺服器沒有讀到 COINGLASS_API_KEY，因此不能下載真實 CoinGlass 資料。", "action": "請在 Zeabur 環境變數確認 COINGLASS_API_KEY 已設定後重新部署。"}
    if "timeout" in lower or "network" in lower or "connection" in lower:
        return {"code": "NETWORK_ERROR", "title": "資料來源連線暫時失敗", "message": "Bitget 或 CoinGlass 的網路請求沒有正常完成。系統沒有使用缺失資料繼續算績效。", "action": "稍後重新執行即可；若持續發生再檢查 Zeabur 網路與 API 狀態。"}
    return {"code": "BACKTEST_FAILED", "title": "回測沒有完成", "message": "系統在取得或處理真實資料時遇到錯誤，因此已停止，沒有產生假的回測結果。", "action": "技術記錄保留在伺服器；網頁只顯示白話診斷，不顯示 Python traceback。"}


def humanize_runtime_error(exc: Exception) -> dict[str, str]:
    return humanize_backtest_failure(str(exc))

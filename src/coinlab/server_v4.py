from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from .config import Settings
from .exchange import BitgetAPIError, BitgetV2Client
from .history_policy import humanize_backtest_failure, normalize_backtest_window, standard_policy
from .scanner import ScanConfig, next_completed_bar_time, scan_market, seconds_until_next_completed_bar


app = FastAPI(
    title="CoinGlass × Bitget Strategy Lab",
    version="0.4.0",
    description="Real-data strategy research, detailed backtests and continuous Bitget/CoinGlass market scanning.",
)

ARTIFACT_ROOT = Path(os.getenv("ARTIFACT_ROOT", "artifacts/zeabur"))
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
SCANNER_ROOT = ARTIFACT_ROOT / "scanner"
SCANNER_ROOT.mkdir(parents=True, exist_ok=True)

_backtest_lock = threading.Lock()
_backtest_job: dict[str, Any] = {
    "state": "idle", "run_id": None, "started_at": None, "finished_at": None,
    "output_dir": None, "log_file": None, "error": None,
    "message": "尚未開始回測。", "symbol": None, "timeframe": None, "window": None,
}

_scan_lock = threading.Lock()
_scan_pause_event = threading.Event()
_scan_wake_event = threading.Event()
_scan_thread: threading.Thread | None = None
_scan_runtime_cfg: ScanConfig | None = None
_scan_seen_keys: set[str] = set()
_scan_job: dict[str, Any] = {
    "state": "idle", "mode": "paused", "run_id": None, "started_at": None,
    "finished_at": None, "next_scan_at": None, "progress": None, "result": None,
    "new_signals": [], "error": None, "message": "掃描器尚未啟動。",
}


class BacktestRequest(BaseModel):
    symbol: str | None = None
    timeframe: str | None = None
    risk_per_trade: float | None = Field(default=None, gt=0, le=0.05)
    initial_equity: float | None = Field(default=None, gt=0)
    taker_fee_bps: float | None = Field(default=None, ge=0, le=100)
    slippage_bps: float | None = Field(default=None, ge=0, le=100)

    @field_validator("symbol")
    @classmethod
    def valid_symbol(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = BitgetV2Client.normalize_symbol(value)
        if not normalized.endswith("USDT") or len(normalized) <= 4:
            raise ValueError("請選擇 Bitget 的 USDT 永續合約，例如 BTCUSDT。")
        return normalized

    @field_validator("timeframe")
    @classmethod
    def valid_timeframe(cls, value: str | None) -> str | None:
        if value is None:
            return None
        allowed = {"5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"}
        if value not in allowed:
            raise ValueError(f"不支援的回測週期：{value}")
        return value


class ScanRequest(BaseModel):
    timeframe: str | None = None
    lookback_bars: int | None = Field(default=None, ge=120, le=1000)
    min_aligned_rows: int | None = Field(default=None, ge=96, le=900)
    min_turnover_usdt: float | None = Field(default=None, ge=0)
    max_spread_pct: float | None = Field(default=None, ge=0.01, le=10)
    max_symbols: int | None = Field(default=None, ge=0, le=1000)

    @field_validator("timeframe")
    @classmethod
    def valid_scan_timeframe(cls, value: str | None) -> str | None:
        allowed = {"5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"}
        if value is not None and value not in allowed:
            raise ValueError(f"不支援的掃描週期：{value}")
        return value


class LiveExecuteRequest(BaseModel):
    symbol: str
    direction: Literal["long", "short"]
    strategy_entry: float = Field(gt=0)
    strategy_stop: float = Field(gt=0)
    strategy_take_profit: float = Field(gt=0)
    risk_per_trade: float | None = Field(default=None, gt=0, le=0.05)
    leverage: int | None = Field(default=None, ge=1, le=125)
    order_type: Literal["market", "limit"] = "market"



def _detail(title: str, message: str, action: str = "") -> dict[str, str]:
    return {"title": title, "message": message, "action": action}


def _require_admin(authorization: str | None = Header(default=None)) -> None:
    expected = os.getenv("ADMIN_BEARER_TOKEN", "")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail=_detail("管理密碼尚未設定", "Zeabur 尚未設定 ADMIN_BEARER_TOKEN。", "請先新增環境變數並重新部署。"),
        )
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail=_detail("需要管理密碼", "請先輸入 ADMIN_BEARER_TOKEN。"))
    if not secrets.compare_digest(authorization[7:], expected):
        raise HTTPException(status_code=403, detail=_detail("管理密碼不正確", "輸入的管理密碼與伺服器設定不一致。"))


def _settings() -> Settings:
    return Settings()


def _bitget(*, live: bool | None = None) -> BitgetV2Client:
    s = _settings()
    return BitgetV2Client(
        api_key=s.bitget_api_key,
        api_secret=s.bitget_api_secret,
        passphrase=s.bitget_api_passphrase,
        base_url=s.bitget_rest_base_url,
        product_type=s.bitget_product_type,
        margin_coin=s.bitget_margin_coin,
        live_enabled=s.live_trading_enabled if live is None else live,
    )


def _safe_public_config() -> dict[str, Any]:
    s = _settings()
    return {
        "symbol": s.symbol,
        "timeframe": s.timeframe,
        "risk_per_trade": s.risk_per_trade,
        "initial_equity": s.initial_equity,
        "taker_fee_bps": s.taker_fee_bps,
        "slippage_bps": s.slippage_bps,
        "coinglass_exchange": s.coinglass_exchange,
        "coinglass_key_configured": bool(s.coinglass_api_key),
        "bitget_credentials_configured": bool(s.bitget_api_key and s.bitget_api_secret and s.bitget_api_passphrase),
        "live_trading_enabled": s.live_trading_enabled,
        "scan_timeframe": s.scan_timeframe,
        "scan_auto_start": s.scan_auto_start,
        "scan_min_turnover_usdt": s.scan_min_turnover_usdt,
        "scan_max_spread_pct": s.scan_max_spread_pct,
    }


def _humanize_bitget_error(exc: Exception) -> dict[str, str]:
    text = str(exc).lower()
    if any(k in text for k in ("api key", "passphrase", "credentials", "authentication")):
        return _detail(
            "Bitget 私人 API 尚未連線",
            "讀取帳戶需要 BITGET_API_KEY、BITGET_API_SECRET、BITGET_API_PASSPHRASE。",
            "請在 Zeabur 設定三個環境變數；不要把密鑰貼到聊天。",
        )
    if "live_trading_enabled=false" in text:
        return _detail("真實下單仍被鎖住", "LIVE_TRADING_ENABLED=false，因此系統拒絕送單。")
    if "not tradable" in text:
        return _detail("這個合約目前不可交易", "Bitget 回報此合約目前不是正常可交易狀態。")
    return _detail("Bitget 操作沒有完成", "Bitget 回傳錯誤，系統已停止這次操作，沒有猜測或強行送單。")


def _read_stage(log_file: str | None) -> str | None:
    if not log_file:
        return None
    path = Path(log_file)
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-120:]
    for line in reversed(lines):
        if line.startswith("COINLAB_STAGE:"):
            parts = line.split(":", 2)
            return parts[2] if len(parts) == 3 else None
    return None


def _run_backtest(run_id: str, output_dir: Path, log_file: Path, env: dict[str, str], timeframe: str) -> None:
    global _backtest_job
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                [sys.executable, "-m", "coinlab.cli", "backtest", "--out", str(output_dir)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        with _backtest_lock:
            if _backtest_job.get("run_id") != run_id:
                return
            _backtest_job["finished_at"] = datetime.now(timezone.utc).isoformat()
            if process.returncode == 0:
                _backtest_job["state"] = "completed"
                _backtest_job["error"] = None
                _backtest_job["message"] = "回測完成。每筆入場、SL、TP、出場、費用、Funding 與淨損益都已寫入報告。"
            else:
                text = log_file.read_text(encoding="utf-8", errors="replace") if log_file.exists() else ""
                friendly = humanize_backtest_failure(text, timeframe)
                _backtest_job["state"] = "failed"
                _backtest_job["error"] = friendly
                _backtest_job["message"] = friendly.get("message", "回測沒有完成。")
    except Exception as exc:
        with _backtest_lock:
            if _backtest_job.get("run_id") == run_id:
                _backtest_job["state"] = "failed"
                _backtest_job["finished_at"] = datetime.now(timezone.utc).isoformat()
                _backtest_job["error"] = _detail(
                    "回測服務沒有完成",
                    "伺服器執行回測時遇到未預期問題，這次沒有產生任何假績效。",
                    f"錯誤類型：{type(exc).__name__}。把這段白話訊息傳給我即可。",
                )


def _load_seen_keys() -> None:
    path = SCANNER_ROOT / "signal_history.ndjson"
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-5000:]
        for line in lines:
            try:
                row = json.loads(line)
                key = row.get("signal_key")
                if key:
                    _scan_seen_keys.add(str(key))
            except json.JSONDecodeError:
                continue
    except OSError:
        pass


def _persist_scan_result(result: dict[str, Any], new_signals: list[dict[str, Any]]) -> None:
    (SCANNER_ROOT / "latest_scan.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if new_signals:
        with (SCANNER_ROOT / "signal_history.ndjson").open("a", encoding="utf-8") as f:
            for row in new_signals:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _scan_config_from_settings() -> ScanConfig:
    s = _settings()
    return ScanConfig(
        timeframe=s.scan_timeframe,
        lookback_bars=s.scan_lookback_bars,
        min_aligned_rows=s.scan_min_aligned_rows,
        min_turnover_usdt=s.scan_min_turnover_usdt,
        max_spread_pct=s.scan_max_spread_pct,
        max_symbols=s.scan_max_symbols,
        coinglass_exchange=s.coinglass_exchange,
    )


def _current_scan_cfg() -> ScanConfig:
    with _scan_lock:
        cfg = _scan_runtime_cfg
    return cfg or _scan_config_from_settings()


def _scanner_loop() -> None:
    global _scan_job
    while True:
        if _scan_pause_event.is_set():
            with _scan_lock:
                _scan_job["state"] = "paused"
                _scan_job["mode"] = "paused"
                _scan_job["next_scan_at"] = None
                _scan_job["message"] = "掃描已暫停，不會再消耗 CoinGlass API。"
            _scan_wake_event.wait()
            _scan_wake_event.clear()
            continue

        s = _settings()
        if not s.coinglass_api_key:
            with _scan_lock:
                _scan_job["state"] = "failed"
                _scan_job["mode"] = "paused"
                _scan_job["error"] = _detail("CoinGlass API 尚未設定", "沒有 COINGLASS_API_KEY，無法執行自動掃幣。")
                _scan_job["message"] = "自動掃描已停止。"
            _scan_pause_event.set()
            continue

        cfg = _current_scan_cfg()
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        with _scan_lock:
            _scan_job.update({
                "state": "running", "mode": "active", "run_id": run_id,
                "started_at": datetime.now(timezone.utc).isoformat(), "finished_at": None,
                "next_scan_at": None, "progress": {"current": 0, "total": 0}, "error": None,
                "message": "正在掃描 Bitget 市場；只對通過流動性篩選的幣讀 CoinGlass。",
            })

        def progress(value: dict[str, Any]) -> None:
            with _scan_lock:
                if _scan_job.get("run_id") == run_id:
                    _scan_job["progress"] = value
                    total = value.get("total") or 0
                    current = value.get("current") or 0
                    symbol = value.get("symbol") or ""
                    _scan_job["message"] = f"正在檢查 {symbol}：{current} / {total}" if total else "正在建立候選清單。"

        try:
            result = scan_market(
                coinglass_api_key=s.coinglass_api_key,
                cfg=cfg,
                progress=progress,
                should_stop=_scan_pause_event.is_set,
            )
            if result.get("status") == "paused" or _scan_pause_event.is_set():
                with _scan_lock:
                    _scan_job["state"] = "paused"
                    _scan_job["mode"] = "paused"
                    _scan_job["finished_at"] = datetime.now(timezone.utc).isoformat()
                    _scan_job["result"] = result
                    _scan_job["message"] = "掃描已暫停；目前這輪的部分結果已保留。"
                continue

            new_signals: list[dict[str, Any]] = []
            for row in result.get("matches", []):
                key = str(row.get("signal_key") or "")
                is_new = bool(key and key not in _scan_seen_keys)
                row["is_new"] = is_new
                if is_new:
                    _scan_seen_keys.add(key)
                    new_signals.append(row)
            if len(_scan_seen_keys) > 20_000:
                _scan_seen_keys.clear()
                _load_seen_keys()

            _persist_scan_result(result, new_signals)
            delay = seconds_until_next_completed_bar(cfg.timeframe, grace_seconds=s.scan_grace_seconds)
            next_at = next_completed_bar_time(cfg.timeframe, grace_seconds=s.scan_grace_seconds)
            with _scan_lock:
                _scan_job.update({
                    "state": "waiting", "mode": "active",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "next_scan_at": next_at, "result": result, "new_signals": new_signals, "error": None,
                    "message": f"本輪完成，找到 {len(result.get('matches', []))} 個訊號，其中 {len(new_signals)} 個是新訊號。下一輪會在新 {cfg.timeframe} K 線收完後自動開始。",
                })
            _scan_wake_event.wait(timeout=delay)
            _scan_wake_event.clear()
        except Exception as exc:
            friendly = humanize_backtest_failure(str(exc), cfg.timeframe)
            if friendly.get("code") == "BACKTEST_FAILED":
                friendly = _detail(
                    "自動掃描遇到資料問題",
                    "Bitget / CoinGlass 其中一個資料來源暫時沒有完成，系統沒有用缺資料硬判斷訊號。",
                    "系統會稍後自動重試；也可以按暫停停止掃描。",
                ) | {"code": "SCAN_FAILED"}
            with _scan_lock:
                _scan_job.update({
                    "state": "failed", "mode": "active",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "error": friendly, "message": friendly.get("message", "掃描遇到問題，稍後重試。"),
                })
            if _scan_pause_event.is_set():
                continue
            _scan_wake_event.wait(timeout=max(30, s.scan_error_retry_seconds))
            _scan_wake_event.clear()


def _ensure_scanner_thread() -> None:
    global _scan_thread
    if _scan_thread and _scan_thread.is_alive():
        return
    _scan_thread = threading.Thread(target=_scanner_loop, daemon=True, name="coinlab-continuous-scanner")
    _scan_thread.start()


@app.on_event("startup")
def _startup() -> None:
    global _scan_runtime_cfg
    _load_seen_keys()
    _scan_runtime_cfg = _scan_config_from_settings()
    s = _settings()
    if s.scan_auto_start and s.coinglass_api_key:
        _scan_pause_event.clear()
    else:
        _scan_pause_event.set()
    _ensure_scanner_thread()
    _scan_wake_event.set()


DASHBOARD = r'''<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CoinGlass × Bitget Strategy Lab v0.4</title>
<style>
:root{color-scheme:dark;--bg:#080b11;--panel:#111827;--line:#263249;--muted:#94a3b8;--accent:#f59e0b;--ok:#22c55e;--bad:#ef4444;--warn:#fbbf24}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top,#162137 0,#080b11 46%);font:14px/1.5 Inter,system-ui,sans-serif;color:#e5e7eb}.wrap{max-width:1500px;margin:auto;padding:20px}.top{display:flex;gap:14px;align-items:center;justify-content:space-between;flex-wrap:wrap}.brand{font-size:23px;font-weight:850}.sub{color:var(--muted)}.pill{padding:6px 10px;border:1px solid var(--line);border-radius:999px;background:#0c111b}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px;margin-top:16px}.card{background:rgba(17,24,39,.95);border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 16px 44px #0006}.c4{grid-column:span 4}.c6{grid-column:span 6}.c12{grid-column:span 12}h2{font-size:17px;margin:0 0 12px}label{display:block;color:var(--muted);font-size:12px;margin:8px 0 5px}input,select,button{width:100%;border:1px solid var(--line);border-radius:10px;background:#0a101a;color:#e5e7eb;padding:11px}input[readonly]{opacity:.82;background:#0b111b}button{cursor:pointer;background:#1c2738;font-weight:780}button.primary{background:var(--accent);color:#111;border-color:#fbbf24}button.pause{background:#3a1720;border-color:#7f1d1d;color:#fecaca}button:disabled{opacity:.45;cursor:not-allowed}.row{display:flex;gap:9px}.row>*{flex:1}.search{position:relative}.suggest{position:absolute;z-index:20;left:0;right:0;top:68px;background:#0a101a;border:1px solid var(--line);border-radius:10px;max-height:280px;overflow:auto;display:none}.sitem{padding:9px 11px;cursor:pointer;border-bottom:1px solid #1f2937}.sitem:hover{background:#172033}.sitem small{float:right;color:var(--muted)}.hint{color:var(--muted);font-size:12px;margin-top:7px}.notice{margin:10px 0;padding:10px 12px;border:1px solid #3b4963;background:#0a111d;border-radius:10px;color:#cbd5e1}.status{margin-top:10px;border-radius:12px;padding:13px 14px;border:1px solid var(--line);background:#090e17}.status.good{border-color:#166534;background:#0d2418}.status.bad{border-color:#7f1d1d;background:#2b1014}.status.warn{border-color:#854d0e;background:#2b210d}.status-title{font-weight:820;font-size:15px;margin-bottom:4px}.status-action{color:#aab6c8;margin-top:7px;font-size:12px}.live-off{color:#fecaca;background:#3f1117;border:1px solid #7f1d1d;padding:10px;border-radius:10px}.live-on{color:#bbf7d0;background:#0f3b25;border:1px solid #166534;padding:10px;border-radius:10px}.scroll{overflow:auto;max-height:560px;border-radius:10px}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:9px 7px;border-bottom:1px solid #222d40;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}.long{color:#4ade80}.short{color:#fb7185}.positive{color:#4ade80}.negative{color:#fb7185}.kpi{font-size:25px;font-weight:850}.hidden{display:none!important}.section-title{display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap}.metric-grid{display:grid;grid-template-columns:repeat(5,minmax(130px,1fr));gap:9px;margin:10px 0}.metric{background:#0a101a;border:1px solid #202b3e;border-radius:10px;padding:10px}.metric b{display:block;font-size:18px}.metric span{color:var(--muted);font-size:11px}.tag{display:inline-block;padding:3px 7px;border-radius:999px;background:#162033;border:1px solid #2b3a53;font-size:11px}.newtag{color:#fde68a;border-color:#854d0e;background:#2b210d}@media(max-width:950px){.c4,.c6{grid-column:span 12}.row{flex-direction:column}.metric-grid{grid-template-columns:repeat(2,1fr)}}
</style></head><body><div class="wrap">
<div class="top"><div><div class="brand">CoinGlass × Bitget Strategy Lab</div><div class="sub">v0.4 · 多幣種 · 詳細真實回測 · 自動掃幣 · Bitget 執行層</div></div><div class="pill" id="health">檢查服務中…</div></div>
<div class="grid">
<section class="card c4"><h2>Bitget 合約搜尋</h2><div class="search"><label>搜尋 USDT 永續合約</label><input id="symbol" autocomplete="off" placeholder="輸入 b、btc、bnb…"><div id="suggest" class="suggest"></div></div><div class="hint">直接讀 Bitget 目前可交易合約，不寫死幣種。</div><div style="margin-top:12px"><span class="sub">目前選擇：</span> <b id="selected">—</b></div></section>
<section class="card c4"><h2>管理驗證</h2><label>ADMIN_BEARER_TOKEN</label><input id="token" type="password" placeholder="輸入你在 Zeabur 設定的管理密碼"><div class="hint">只存在此瀏覽器 session，不會寫進報告。</div></section>
<section class="card c4"><h2>交易所狀態</h2><div id="liveBox" class="live-off">讀取中…</div><div class="row" style="margin-top:10px"><button onclick="loadAccount()">讀帳戶</button><button onclick="loadPositions()">讀持倉</button></div><div id="exchangeStatus" class="status"><div class="status-title">尚未讀取</div></div></section>
<section class="card c6"><h2>單幣種真實資料回測</h2><div class="notice" id="historyHint">系統會依 CoinGlass Standard 與週期自動選最大可用區間。</div><div class="row"><div><label>自動開始 UTC</label><input id="btStart" readonly></div><div><label>自動結束 UTC</label><input id="btEnd" readonly></div></div><div class="row"><div><label>Timeframe</label><select id="btTf"><option>5m</option><option selected>15m</option><option>30m</option><option>1h</option><option>4h</option></select></div><div><label>風險 / 每單</label><input id="btRisk" type="number" min="0.001" max="0.05" step="0.001"></div></div><button id="btBtn" class="primary" style="margin-top:12px" onclick="startBacktest()">開始完整回測目前幣種</button><button id="copyReportBtn" class="hidden" style="margin-top:9px" onclick="copyReport()">複製完整回測報告給 ChatGPT</button><div id="btStatus" class="status"><div class="status-title">尚未開始</div><div>日期會自動選好，不需要手動輸入。</div></div></section>
<section class="card c6"><h2>Bitget 全市場自動掃描</h2><div class="notice">啟動後會在每根所選週期 K 線收完後自動再掃一次；不會每幾秒重複浪費 CoinGlass API。</div><div class="row"><div><label>Timeframe</label><select id="scanTf"><option>5m</option><option selected>15m</option><option>30m</option><option>1h</option><option>4h</option></select></div><div><label>最低 24h USDT 成交額</label><input id="scanTurnover" type="number"></div><div><label>最大 Spread %</label><input id="scanSpread" type="number" step="0.05"></div></div><div class="row" style="margin-top:12px"><button class="primary" onclick="startScan()">開始 / 繼續自動掃描</button><button class="pause" onclick="pauseScan()">暫停掃描</button></div><div id="scanStatus" class="status"><div class="status-title">讀取掃描器狀態中…</div></div></section>
<section id="resultCard" class="card c12 hidden"><div class="section-title"><h2>回測策略總覽</h2><span class="tag" id="resultMeta"></span></div><div id="metricGrid" class="metric-grid"></div><div class="scroll"><table><thead><tr><th>策略</th><th>交易數</th><th>勝率</th><th>PF</th><th>期望 R</th><th>淨損益 U</th><th>報酬%</th><th>Max DD%</th><th>費用 U</th><th>滑價 U</th><th>Funding U</th></tr></thead><tbody id="strategyRows"></tbody></table></div></section>
<section id="tradesCard" class="card c12 hidden"><div class="section-title"><h2>逐筆回測交易</h2><div style="min-width:220px"><select id="tradeStrategy" onchange="renderTrades()"></select></div></div><div class="hint">每單都顯示策略當時的 SL / TP、實際出場、費用與淨損益；不只看勝率。</div><div class="scroll"><table><thead><tr><th>#</th><th>策略</th><th>方向</th><th>入場時間</th><th>入場</th><th>初始 SL</th><th>出場時 SL</th><th>TP</th><th>出場</th><th>原因</th><th>數量</th><th>名目 U</th><th>毛利 U</th><th>滑價 U</th><th>手續費 U</th><th>Funding U</th><th>淨損益 U</th><th>R</th><th>MFE R</th><th>MAE R</th><th>K數</th></tr></thead><tbody id="tradeRows"></tbody></table></div></section>
<section class="card c12"><div class="section-title"><h2>目前符合策略邏輯的幣</h2><div><span class="sub">訊號：</span><span class="kpi" id="matchCount">0</span></div></div><div class="hint">掃描候選不等於下單；真實下單仍由 LIVE_TRADING_ENABLED 控制。</div><div class="scroll"><table><thead><tr><th>Symbol</th><th>策略</th><th>方向</th><th>訊號時間</th><th>參考入場</th><th>SL</th><th>SL%</th><th>TP</th><th>TP%</th><th>R</th><th>24h 成交額</th><th>Spread%</th><th>CoinGlass</th></tr></thead><tbody id="matches"></tbody></table></div></section>
</div></div>
<script>
const $=id=>document.getElementById(id);let btTimer=null,scanTimer=null,currentReport=null;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function headers(){const t=$('token').value.trim();if(t)sessionStorage.setItem('coinlab_token',t);return {'Content-Type':'application/json','Authorization':'Bearer '+t}}
$('token').value=sessionStorage.getItem('coinlab_token')||'';
class FriendlyError extends Error{constructor(info){super(info.message||info.title||'操作失敗');this.info=info}}
async function jfetch(url,opt={}){const r=await fetch(url,opt);let x;try{x=await r.json()}catch{x={detail:{title:'伺服器回應異常',message:'伺服器沒有回傳可讀資料。'}}}if(!r.ok){let d=x.detail||x;if(typeof d==='string')d={title:'操作沒有完成',message:d};throw new FriendlyError(d)}return x}
function showStatus(id,title,message,action='',kind=''){const el=$(id);el.className='status'+(kind?' '+kind:'');el.innerHTML=`<div class="status-title">${esc(title)}</div><div>${esc(message)}</div>${action?`<div class="status-action">${esc(action)}</div>`:''}`}
function showErr(id,e){const i=e.info||{title:'操作沒有完成',message:e.message||'未知錯誤'};showStatus(id,i.title||'操作沒有完成',i.message||'',i.action||'','bad')}
function fmt(n,d=6){if(n===null||n===undefined||n==='')return '—';const x=Number(n);return Number.isFinite(x)?x.toLocaleString(undefined,{maximumFractionDigits:d}):esc(n)}
function pct(n){if(n===null||n===undefined)return '—';return (Number(n)*100).toFixed(2)+'%'}
function pnlClass(n){return Number(n)>0?'positive':Number(n)<0?'negative':''}
async function boot(){try{const h=await jfetch('/health');$('health').textContent='● 服務正常 · v'+h.version}catch(e){$('health').textContent='服務異常'}try{const c=await jfetch('/config');$('symbol').value=c.symbol;$('selected').textContent=c.symbol;$('btRisk').value=c.risk_per_trade;$('scanTf').value=c.scan_timeframe;$('scanTurnover').value=c.scan_min_turnover_usdt;$('scanSpread').value=c.scan_max_spread_pct;$('liveBox').className=c.live_trading_enabled?'live-on':'live-off';$('liveBox').textContent=c.live_trading_enabled?'真實下單已解鎖':'真實下單硬鎖（LIVE_TRADING_ENABLED=false）'}catch(e){$('liveBox').textContent='讀取設定失敗'}await applyHistoryPolicy();startScanPolling();}
async function applyHistoryPolicy(){try{const tf=$('btTf').value;const p=await jfetch('/api/backtest/history-policy?timeframe='+encodeURIComponent(tf));$('btStart').value=p.earliest_safe_start||'';$('btEnd').value=p.latest_completed_bar||'';$('historyHint').textContent=p.max_history_days?`CoinGlass Standard 的 ${tf} 最多查最近 ${p.max_history_days} 天；本次會自動用滿目前可取得區間。`:`${tf} 沒有短週期 rolling 天數限制；系統仍只使用已完成 K 線。`}catch(e){$('historyHint').textContent='無法取得目前方案可用日期。'}}
$('btTf').addEventListener('change',applyHistoryPolicy);
let st=null;$('symbol').addEventListener('input',()=>{clearTimeout(st);st=setTimeout(searchSymbols,120)});$('symbol').addEventListener('focus',searchSymbols);
async function searchSymbols(){const q=$('symbol').value.trim();try{const x=await jfetch('/api/bitget/contracts?q='+encodeURIComponent(q)+'&limit=80');const box=$('suggest');box.innerHTML=x.contracts.map(c=>`<div class="sitem" data-s="${esc(c.symbol)}"><b>${esc(c.symbol)}</b><small>${esc(c.base_coin)} · max ${esc(c.max_leverage)}x</small></div>`).join('');box.style.display=x.contracts.length?'block':'none';[...box.children].forEach(el=>el.onclick=()=>{$('symbol').value=el.dataset.s;$('selected').textContent=el.dataset.s;box.style.display='none'})}catch(e){$('suggest').style.display='none'}}
document.addEventListener('click',e=>{if(!e.target.closest('.search'))$('suggest').style.display='none'});
async function startBacktest(){try{$('btBtn').disabled=true;$('copyReportBtn').classList.add('hidden');$('resultCard').classList.add('hidden');$('tradesCard').classList.add('hidden');currentReport=null;showStatus('btStatus','準備回測','正在自動確認 CoinGlass 可用的最大日期範圍。','','warn');const body={symbol:$('symbol').value,timeframe:$('btTf').value,risk_per_trade:Number($('btRisk').value)};const x=await jfetch('/backtest/start',{method:'POST',headers:headers(),body:JSON.stringify(body)});$('btStart').value=x.window.used_start;$('btEnd').value=x.window.used_end;showStatus('btStatus','回測已開始',x.message||'正在下載真實資料。','日期由系統自動決定，不會超出方案範圍。','warn');clearInterval(btTimer);btTimer=setInterval(pollBacktest,2200)}catch(e){$('btBtn').disabled=false;showErr('btStatus',e)}}
async function pollBacktest(){try{const x=await jfetch('/backtest/status',{headers:headers()});if(x.state==='running'){showStatus('btStatus','回測進行中',x.stage_message||x.message||'正在處理真實資料…','失敗時只顯示白話原因，不顯示 Python traceback。','warn')}else if(x.state==='completed'){clearInterval(btTimer);$('btBtn').disabled=false;$('copyReportBtn').classList.remove('hidden');showStatus('btStatus','回測完成',x.message,'下方已展開策略總覽與逐筆交易。','good');await loadReport()}else if(x.state==='failed'){clearInterval(btTimer);$('btBtn').disabled=false;const i=x.error||{title:'回測沒有完成',message:x.message||'回測失敗'};showStatus('btStatus',i.title||'回測沒有完成',i.message||'',i.action||'','bad')}}catch(e){clearInterval(btTimer);$('btBtn').disabled=false;showErr('btStatus',e)}}
async function loadReport(){try{currentReport=await jfetch('/backtest/report',{headers:headers()});renderReport(currentReport)}catch(e){showErr('btStatus',e)}}
function renderReport(r){$('resultCard').classList.remove('hidden');$('tradesCard').classList.remove('hidden');const m=r.metadata||{};$('resultMeta').textContent=`${m.symbol||''} · ${m.timeframe||''} · ${m.used_start||''} → ${m.used_end||''}`;const ps=r.portfolio_summary||{};$('metricGrid').innerHTML=`<div class="metric"><b>${fmt(ps.total_strategy_trades,0)}</b><span>全部策略交易筆數</span></div><div class="metric"><b class="${pnlClass(ps.sum_strategy_net_pnl)}">${fmt(ps.sum_strategy_net_pnl,2)} U</b><span>策略診斷淨損益合計</span></div><div class="metric"><b>${fmt(ps.sum_strategy_fees,2)} U</b><span>手續費合計</span></div><div class="metric"><b>${fmt(ps.sum_strategy_slippage_cost,2)} U</b><span>滑價成本合計</span></div><div class="metric"><b class="${pnlClass(ps.sum_strategy_funding_pnl)}">${fmt(ps.sum_strategy_funding_pnl,2)} U</b><span>Funding 合計</span></div>`;const ss=r.strategies||{};$('strategyRows').innerHTML=Object.entries(ss).map(([name,v])=>{const x=v.metrics||{};return `<tr><td><b>${esc(name)}</b></td><td>${fmt(x.trades,0)}</td><td>${pct(x.win_rate)}</td><td>${fmt(x.profit_factor,3)}</td><td>${fmt(x.expectancy_r,3)}</td><td class="${pnlClass(x.net_pnl)}">${fmt(x.net_pnl,2)}</td><td class="${pnlClass(x.return_pct)}">${pct(x.return_pct)}</td><td>${pct(x.max_drawdown_pct)}</td><td>${fmt(x.total_fees,2)}</td><td>${fmt(x.total_slippage_cost,2)}</td><td class="${pnlClass(x.total_funding_pnl)}">${fmt(x.total_funding_pnl,2)}</td></tr>`}).join('');const sel=$('tradeStrategy');sel.innerHTML='<option value="ALL">全部策略</option>'+Object.keys(ss).map(k=>`<option value="${esc(k)}">${esc(k)}</option>`).join('');renderTrades()}
function renderTrades(){if(!currentReport)return;const chosen=$('tradeStrategy').value;const rows=(currentReport.all_trades||[]).filter(r=>chosen==='ALL'||r.strategy===chosen);$('tradeRows').innerHTML=rows.map((r,i)=>`<tr><td>${i+1}</td><td>${esc(r.strategy)}</td><td class="${r.direction>0?'long':'short'}">${esc(r.direction_text)}</td><td>${esc(r.entry_time)}</td><td>${fmt(r.entry)}</td><td>${fmt(r.initial_stop)}</td><td>${fmt(r.stop_at_exit)}</td><td>${fmt(r.target)}</td><td>${fmt(r.exit)}</td><td>${esc(r.reason_text)}</td><td>${fmt(r.size)}</td><td>${fmt(r.entry_notional,2)}</td><td class="${pnlClass(r.gross_pnl)}">${fmt(r.gross_pnl,2)}</td><td>${fmt(r.slippage_cost,2)}</td><td>${fmt(r.fees,2)}</td><td class="${pnlClass(r.funding_pnl)}">${fmt(r.funding_pnl,2)}</td><td class="${pnlClass(r.net_pnl)}"><b>${fmt(r.net_pnl,2)}</b></td><td class="${pnlClass(r.r_multiple)}">${fmt(r.r_multiple,3)}</td><td>${fmt(r.mfe_r,2)}</td><td>${fmt(r.mae_r,2)}</td><td>${fmt(r.holding_bars,0)}</td></tr>`).join('')}
async function copyReport(){try{const x=currentReport||await jfetch('/backtest/report',{headers:headers()});await navigator.clipboard.writeText(JSON.stringify(x,null,2));showStatus('btStatus','回測完成 · 已複製','完整報告與逐筆交易已複製到剪貼簿。','直接貼給 ChatGPT 即可繼續檢討策略。','good')}catch(e){showErr('btStatus',e)}}
async function startScan(){try{const body={timeframe:$('scanTf').value,min_turnover_usdt:Number($('scanTurnover').value),max_spread_pct:Number($('scanSpread').value)};const x=await jfetch('/api/scan/start',{method:'POST',headers:headers(),body:JSON.stringify(body)});showStatus('scanStatus','自動掃描已啟動',x.message||'正在開始第一輪掃描。','之後會在每根新 K 線收完後自動再掃。','good')}catch(e){showErr('scanStatus',e)}}
async function pauseScan(){try{const x=await jfetch('/api/scan/pause',{method:'POST',headers:headers(),body:'{}'});showStatus('scanStatus','掃描已暫停',x.message||'已停止後續自動掃描。','再次按「開始 / 繼續」即可恢復。','warn')}catch(e){showErr('scanStatus',e)}}
function startScanPolling(){clearInterval(scanTimer);pollScan();scanTimer=setInterval(pollScan,3000)}
async function pollScan(){try{const x=await jfetch('/api/scan/status',{headers:headers()});if(x.state==='running'||x.state==='starting'){const p=x.progress||{};showStatus('scanStatus','自動掃描進行中',x.message||'正在檢查市場…',p.total?`進度 ${p.current||0} / ${p.total}`:'','warn')}else if(x.state==='waiting'){showStatus('scanStatus','自動掃描運作中',x.message||'本輪完成。',x.next_scan_at?`下一輪：約 ${x.next_scan_at}`:'','good')}else if(x.state==='paused'){showStatus('scanStatus','掃描已暫停',x.message||'目前不會自動掃描。','按開始 / 繼續即可恢復。','warn')}else if(x.state==='failed'){const i=x.error||{title:'掃描遇到問題',message:x.message||''};showStatus('scanStatus',i.title||'掃描遇到問題',i.message||'',i.action||'系統會稍後重試。','bad')}if(x.result)renderMatches(x.result.matches||[])}catch(e){showErr('scanStatus',e)}}
function renderMatches(rows){$('matchCount').textContent=rows.length;$('matches').innerHTML=rows.map(r=>`<tr><td><b>${esc(r.symbol)}</b>${r.is_new?' <span class="tag newtag">NEW</span>':''}</td><td>${esc(r.strategy)}</td><td class="${esc(r.direction)}">${esc(r.direction_text||r.direction)}</td><td>${esc(r.signal_time)}</td><td>${fmt(r.reference_price)}</td><td>${fmt(r.stop_loss)}</td><td>${pct(r.stop_pct)}</td><td>${fmt(r.take_profit)}</td><td>${pct(r.take_profit_pct)}</td><td>${fmt(r.reward_r,2)}</td><td>${fmt(r.volume_24h_usdt,0)}</td><td>${fmt(r.spread_pct,3)}</td><td>${esc(r.coinglass_instrument)}</td></tr>`).join('')}
async function loadAccount(){try{showStatus('exchangeStatus','讀取中','正在向 Bitget 讀取帳戶…','','warn');const x=await jfetch('/api/bitget/account',{headers:headers()});showStatus('exchangeStatus','Bitget 帳戶已連線',`權益 ${fmt(x.equity,2)} U｜可用 ${fmt(x.available,2)} U｜鎖定 ${fmt(x.locked,2)} U`,'只讀取，不會下單。','good')}catch(e){showErr('exchangeStatus',e)}}
async function loadPositions(){try{showStatus('exchangeStatus','讀取中','正在向 Bitget 讀取持倉…','','warn');const x=await jfetch('/api/bitget/positions',{headers:headers()});const rows=x.positions||[];if(!rows.length)showStatus('exchangeStatus','目前沒有持倉','Bitget USDT 永續帳戶沒有開放中的部位。','','good');else showStatus('exchangeStatus',`目前 ${rows.length} 個持倉`,rows.map(p=>`${p.symbol} ${p.direction==='long'?'多':'空'}｜數量 ${fmt(p.size)}｜均價 ${fmt(p.entry_price)}｜未實現 ${fmt(p.unrealized_pnl,2)}U`).join('；'),'只讀取，不會修改部位。','good')}catch(e){showErr('exchangeStatus',e)}}
boot();
</script></body></html>'''


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return DASHBOARD


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "utc": datetime.now(timezone.utc).isoformat(), "version": "0.4.0"}


@app.get("/config")
def config() -> dict[str, Any]:
    return _safe_public_config()


@app.get("/api/backtest/history-policy")
def backtest_history_policy(timeframe: str = Query(default="15m", max_length=8)) -> dict[str, Any]:
    try:
        return standard_policy(timeframe)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=_detail("週期不支援", str(exc))) from exc


@app.get("/api/bitget/contracts")
def bitget_contracts(q: str = Query(default="", max_length=40), limit: int = Query(default=80, ge=1, le=300)) -> dict[str, Any]:
    try:
        rows = _bitget(live=False).get_contracts()
    except BitgetAPIError as exc:
        raise HTTPException(status_code=502, detail=_humanize_bitget_error(exc)) from exc
    term = BitgetV2Client.normalize_symbol(q)
    if term:
        rows = [r for r in rows if term in r["symbol"] or term in r["base_coin"]]
        rows.sort(key=lambda r: (not r["base_coin"].startswith(term), not r["symbol"].startswith(term), r["symbol"]))
    else:
        rows.sort(key=lambda r: r["symbol"])
    return {
        "count": min(len(rows), limit),
        "contracts": [{
            "symbol": r["symbol"], "base_coin": r["base_coin"], "quote_coin": r["quote_coin"],
            "min_size": r["min_size"], "size_step": r["size_step"], "price_step": r["price_step"],
            "min_notional": r["min_notional"], "min_leverage": r["min_leverage"], "max_leverage": r["max_leverage"],
        } for r in rows[:limit]],
    }


@app.post("/backtest/start", dependencies=[Depends(_require_admin)])
def start_backtest(req: BacktestRequest) -> dict[str, Any]:
    global _backtest_job
    s = _settings()
    if not s.coinglass_api_key:
        raise HTTPException(status_code=503, detail=_detail("CoinGlass API 尚未設定", "沒有 COINGLASS_API_KEY，無法回測。"))
    symbol = req.symbol or s.symbol
    timeframe = req.timeframe or s.timeframe
    window = normalize_backtest_window(timeframe=timeframe, requested_start=None, requested_end=None)
    with _backtest_lock:
        if _backtest_job.get("state") == "running":
            raise HTTPException(status_code=409, detail=_detail("已有回測正在進行", "請等目前回測完成後再開始下一次。"))
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = ARTIFACT_ROOT / "backtests" / run_id
        log_file = output_dir / "run.log"
        env = os.environ.copy()
        env.update({
            "SYMBOL": symbol,
            "COINGLASS_SYMBOL": "AUTO",
            "COINGLASS_EXCHANGE": s.coinglass_exchange,
            "TIMEFRAME": timeframe,
            "START": window.used_start,
            "END": window.used_end,
            "INITIAL_EQUITY": str(req.initial_equity or s.initial_equity),
            "RISK_PER_TRADE": str(req.risk_per_trade or s.risk_per_trade),
            "TAKER_FEE_BPS": str(s.taker_fee_bps if req.taker_fee_bps is None else req.taker_fee_bps),
            "SLIPPAGE_BPS": str(s.slippage_bps if req.slippage_bps is None else req.slippage_bps),
            "LIVE_TRADING_ENABLED": "false",
        })
        _backtest_job = {
            "state": "running", "run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None, "output_dir": str(output_dir), "log_file": str(log_file), "error": None,
            "message": f"已自動選用 {window.used_start} ～ {window.used_end}。",
            "symbol": symbol, "timeframe": timeframe, "window": window.to_dict(),
        }
        threading.Thread(
            target=_run_backtest,
            args=(run_id, output_dir, log_file, env, timeframe),
            daemon=True,
            name=f"backtest-{run_id}",
        ).start()
    return {"accepted": True, "state": "running", "run_id": run_id, "symbol": symbol, "timeframe": timeframe, "window": window.to_dict(), "message": _backtest_job["message"]}


@app.get("/backtest/status", dependencies=[Depends(_require_admin)])
def backtest_status() -> dict[str, Any]:
    with _backtest_lock:
        raw = dict(_backtest_job)
    stage = _read_stage(raw.get("log_file"))
    report_path = Path(raw["output_dir"]) / "BACKTEST_REPORT.json" if raw.get("output_dir") else None
    return {
        "state": raw.get("state"), "run_id": raw.get("run_id"), "started_at": raw.get("started_at"),
        "finished_at": raw.get("finished_at"), "symbol": raw.get("symbol"), "timeframe": raw.get("timeframe"),
        "window": raw.get("window"), "message": raw.get("message"), "stage_message": stage,
        "error": raw.get("error"), "report_ready": bool(report_path and report_path.exists()),
    }


@app.get("/backtest/report", dependencies=[Depends(_require_admin)])
def backtest_report() -> dict[str, Any]:
    with _backtest_lock:
        output_dir = _backtest_job.get("output_dir")
    if not output_dir:
        raise HTTPException(status_code=404, detail=_detail("還沒有回測報告", "目前尚未執行回測。"))
    path = Path(output_dir) / "BACKTEST_REPORT.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=_detail("報告尚未完成", "這次回測還在處理或已停止。"))
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/api/scan/start", dependencies=[Depends(_require_admin)])
def start_scan(req: ScanRequest) -> dict[str, Any]:
    global _scan_runtime_cfg
    s = _settings()
    if not s.coinglass_api_key:
        raise HTTPException(status_code=503, detail=_detail("CoinGlass API 尚未設定", "沒有 COINGLASS_API_KEY，無法自動掃幣。"))
    current = _current_scan_cfg()
    _scan_runtime_cfg = ScanConfig(
        timeframe=req.timeframe or current.timeframe,
        lookback_bars=req.lookback_bars or current.lookback_bars,
        min_aligned_rows=req.min_aligned_rows or current.min_aligned_rows,
        min_turnover_usdt=current.min_turnover_usdt if req.min_turnover_usdt is None else req.min_turnover_usdt,
        max_spread_pct=current.max_spread_pct if req.max_spread_pct is None else req.max_spread_pct,
        max_symbols=current.max_symbols if req.max_symbols is None else req.max_symbols,
        coinglass_exchange=s.coinglass_exchange,
    )
    _scan_pause_event.clear()
    _ensure_scanner_thread()
    _scan_wake_event.set()
    with _scan_lock:
        _scan_job["mode"] = "active"
        _scan_job["state"] = "starting"
        _scan_job["message"] = "自動掃描已啟動，正在開始新一輪。"
    return {"accepted": True, "state": "starting", "message": "掃描已開始；完成後會等下一根 K 線收完再自動掃一次。"}


@app.post("/api/scan/pause", dependencies=[Depends(_require_admin)])
def pause_scan() -> dict[str, Any]:
    _scan_pause_event.set()
    _scan_wake_event.set()
    with _scan_lock:
        _scan_job["mode"] = "paused"
        _scan_job["message"] = "已收到暫停指令；若正在檢查某個幣，會在這個幣完成後停止。"
    return {"accepted": True, "state": "paused", "message": "掃描已要求暫停，不會啟動下一輪。"}


@app.get("/api/scan/status", dependencies=[Depends(_require_admin)])
def scan_status() -> dict[str, Any]:
    with _scan_lock:
        raw = dict(_scan_job)
    return {
        "state": raw.get("state"), "mode": raw.get("mode"), "run_id": raw.get("run_id"),
        "started_at": raw.get("started_at"), "finished_at": raw.get("finished_at"),
        "next_scan_at": raw.get("next_scan_at"), "progress": raw.get("progress"),
        "result": raw.get("result"), "new_signals": raw.get("new_signals"),
        "message": raw.get("message"), "error": raw.get("error"),
    }


@app.get("/api/bitget/account", dependencies=[Depends(_require_admin)])
def bitget_account() -> dict[str, Any]:
    try:
        return _bitget().get_account()
    except BitgetAPIError as exc:
        raise HTTPException(status_code=502, detail=_humanize_bitget_error(exc)) from exc


@app.get("/api/bitget/positions", dependencies=[Depends(_require_admin)])
def bitget_positions() -> dict[str, Any]:
    try:
        return {"positions": _bitget().get_positions()}
    except BitgetAPIError as exc:
        raise HTTPException(status_code=502, detail=_humanize_bitget_error(exc)) from exc


@app.get("/api/bitget/orders", dependencies=[Depends(_require_admin)])
def bitget_orders() -> dict[str, Any]:
    try:
        client = _bitget()
        return {"orders": client.get_open_orders(), "protection_orders": client.get_plan_orders()}
    except BitgetAPIError as exc:
        raise HTTPException(status_code=502, detail=_humanize_bitget_error(exc)) from exc


@app.post("/api/live/execute", dependencies=[Depends(_require_admin)])
def live_execute(req: LiveExecuteRequest) -> dict[str, Any]:
    s = _settings()
    if not s.live_trading_enabled:
        raise HTTPException(status_code=423, detail=_detail("真實下單仍被鎖住", "LIVE_TRADING_ENABLED=false，不會送出任何真實訂單。"))
    try:
        return _bitget().execute_strategy_order(
            symbol=req.symbol,
            direction=req.direction,
            strategy_entry=req.strategy_entry,
            strategy_stop=req.strategy_stop,
            strategy_take_profit=req.strategy_take_profit,
            risk_per_trade=req.risk_per_trade or s.risk_per_trade,
            leverage=req.leverage or s.bitget_leverage,
            margin_mode=s.bitget_margin_mode,
            position_mode=s.bitget_position_mode,
            order_type=req.order_type,
            max_position_notional_equity_multiple=s.max_position_notional_equity_multiple,
            max_portfolio_notional_equity_multiple=s.max_portfolio_notional_equity_multiple,
            available_margin_utilization_pct=s.available_margin_utilization_pct,
        )
    except BitgetAPIError as exc:
        raise HTTPException(status_code=409, detail=_humanize_bitget_error(exc)) from exc


@app.post("/api/live/close/{symbol}", dependencies=[Depends(_require_admin)])
def live_close(symbol: str) -> dict[str, Any]:
    s = _settings()
    if not s.live_trading_enabled:
        raise HTTPException(status_code=423, detail=_detail("真實平倉仍被鎖住", "LIVE_TRADING_ENABLED=false，不會修改持倉。"))
    try:
        return _bitget().close_position_market(symbol)
    except BitgetAPIError as exc:
        raise HTTPException(status_code=409, detail=_humanize_bitget_error(exc)) from exc

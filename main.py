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

from coinlab.config import Settings
from coinlab.exchange import BitgetAPIError, BitgetV2Client
from coinlab.scanner import ScanConfig, scan_market


app = FastAPI(
    title="CoinGlass × Bitget Strategy Lab",
    version="0.2.0",
    description=(
        "Multi-symbol CoinGlass research, Bitget market scanner and a strictly separated "
        "Bitget execution layer. Strategy and stop/take-profit decisions remain in coinlab.strategies/backtest."
    ),
)

ARTIFACT_ROOT = Path(os.getenv("ARTIFACT_ROOT", "artifacts/zeabur"))
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

_backtest_lock = threading.Lock()
_backtest_job: dict[str, Any] = {
    "state": "idle", "run_id": None, "started_at": None, "finished_at": None,
    "return_code": None, "output_dir": None, "log_file": None, "error": None,
}
_scan_lock = threading.Lock()
_scan_job: dict[str, Any] = {
    "state": "idle", "run_id": None, "started_at": None, "finished_at": None,
    "progress": None, "result": None, "error": None,
}


class BacktestRequest(BaseModel):
    symbol: str | None = None
    start: str | None = None
    end: str | None = None
    timeframe: str | None = None
    coinglass_exchange: str | None = None
    initial_equity: float | None = Field(default=None, gt=0)
    risk_per_trade: float | None = Field(default=None, gt=0, le=0.05)
    taker_fee_bps: float | None = Field(default=None, ge=0, le=100)
    slippage_bps: float | None = Field(default=None, ge=0, le=100)
    min_aligned_coverage: float | None = Field(default=None, ge=0.5, le=1.0)

    @field_validator("symbol")
    @classmethod
    def valid_symbol(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = BitgetV2Client.normalize_symbol(value)
        if not normalized.endswith("USDT") or len(normalized) <= 4:
            raise ValueError("symbol must be a Bitget USDT futures symbol, e.g. BTCUSDT")
        return normalized

    @field_validator("timeframe")
    @classmethod
    def valid_timeframe(cls, value: str | None) -> str | None:
        if value is None:
            return value
        allowed = {"1m", "3m", "5m", "15m", "30m", "1h", "4h", "6h", "8h", "12h", "1d", "1w"}
        if value not in allowed:
            raise ValueError(f"timeframe must be one of {sorted(allowed)}")
        return value

    @field_validator("start", "end")
    @classmethod
    def valid_iso_time(cls, value: str | None) -> str | None:
        if value is None:
            return value
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("must be ISO-8601, e.g. 2025-01-01T00:00:00Z") from exc
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
        allowed = {"1m", "3m", "5m", "15m", "30m", "1h", "4h", "6h", "8h", "12h", "1d", "1w"}
        if value is not None and value not in allowed:
            raise ValueError(f"timeframe must be one of {sorted(allowed)}")
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

    @field_validator("symbol")
    @classmethod
    def normalize_live_symbol(cls, value: str) -> str:
        normalized = BitgetV2Client.normalize_symbol(value)
        if not normalized.endswith("USDT"):
            raise ValueError("symbol must end with USDT")
        return normalized


def _require_admin(authorization: str | None = Header(default=None)) -> None:
    expected = os.getenv("ADMIN_BEARER_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=503, detail="ADMIN_BEARER_TOKEN is not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required")
    if not secrets.compare_digest(authorization[7:], expected):
        raise HTTPException(status_code=403, detail="Invalid bearer token")


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
        "coinglass_symbol": s.coinglass_symbol,
        "coinglass_exchange": s.coinglass_exchange,
        "timeframe": s.timeframe,
        "start": s.start,
        "end": s.end,
        "live_trading_enabled": s.live_trading_enabled,
        "coinglass_key_configured": bool(s.coinglass_api_key),
        "bitget_credentials_configured": bool(s.bitget_api_key and s.bitget_api_secret and s.bitget_api_passphrase),
        "admin_token_configured": bool(os.getenv("ADMIN_BEARER_TOKEN")),
        "bitget_product_type": s.bitget_product_type,
        "bitget_margin_mode": s.bitget_margin_mode,
        "bitget_position_mode": s.bitget_position_mode,
        "bitget_leverage": s.bitget_leverage,
        "risk_per_trade": s.risk_per_trade,
        "scan_timeframe": s.scan_timeframe,
    }


DASHBOARD = r'''<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CoinGlass × Bitget Strategy Lab</title>
<style>
:root{color-scheme:dark;--bg:#090b10;--panel:#121722;--line:#263044;--muted:#94a3b8;--accent:#f59e0b;--ok:#22c55e;--bad:#ef4444;--blue:#38bdf8}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top,#172033 0,#090b10 42%);font:14px/1.45 Inter,system-ui,sans-serif;color:#e5e7eb}.wrap{max-width:1380px;margin:auto;padding:22px}.top{display:flex;gap:16px;align-items:center;justify-content:space-between;flex-wrap:wrap}.brand{font-size:23px;font-weight:800}.sub{color:var(--muted)}.pill{padding:6px 10px;border:1px solid var(--line);border-radius:999px;background:#0d111a}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px;margin-top:16px}.card{background:rgba(18,23,34,.94);border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 14px 40px #0005}.c4{grid-column:span 4}.c6{grid-column:span 6}.c8{grid-column:span 8}.c12{grid-column:span 12}h2{font-size:16px;margin:0 0 12px}label{display:block;color:var(--muted);font-size:12px;margin:8px 0 5px}input,select,button{width:100%;border:1px solid var(--line);border-radius:10px;background:#0b1019;color:#e5e7eb;padding:10px 11px}button{cursor:pointer;background:#1b2434;font-weight:700}button.primary{background:var(--accent);color:#111;border-color:#fbbf24}button:disabled{opacity:.45;cursor:not-allowed}.row{display:flex;gap:9px}.row>*{flex:1}.search{position:relative}.suggest{position:absolute;z-index:20;left:0;right:0;top:68px;background:#0b1019;border:1px solid var(--line);border-radius:10px;max-height:280px;overflow:auto;display:none}.sitem{padding:9px 11px;cursor:pointer;border-bottom:1px solid #1b2434}.sitem:hover{background:#172033}.sitem small{float:right;color:var(--muted)}.status{white-space:pre-wrap;background:#090d14;border-radius:10px;padding:10px;min-height:54px;color:#cbd5e1;overflow:auto}.live-off{color:#fecaca;background:#3f1117;border:1px solid #7f1d1d;padding:10px;border-radius:10px}.live-on{color:#bbf7d0;background:#0f3b25;border:1px solid #166534;padding:10px;border-radius:10px}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:9px 7px;border-bottom:1px solid #222b3b;text-align:right;white-space:nowrap}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}.long{color:#4ade80}.short{color:#fb7185}.scroll{overflow:auto;max-height:520px}.kpi{font-size:25px;font-weight:800}.hint{color:var(--muted);font-size:12px;margin-top:7px}@media(max-width:900px){.c4,.c6,.c8{grid-column:span 12}}
</style></head><body><div class="wrap">
<div class="top"><div><div class="brand">CoinGlass × Bitget Strategy Lab</div><div class="sub">多幣種研究 · 全市場掃描 · Bitget 執行層與策略層嚴格分離</div></div><div class="pill" id="health">checking…</div></div>
<div class="grid">
<section class="card c4"><h2>Bitget 合約搜尋</h2><div class="search"><label>搜尋 USDT 永續合約</label><input id="symbol" value="ETHUSDT" autocomplete="off" placeholder="輸入 b、btc、bnb…"><div id="suggest" class="suggest"></div></div><div class="hint">清單直接來自 Bitget，不寫死幣種。</div><div style="margin-top:12px"><span class="sub">目前選擇：</span> <b id="selected">ETHUSDT</b></div></section>
<section class="card c4"><h2>管理驗證</h2><label>ADMIN_BEARER_TOKEN</label><input id="token" type="password" placeholder="只存在此瀏覽器 session"><div class="hint">Token 不會顯示在頁面或回測報告。</div></section>
<section class="card c4"><h2>交易所狀態</h2><div id="liveBox" class="live-off">讀取中…</div><div class="row" style="margin-top:10px"><button onclick="loadAccount()">讀帳戶</button><button onclick="loadPositions()">讀持倉</button></div><div id="exchangeStatus" class="status" style="margin-top:10px">尚未讀取</div></section>

<section class="card c6"><h2>單幣種真實資料回測</h2><div class="row"><div><label>開始 UTC</label><input id="btStart" value="2025-01-01T00:00:00Z"></div><div><label>結束 UTC</label><input id="btEnd" value="2026-01-01T00:00:00Z"></div></div><div class="row"><div><label>Timeframe</label><select id="btTf"><option>5m</option><option selected>15m</option><option>30m</option><option>1h</option><option>4h</option></select></div><div><label>風險 / 每單</label><input id="btRisk" type="number" step="0.001" value="0.01"></div></div><button class="primary" style="margin-top:12px" onclick="startBacktest()">開始回測目前幣種</button><div id="btStatus" class="status" style="margin-top:10px">idle</div></section>

<section class="card c6"><h2>全市場策略掃描</h2><div class="row"><div><label>Timeframe</label><select id="scanTf"><option>5m</option><option selected>15m</option><option>30m</option><option>1h</option><option>4h</option></select></div><div><label>最低 24h USDT 成交額</label><input id="scanTurnover" type="number" value="1000000"></div><div><label>最大 Spread %</label><input id="scanSpread" type="number" step="0.05" value="0.50"></div></div><button class="primary" style="margin-top:12px" onclick="startScan()">掃描 Bitget 全部可交易幣</button><div id="scanStatus" class="status" style="margin-top:10px">idle</div></section>

<section class="card c12"><div class="top"><h2>符合目前 CoinGlass 策略邏輯的幣</h2><div><span class="sub">訊號：</span><span class="kpi" id="matchCount">0</span></div></div><div class="scroll"><table><thead><tr><th>Symbol</th><th>Strategy</th><th>Direction</th><th>Signal time</th><th>Ref</th><th>SL</th><th>TP</th><th>R</th><th>24h vol</th><th>spread%</th><th>CG instrument</th></tr></thead><tbody id="matches"></tbody></table></div></section>
</div></div>
<script>
const $=id=>document.getElementById(id); let scanTimer=null,btTimer=null;
function headers(){const t=$('token').value.trim(); if(t) sessionStorage.setItem('coinlab_token',t); return {'Content-Type':'application/json','Authorization':'Bearer '+t}}
$('token').value=sessionStorage.getItem('coinlab_token')||'';
async function jfetch(url,opt={}){const r=await fetch(url,opt);let x;try{x=await r.json()}catch{x={detail:await r.text()}}if(!r.ok)throw new Error(x.detail||JSON.stringify(x));return x}
async function boot(){try{const h=await jfetch('/health');$('health').textContent='● service '+(h.ok?'online':'offline')}catch(e){$('health').textContent='service error'}try{const c=await jfetch('/config');$('symbol').value=c.symbol;$('selected').textContent=c.symbol;$('liveBox').className=c.live_trading_enabled?'live-on':'live-off';$('liveBox').textContent=c.live_trading_enabled?'LIVE_TRADING_ENABLED=true — 真實下單已解鎖':'LIVE_TRADING_ENABLED=false — 真實下單硬鎖';}catch(e){$('liveBox').textContent=e.message}}
let st=null;$('symbol').addEventListener('input',()=>{clearTimeout(st);st=setTimeout(searchSymbols,120)});$('symbol').addEventListener('focus',searchSymbols);
async function searchSymbols(){const q=$('symbol').value.trim();try{const x=await jfetch('/api/bitget/contracts?q='+encodeURIComponent(q)+'&limit=80');const box=$('suggest');box.innerHTML=x.contracts.map(c=>`<div class="sitem" data-s="${c.symbol}"><b>${c.symbol}</b><small>${c.base_coin} · max ${c.max_leverage}x</small></div>`).join('');box.style.display=x.contracts.length?'block':'none';[...box.children].forEach(el=>el.onclick=()=>{$('symbol').value=el.dataset.s;$('selected').textContent=el.dataset.s;box.style.display='none'})}catch(e){$('suggest').style.display='none'}}
document.addEventListener('click',e=>{if(!e.target.closest('.search'))$('suggest').style.display='none'});
async function startBacktest(){try{const body={symbol:$('symbol').value,start:$('btStart').value,end:$('btEnd').value,timeframe:$('btTf').value,risk_per_trade:Number($('btRisk').value)};const x=await jfetch('/backtest/start',{method:'POST',headers:headers(),body:JSON.stringify(body)});$('btStatus').textContent=JSON.stringify(x,null,2);clearInterval(btTimer);btTimer=setInterval(pollBacktest,3000)}catch(e){$('btStatus').textContent=e.message}}
async function pollBacktest(){try{const x=await jfetch('/backtest/status',{headers:headers()});$('btStatus').textContent=JSON.stringify({state:x.state,run_id:x.run_id,report_ready:x.report_ready,error:x.error,log_tail:(x.log_tail||[]).slice(-8)},null,2);if(x.state!=='running')clearInterval(btTimer)}catch(e){$('btStatus').textContent=e.message;clearInterval(btTimer)}}
async function startScan(){try{const body={timeframe:$('scanTf').value,min_turnover_usdt:Number($('scanTurnover').value),max_spread_pct:Number($('scanSpread').value)};const x=await jfetch('/api/scan/start',{method:'POST',headers:headers(),body:JSON.stringify(body)});$('scanStatus').textContent=JSON.stringify(x,null,2);clearInterval(scanTimer);scanTimer=setInterval(pollScan,2500)}catch(e){$('scanStatus').textContent=e.message}}
async function pollScan(){try{const x=await jfetch('/api/scan/status',{headers:headers()});$('scanStatus').textContent=JSON.stringify({state:x.state,progress:x.progress,error:x.error,stats:x.result&&x.result.stats},null,2);if(x.result)renderMatches(x.result.matches||[]);if(x.state!=='running')clearInterval(scanTimer)}catch(e){$('scanStatus').textContent=e.message;clearInterval(scanTimer)}}
function fmt(n){if(n===null||n===undefined)return '';const x=Number(n);return Number.isFinite(x)?x.toLocaleString(undefined,{maximumSignificantDigits:8}):n}
function renderMatches(rows){$('matchCount').textContent=rows.length;$('matches').innerHTML=rows.map(r=>`<tr><td><b>${r.symbol}</b></td><td>${r.strategy}</td><td class="${r.direction}">${r.direction}</td><td>${r.signal_time}</td><td>${fmt(r.reference_price)}</td><td>${fmt(r.stop_loss)}</td><td>${fmt(r.take_profit)}</td><td>${fmt(r.reward_r)}</td><td>${fmt(r.volume_24h_usdt)}</td><td>${fmt(r.spread_pct)}</td><td>${r.coinglass_instrument}</td></tr>`).join('')}
async function loadAccount(){try{const x=await jfetch('/api/bitget/account',{headers:headers()});$('exchangeStatus').textContent=JSON.stringify(x,null,2)}catch(e){$('exchangeStatus').textContent=e.message}}
async function loadPositions(){try{const x=await jfetch('/api/bitget/positions',{headers:headers()});$('exchangeStatus').textContent=JSON.stringify(x,null,2)}catch(e){$('exchangeStatus').textContent=e.message}}
boot();
</script></body></html>'''


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return DASHBOARD


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "utc": datetime.now(timezone.utc).isoformat(), "version": "0.2.0"}


@app.get("/config")
def config() -> dict[str, Any]:
    return _safe_public_config()


@app.get("/strategies")
def strategies() -> dict[str, Any]:
    from coinlab.strategies import STRATEGIES
    return {"strategies": sorted(STRATEGIES.keys()), "count": len(STRATEGIES)}


@app.get("/api/bitget/contracts")
def bitget_contracts(q: str = Query(default="", max_length=40), limit: int = Query(default=80, ge=1, le=300)) -> dict[str, Any]:
    try:
        rows = _bitget(live=False).get_contracts()
    except BitgetAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    term = BitgetV2Client.normalize_symbol(q)
    if term:
        rows = [r for r in rows if term in r["symbol"] or term in r["base_coin"]]
        rows.sort(key=lambda r: (not r["base_coin"].startswith(term), not r["symbol"].startswith(term), r["symbol"]))
    else:
        rows.sort(key=lambda r: r["symbol"])
    public = [{
        "symbol": r["symbol"], "base_coin": r["base_coin"], "quote_coin": r["quote_coin"],
        "min_size": r["min_size"], "size_step": r["size_step"], "price_step": r["price_step"],
        "min_notional": r["min_notional"], "min_leverage": r["min_leverage"], "max_leverage": r["max_leverage"],
    } for r in rows[:limit]]
    return {"count": len(public), "contracts": public}


def _run_backtest(run_id: str, output_dir: Path, log_file: Path, env: dict[str, str]) -> None:
    global _backtest_job
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                [sys.executable, "-m", "coinlab.cli", "backtest", "--out", str(output_dir)],
                env=env, stdout=log, stderr=subprocess.STDOUT, text=True, check=False,
            )
        with _backtest_lock:
            _backtest_job["state"] = "completed" if process.returncode == 0 else "failed"
            _backtest_job["return_code"] = process.returncode
            _backtest_job["finished_at"] = datetime.now(timezone.utc).isoformat()
            if process.returncode != 0:
                _backtest_job["error"] = "Backtest failed; inspect log_tail"
    except Exception as exc:
        with _backtest_lock:
            _backtest_job["state"] = "failed"
            _backtest_job["finished_at"] = datetime.now(timezone.utc).isoformat()
            _backtest_job["error"] = f"{type(exc).__name__}: {exc}"


@app.post("/backtest/start", dependencies=[Depends(_require_admin)])
def start_backtest(req: BacktestRequest) -> dict[str, Any]:
    global _backtest_job
    with _backtest_lock:
        if _backtest_job["state"] == "running":
            raise HTTPException(status_code=409, detail="A backtest is already running")
        if not os.getenv("COINGLASS_API_KEY"):
            raise HTTPException(status_code=503, detail="COINGLASS_API_KEY is not configured")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = ARTIFACT_ROOT / "backtests" / run_id
        log_file = output_dir / "run.log"
        env = os.environ.copy()
        overrides = {
            "SYMBOL": req.symbol, "START": req.start, "END": req.end, "TIMEFRAME": req.timeframe,
            "COINGLASS_EXCHANGE": req.coinglass_exchange, "INITIAL_EQUITY": req.initial_equity,
            "RISK_PER_TRADE": req.risk_per_trade, "TAKER_FEE_BPS": req.taker_fee_bps,
            "SLIPPAGE_BPS": req.slippage_bps, "MIN_ALIGNED_COVERAGE": req.min_aligned_coverage,
        }
        for key, value in overrides.items():
            if value is not None:
                env[key] = str(value)
        if req.symbol is not None:
            env["COINGLASS_SYMBOL"] = "AUTO"
        env["LIVE_TRADING_ENABLED"] = "false"
        _backtest_job = {
            "state": "running", "run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None, "return_code": None, "output_dir": str(output_dir),
            "log_file": str(log_file), "error": None,
        }
        threading.Thread(target=_run_backtest, args=(run_id, output_dir, log_file, env), daemon=True, name=f"backtest-{run_id}").start()
        return {"accepted": True, "run_id": run_id, "state": "running", "symbol": env.get("SYMBOL")}


@app.get("/backtest/status", dependencies=[Depends(_require_admin)])
def backtest_status() -> dict[str, Any]:
    with _backtest_lock:
        payload = dict(_backtest_job)
    log_path = Path(payload["log_file"]) if payload.get("log_file") else None
    if log_path and log_path.exists():
        payload["log_tail"] = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
    else:
        payload["log_tail"] = []
    report_path = Path(payload["output_dir"]) / "BACKTEST_REPORT.json" if payload.get("output_dir") else None
    payload["report_ready"] = bool(report_path and report_path.exists())
    return payload


@app.get("/backtest/report", dependencies=[Depends(_require_admin)])
def backtest_report() -> dict[str, Any]:
    with _backtest_lock:
        output_dir = _backtest_job.get("output_dir")
    if not output_dir:
        raise HTTPException(status_code=404, detail="No backtest has been run")
    report_path = Path(output_dir) / "BACKTEST_REPORT.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Backtest report is not ready")
    return json.loads(report_path.read_text(encoding="utf-8"))


def _run_scan(run_id: str, cfg: ScanConfig, api_key: str) -> None:
    global _scan_job
    try:
        def progress(value: dict[str, Any]) -> None:
            with _scan_lock:
                if _scan_job.get("run_id") == run_id:
                    _scan_job["progress"] = value
        result = scan_market(coinglass_api_key=api_key, cfg=cfg, progress=progress)
        with _scan_lock:
            _scan_job["state"] = "completed"
            _scan_job["finished_at"] = datetime.now(timezone.utc).isoformat()
            _scan_job["result"] = result
    except Exception as exc:
        with _scan_lock:
            _scan_job["state"] = "failed"
            _scan_job["finished_at"] = datetime.now(timezone.utc).isoformat()
            _scan_job["error"] = f"{type(exc).__name__}: {exc}"


@app.post("/api/scan/start", dependencies=[Depends(_require_admin)])
def start_scan(req: ScanRequest) -> dict[str, Any]:
    global _scan_job
    s = _settings()
    if not s.coinglass_api_key:
        raise HTTPException(status_code=503, detail="COINGLASS_API_KEY is not configured")
    cfg = ScanConfig(
        timeframe=req.timeframe or s.scan_timeframe,
        lookback_bars=req.lookback_bars or s.scan_lookback_bars,
        min_aligned_rows=req.min_aligned_rows or s.scan_min_aligned_rows,
        min_turnover_usdt=s.scan_min_turnover_usdt if req.min_turnover_usdt is None else req.min_turnover_usdt,
        max_spread_pct=s.scan_max_spread_pct if req.max_spread_pct is None else req.max_spread_pct,
        max_symbols=s.scan_max_symbols if req.max_symbols is None else req.max_symbols,
        coinglass_exchange=s.coinglass_exchange,
    )
    with _scan_lock:
        if _scan_job["state"] == "running":
            raise HTTPException(status_code=409, detail="A market scan is already running")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        _scan_job = {
            "state": "running", "run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None, "progress": {"current": 0, "total": 0}, "result": None, "error": None,
        }
        threading.Thread(target=_run_scan, args=(run_id, cfg, s.coinglass_api_key), daemon=True, name=f"scan-{run_id}").start()
    return {"accepted": True, "run_id": run_id, "state": "running", "config": cfg.__dict__}


@app.get("/api/scan/status", dependencies=[Depends(_require_admin)])
def scan_status() -> dict[str, Any]:
    with _scan_lock:
        return dict(_scan_job)


@app.get("/api/bitget/account", dependencies=[Depends(_require_admin)])
def bitget_account() -> dict[str, Any]:
    try:
        return _bitget().get_account()
    except BitgetAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/bitget/positions", dependencies=[Depends(_require_admin)])
def bitget_positions() -> dict[str, Any]:
    try:
        return {"positions": _bitget().get_positions()}
    except BitgetAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/bitget/orders", dependencies=[Depends(_require_admin)])
def bitget_orders() -> dict[str, Any]:
    try:
        client = _bitget()
        return {"orders": client.get_open_orders(), "protection_orders": client.get_plan_orders()}
    except BitgetAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/live/execute", dependencies=[Depends(_require_admin)])
def live_execute(req: LiveExecuteRequest) -> dict[str, Any]:
    """Submit EXACT strategy entry/SL/TP. The exchange layer may size/round quantity only."""
    s = _settings()
    if not s.live_trading_enabled:
        raise HTTPException(status_code=423, detail="LIVE_TRADING_ENABLED=false; live trading is locked")
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
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/live/close/{symbol}", dependencies=[Depends(_require_admin)])
def live_close(symbol: str) -> dict[str, Any]:
    s = _settings()
    if not s.live_trading_enabled:
        raise HTTPException(status_code=423, detail="LIVE_TRADING_ENABLED=false; live trading is locked")
    try:
        return _bitget().close_position_market(symbol)
    except BitgetAPIError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))

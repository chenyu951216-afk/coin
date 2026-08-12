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
from coinlab.history_policy import (
    humanize_backtest_failure,
    normalize_backtest_window,
    standard_policy,
)
from coinlab.scanner import ScanConfig, scan_market


app = FastAPI(
    title="CoinGlass × Bitget Strategy Lab",
    version="0.3.0",
    description=(
        "Multi-symbol CoinGlass research, Bitget market scanner and a strictly separated "
        "Bitget execution layer. Strategy and stop/take-profit decisions remain unchanged."
    ),
)

ARTIFACT_ROOT = Path(os.getenv("ARTIFACT_ROOT", "artifacts/zeabur"))
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

_backtest_lock = threading.Lock()
_backtest_job: dict[str, Any] = {
    "state": "idle",
    "run_id": None,
    "started_at": None,
    "finished_at": None,
    "return_code": None,
    "output_dir": None,
    "log_file": None,
    "error": None,
    "message": "尚未開始回測。",
    "symbol": None,
    "timeframe": None,
    "window": None,
}
_scan_lock = threading.Lock()
_scan_job: dict[str, Any] = {
    "state": "idle",
    "run_id": None,
    "started_at": None,
    "finished_at": None,
    "progress": None,
    "result": None,
    "error": None,
    "message": "尚未開始掃描。",
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
            raise ValueError("請選擇 Bitget 的 USDT 永續合約，例如 BTCUSDT。")
        return normalized

    @field_validator("timeframe")
    @classmethod
    def valid_timeframe(cls, value: str | None) -> str | None:
        if value is None:
            return value
        allowed = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "1w"}
        if value not in allowed:
            raise ValueError(f"不支援的週期：{value}")
        return value

    @field_validator("start", "end")
    @classmethod
    def valid_iso_time(cls, value: str | None) -> str | None:
        if value is None:
            return value
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("日期格式不正確，請使用 YYYY-MM-DDTHH:MM:SSZ。") from exc
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
        allowed = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "1w"}
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

    @field_validator("symbol")
    @classmethod
    def normalize_live_symbol(cls, value: str) -> str:
        normalized = BitgetV2Client.normalize_symbol(value)
        if not normalized.endswith("USDT"):
            raise ValueError("幣種必須是 USDT 永續合約。")
        return normalized


def _detail(title: str, message: str, action: str = "") -> dict[str, str]:
    return {"title": title, "message": message, "action": action}


def _require_admin(authorization: str | None = Header(default=None)) -> None:
    expected = os.getenv("ADMIN_BEARER_TOKEN", "")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail=_detail(
                "管理密碼尚未設定",
                "Zeabur 沒有設定 ADMIN_BEARER_TOKEN，所以不能從網頁啟動受保護操作。",
                "請先在 Zeabur 環境變數新增 ADMIN_BEARER_TOKEN，重新部署後再試。",
            ),
        )
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail=_detail("需要管理密碼", "請先在頁面的管理驗證欄位輸入 ADMIN_BEARER_TOKEN。"),
        )
    if not secrets.compare_digest(authorization[7:], expected):
        raise HTTPException(
            status_code=403,
            detail=_detail("管理密碼不正確", "輸入的 ADMIN_BEARER_TOKEN 與伺服器設定不一致。"),
        )


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
        "live_trading_enabled": s.live_trading_enabled,
        "coinglass_key_configured": bool(s.coinglass_api_key),
        "bitget_credentials_configured": bool(
            s.bitget_api_key and s.bitget_api_secret and s.bitget_api_passphrase
        ),
        "admin_token_configured": bool(os.getenv("ADMIN_BEARER_TOKEN")),
        "bitget_product_type": s.bitget_product_type,
        "bitget_margin_mode": s.bitget_margin_mode,
        "bitget_position_mode": s.bitget_position_mode,
        "bitget_leverage": s.bitget_leverage,
        "risk_per_trade": s.risk_per_trade,
        "scan_timeframe": s.scan_timeframe,
    }


def _humanize_bitget_error(exc: Exception) -> dict[str, str]:
    text = str(exc)
    lower = text.lower()
    if "api key" in lower or "passphrase" in lower or "credentials" in lower or "authentication" in lower:
        return _detail(
            "Bitget API 尚未完整連線",
            "讀取私人帳戶需要 BITGET_API_KEY、BITGET_API_SECRET、BITGET_API_PASSPHRASE。",
            "請在 Zeabur 設定三個 Bitget API 環境變數；不要把密鑰貼到聊天。",
        )
    if "not tradable" in lower:
        return _detail("這個合約目前不能交易", "Bitget 回報此 USDT 永續合約目前不是可交易狀態。")
    if "live_trading_enabled=false" in lower:
        return _detail("真實下單仍被鎖住", "目前 LIVE_TRADING_ENABLED=false，所以系統拒絕送出真實訂單。")
    return _detail(
        "Bitget 操作沒有完成",
        "Bitget 回傳錯誤，系統沒有猜測或強行繼續操作。",
        "稍後再試；若持續發生，把這段白話訊息和操作時間傳給我。",
    )


DASHBOARD = r'''<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CoinGlass × Bitget Strategy Lab</title>
<style>
:root{color-scheme:dark;--bg:#090b10;--panel:#121722;--line:#263044;--muted:#94a3b8;--accent:#f59e0b;--ok:#22c55e;--bad:#ef4444;--warn:#fbbf24;--blue:#38bdf8}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top,#172033 0,#090b10 42%);font:14px/1.5 Inter,system-ui,sans-serif;color:#e5e7eb}.wrap{max-width:1380px;margin:auto;padding:22px}.top{display:flex;gap:16px;align-items:center;justify-content:space-between;flex-wrap:wrap}.brand{font-size:23px;font-weight:800}.sub{color:var(--muted)}.pill{padding:6px 10px;border:1px solid var(--line);border-radius:999px;background:#0d111a}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px;margin-top:16px}.card{background:rgba(18,23,34,.94);border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 14px 40px #0005}.c4{grid-column:span 4}.c6{grid-column:span 6}.c12{grid-column:span 12}h2{font-size:17px;margin:0 0 12px}label{display:block;color:var(--muted);font-size:12px;margin:8px 0 5px}input,select,button{width:100%;border:1px solid var(--line);border-radius:10px;background:#0b1019;color:#e5e7eb;padding:11px}button{cursor:pointer;background:#1b2434;font-weight:750}button.primary{background:var(--accent);color:#111;border-color:#fbbf24}button:disabled{opacity:.45;cursor:not-allowed}.row{display:flex;gap:9px}.row>*{flex:1}.search{position:relative}.suggest{position:absolute;z-index:20;left:0;right:0;top:68px;background:#0b1019;border:1px solid var(--line);border-radius:10px;max-height:280px;overflow:auto;display:none}.sitem{padding:9px 11px;cursor:pointer;border-bottom:1px solid #1b2434}.sitem:hover{background:#172033}.sitem small{float:right;color:var(--muted)}.hint{color:var(--muted);font-size:12px;margin-top:7px}.history-hint{margin:10px 0;padding:10px 12px;border:1px solid #3a455d;background:#0b111c;border-radius:10px;color:#cbd5e1}.status{margin-top:10px;border-radius:12px;padding:13px 14px;border:1px solid var(--line);background:#090d14}.status.good{border-color:#166534;background:#0d2418}.status.bad{border-color:#7f1d1d;background:#2b1014}.status.warn{border-color:#854d0e;background:#2b210d}.status-title{font-weight:800;font-size:15px;margin-bottom:4px}.status-text{color:#d7dee9}.status-action{color:#aab6c8;margin-top:7px;font-size:12px}.live-off{color:#fecaca;background:#3f1117;border:1px solid #7f1d1d;padding:10px;border-radius:10px}.live-on{color:#bbf7d0;background:#0f3b25;border:1px solid #166534;padding:10px;border-radius:10px}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:9px 7px;border-bottom:1px solid #222b3b;text-align:right;white-space:nowrap}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}.long{color:#4ade80}.short{color:#fb7185}.scroll{overflow:auto;max-height:520px}.kpi{font-size:25px;font-weight:800}.hidden{display:none!important}@media(max-width:900px){.c4,.c6{grid-column:span 12}.row{flex-direction:column}}
</style>
</head>
<body><div class="wrap">
<div class="top"><div><div class="brand">CoinGlass × Bitget Strategy Lab</div><div class="sub">多幣種研究 · 真實資料回測 · 全市場掃描 · Bitget 執行層</div></div><div class="pill" id="health">檢查服務中…</div></div>
<div class="grid">
<section class="card c4"><h2>Bitget 合約搜尋</h2><div class="search"><label>搜尋 USDT 永續合約</label><input id="symbol" value="ETHUSDT" autocomplete="off" placeholder="輸入 b、btc、bnb…"><div id="suggest" class="suggest"></div></div><div class="hint">清單直接讀 Bitget，不寫死幣種。</div><div style="margin-top:12px"><span class="sub">目前選擇：</span> <b id="selected">ETHUSDT</b></div></section>
<section class="card c4"><h2>管理驗證</h2><label>ADMIN_BEARER_TOKEN</label><input id="token" type="password" placeholder="輸入你在 Zeabur 設定的管理密碼"><div class="hint">只存在這個瀏覽器 session，不會進回測報告。</div></section>
<section class="card c4"><h2>交易所狀態</h2><div id="liveBox" class="live-off">讀取中…</div><div class="row" style="margin-top:10px"><button onclick="loadAccount()">讀帳戶</button><button onclick="loadPositions()">讀持倉</button></div><div id="exchangeStatus" class="status"><div class="status-text">尚未讀取私人帳戶。</div></div></section>

<section class="card c6"><h2>單幣種真實資料回測</h2>
<div class="row"><div><label>開始 UTC</label><input id="btStart" placeholder="自動依 Standard 方案帶入"></div><div><label>結束 UTC</label><input id="btEnd" placeholder="最新已收 K"></div></div>
<div class="row"><div><label>Timeframe</label><select id="btTf"><option>5m</option><option selected>15m</option><option>30m</option><option>1h</option><option>4h</option></select></div><div><label>風險 / 每單</label><input id="btRisk" type="number" step="0.001" min="0.001" max="0.05" value="0.01"></div></div>
<div id="historyHint" class="history-hint">正在讀取 CoinGlass Standard 可用歷史範圍…</div>
<button id="btBtn" class="primary" onclick="startBacktest()">開始回測目前幣種</button>
<div id="btStatus" class="status"><div class="status-title">尚未開始</div><div class="status-text">按下按鈕後，這裡只會顯示白話進度或錯誤原因。</div></div>
<button id="copyReportBtn" class="hidden" style="margin-top:10px" onclick="copyReport()">複製完整回測報告</button>
</section>

<section class="card c6"><h2>全市場策略掃描</h2><div class="row"><div><label>Timeframe</label><select id="scanTf"><option>5m</option><option selected>15m</option><option>30m</option><option>1h</option><option>4h</option></select></div><div><label>最低 24h USDT 成交額</label><input id="scanTurnover" type="number" value="1000000"></div><div><label>最大 Spread %</label><input id="scanSpread" type="number" step="0.05" value="0.50"></div></div><button id="scanBtn" class="primary" style="margin-top:12px" onclick="startScan()">掃描 Bitget 全部可交易幣</button><div id="scanStatus" class="status"><div class="status-title">尚未開始</div><div class="status-text">先用 Bitget 流動性過濾，再耗用 CoinGlass API。</div></div></section>

<section class="card c12"><div class="top"><h2>符合目前 CoinGlass 策略邏輯的幣</h2><div><span class="sub">訊號：</span><span class="kpi" id="matchCount">0</span></div></div><div class="scroll"><table><thead><tr><th>Symbol</th><th>Strategy</th><th>Direction</th><th>Signal time</th><th>Ref</th><th>SL</th><th>TP</th><th>R</th><th>24h vol</th><th>spread%</th><th>CG instrument</th></tr></thead><tbody id="matches"></tbody></table></div></section>
</div></div>
<script>
const $=id=>document.getElementById(id);let scanTimer=null,btTimer=null;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function headers(){const t=$('token').value.trim();if(t)sessionStorage.setItem('coinlab_token',t);return {'Content-Type':'application/json','Authorization':'Bearer '+t}}
$('token').value=sessionStorage.getItem('coinlab_token')||'';
class FriendlyError extends Error{constructor(info){super(info.message||info.title||'操作失敗');this.info=info}}
async function jfetch(url,opt={}){const r=await fetch(url,opt);let x;try{x=await r.json()}catch{x={detail:'伺服器回應格式異常'}}if(!r.ok){let d=x.detail||x;if(typeof d==='string')d={title:'操作沒有完成',message:d};throw new FriendlyError(d)}return x}
function showStatus(id,title,message,action='',kind=''){const el=$(id);el.className='status'+(kind?' '+kind:'');el.innerHTML=`<div class="status-title">${esc(title)}</div><div class="status-text">${esc(message)}</div>${action?`<div class="status-action">${esc(action)}</div>`:''}`}
function showFriendlyError(id,e){const i=e.info||{title:'操作沒有完成',message:e.message||'未知錯誤'};showStatus(id,i.title||'操作沒有完成',i.message||'',i.action||'','bad')}
async function boot(){try{const h=await jfetch('/health');$('health').textContent='● 服務正常 · v'+h.version}catch(e){$('health').textContent='服務異常'}try{const c=await jfetch('/config');$('symbol').value=c.symbol;$('selected').textContent=c.symbol;$('btRisk').value=c.risk_per_trade;$('liveBox').className=c.live_trading_enabled?'live-on':'live-off';$('liveBox').textContent=c.live_trading_enabled?'真實下單已解鎖':'真實下單硬鎖（LIVE_TRADING_ENABLED=false）';}catch(e){$('liveBox').textContent='讀取設定失敗'}await applyHistoryPolicy(true)}
async function applyHistoryPolicy(resetDates=true){try{const tf=$('btTf').value;const p=await jfetch('/api/backtest/history-policy?timeframe='+encodeURIComponent(tf));if(resetDates){if(p.earliest_safe_start)$('btStart').value=p.earliest_safe_start;$('btEnd').value=p.latest_completed_bar}const range=p.max_history_days?`Standard 的 ${tf} 最多查最近 ${p.max_history_days} 天。`:`Standard 的 ${tf} 可查完整歷史。`;const actual=p.earliest_safe_start?`目前安全可用：約 ${p.earliest_safe_start} ～ ${p.latest_completed_bar}`:`最新已收 K：${p.latest_completed_bar}`;$('historyHint').textContent=range+' '+actual}catch(e){$('historyHint').textContent='無法讀取方案歷史限制；請重新整理頁面。'}}
$('btTf').addEventListener('change',()=>applyHistoryPolicy(true));
let st=null;$('symbol').addEventListener('input',()=>{clearTimeout(st);st=setTimeout(searchSymbols,120)});$('symbol').addEventListener('focus',searchSymbols);
async function searchSymbols(){const q=$('symbol').value.trim();try{const x=await jfetch('/api/bitget/contracts?q='+encodeURIComponent(q)+'&limit=80');const box=$('suggest');box.innerHTML=x.contracts.map(c=>`<div class="sitem" data-s="${esc(c.symbol)}"><b>${esc(c.symbol)}</b><small>${esc(c.base_coin)} · max ${esc(c.max_leverage)}x</small></div>`).join('');box.style.display=x.contracts.length?'block':'none';[...box.children].forEach(el=>el.onclick=()=>{$('symbol').value=el.dataset.s;$('selected').textContent=el.dataset.s;box.style.display='none'})}catch(e){$('suggest').style.display='none'}}
document.addEventListener('click',e=>{if(!e.target.closest('.search'))$('suggest').style.display='none'});
async function startBacktest(){try{$('btBtn').disabled=true;$('copyReportBtn').classList.add('hidden');showStatus('btStatus','準備回測','正在檢查日期是否符合 CoinGlass Standard 方案範圍…','','warn');const body={symbol:$('symbol').value,start:$('btStart').value,end:$('btEnd').value,timeframe:$('btTf').value,risk_per_trade:Number($('btRisk').value)};const x=await jfetch('/backtest/start',{method:'POST',headers:headers(),body:JSON.stringify(body)});if(x.window){$('btStart').value=x.window.used_start;$('btEnd').value=x.window.used_end}showStatus('btStatus',x.window&&x.window.adjusted?'日期已自動調整，回測開始':'回測已開始',x.message||'正在取得真實資料。',x.window&&x.window.adjusted?'系統只會使用 CoinGlass 方案真正允許取得的資料，不會補假資料。':'','warn');clearInterval(btTimer);btTimer=setInterval(pollBacktest,2500)}catch(e){$('btBtn').disabled=false;showFriendlyError('btStatus',e)}}
async function pollBacktest(){try{const x=await jfetch('/backtest/status',{headers:headers()});if(x.state==='running'){showStatus('btStatus','回測進行中',x.message||'正在下載、對時並驗證真實資料…','依期間與 CoinGlass API 速度，可能需要幾分鐘。','warn')}else if(x.state==='completed'){clearInterval(btTimer);$('btBtn').disabled=false;$('copyReportBtn').classList.remove('hidden');showStatus('btStatus','回測完成',x.message||'真實資料回測完成，報告已產生。','可按下「複製完整回測報告」直接貼給我分析。','good')}else if(x.state==='failed'){clearInterval(btTimer);$('btBtn').disabled=false;const i=x.error||{title:'回測沒有完成',message:x.message||'回測失敗'};showStatus('btStatus',i.title||'回測沒有完成',i.message||'',i.action||'','bad')}}catch(e){clearInterval(btTimer);$('btBtn').disabled=false;showFriendlyError('btStatus',e)}}
async function copyReport(){try{const x=await jfetch('/backtest/report',{headers:headers()});await navigator.clipboard.writeText(JSON.stringify(x,null,2));showStatus('btStatus','回測完成 · 已複製','完整 BACKTEST_REPORT.json 已複製到剪貼簿。','直接貼到 ChatGPT，我就能依 OOS、PF、Expectancy、Max DD 與交易明細繼續改策略。','good')}catch(e){showFriendlyError('btStatus',e)}}
async function startScan(){try{$('scanBtn').disabled=true;showStatus('scanStatus','開始掃描','正在讀取 Bitget 合約與流動性資料…','','warn');const body={timeframe:$('scanTf').value,min_turnover_usdt:Number($('scanTurnover').value),max_spread_pct:Number($('scanSpread').value)};const x=await jfetch('/api/scan/start',{method:'POST',headers:headers(),body:JSON.stringify(body)});showStatus('scanStatus','掃描已開始',x.message||'先過濾 Bitget 市場，再檢查 CoinGlass 策略。','','warn');clearInterval(scanTimer);scanTimer=setInterval(pollScan,2500)}catch(e){$('scanBtn').disabled=false;showFriendlyError('scanStatus',e)}}
async function pollScan(){try{const x=await jfetch('/api/scan/status',{headers:headers()});if(x.state==='running'){const p=x.progress||{};const progress=p.total?`目前 ${p.current||0} / ${p.total}`:'正在建立候選清單';showStatus('scanStatus','掃描進行中',x.message||progress,progress,'warn')}else if(x.state==='completed'){clearInterval(scanTimer);$('scanBtn').disabled=false;const count=x.result&&x.result.matches?x.result.matches.length:0;showStatus('scanStatus','掃描完成',`目前找到 ${count} 個符合策略條件的訊號。`,'下方表格只顯示真正符合目前策略邏輯的項目。','good')}else if(x.state==='failed'){clearInterval(scanTimer);$('scanBtn').disabled=false;const i=x.error||{title:'掃描沒有完成',message:x.message||'掃描失敗'};showStatus('scanStatus',i.title||'掃描沒有完成',i.message||'',i.action||'','bad')}if(x.result)renderMatches(x.result.matches||[])}catch(e){clearInterval(scanTimer);$('scanBtn').disabled=false;showFriendlyError('scanStatus',e)}}
function fmt(n){if(n===null||n===undefined)return '';const x=Number(n);return Number.isFinite(x)?x.toLocaleString(undefined,{maximumSignificantDigits:8}):n}
function renderMatches(rows){$('matchCount').textContent=rows.length;$('matches').innerHTML=rows.map(r=>`<tr><td><b>${esc(r.symbol)}</b></td><td>${esc(r.strategy)}</td><td class="${esc(r.direction)}">${esc(r.direction)}</td><td>${esc(r.signal_time)}</td><td>${fmt(r.reference_price)}</td><td>${fmt(r.stop_loss)}</td><td>${fmt(r.take_profit)}</td><td>${fmt(r.reward_r)}</td><td>${fmt(r.volume_24h_usdt)}</td><td>${fmt(r.spread_pct)}</td><td>${esc(r.coinglass_instrument)}</td></tr>`).join('')}
async function loadAccount(){try{showStatus('exchangeStatus','讀取中','正在向 Bitget 讀取 USDT 合約帳戶…','','warn');const x=await jfetch('/api/bitget/account',{headers:headers()});showStatus('exchangeStatus','Bitget 帳戶已連線',`帳戶權益：${fmt(x.equity)} USDT｜可用：${fmt(x.available)} USDT｜鎖定：${fmt(x.locked)} USDT`,'這只是讀取，不會下單。','good')}catch(e){showFriendlyError('exchangeStatus',e)}}
async function loadPositions(){try{showStatus('exchangeStatus','讀取中','正在向 Bitget 讀取目前持倉…','','warn');const x=await jfetch('/api/bitget/positions',{headers:headers()});const rows=x.positions||[];if(!rows.length){showStatus('exchangeStatus','目前沒有持倉','Bitget USDT 永續帳戶目前沒有開放中的部位。','','good')}else{const summary=rows.map(p=>`${p.symbol} ${p.direction==='long'?'多':'空'}｜數量 ${fmt(p.size)}｜均價 ${fmt(p.entry_price)}｜未實現 ${fmt(p.unrealized_pnl)} U`).join('；');showStatus('exchangeStatus',`目前 ${rows.length} 個持倉`,summary,'這只是讀取，不會修改部位。','good')}}catch(e){showFriendlyError('exchangeStatus',e)}}
boot();
</script>
</body></html>'''


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return DASHBOARD


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "utc": datetime.now(timezone.utc).isoformat(), "version": "0.3.0"}


@app.get("/config")
def config() -> dict[str, Any]:
    return _safe_public_config()


@app.get("/strategies")
def strategies() -> dict[str, Any]:
    from coinlab.strategies import STRATEGIES

    return {"strategies": sorted(STRATEGIES.keys()), "count": len(STRATEGIES)}


@app.get("/api/backtest/history-policy")
def backtest_history_policy(
    timeframe: str = Query(default="15m", max_length=8),
) -> dict[str, Any]:
    try:
        return standard_policy(timeframe)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=_detail("週期不支援", str(exc), "請選擇畫面提供的回測週期。"),
        ) from exc


@app.get("/api/bitget/contracts")
def bitget_contracts(
    q: str = Query(default="", max_length=40),
    limit: int = Query(default=80, ge=1, le=300),
) -> dict[str, Any]:
    try:
        rows = _bitget(live=False).get_contracts()
    except BitgetAPIError as exc:
        raise HTTPException(status_code=502, detail=_humanize_bitget_error(exc)) from exc
    term = BitgetV2Client.normalize_symbol(q)
    if term:
        rows = [r for r in rows if term in r["symbol"] or term in r["base_coin"]]
        rows.sort(
            key=lambda r: (
                not r["base_coin"].startswith(term),
                not r["symbol"].startswith(term),
                r["symbol"],
            )
        )
    else:
        rows.sort(key=lambda r: r["symbol"])
    public = [
        {
            "symbol": r["symbol"],
            "base_coin": r["base_coin"],
            "quote_coin": r["quote_coin"],
            "min_size": r["min_size"],
            "size_step": r["size_step"],
            "price_step": r["price_step"],
            "min_notional": r["min_notional"],
            "min_leverage": r["min_leverage"],
            "max_leverage": r["max_leverage"],
        }
        for r in rows[:limit]
    ]
    return {"count": len(public), "contracts": public}


def _run_backtest(
    run_id: str,
    output_dir: Path,
    log_file: Path,
    env: dict[str, str],
    timeframe: str,
) -> None:
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
            _backtest_job["state"] = "completed" if process.returncode == 0 else "failed"
            _backtest_job["return_code"] = process.returncode
            _backtest_job["finished_at"] = datetime.now(timezone.utc).isoformat()
            if process.returncode == 0:
                _backtest_job["error"] = None
                _backtest_job["message"] = "真實資料回測已完成，沒有使用虛構資料。"
            else:
                text = log_file.read_text(encoding="utf-8", errors="replace")
                friendly = humanize_backtest_failure(text, timeframe)
                _backtest_job["error"] = friendly
                _backtest_job["message"] = friendly["message"]
    except Exception as exc:
        with _backtest_lock:
            if _backtest_job.get("run_id") != run_id:
                return
            friendly = humanize_backtest_failure(str(exc), timeframe)
            _backtest_job["state"] = "failed"
            _backtest_job["finished_at"] = datetime.now(timezone.utc).isoformat()
            _backtest_job["error"] = friendly
            _backtest_job["message"] = friendly["message"]


@app.post("/backtest/start", dependencies=[Depends(_require_admin)])
def start_backtest(req: BacktestRequest) -> dict[str, Any]:
    global _backtest_job
    s = _settings()
    if not s.coinglass_api_key:
        raise HTTPException(
            status_code=503,
            detail=_detail(
                "CoinGlass API 尚未設定",
                "Zeabur 沒有讀到 COINGLASS_API_KEY，所以不能取得真實衍生品資料。",
                "請在 Zeabur 環境變數設定 COINGLASS_API_KEY 後重新部署。",
            ),
        )

    timeframe = req.timeframe or s.timeframe
    symbol = req.symbol or s.symbol
    try:
        window = normalize_backtest_window(
            timeframe=timeframe,
            requested_start=req.start,
            requested_end=req.end,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=_detail("回測日期不正確", str(exc), "請重新選擇日期後再執行。"),
        ) from exc

    with _backtest_lock:
        if _backtest_job["state"] == "running":
            raise HTTPException(
                status_code=409,
                detail=_detail("已有回測正在執行", "請等目前這次回測完成後再開始下一次。"),
            )

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = ARTIFACT_ROOT / "backtests" / run_id
        log_file = output_dir / "run.log"
        env = os.environ.copy()
        overrides: dict[str, Any] = {
            "SYMBOL": symbol,
            "START": window.used_start,
            "END": window.used_end,
            "TIMEFRAME": timeframe,
            "COINGLASS_EXCHANGE": req.coinglass_exchange or s.coinglass_exchange,
            "INITIAL_EQUITY": req.initial_equity,
            "RISK_PER_TRADE": req.risk_per_trade,
            "TAKER_FEE_BPS": req.taker_fee_bps,
            "SLIPPAGE_BPS": req.slippage_bps,
            "MIN_ALIGNED_COVERAGE": req.min_aligned_coverage,
        }
        for key, value in overrides.items():
            if value is not None:
                env[key] = str(value)
        env["COINGLASS_SYMBOL"] = "AUTO"
        env["LIVE_TRADING_ENABLED"] = "false"

        _backtest_job = {
            "state": "running",
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "return_code": None,
            "output_dir": str(output_dir),
            "log_file": str(log_file),
            "error": None,
            "message": window.message if window.adjusted else "正在下載 Bitget K 線與 CoinGlass 衍生品資料，接著會做精確時間對齊。",
            "symbol": symbol,
            "timeframe": timeframe,
            "window": window.to_dict(),
        }
        threading.Thread(
            target=_run_backtest,
            args=(run_id, output_dir, log_file, env, timeframe),
            daemon=True,
            name=f"backtest-{run_id}",
        ).start()

    return {
        "accepted": True,
        "run_id": run_id,
        "state": "running",
        "symbol": symbol,
        "timeframe": timeframe,
        "message": window.message,
        "window": window.to_dict(),
    }


@app.get("/backtest/status", dependencies=[Depends(_require_admin)])
def backtest_status() -> dict[str, Any]:
    with _backtest_lock:
        raw = dict(_backtest_job)
    report_path = Path(raw["output_dir"]) / "BACKTEST_REPORT.json" if raw.get("output_dir") else None
    return {
        "state": raw.get("state"),
        "run_id": raw.get("run_id"),
        "started_at": raw.get("started_at"),
        "finished_at": raw.get("finished_at"),
        "symbol": raw.get("symbol"),
        "timeframe": raw.get("timeframe"),
        "window": raw.get("window"),
        "message": raw.get("message"),
        "error": raw.get("error"),
        "report_ready": bool(report_path and report_path.exists()),
    }


@app.get("/backtest/report", dependencies=[Depends(_require_admin)])
def backtest_report() -> dict[str, Any]:
    with _backtest_lock:
        output_dir = _backtest_job.get("output_dir")
    if not output_dir:
        raise HTTPException(
            status_code=404,
            detail=_detail("還沒有回測報告", "目前尚未執行過回測。"),
        )
    report_path = Path(output_dir) / "BACKTEST_REPORT.json"
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail=_detail("報告尚未完成", "這次回測還在執行，或已因資料問題停止。"),
        )
    return json.loads(report_path.read_text(encoding="utf-8"))


def _run_scan(run_id: str, cfg: ScanConfig, api_key: str) -> None:
    global _scan_job
    try:
        def progress(value: dict[str, Any]) -> None:
            with _scan_lock:
                if _scan_job.get("run_id") == run_id:
                    _scan_job["progress"] = value
                    current = value.get("current", 0)
                    total = value.get("total", 0)
                    _scan_job["message"] = (
                        f"正在檢查候選幣種：{current} / {total}。" if total else "正在建立 Bitget 候選清單。"
                    )

        result = scan_market(coinglass_api_key=api_key, cfg=cfg, progress=progress)
        with _scan_lock:
            if _scan_job.get("run_id") != run_id:
                return
            _scan_job["state"] = "completed"
            _scan_job["finished_at"] = datetime.now(timezone.utc).isoformat()
            _scan_job["result"] = result
            count = len(result.get("matches", [])) if isinstance(result, dict) else 0
            _scan_job["message"] = f"掃描完成，目前找到 {count} 個符合策略條件的訊號。"
    except Exception as exc:
        with _scan_lock:
            if _scan_job.get("run_id") != run_id:
                return
            friendly = humanize_backtest_failure(str(exc), cfg.timeframe)
            if friendly["code"] == "BACKTEST_FAILED":
                friendly = _detail(
                    "全市場掃描沒有完成",
                    "掃描真實 Bitget / CoinGlass 資料時遇到問題，系統已停止，沒有用缺失資料硬判斷訊號。",
                    "稍後再試；若持續發生，把這段白話訊息傳給我。",
                ) | {"code": "SCAN_FAILED"}
            _scan_job["state"] = "failed"
            _scan_job["finished_at"] = datetime.now(timezone.utc).isoformat()
            _scan_job["error"] = friendly
            _scan_job["message"] = friendly["message"]


@app.post("/api/scan/start", dependencies=[Depends(_require_admin)])
def start_scan(req: ScanRequest) -> dict[str, Any]:
    global _scan_job
    s = _settings()
    if not s.coinglass_api_key:
        raise HTTPException(
            status_code=503,
            detail=_detail(
                "CoinGlass API 尚未設定",
                "沒有 COINGLASS_API_KEY，無法對候選幣做衍生品策略掃描。",
            ),
        )
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
            raise HTTPException(
                status_code=409,
                detail=_detail("已有掃描正在進行", "請等目前的全市場掃描完成後再重新執行。"),
            )
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        _scan_job = {
            "state": "running",
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "progress": {"current": 0, "total": 0},
            "result": None,
            "error": None,
            "message": "正在讀取 Bitget 全市場並先做流動性過濾。",
        }
        threading.Thread(
            target=_run_scan,
            args=(run_id, cfg, s.coinglass_api_key),
            daemon=True,
            name=f"scan-{run_id}",
        ).start()
    return {
        "accepted": True,
        "run_id": run_id,
        "state": "running",
        "message": "掃描已開始：先用 Bitget 篩選，再對候選幣讀 CoinGlass 資料。",
    }


@app.get("/api/scan/status", dependencies=[Depends(_require_admin)])
def scan_status() -> dict[str, Any]:
    with _scan_lock:
        raw = dict(_scan_job)
    return {
        "state": raw.get("state"),
        "run_id": raw.get("run_id"),
        "started_at": raw.get("started_at"),
        "finished_at": raw.get("finished_at"),
        "progress": raw.get("progress"),
        "result": raw.get("result"),
        "message": raw.get("message"),
        "error": raw.get("error"),
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
    """Submit EXACT strategy entry/SL/TP. Exchange code may size/round quantity only."""
    s = _settings()
    if not s.live_trading_enabled:
        raise HTTPException(
            status_code=423,
            detail=_detail("真實下單仍被鎖住", "目前 LIVE_TRADING_ENABLED=false，不會送出任何真實訂單。"),
        )
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
        raise HTTPException(
            status_code=423,
            detail=_detail("真實平倉仍被鎖住", "目前 LIVE_TRADING_ENABLED=false，不會修改 Bitget 持倉。"),
        )
    try:
        return _bitget().close_position_market(symbol)
    except BitgetAPIError as exc:
        raise HTTPException(status_code=409, detail=_humanize_bitget_error(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))

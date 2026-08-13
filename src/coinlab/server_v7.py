from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field, field_validator

from . import server_v5 as v6
from .config import Settings
from .market_backtest import MarketBacktestConfig, run_market_backtest


app = FastAPI(
    title="CoinGlass × Bitget Strategy Lab",
    version="0.7.0",
    description="Market-wide no-lookahead historical simulation, fixed paper notionals, continuous Bitget scanning and locked-test research packages.",
)

_SKIP_PATHS = {"/", "/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc", "/health"}
for route in v6.app.router.routes:
    if getattr(route, "path", None) not in _SKIP_PATHS:
        app.router.routes.append(route)
app.router.on_startup.extend(v6.app.router.on_startup)
app.router.on_shutdown.extend(v6.app.router.on_shutdown)

MARKET_ROOT = v6.v4.ARTIFACT_ROOT / "market_backtests"
MARKET_ROOT.mkdir(parents=True, exist_ok=True)
_market_lock = threading.Lock()
_market_stop = threading.Event()
_market_thread: threading.Thread | None = None
_market_job: dict[str, Any] = {
    "state": "idle", "run_id": None, "started_at": None, "finished_at": None,
    "output_dir": None, "timeframe": None, "progress": None, "message": "尚未開始全市場回測。",
    "error": None, "summary": None,
}


class MarketBacktestRequest(BaseModel):
    timeframe: str = "15m"
    min_historical_24h_turnover_usdt: float | None = Field(default=None, ge=0)
    max_symbols: int | None = Field(default=None, ge=0, le=2000)

    @field_validator("timeframe")
    @classmethod
    def valid_timeframe(cls, value: str) -> str:
        allowed = {"5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"}
        if value not in allowed:
            raise ValueError(f"不支援的回測週期：{value}")
        return value


def _admin(authorization: str | None = Header(default=None)) -> None:
    return v6.v4._require_admin(authorization)


def _detail(title: str, message: str, action: str = "") -> dict[str, str]:
    return v6.v4._detail(title, message, action)


def _run_market_job(run_id: str, outdir: Path, req: MarketBacktestRequest) -> None:
    global _market_job
    s = Settings()
    cfg = MarketBacktestConfig(
        timeframe=req.timeframe,
        min_historical_24h_turnover_usdt=(
            s.market_backtest_min_24h_turnover_usdt
            if req.min_historical_24h_turnover_usdt is None
            else req.min_historical_24h_turnover_usdt
        ),
        max_symbols=s.market_backtest_max_symbols if req.max_symbols is None else req.max_symbols,
        fee_bps=s.taker_fee_bps,
        slippage_bps=s.slippage_bps,
        max_estimated_cost_r=s.max_estimated_cost_r,
        low_notional_usdt=s.paper_low_notional_usdt,
        high_notional_usdt=s.paper_high_notional_usdt,
        high_price_threshold_usdt=s.paper_high_price_threshold,
        initial_equity=s.initial_equity,
        min_aligned_coverage=s.min_aligned_coverage,
        min_aligned_rows=s.market_backtest_min_aligned_rows,
        coinglass_exchange=s.coinglass_exchange,
        bitget_base_url=s.bitget_rest_base_url,
        cache_root=s.market_cache_root,
    )

    def progress(value: dict[str, Any]) -> None:
        stage_labels = {
            "price": "Bitget 歷史 K 線", "oi": "CoinGlass OI", "funding": "CoinGlass Funding",
            "liq": "CoinGlass 爆倉", "ls": "CoinGlass 多空比", "taker": "CoinGlass Taker Flow",
            "orderbook": "CoinGlass Orderbook",
        }
        with _market_lock:
            if _market_job.get("run_id") != run_id:
                return
            _market_job["progress"] = value
            current = value.get("current") or 0
            total = value.get("total") or 0
            symbol = value.get("symbol") or ""
            stage = stage_labels.get(str(value.get("stage") or ""), str(value.get("stage") or ""))
            _market_job["message"] = f"正在處理 {symbol}（{current}/{total}）· {stage}。這是全市場真實資料工作，可能需要較久。"

    try:
        report = run_market_backtest(
            coinglass_api_key=s.coinglass_api_key,
            requested_start=s.start,
            requested_end=s.end,
            cfg=cfg,
            outdir=outdir,
            progress=progress,
            should_stop=_market_stop.is_set,
        )
        with _market_lock:
            if _market_job.get("run_id") != run_id:
                return
            _market_job["finished_at"] = datetime.now(timezone.utc).isoformat()
            if _market_stop.is_set():
                _market_job["state"] = "stopped"
                _market_job["message"] = "全市場回測已停止。已取得的快取資料會保留，下次重新跑可重用。"
                return
            summary = report.get("portfolio_summary") or {}
            _market_job["state"] = "completed"
            _market_job["summary"] = summary
            _market_job["message"] = (
                f"全市場回測完成：{int(summary.get('trades') or 0)} 筆模擬單，"
                f"淨損益 {float(summary.get('net_pnl') or 0):,.2f} U。每筆交易已寫入 CSV。"
            )
            _market_job["error"] = None
    except Exception as exc:
        with _market_lock:
            if _market_job.get("run_id") == run_id:
                _market_job["finished_at"] = datetime.now(timezone.utc).isoformat()
                _market_job["state"] = "failed"
                _market_job["error"] = _detail(
                    "全市場回測沒有完成",
                    "系統在下載或處理真實 Bitget / CoinGlass 資料時遇到問題，因此停止，沒有補假資料或製造績效。",
                    f"錯誤類型：{type(exc).__name__}。把這段白話訊息傳給我即可。",
                )
                _market_job["message"] = _market_job["error"]["message"]


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "version": "0.7.0", "market_wide_backtest": True, "fixed_paper_notional": True}


@app.post("/api/market-backtest/start", dependencies=[Depends(_admin)])
def start_market_backtest(req: MarketBacktestRequest) -> dict[str, Any]:
    global _market_thread, _market_job
    s = Settings()
    if not s.coinglass_api_key:
        raise HTTPException(status_code=503, detail=_detail("CoinGlass API 尚未設定", "沒有 COINGLASS_API_KEY，無法執行全市場歷史回測。"))
    with _market_lock:
        if _market_thread and _market_thread.is_alive():
            raise HTTPException(status_code=409, detail=_detail("全市場回測正在進行", "目前已有一個全市場回測工作，請等完成或先停止。"))
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        outdir = MARKET_ROOT / run_id
        _market_stop.clear()
        _market_job = {
            "state": "running", "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(), "finished_at": None,
            "output_dir": str(outdir), "timeframe": req.timeframe,
            "progress": {"current": 0, "total": 0}, "message": "正在建立 Bitget × CoinGlass 歷史市場 universe。",
            "error": None, "summary": None,
        }
        _market_thread = threading.Thread(target=_run_market_job, args=(run_id, outdir, req), daemon=True, name="coinlab-market-backtest")
        _market_thread.start()
    return {"accepted": True, "run_id": run_id, "state": "running"}


@app.post("/api/market-backtest/stop", dependencies=[Depends(_admin)])
def stop_market_backtest() -> dict[str, Any]:
    _market_stop.set()
    with _market_lock:
        if _market_job.get("state") == "running":
            _market_job["message"] = "已收到停止要求；系統會在目前資料請求／幣種安全結束後停止。"
    return {"accepted": True, "message": "停止要求已送出。"}


@app.get("/api/market-backtest/status", dependencies=[Depends(_admin)])
def market_backtest_status() -> dict[str, Any]:
    with _market_lock:
        return dict(_market_job)


def _market_file(kind: str) -> Path:
    with _market_lock:
        output = _market_job.get("output_dir")
        state = _market_job.get("state")
    if not output or state != "completed":
        raise HTTPException(status_code=409, detail=_detail("全市場回測尚未完成", "完成後才能下載研究包。"))
    root = Path(output)
    patterns = {
        "research": "COINLAB_MARKET_RESEARCH_*.zip",
        "audit": "COINLAB_MARKET_AUDIT_*.zip",
        "trades": "all_market_trades.csv",
        "report": "MARKET_BACKTEST_REPORT.json",
    }
    pattern = patterns[kind]
    matches = sorted(root.glob(pattern)) if "*" in pattern else [root / pattern]
    path = matches[-1] if matches and matches[-1].exists() else None
    if path is None:
        raise HTTPException(status_code=404, detail=_detail("下載檔案不存在", "這次回測沒有產生指定檔案。"))
    return path


@app.get("/api/market-backtest/download/research", dependencies=[Depends(_admin)])
def market_research_download() -> FileResponse:
    p = _market_file("research")
    return FileResponse(p, media_type="application/zip", filename=p.name)


@app.get("/api/market-backtest/download/audit", dependencies=[Depends(_admin)])
def market_audit_download() -> FileResponse:
    p = _market_file("audit")
    return FileResponse(p, media_type="application/zip", filename=p.name)


@app.get("/api/market-backtest/download/trades", dependencies=[Depends(_admin)])
def market_trades_download() -> FileResponse:
    p = _market_file("trades")
    return FileResponse(p, media_type="text/csv", filename=p.name)


# Replace only the old single-symbol card. Keeping btTf/btStart/btEnd/btRisk IDs
# preserves the already-tested date-policy boot JS without using the old start action.
_new_market_card = r'''<section class="card c6"><h2>Bitget 全市場歷史真實回測</h2>
<div class="notice">不再只回測目前選擇的單一幣。系統會依這段歷史時間，逐一檢查 Bitget 可研究合約；歷史交易資格只看當時以前已收 K 的 24h 成交額，不用今天成交額回頭選過去贏家。</div>
<div class="notice">模擬名目固定：進場當下市場價格 &gt; 50U → 20,000U；≤ 50U → 2,000U。所有實際模擬單都會誠實寫入 all_market_trades.csv。</div>
<div class="row"><div><label>自動開始 UTC</label><input id="btStart" readonly></div><div><label>自動結束 UTC</label><input id="btEnd" readonly></div></div>
<div class="row"><div><label>Timeframe</label><select id="btTf"><option>5m</option><option selected>15m</option><option>30m</option><option>1h</option><option>4h</option></select></div><div><label>歷史 24h 最低成交額 U</label><input id="marketTurnover" type="number" value="1000000"></div><div><label>最多幣種（0=全部）</label><input id="marketMaxSymbols" type="number" value="0" min="0"></div></div>
<input id="btRisk" class="hidden" value="0.01">
<div class="row" style="margin-top:12px"><button id="marketBtBtn" class="primary" onclick="startMarketBacktest()">開始全市場完整回測</button><button class="pause" onclick="stopMarketBacktest()">停止回測</button></div>
<div id="btStatus" class="status"><div class="status-title">尚未開始</div><div>日期會依 CoinGlass Standard 自動選滿可用範圍。</div></div>
<div id="marketSummary" class="notice hidden"></div>
<div class="row" style="margin-top:9px"><button id="marketResearchBtn" class="hidden primary" onclick="downloadMarket('/api/market-backtest/download/research')">下載全市場研究包 ZIP（傳給 ChatGPT）</button><button id="marketTradesBtn" class="hidden" onclick="downloadMarket('/api/market-backtest/download/trades')">下載全部逐筆交易 CSV</button><button id="marketAuditBtn" class="hidden" onclick="downloadMarket('/api/market-backtest/download/audit')">下載完整稽核包 ZIP</button></div>
</section>'''

DASHBOARD = v6.DASHBOARD
DASHBOARD = DASHBOARD.replace("Strategy Lab v0.6", "Strategy Lab v0.7")
DASHBOARD = DASHBOARD.replace("v0.6 · 成本感知 · 多幣種", "v0.7 · 全市場真實回測 · 固定名目 · 無未來資料")
DASHBOARD = re.sub(
    r'<section class="card c6"><h2>單幣種真實資料回測</h2>.*?</section>(?=\s*<section class="card c6"><h2>Bitget 全市場自動掃描</h2>)',
    _new_market_card,
    DASHBOARD,
    count=1,
    flags=re.S,
)

_market_js = r'''
let marketTimer=null;
async function startMarketBacktest(){
  $('marketResearchBtn').classList.add('hidden');$('marketTradesBtn').classList.add('hidden');$('marketAuditBtn').classList.add('hidden');$('marketSummary').classList.add('hidden');
  try{
    await jfetch('/api/market-backtest/start',{method:'POST',headers:headers(),body:JSON.stringify({timeframe:$('btTf').value,min_historical_24h_turnover_usdt:Number($('marketTurnover').value),max_symbols:Number($('marketMaxSymbols').value)})});
    showStatus('btStatus','全市場回測已開始','正在建立歷史 universe 並逐幣下載真實資料。','全部幣種可能需要較長時間；系統會顯示進度。','warn');
    if(marketTimer)clearInterval(marketTimer);marketTimer=setInterval(pollMarketBacktest,2500);await pollMarketBacktest();
  }catch(e){showErr('btStatus',e)}
}
async function stopMarketBacktest(){try{await jfetch('/api/market-backtest/stop',{method:'POST',headers:headers()});showStatus('btStatus','已要求停止','會在目前幣種／資料請求安全結束後停止。','','warn')}catch(e){showErr('btStatus',e)}}
async function pollMarketBacktest(){
  try{
    const x=await jfetch('/api/market-backtest/status',{headers:headers()});const p=x.progress||{};
    if(x.state==='running'){showStatus('btStatus','全市場回測進行中',x.message||'處理中…',p.total?`進度 ${p.current||0} / ${p.total} · ${p.symbol||''}`:'正在建立 universe','warn');return}
    if(x.state==='completed'){
      if(marketTimer){clearInterval(marketTimer);marketTimer=null}const s=x.summary||{};
      showStatus('btStatus','全市場回測完成',x.message||'完成。','請下載研究包傳給 ChatGPT；不要貼大型 JSON。','good');
      $('marketSummary').classList.remove('hidden');$('marketSummary').innerHTML=`模擬交易：<b>${fmt(s.trades,0)}</b>　淨損益：<b class="${pnlClass(s.net_pnl)}">${fmt(s.net_pnl,2)} U</b>　PF：<b>${fmt(s.profit_factor,3)}</b>　期望R：<b>${fmt(s.expectancy_r,3)}</b>　峰值同時單數：<b>${fmt(s.peak_open_tickets,0)}</b>　峰值總名目：<b>${fmt(s.peak_gross_notional_usdt,0)} U</b>`;
      $('marketResearchBtn').classList.remove('hidden');$('marketTradesBtn').classList.remove('hidden');$('marketAuditBtn').classList.remove('hidden');return
    }
    if(x.state==='failed'){if(marketTimer){clearInterval(marketTimer);marketTimer=null}showErr('btStatus',new FriendlyError(x.error||{title:'全市場回測沒有完成',message:x.message||'資料處理失敗。'}));return}
    if(x.state==='stopped'){if(marketTimer){clearInterval(marketTimer);marketTimer=null}showStatus('btStatus','全市場回測已停止',x.message||'已停止。','','warn')}
  }catch(e){showErr('btStatus',e)}
}
async function downloadMarket(url){
  try{const r=await fetch(url,{headers:headers()});if(!r.ok){let x={};try{x=await r.json()}catch{};throw new FriendlyError(x.detail||{title:'下載失敗',message:'無法下載檔案。'})}const blob=await r.blob();const cd=r.headers.get('content-disposition')||'';const m=cd.match(/filename="?([^";]+)"?/i);const name=m?m[1]:'coinlab_market_file';const u=URL.createObjectURL(blob);const a=document.createElement('a');a.href=u;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(u),1500)}catch(e){showErr('btStatus',e)}
}
'''
DASHBOARD = DASHBOARD.replace("</script></body></html>", _market_js + "\n</script></body></html>")


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return DASHBOARD

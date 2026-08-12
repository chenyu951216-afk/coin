from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from . import server_v4 as v4
from .research_package import build_audit_package, build_research_package


app = FastAPI(
    title="CoinGlass × Bitget Strategy Lab",
    version="0.5.0",
    description="Real-data strategy research with anti-overfit research packages, detailed backtests and continuous scanning.",
)

# Reuse the already-tested v0.4 API routes and their module state, but own the
# root page and new downloadable-artifact endpoints in v0.5.
_SKIP_PATHS = {"/", "/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
for route in v4.app.router.routes:
    if getattr(route, "path", None) not in _SKIP_PATHS:
        app.router.routes.append(route)
# FastAPI/Starlette versions in this project expose startup/shutdown handlers on
# the router. Copy the tested v0.4 handlers directly rather than relying on the
# removed app.add_event_handler compatibility API.
app.router.on_startup.extend(v4.app.router.on_startup)
app.router.on_shutdown.extend(v4.app.router.on_shutdown)


_old_button = '<button id="copyReportBtn" class="hidden" style="margin-top:9px" onclick="copyReport()">複製完整回測報告給 ChatGPT</button>'
_new_buttons = (
    '<div class="notice" style="margin-top:10px">平常要傳給 ChatGPT 改策略，請下載「研究包」。'
    '研究包會刻意排除 locked test 明細，避免反覆調參污染 OOS。完整稽核包只在策略候選凍結後使用。</div>'
    '<div class="row" style="margin-top:9px">'
    '<button id="researchDownloadBtn" class="hidden primary" onclick="downloadResearchPackage()">下載研究包 ZIP（傳給 ChatGPT）</button>'
    '<button id="auditDownloadBtn" class="hidden" onclick="downloadAuditPackage()">下載完整稽核包 ZIP（最終驗證）</button>'
    '</div>'
)

DASHBOARD = v4.DASHBOARD
DASHBOARD = DASHBOARD.replace("Strategy Lab v0.4", "Strategy Lab v0.5")
DASHBOARD = DASHBOARD.replace("v0.4 · 多幣種", "v0.5 · 多幣種")
DASHBOARD = DASHBOARD.replace(_old_button, _new_buttons)
DASHBOARD = DASHBOARD.replace("$('copyReportBtn').classList.add('hidden');", "$('researchDownloadBtn').classList.add('hidden');$('auditDownloadBtn').classList.add('hidden');")
DASHBOARD = DASHBOARD.replace("$('copyReportBtn').classList.remove('hidden');", "$('researchDownloadBtn').classList.remove('hidden');$('auditDownloadBtn').classList.remove('hidden');")

_download_js = r'''
async function downloadBacktestFile(url,label){
  try{
    const r=await fetch(url,{headers:headers()});
    if(!r.ok){
      let x={};try{x=await r.json()}catch{}
      let d=x.detail||x||{};if(typeof d==='string')d={title:'下載失敗',message:d};
      throw new FriendlyError(d);
    }
    const blob=await r.blob();
    let filename=label+'.zip';
    const cd=r.headers.get('content-disposition')||'';
    const m=cd.match(/filename="?([^";]+)"?/i);if(m)filename=m[1];
    const u=URL.createObjectURL(blob);const a=document.createElement('a');a.href=u;a.download=filename;
    document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(u),1500);
    showStatus('btStatus','回測完成 · 檔案已下載',label+' 已下載，不需要複製大型 JSON。','研究調整請傳研究包；完整稽核包不要拿來反覆調參。','good');
  }catch(e){showErr('btStatus',e)}
}
async function downloadResearchPackage(){return downloadBacktestFile('/backtest/download/research','CoinLab 研究包')}
async function downloadAuditPackage(){return downloadBacktestFile('/backtest/download/audit','CoinLab 完整稽核包')}
'''
DASHBOARD = DASHBOARD.replace("</script></body></html>", _download_js + "\n</script></body></html>")


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return DASHBOARD


def _current_output_dir() -> Path:
    with v4._backtest_lock:
        output_dir = v4._backtest_job.get("output_dir")
        state = v4._backtest_job.get("state")
    if not output_dir:
        raise HTTPException(status_code=404, detail=v4._detail("還沒有回測檔案", "請先完成一次回測。"))
    if state != "completed":
        raise HTTPException(status_code=409, detail=v4._detail("回測尚未完成", "等回測完成後才能下載檔案。"))
    root = Path(output_dir)
    if not (root / "BACKTEST_REPORT.json").exists():
        raise HTTPException(status_code=404, detail=v4._detail("回測報告不存在", "這次回測沒有產生可下載報告。"))
    return root


@app.get("/backtest/download/research", dependencies=[Depends(v4._require_admin)])
def download_research_package() -> FileResponse:
    try:
        path = build_research_package(_current_output_dir())
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=v4._detail("研究包建立失敗", "完整回測已保留，但壓縮研究包時遇到問題。", f"錯誤類型：{type(exc).__name__}"),
        ) from exc
    return FileResponse(path, media_type="application/zip", filename=path.name)


@app.get("/backtest/download/audit", dependencies=[Depends(v4._require_admin)])
def download_audit_package() -> FileResponse:
    try:
        path = build_audit_package(_current_output_dir())
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=v4._detail("稽核包建立失敗", "完整回測已保留，但壓縮稽核包時遇到問題。", f"錯誤類型：{type(exc).__name__}"),
        ) from exc
    return FileResponse(path, media_type="application/zip", filename=path.name)


@app.get("/health-v5")
def health_v5() -> dict[str, Any]:
    return {"ok": True, "version": "0.5.0", "downloadable_research_packages": True}

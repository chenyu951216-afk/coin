from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator


app = FastAPI(
    title="CoinGlass × Bitget Strategy Lab",
    version="0.1.1",
    description="Research-only CoinGlass/Bitget backtest service. Live trading remains disabled unless explicitly enabled elsewhere.",
)

ARTIFACT_ROOT = Path(os.getenv("ARTIFACT_ROOT", "artifacts/zeabur"))
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

_job_lock = threading.Lock()
_job: dict[str, Any] = {
    "state": "idle",
    "run_id": None,
    "started_at": None,
    "finished_at": None,
    "return_code": None,
    "output_dir": None,
    "log_file": None,
    "error": None,
}


class BacktestRequest(BaseModel):
    start: str | None = None
    end: str | None = None
    timeframe: str | None = None
    coinglass_exchange: str | None = None
    initial_equity: float | None = Field(default=None, gt=0)
    risk_per_trade: float | None = Field(default=None, gt=0, le=0.05)
    taker_fee_bps: float | None = Field(default=None, ge=0, le=100)
    slippage_bps: float | None = Field(default=None, ge=0, le=100)
    min_aligned_coverage: float | None = Field(default=None, ge=0.5, le=1.0)

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


def _require_admin(authorization: str | None = Header(default=None)) -> None:
    expected = os.getenv("ADMIN_BEARER_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=503, detail="ADMIN_BEARER_TOKEN is not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required")
    supplied = authorization[7:]
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=403, detail="Invalid bearer token")


def _safe_public_config() -> dict[str, Any]:
    return {
        "symbol": os.getenv("SYMBOL", "ETHUSDT"),
        "coinglass_symbol": os.getenv("COINGLASS_SYMBOL", "ETHUSDT"),
        "coinglass_exchange": os.getenv("COINGLASS_EXCHANGE", "Binance"),
        "timeframe": os.getenv("TIMEFRAME", "15m"),
        "start": os.getenv("START", "2025-01-01T00:00:00Z"),
        "end": os.getenv("END", "2026-01-01T00:00:00Z"),
        "live_trading_enabled": os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true",
        "coinglass_key_configured": bool(os.getenv("COINGLASS_API_KEY")),
        "admin_token_configured": bool(os.getenv("ADMIN_BEARER_TOKEN")),
    }


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "CoinGlass × Bitget Strategy Lab",
        "status": "running",
        "mode": "research",
        "live_trading_enabled": os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "utc": datetime.now(timezone.utc).isoformat()}


@app.get("/config")
def config() -> dict[str, Any]:
    return _safe_public_config()


@app.get("/strategies")
def strategies() -> dict[str, Any]:
    from coinlab.strategies import STRATEGIES

    return {"strategies": sorted(STRATEGIES.keys()), "count": len(STRATEGIES)}


def _run_backtest(run_id: str, output_dir: Path, log_file: Path, env: dict[str, str]) -> None:
    global _job
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
        with _job_lock:
            _job["state"] = "completed" if process.returncode == 0 else "failed"
            _job["return_code"] = process.returncode
            _job["finished_at"] = datetime.now(timezone.utc).isoformat()
            if process.returncode != 0:
                _job["error"] = "Backtest process failed; inspect log_tail via /backtest/status"
    except Exception as exc:  # service must survive a failed research run
        with _job_lock:
            _job["state"] = "failed"
            _job["finished_at"] = datetime.now(timezone.utc).isoformat()
            _job["error"] = f"{type(exc).__name__}: {exc}"


@app.post("/backtest/start", dependencies=[Depends(_require_admin)])
def start_backtest(req: BacktestRequest) -> dict[str, Any]:
    global _job
    with _job_lock:
        if _job["state"] == "running":
            raise HTTPException(status_code=409, detail="A backtest is already running")

        if not os.getenv("COINGLASS_API_KEY"):
            raise HTTPException(status_code=503, detail="COINGLASS_API_KEY is not configured")

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = ARTIFACT_ROOT / run_id
        log_file = output_dir / "run.log"
        env = os.environ.copy()
        overrides = {
            "START": req.start,
            "END": req.end,
            "TIMEFRAME": req.timeframe,
            "COINGLASS_EXCHANGE": req.coinglass_exchange,
            "INITIAL_EQUITY": req.initial_equity,
            "RISK_PER_TRADE": req.risk_per_trade,
            "TAKER_FEE_BPS": req.taker_fee_bps,
            "SLIPPAGE_BPS": req.slippage_bps,
            "MIN_ALIGNED_COVERAGE": req.min_aligned_coverage,
        }
        for key, value in overrides.items():
            if value is not None:
                env[key] = str(value)

        # This HTTP service is research-only. Never let a backtest request flip live trading on.
        env["LIVE_TRADING_ENABLED"] = "false"

        _job = {
            "state": "running",
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "return_code": None,
            "output_dir": str(output_dir),
            "log_file": str(log_file),
            "error": None,
        }
        thread = threading.Thread(
            target=_run_backtest,
            args=(run_id, output_dir, log_file, env),
            daemon=True,
            name=f"coinlab-backtest-{run_id}",
        )
        thread.start()
        return {"accepted": True, "run_id": run_id, "state": "running"}


@app.get("/backtest/status", dependencies=[Depends(_require_admin)])
def backtest_status() -> dict[str, Any]:
    with _job_lock:
        payload = dict(_job)
    log_path = Path(payload["log_file"]) if payload.get("log_file") else None
    if log_path and log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        payload["log_tail"] = lines[-80:]
    else:
        payload["log_tail"] = []
    report_path = Path(payload["output_dir"]) / "BACKTEST_REPORT.json" if payload.get("output_dir") else None
    payload["report_ready"] = bool(report_path and report_path.exists())
    return payload


@app.get("/backtest/report", dependencies=[Depends(_require_admin)])
def backtest_report() -> dict[str, Any]:
    with _job_lock:
        output_dir = _job.get("output_dir")
    if not output_dir:
        raise HTTPException(status_code=404, detail="No backtest has been run")
    report_path = Path(output_dir) / "BACKTEST_REPORT.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Backtest report is not ready")
    return json.loads(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))

"""
api.py — FastAPI REST layer for ZORO

Endpoints:
  GET  /                  health check
  GET  /signal            latest signal for all coins (or ?symbol=BNB-USD)
  GET  /trades            recent trade log (?symbol=BNB-USD&limit=50)
  GET  /backtest          P&L summary per coin
  GET  /status            bot status (uptime, open positions count)

Run standalone (dev):
  uvicorn api:app --reload --port 8000

In production this runs inside Docker alongside the bot.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

import pytz
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from zoro.database import fetch_pnl_summary, fetch_trades, init_db
from zoro.explainability import explain_trade
from zoro.data import fetch_crypto_data

app = FastAPI(
    title="ZORO Crypto Bot API",
    description="REST interface for the ZORO paper trading bot",
    version="2.0.0",
)

# Allow the Streamlit dashboard (any origin in dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

INDIA_TZ   = pytz.timezone("Asia/Kolkata")
_start_time = datetime.now(timezone.utc)

# Shared state written by trader.py (imported when running together)
# When running as standalone API, these stay empty — that's fine.
_latest_signals: dict = {}
_open_positions: dict = {}


# ── Helpers ────────────────────────────────────────────────────────────────

def _utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialise(obj):
    """Make datetime objects JSON-safe."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def _clean(rows: list[dict]) -> list[dict]:
    return [{k: _serialise(v) for k, v in row.items()} for row in rows]


# ── Startup ────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    init_db()


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "ZORO Crypto Bot API",
        "version": "2.0.0",
        "timestamp": _utc_now_str(),
    }


@app.get("/signal")
def get_signal(symbol: Optional[str] = Query(None, description="e.g. BNB-USD")):
    """
    Returns the most recent signal snapshot from the live bot.
    If the API is running separately from the bot this returns
    the latest data from the signals table instead.
    """
    if _latest_signals:
        if symbol:
            data = _latest_signals.get(symbol)
            if data is None:
                return {"error": f"No signal found for {symbol}"}
            return data
        return _latest_signals

    # Fallback: read from DB
    try:
        from sqlalchemy import text
        from zoro.database import get_engine
        engine = get_engine()
        with engine.connect() as conn:
            if symbol:
                sql = text("""
                    SELECT * FROM signals
                    WHERE symbol = :sym
                    ORDER BY timestamp DESC LIMIT 1
                """)
                rows = conn.execute(sql, {"sym": symbol}).mappings().all()
            else:
                sql = text("""
                    SELECT DISTINCT ON (symbol) *
                    FROM signals
                    ORDER BY symbol, timestamp DESC
                """)
                rows = conn.execute(sql).mappings().all()
        return _clean([dict(r) for r in rows])
    except Exception as e:
        return {"error": str(e), "hint": "DB may not be running"}


@app.get("/trades")
def get_trades(
    symbol: Optional[str] = Query(None, description="e.g. BNB-USD"),
    limit:  int           = Query(50, ge=1, le=500),
):
    """Recent trades from the PostgreSQL trade log."""
    rows = fetch_trades(symbol=symbol, limit=limit)
    return _clean(rows)


@app.get("/backtest")
def get_backtest():
    """
    P&L summary per coin: total_pnl, avg_pnl, win_rate_pct, trade_count.
    This is live paper trading data — not a historical simulation.
    """
    summary = fetch_pnl_summary()
    return {
        "generated_at": _utc_now_str(),
        "note": "Live paper trading P&L from PostgreSQL — not backtested simulation",
        "coins": _clean(summary),
    }


@app.get("/status")
def get_status():
    uptime_seconds = (datetime.now(timezone.utc) - _start_time).total_seconds()
    return {
        "status":          "running",
        "uptime_seconds":  round(uptime_seconds),
        "open_positions":  len(_open_positions),
        "coins_tracked":   list(_open_positions.keys()) if _open_positions else [],
        "timestamp":       _utc_now_str(),
    }


@app.get("/explain")
async def explain(symbol: str = "BNB-USD"):
    try:
        df = fetch_crypto_data(symbol)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        from zoro.signals import run_signal_engine
        result = run_signal_engine(symbol, df)
        if result is None:
            raise HTTPException(status_code=404, detail="No signal generated")
        explanation = explain_trade(symbol, df, result)
        return explanation
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
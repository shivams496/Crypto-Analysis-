"""
database.py — PostgreSQL connection and trade log persistence

All trade data goes here. Every trade is a row in the `trades` table.
SQLAlchemy core (no ORM) — keeps it simple and fast.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, DateTime, Float, Integer, MetaData, String,
    Table, Text, create_engine, text,
)
from sqlalchemy.engine import Engine

log = logging.getLogger(__name__)

# ── Engine (lazy singleton) ────────────────────────────────────────────────
_engine: Optional[Engine] = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = os.environ.get(
            "DATABASE_URL",
            "postgresql://zoro:zoro@localhost:5432/zorodb",
        )
        _engine = create_engine(url, pool_pre_ping=True, future=True)
    return _engine


# ── Schema ─────────────────────────────────────────────────────────────────
metadata = MetaData()

trades_table = Table(
    "trades",
    metadata,
    Column("id",             Integer, primary_key=True, autoincrement=True),
    Column("timestamp",      DateTime(timezone=True), nullable=False),
    Column("symbol",         String(20),  nullable=False),
    Column("action",         String(10),  nullable=False),   # LONG / SHORT / CLOSE
    Column("price",          Float,       nullable=False),
    Column("qty",            Float,       nullable=False),
    Column("position_value", Float,       nullable=True),
    Column("stop_loss",      Float,       nullable=True),
    Column("take_profit",    Float,       nullable=True),
    Column("pnl",            Float,       nullable=True),    # NULL until CLOSE
    Column("rsi",            Float,       nullable=True),
    Column("confidence",     Integer,     nullable=True),
    Column("direction",      String(10),  nullable=True),
    Column("rl_signal",      String(10),  nullable=True),
    Column("lstm_prob",      Float,       nullable=True),
    Column("coin",           String(10),  nullable=True),    # ETH / BNB etc
)

signals_table = Table(
    "signals",
    metadata,
    Column("id",         Integer, primary_key=True, autoincrement=True),
    Column("timestamp",  DateTime(timezone=True), nullable=False),
    Column("symbol",     String(20),  nullable=False),
    Column("direction",  String(10),  nullable=False),
    Column("confidence", Integer,     nullable=False),
    Column("rsi",        Float,       nullable=True),
    Column("lstm_prob",  Float,       nullable=True),
    Column("rl_signal",  String(10),  nullable=True),
    Column("price",      Float,       nullable=True),
    Column("actionable", String(5),   nullable=True),        # "true" / "false"
)


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    engine = get_engine()
    try:
        metadata.create_all(engine)
        log.info("[DB] Tables ready (trades, signals)")
    except Exception as e:
        log.warning(f"[DB] Could not create tables: {e} — running without DB")


def insert_trade(trade: dict) -> None:
    """Insert one row into the trades table. Silently skips on DB error."""
    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(trades_table.insert().values(**trade))
    except Exception as e:
        log.warning(f"[DB] insert_trade failed: {e}")


def insert_signal(signal: dict) -> None:
    """Insert one row into the signals table. Silently skips on DB error."""
    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(signals_table.insert().values(**signal))
    except Exception as e:
        log.warning(f"[DB] insert_signal failed: {e}")


def fetch_trades(symbol: str | None = None, limit: int = 100) -> list[dict]:
    """Return recent trades, optionally filtered by symbol."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            query = trades_table.select().order_by(
                trades_table.c.timestamp.desc()
            ).limit(limit)
            if symbol:
                query = query.where(trades_table.c.symbol == symbol)
            rows = conn.execute(query).mappings().all()
            return [dict(r) for r in rows]
    except Exception as e:
        log.warning(f"[DB] fetch_trades failed: {e}")
        return []


def fetch_pnl_summary() -> list[dict]:
    """P&L per coin — total pnl, trade count, win rate."""
    engine = get_engine()
    try:
        sql = text("""
            SELECT
                coin,
                COUNT(*)                                      AS total_trades,
                ROUND(SUM(pnl)::numeric, 4)                  AS total_pnl,
                ROUND(AVG(pnl)::numeric, 4)                  AS avg_pnl,
                ROUND(
                    100.0 * COUNT(*) FILTER (WHERE pnl > 0)
                    / NULLIF(COUNT(*) FILTER (WHERE pnl IS NOT NULL), 0),
                    1
                )                                             AS win_rate_pct
            FROM trades
            WHERE action = 'CLOSE'
            GROUP BY coin
            ORDER BY total_pnl DESC
        """)
        with engine.connect() as conn:
            rows = conn.execute(sql).mappings().all()
            return [dict(r) for r in rows]
    except Exception as e:
        log.warning(f"[DB] fetch_pnl_summary failed: {e}")
        return []

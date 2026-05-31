"""
trader.py — Paper trading loop (Phase 3 — PostgreSQL + FastAPI)

Run with:  python trader.py

No kernel-locking cells. No "NEVER press Run All" warnings.
Ctrl-C cleanly stops the loop and saves the trade log (JSON + PostgreSQL).
"""
from __future__ import annotations

import json
import logging
import os
import signal
import time
from datetime import datetime

# Suppress TensorFlow and OneDNN noise
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import pytz

from zoro.alerts import send_email, send_telegram, signal_message
from zoro.config import config
from zoro.data import fetch_crypto_data
from zoro.database import init_db, insert_signal, insert_trade
from zoro.signals import run_signal_engine

# Also expose these for the FastAPI module (imported together in Docker)
import api as _api

INDIA_TZ         = pytz.timezone(config.TIMEZONE)
_trade_log: list[dict] = []
_running         = True

TAKE_PROFIT_RATIO = 2.0


def _handle_interrupt(signum, frame):
    global _running
    print("\n[INFO] Stopping ZORO trader… saving log.")
    _running = False


def _save_log() -> None:
    path = "zoro_trade_log.json"
    with open(path, "w") as f:
        json.dump(_trade_log, f, indent=2, default=str)
    print(f"[INFO] Trade log saved → {path}  ({len(_trade_log)} entries)")


def _calc_pnl(side: str, entry: float, exit_: float, qty: float) -> float:
    if side == "SHORT":
        return (entry - exit_) * qty
    return (exit_ - entry) * qty


def _record_trade(result, action: str, qty: float,
                  entry_price: float | None = None,
                  side: str | None = None,
                  take_profit: float | None = None) -> None:
    """Append to in-memory log AND insert into PostgreSQL."""
    now = datetime.now(INDIA_TZ)
    pnl = None
    if action == "CLOSE" and entry_price is not None and side is not None:
        pnl = round(_calc_pnl(side, entry_price, result.price, qty), 4)

    entry: dict = {
        "timestamp":      now,
        "symbol":         result.symbol,
        "action":         action,
        "price":          result.price,
        "rsi":            result.rsi,
        "direction":      result.direction,
        "confidence":     result.confidence,
        "rl_signal":      result.rl_signal,
        "lstm_prob":      result.lstm_prob,
        "qty":            qty,
        "position_value": result.price * qty,
        "stop_loss":      result.stop_loss,
        "take_profit":    take_profit,
        "pnl":            pnl,
        "coin":           result.symbol.replace("-USD", ""),
    }

    _trade_log.append(entry)

    # ── Write to PostgreSQL ──────────────────────────────────────────────
    db_row = {k: v for k, v in entry.items()
              if k not in ("direction",)}  # direction already in signal table
    db_row["direction"] = result.direction
    insert_trade(db_row)


def _record_signal(result, now: datetime) -> None:
    """Log every scan result to the signals table + share with API."""
    row = {
        "timestamp":  now,
        "symbol":     result.symbol,
        "direction":  result.direction,
        "confidence": result.confidence,
        "rsi":        result.rsi,
        "lstm_prob":  result.lstm_prob,
        "rl_signal":  result.rl_signal,
        "price":      result.price,
        "actionable": str(result.actionable).lower(),
    }
    insert_signal(row)
    # Share live data with FastAPI (same process)
    _api._latest_signals[result.symbol] = {
        **row,
        "timestamp": now.isoformat(),
    }


def run_loop() -> None:
    global _running
    signal.signal(signal.SIGINT, _handle_interrupt)

    # Initialise DB (creates tables if missing)
    init_db()

    print("=" * 60)
    print("  ⚔️  ZORO Crypto Bot — Paper Trader  [Phase 3]")
    print(f"  Coins    : {', '.join(config.COINS)}")
    print(f"  Interval : {config.CHECK_INTERVAL_SEC}s")
    print(f"  Threshold: {config.SIGNAL_THRESHOLD}/100")
    print("  (Ctrl-C to stop cleanly)")
    print("=" * 60)

    open_positions: dict[str, dict] = {}
    # Share open positions with FastAPI
    _api._open_positions = open_positions

    while _running:
        now     = datetime.now(INDIA_TZ)
        now_str = now.strftime("%H:%M:%S IST")
        print(f"\n[{now_str}] Scanning {len(config.COINS)} coins…")

        for symbol in config.COINS:
            df = fetch_crypto_data(symbol)
            if df is None:
                continue

            result = run_signal_engine(symbol, df)
            if result is None:
                continue

            # Log every signal to DB + API
            _record_signal(result, now)

            emoji = {"LONG": "🟢", "SHORT": "🔴", "FLAT": "⏳"}[result.direction]
            print(
                f"  {symbol:<9} ${result.price:>10,.2f}  "
                f"RSI={result.rsi:5.1f}  conf={result.confidence}/100  {emoji}"
            )

            pos = open_positions.get(symbol)

            # ── Open new position ─────────────────────────────────────────
            if result.actionable and pos is None:
                qty       = config.POSITION_SIZE_USD / result.price
                stop_dist = abs(result.price - result.stop_loss)
                if result.direction == "SHORT":
                    take_profit = result.price - TAKE_PROFIT_RATIO * stop_dist
                else:
                    take_profit = result.price + TAKE_PROFIT_RATIO * stop_dist

                open_positions[symbol] = {
                    "entry_price": result.price,
                    "qty":         qty,
                    "side":        result.direction,
                    "stop_loss":   result.stop_loss,
                    "take_profit": take_profit,
                }
                _record_trade(result, result.direction, qty,
                              take_profit=take_profit)
                msg = signal_message(
                    symbol, result.direction, result.price,
                    result.confidence, result.gates, result.stop_loss,
                )
                print(
                    f"  → OPENED {result.direction}  qty={qty:.6f}  "
                    f"stop=${result.stop_loss:,.2f}  tp=${take_profit:,.2f}"
                )
                send_telegram(msg)
                send_email(f"ZORO {result.direction} — {symbol}", msg.replace("<b>","").replace("</b>",""))

            # ── Manage open position ──────────────────────────────────────
            elif pos is not None:
                entry_price = pos["entry_price"]
                qty         = pos["qty"]
                side        = pos["side"]
                stop_loss   = pos["stop_loss"]
                take_profit = pos["take_profit"]
                pnl         = _calc_pnl(side, entry_price, result.price, qty)

                stop_hit = (
                    (side == "LONG"  and result.price <= stop_loss) or
                    (side == "SHORT" and result.price >= stop_loss)
                )
                tp_hit = (
                    (side == "LONG"  and result.price >= take_profit) or
                    (side == "SHORT" and result.price <= take_profit)
                )

                if stop_hit:
                    print(f"  → STOP HIT  {symbol}  P&L=${pnl:+.2f}")
                    _record_trade(result, "CLOSE", qty, entry_price, side)
                    del open_positions[symbol]
                    send_email(f"ZORO STOP HIT — {symbol}", f"Stop loss hit on {symbol}\nEntry: ${entry_price:,.2f}\nExit: ${result.price:,.2f}\nP&L: ${pnl:+.2f}")
                elif tp_hit:
                    print(f"  → TAKE PROFIT {symbol}  P&L=${pnl:+.2f} ✅")
                    _record_trade(result, "CLOSE", qty, entry_price, side)
                    del open_positions[symbol]
                    send_email(f"ZORO TAKE PROFIT ✅ — {symbol}", f"Take profit hit on {symbol}\nEntry: ${entry_price:,.2f}\nExit: ${result.price:,.2f}\nP&L: ${pnl:+.2f}")
                else:
                    print(
                        f"    ↳ open {side}  entry=${entry_price:,.2f}  "
                        f"P&L=${pnl:+.2f}  tp=${take_profit:,.2f}"
                    )

        time.sleep(config.CHECK_INTERVAL_SEC)

    _save_log()


if __name__ == "__main__":
    run_loop()

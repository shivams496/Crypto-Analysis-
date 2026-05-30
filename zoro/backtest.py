"""
zoro/backtest.py — Backtesting engine with real metrics

Metrics computed
----------------
- Sharpe ratio (annualised, hourly data)
- Max drawdown %
- Average win / average loss
- Risk-reward ratio (avg_win / |avg_loss|)
- Win rate
- Total return %

Usage
-----
    from zoro.backtest import run_backtest, print_backtest_summary
    result = run_backtest("ETH-USD", strategy="rsi")
    print_backtest_summary(result)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf

from .config import config
from .data import compute_indicators

Strategy = Literal["rsi", "buy_hold", "rl"]


@dataclass
class BacktestResult:
    symbol:       str
    strategy:     str
    total_return: float    # e.g. 0.12 = +12 %
    sharpe:       float
    max_drawdown: float    # e.g. -0.25 = -25 %
    avg_win:      float
    avg_loss:     float
    rr_ratio:     float    # avg_win / |avg_loss|, want > 1.5
    win_rate:     float    # 0-1
    n_trades:     int
    trade_log:    list[dict] = field(default_factory=list)

    def flag(self) -> str:
        """Return a one-line health flag."""
        flags = []
        if self.sharpe < config.MIN_SHARPE:
            flags.append(f"Sharpe {self.sharpe:.2f} < {config.MIN_SHARPE} ⚠️")
        if self.rr_ratio < 1.0:
            flags.append(f"R:R {self.rr_ratio:.2f} < 1.0 ⚠️")
        if self.max_drawdown < -0.30:
            flags.append(f"Drawdown {self.max_drawdown:.1%} ⚠️")
        return " | ".join(flags) if flags else "OK ✅"


def _compute_metrics(trade_log: list[dict], equity_curve: pd.Series,
                     symbol: str, strategy: str) -> BacktestResult:
    """Crunch the numbers from raw trade log and equity curve."""
    returns = equity_curve.pct_change().dropna()

    # Sharpe (annualised, hourly → ×√(252×24))
    if returns.std() == 0:
        sharpe = 0.0
    else:
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252 * 24))

    # Max drawdown
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    max_dd   = float(drawdown.min())

    # Win / loss stats from trade log
    pnls = [t["pnl"] for t in trade_log if "pnl" in t and t["pnl"] != 0]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    avg_win  = float(np.mean(wins))   if wins   else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
    win_rate = len(wins) / len(pnls) if pnls else 0.0
    total_return = float((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1)

    return BacktestResult(
        symbol=symbol,
        strategy=strategy,
        total_return=total_return,
        sharpe=sharpe,
        max_drawdown=max_dd,
        avg_win=avg_win,
        avg_loss=avg_loss,
        rr_ratio=rr_ratio,
        win_rate=win_rate,
        n_trades=len(pnls),
        trade_log=trade_log,
    )


def _rsi_strategy(df: pd.DataFrame) -> tuple[list[dict], pd.Series]:
    """Simple RSI mean-reversion strategy."""
    capital    = 10_000.0
    position   = 0.0        # units held
    entry_price = 0.0
    equity     = []
    trade_log  = []

    for ts, row in df.iterrows():
        price = float(row["Close"])
        rsi   = float(row["RSI"]) if pd.notna(row["RSI"]) else 50.0

        if position == 0 and rsi < config.RSI_OVERSOLD:
            # BUY
            position    = capital / price
            entry_price = price

        elif position > 0 and rsi > config.RSI_OVERBOUGHT:
            # SELL
            pnl = (price - entry_price) * position
            trade_log.append({
                "timestamp": str(ts),
                "action": "SELL",
                "entry": entry_price,
                "exit": price,
                "qty": position,
                "position_value": price * position,
                "pnl": pnl,
            })
            capital  += pnl
            position  = 0.0

        equity.append(capital + position * price)

    # Close any open position at last bar
    if position > 0:
        last_price = float(df["Close"].iloc[-1])
        pnl = (last_price - entry_price) * position
        trade_log.append({
            "timestamp": str(df.index[-1]),
            "action": "CLOSE",
            "entry": entry_price,
            "exit": last_price,
            "qty": position,
            "position_value": last_price * position,
            "pnl": pnl,
        })
        capital += pnl

    return trade_log, pd.Series(equity, index=df.index)


def _buy_hold_strategy(df: pd.DataFrame) -> tuple[list[dict], pd.Series]:
    """Baseline: buy on day 1, hold to end."""
    capital    = 10_000.0
    entry_price = float(df["Close"].iloc[0])
    position   = capital / entry_price
    equity     = df["Close"] * position
    last_price = float(df["Close"].iloc[-1])
    pnl        = (last_price - entry_price) * position
    trade_log  = [
        {"timestamp": str(df.index[0]),  "action": "BUY",  "entry": entry_price,
         "exit": None, "qty": position, "position_value": capital, "pnl": 0},
        {"timestamp": str(df.index[-1]), "action": "SELL", "entry": entry_price,
         "exit": last_price, "qty": position,
         "position_value": last_price * position, "pnl": pnl},
    ]
    return trade_log, equity.astype(float)


def run_backtest(symbol: str, strategy: Strategy = "rsi") -> BacktestResult | None:
    """
    Download data and run *strategy* on *symbol*.

    Returns None if data download fails.
    """
    try:
        df = yf.download(
            symbol,
            period=config.BACKTEST_PERIOD,
            interval=config.BACKTEST_INTERVAL,
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        print(f"[WARN] backtest.run_backtest({symbol}): download failed — {e}")
        return None

    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        print(f"[WARN] backtest.run_backtest({symbol}): empty data")
        return None

    df = compute_indicators(df)
    df.dropna(inplace=True)

    if strategy == "rsi":
        trade_log, equity = _rsi_strategy(df)
    elif strategy == "buy_hold":
        trade_log, equity = _buy_hold_strategy(df)
    else:
        print(f"[WARN] backtest: unknown strategy '{strategy}'")
        return None

    return _compute_metrics(trade_log, equity, symbol, strategy)


def print_backtest_summary(result: BacktestResult) -> None:
    """Print a formatted backtest summary table."""
    w = 55
    print("=" * w)
    print(f"  BACKTEST — {result.symbol}  [{result.strategy.upper()}]")
    print("=" * w)
    print(f"  Total Return   : {result.total_return:>+8.1%}")
    print(f"  Sharpe Ratio   : {result.sharpe:>8.2f}")
    print(f"  Max Drawdown   : {result.max_drawdown:>8.1%}")
    print(f"  Win Rate       : {result.win_rate:>8.1%}  ({result.n_trades} trades)")
    print(f"  Avg Win        : {result.avg_win:>+8.2f}")
    print(f"  Avg Loss       : {result.avg_loss:>+8.2f}")
    print(f"  Risk/Reward    : {result.rr_ratio:>8.2f}  (want > 1.5)")
    print(f"  Health         : {result.flag()}")
    print("=" * w)

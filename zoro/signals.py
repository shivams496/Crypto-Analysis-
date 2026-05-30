"""
zoro/signals.py — 7-Gate signal engine (Phase 2 Step 3: FinBERT sentiment)

Gates
-----
1. RSI (25 / 75 thresholds)
2. SMC/ICT structure  (price vs SMA20 vs SMA50)
3. MACD histogram direction
4. Bollinger Band position
5. LSTM probability
6. Sentiment score  ← now FinBERT (was VADER)
7. RL PPO agent

Change vs Phase 1
-----------------
- Gate 6 now calls sentiment.get_sentiment_score() which uses
  ProsusAI/finbert instead of VADER
- sentiment_score parameter still accepted for manual override
  (useful for testing or if you want to inject a custom score)
- Everything else is identical — same SignalResult, same gates,
  same confidence math, same stop-loss formula
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from .config import config
from .data import latest_scalars
from .lstm_model import get_lstm_prob
from .rl_agent import get_rl_signal
from .sentiment import get_sentiment_score   # ← FinBERT, replaces VADER

Direction = Literal["LONG", "SHORT", "FLAT"]


@dataclass
class SignalResult:
    symbol:     str
    price:      float
    rsi:        float
    direction:  Direction
    confidence: int
    gates:      list[str] = field(default_factory=list)
    atr:        float = 0.0
    stop_loss:  float = 0.0
    rl_signal:  str   = "HOLD"
    lstm_prob:  float = 0.528
    sentiment:  float = 0.0    # ← new: logged for analysis

    @property
    def actionable(self) -> bool:
        return self.confidence >= config.SIGNAL_THRESHOLD and self.direction != "FLAT"


def run_signal_engine(symbol: str, df: pd.DataFrame,
                      sentiment_score: float | None = None) -> SignalResult | None:
    """
    Run the 7-gate signal engine on the latest bar of *df*.

    Parameters
    ----------
    symbol          : coin ticker e.g. "ETH-USD"
    df              : DataFrame from data.fetch_crypto_data()
    sentiment_score : float in [-1, 1] or None.
                      If None (default), FinBERT fetches live headlines.
                      Pass a float to override (useful for testing).

    Returns None if df is invalid.
    """
    if df is None or df.empty:
        return None

    row = df.iloc[-1]

    price   = float(row["Close"])
    rsi     = float(row["RSI"])
    macd_h  = float(row["MACD_Hist"])
    bb_pos  = float(row["BB_Position"])   # 0 = lower band, 1 = upper
    sma20   = float(row["SMA_20"])
    sma50   = float(row["SMA_50"])
    atr     = float(row["ATR"])

    scalars   = latest_scalars(df)
    lstm_prob = get_lstm_prob(df["Close"])

    # ── Gate 6 data: FinBERT sentiment ────────────────────────────────────
    # Fetched once per 5 min (cached), so all 5 coins share the same score
    # per scan cycle — this is correct since headlines affect all coins.
    if sentiment_score is None:
        sentiment_score = get_sentiment_score()   # FinBERT live fetch

    rl_signal = get_rl_signal(symbol, scalars, lstm_prob, sentiment_score)

    confidence = 50
    direction: Direction = "FLAT"
    gates: list[str] = []

    # ── Gate 1: RSI ───────────────────────────────────────────────────────
    if rsi < config.RSI_OVERSOLD:
        confidence += 20
        direction = "LONG"
        gates.append(f"RSI={rsi:.1f} oversold")
    elif rsi > config.RSI_OVERBOUGHT:
        confidence += 20
        direction = "SHORT"
        gates.append(f"RSI={rsi:.1f} overbought")

    # ── Gate 2: SMC/ICT structure ─────────────────────────────────────────
    if price > sma20 > sma50:
        if direction == "LONG":
            confidence += 10
        elif direction == "SHORT":
            confidence -= 5
        gates.append("SMC bullish structure")
    elif price < sma20 < sma50:
        if direction == "SHORT":
            confidence += 10
        elif direction == "LONG":
            confidence -= 5
        gates.append("SMC bearish structure")

    # ── Gate 3: MACD histogram ────────────────────────────────────────────
    if macd_h > 0 and direction == "LONG":
        confidence += 10
        gates.append("MACD bullish")
    elif macd_h < 0 and direction == "SHORT":
        confidence += 10
        gates.append("MACD bearish")

    # ── Gate 4: Bollinger Bands ───────────────────────────────────────────
    if bb_pos <= 0.1 and direction == "LONG":
        confidence += 10
        gates.append(f"BB oversold ({bb_pos:.2f})")
    elif bb_pos >= 0.9 and direction == "SHORT":
        confidence += 10
        gates.append(f"BB overbought ({bb_pos:.2f})")

    # ── Gate 5: LSTM ──────────────────────────────────────────────────────
    if lstm_prob > 0.60 and direction == "LONG":
        confidence += 20
        gates.append(f"LSTM {lstm_prob:.0%} bullish")
    elif lstm_prob < 0.40 and direction == "SHORT":
        confidence += 20
        gates.append(f"LSTM {lstm_prob:.0%} bearish")

    # ── Gate 6: FinBERT Sentiment ─────────────────────────────────────────
    # sentiment_score is in [-1, 1]
    # Scale to ±10 confidence points max
    sent_pts = int(sentiment_score * 10)
    confidence += sent_pts
    if sent_pts != 0:
        label = "bullish" if sentiment_score > 0.15 else \
                "bearish" if sentiment_score < -0.15 else "neutral"
        gates.append(f"FinBERT {label} ({sentiment_score:+.2f}) {sent_pts:+d}pts")

    # ── Gate 7: RL agent ──────────────────────────────────────────────────
    if rl_signal == "BUY" and direction == "LONG":
        confidence += 10
        gates.append("RL BUY")
    elif rl_signal == "SELL" and direction == "SHORT":
        confidence += 10
        gates.append("RL SELL")
    elif rl_signal != "HOLD":
        confidence -= 5
        gates.append(f"RL {rl_signal} (disagrees)")

    confidence = max(0, min(confidence, 100))

    # ── Stop-loss ─────────────────────────────────────────────────────────
    if direction == "LONG":
        stop_loss = price - config.ATR_STOP_MULTIPLIER * atr
    elif direction == "SHORT":
        stop_loss = price + config.ATR_STOP_MULTIPLIER * atr
    else:
        stop_loss = 0.0

    return SignalResult(
        symbol=symbol,
        price=price,
        rsi=rsi,
        direction=direction,
        confidence=confidence,
        gates=gates,
        atr=atr,
        stop_loss=stop_loss,
        rl_signal=rl_signal,
        lstm_prob=lstm_prob,
        sentiment=sentiment_score,
    )

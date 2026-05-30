"""
zoro/data.py — Data fetching and technical indicator computation

Functions
---------
fetch_crypto_data(symbol) → pd.DataFrame | None
compute_indicators(df)    → pd.DataFrame
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import yfinance as yf

from .config import config

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Public API ─────────────────────────────────────────────────────────────

def fetch_crypto_data(symbol: str) -> pd.DataFrame | None:
    """
    Download OHLCV data for *symbol* and attach all technical indicators.
    Returns None if download fails or data is empty.
    """
    try:
        df = yf.download(
            symbol,
            period=config.PERIOD,
            interval=config.INTERVAL,
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        print(f"[WARN] fetch_crypto_data({symbol}): download failed — {e}")
        return None

    # yfinance sometimes returns a MultiIndex on multi-ticker downloads
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        print(f"[WARN] fetch_crypto_data({symbol}): empty dataframe returned")
        return None

    df = compute_indicators(df)
    df.dropna(inplace=True)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RSI, MACD, Bollinger Bands, ATR, SMAs, Momentum, Volume_Ratio
    to a OHLCV dataframe.  All columns added in-place; original df returned.
    """
    c = df["Close"]
    v = df["Volume"]

    # RSI (14-period)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD (12 / 26 / 9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"]      = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger Bands (20-period, 2σ)
    df["BB_Middle"] = c.rolling(20).mean()
    bb_std          = c.rolling(20).std()
    df["BB_Upper"]  = df["BB_Middle"] + 2 * bb_std
    df["BB_Lower"]  = df["BB_Middle"] - 2 * bb_std
    df["BB_Width"]  = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    # 0 = at lower band, 1 = at upper band
    df["BB_Position"] = (c - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-9)

    # ATR (14-period)
    h, l, pc = df["High"], df["Low"], c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # SMAs
    df["SMA_20"] = c.rolling(20).mean()
    df["SMA_50"] = c.rolling(50).mean()

    # Momentum (12-period rate of change)
    df["Momentum"] = c.pct_change(12)

    # Volume vs 20-period average
    vol_avg = v.rolling(20).mean()
    df["Volume_Ratio"] = v / vol_avg.replace(0, np.nan)

    # Daily return
    df["Daily_Return"] = c.pct_change()

    return df


# ── Scalar helpers used by rl_agent.py ────────────────────────────────────

def latest_scalars(df: pd.DataFrame) -> dict:
    """
    Return the most recent bar as a dict of normalised floats for the
    RL observation vector.  All values clipped to [-3, 3] after normalisation.
    """
    row = df.iloc[-1]
    price = float(row["Close"])

    def safe(val, default=0.0):
        v = float(val) if pd.notna(val) else default
        return float(np.clip(v, -3.0, 3.0))

    return {
        "rsi_norm":      safe((row["RSI"] - 50) / 50),
        "macd_norm":     safe(row["MACD"] / price),
        "bb_position":   safe(row["BB_Position"]),           # 0-1
        "trend":         safe(1.0 if row["SMA_20"] > row["SMA_50"] else -1.0),
        "volume_ratio":  safe(row["Volume_Ratio"] - 1.0),   # 0 = average
        "momentum_norm": safe(row["Momentum"] / price),
        "daily_return":  safe(row["Daily_Return"]),
        "atr_norm":      safe(row["ATR"] / price),
    }

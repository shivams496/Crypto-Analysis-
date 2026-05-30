"""
zoro/config.py — Central configuration for ZORO Crypto Bot
All constants live here. Nothing else should define magic numbers.
"""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── Coins ──────────────────────────────────────────────────────────────
    COINS: dict[str, str] = {
        "ETH-USD": "ETHUSDT",
        "BTC-USD": "BTCUSDT",
        "SOL-USD": "SOLUSDT",
        "BNB-USD": "BNBUSDT",
        "ADA-USD": "ADAUSDT",
    }
    PRIMARY_COIN = "ETH-USD"

    # ── Data fetch ─────────────────────────────────────────────────────────
    PERIOD   = "30d"     # reduced from 180d — faster download, still enough for indicators
    INTERVAL = "1h"

    # ── Signal engine ──────────────────────────────────────────────────────
    RSI_OVERSOLD   = 25
    RSI_OVERBOUGHT = 75
    SIGNAL_THRESHOLD = 70

    # ── Risk management ────────────────────────────────────────────────────
    ATR_STOP_MULTIPLIER   = 1.5
    ATR_TRAIL_MULTIPLIER  = 2.0
    POSITION_SIZE_USD     = 100.0

    # ── LSTM ───────────────────────────────────────────────────────────────
    LSTM_MODEL_PATH   = "lstm_model.h5"
    LSTM_SEQUENCE_LEN = 60
    LSTM_FEATURES     = [
        "Close", "RSI", "MACD", "BB_Width", "ATR", "SMA_20", "Volume_Ratio"
    ]

    # ── RL agent ───────────────────────────────────────────────────────────
    RL_AGENT_PATH = "zoro_ppo_agent"
    RL_OBS_SIZE   = 10

    # ── Backtest ───────────────────────────────────────────────────────────
    BACKTEST_PERIOD   = "1y"
    BACKTEST_INTERVAL = "1h"
    MIN_SHARPE        = 0.5

    # ── Monitoring loop ────────────────────────────────────────────────────
    CHECK_INTERVAL_SEC = 60    # 1 minute between scans
    TIMEZONE = "Asia/Kolkata"

    # ── Credentials (from .env) ────────────────────────────────────────────
    TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
    GMAIL_ADDRESS    = os.getenv("GMAIL_ADDRESS", "")
    GMAIL_APP_PW     = os.getenv("GMAIL_APP_PASSWORD", "")
    BINANCE_API_KEY  = os.getenv("BINANCE_TESTNET_API_KEY", "")
    BINANCE_SECRET   = os.getenv("BINANCE_TESTNET_SECRET", "")


# Module-level singleton — import this everywhere
config = Config()

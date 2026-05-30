# ZORO — RL Crypto Trading Bot

> Paper trading bot using Reinforcement Learning + LSTM + FinBERT sentiment. Built from scratch, documented honestly.

**[📊 Live Performance Dashboard](https://huggingface.co/spaces/shivams496/Zoro-crypto-bot)**

---

## What it does

ZORO scans 5 crypto coins (ETH, BTC, SOL, BNB, ADA) every 60 seconds and fires SHORT/LONG signals when a 7-gate confidence engine crosses 70/100. Every trade is logged to PostgreSQL and explained via SHAP feature importance.

## Architecture

```
zoro_v2/
├── zoro/
│   ├── config.py          # All constants in one place
│   ├── data.py            # fetch_crypto_data() + compute_indicators()
│   ├── signals.py         # 7-gate signal engine
│   ├── lstm_model.py      # LSTM wrapper (7 real features, no zeros)
│   ├── rl_agent.py        # PPO wrapper
│   ├── sentiment.py       # FinBERT (replaces VADER)
│   ├── explainability.py  # SHAP on LSTM
│   ├── backtest.py        # Sharpe, max drawdown, avg win/loss
│   └── alerts.py          # Telegram + Email
├── trader.py              # Main loop — python trader.py
├── api.py                 # FastAPI — /signal /trades /backtest /explain
├── docker-compose.yml     # PostgreSQL + bot in one command
└── requirements.txt
```

## Honest results

| Metric | Value |
|---|---|
| LSTM accuracy (walk-forward) | 54% — consistent across 3 windows |
| RL ep_rew_mean | −146 → +8 (genuine learning) |
| RSI Strategy backtest | −40.4% return |
| Buy & Hold backtest | −19.8% return |
| Signal threshold | 70/100 confidence |

The original notebook showed 64.8% win rate — that was fake (lookahead bias in MinMaxScaler). The real number is 54%, which is a genuine directional edge.

## Tech stack

- **RL**: Stable Baselines3 PPO, custom OpenAI Gym environment, Sharpe ratio reward
- **LSTM**: TensorFlow/Keras, 7 features (MACD, BB, RSI, volume, trend, momentum, ATR), walk-forward validation
- **Sentiment**: FinBERT (fine-tuned on financial text), replaces VADER
- **Backend**: FastAPI + PostgreSQL (Docker)
- **Explainability**: SHAP GradientExplainer on LSTM

## Phases completed

- ✅ Phase 0 — Clean repo, honest README, .gitignore, .env.example
- ✅ Phase 1 — Proper Python package, real RL obs vector (7 features, no zeros), real backtest metrics
- ✅ Phase 2 — LSTM retrained (no lookahead bias), RL retrained on 5 coins (300k steps), FinBERT sentiment
- ✅ Phase 3 — Docker + PostgreSQL + FastAPI (5 endpoints live)
- ✅ Phase 4 — SHAP explainability, live performance page, HuggingFace deployment

## Setup

```bash
git clone https://github.com/shivams496/ZORO-Crypto-Bot
cd ZORO-Crypto-Bot
cp .env.example .env        # add your API keys
pip install -r requirements.txt

# Start DB
docker-compose up db -d

# Run bot
python trader.py

# Run API
uvicorn api:app --reload --port 8000
```

## Known limitations

- Paper trading only — no real money
- 1 trade so far (BNB SHORT, bot needs more runtime to collect signals across all 5 coins)
- Walk-forward Sharpe values inflated in training reports (display bug in gym_env.py reward scaling) — agent behaviour is correct

---

*Built as a learning project. Honest results beat cherry-picked numbers.*

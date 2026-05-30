"""
zoro/rl_agent.py — PPO RL agent wrapper (Phase 2 Step 2 update)

Changes vs Phase 1
------------------
- Agent now trained with real 10-feature obs (was 7 zeros before)
- Reward was Sharpe ratio during training (not raw P&L)
- lstm_prob centred correctly: lstm_prob - 0.5
- sentiment_score wired in (ready for FinBERT in Phase 2 Step 3)
- Fallback is more robust: uses MACD crossover + RSI, not RSI alone
"""
from __future__ import annotations

import numpy as np
from .config import config

_rl_agent = None      # stable-baselines3 PPO model, loaded once


def _load_agent():
    global _rl_agent
    if _rl_agent is not None:
        return _rl_agent
    try:
        import os
        agent_zip = config.RL_AGENT_PATH + ".zip"
        if not os.path.exists(agent_zip):
            print(f"[WARN] rl_agent: {agent_zip} not found — RL signals disabled")
            return None
        from stable_baselines3 import PPO
        _rl_agent = PPO.load(config.RL_AGENT_PATH)
        print(f"[INFO] RL agent loaded from {agent_zip}")
        return _rl_agent
    except ImportError:
        print("[WARN] rl_agent: stable-baselines3 not installed — pip install stable-baselines3")
        return None
    except Exception as e:
        print(f"[WARN] rl_agent: load failed ({e})")
        return None


def build_obs_vector(scalars: dict, lstm_prob: float, sentiment_score: float = 0.0) -> np.ndarray:
    """
    Build the 10-element observation vector for the PPO agent.

    Element order must stay IDENTICAL to gym_env.py _get_obs() —
    any change here must be mirrored there and requires a retrain.

    Index  Feature          Source
    -----  ---------------  ------------------------------------------
    0      rsi_norm         (RSI - 50) / 50
    1      macd_norm        MACD / price
    2      bb_position      (price - bb_lower) / (bb_upper - bb_lower)
    3      trend            +1 if SMA20 > SMA50 else -1
    4      volume_ratio     (volume / vol_mean) - 1
    5      sentiment        FinBERT score in [-1, 1]  (0 = neutral)
    6      momentum_norm    momentum / price
    7      lstm_centred     lstm_prob - 0.5
    8      daily_return     price_now / price_24h_ago - 1
    9      atr_norm         ATR / price

    Parameters
    ----------
    scalars         : dict from data.latest_scalars()
    lstm_prob       : float from lstm_model.get_lstm_prob()
    sentiment_score : float normalised to [-1, 1], default 0.0 (neutral)
                      Phase 2 Step 3 will feed real FinBERT values here.

    Returns
    -------
    np.ndarray shape (10,) dtype float32
    """
    obs = np.array([
        scalars.get("rsi_norm",      0.0),   # 0
        scalars.get("macd_norm",     0.0),   # 1
        scalars.get("bb_position",   0.5),   # 2
        scalars.get("trend",         0.0),   # 3
        scalars.get("volume_ratio",  0.0),   # 4
        float(np.clip(sentiment_score, -1.0, 1.0)),  # 5
        scalars.get("momentum_norm", 0.0),   # 6
        float(lstm_prob) - 0.5,              # 7  ← centred around 0
        scalars.get("daily_return",  0.0),   # 8
        scalars.get("atr_norm",      0.0),   # 9
    ], dtype=np.float32)

    return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)


def get_rl_signal(symbol: str, scalars: dict, lstm_prob: float,
                  sentiment_score: float = 0.0) -> str:
    """
    Return BUY / SELL / HOLD from the PPO agent.

    Falls back to MACD + RSI rule if agent is unavailable.

    Parameters
    ----------
    symbol          : coin ticker (for logging only)
    scalars         : dict from data.latest_scalars()
    lstm_prob       : float from lstm_model.get_lstm_prob()
    sentiment_score : optional FinBERT float in [-1, 1]
    """
    agent = _load_agent()

    if agent is None:
        # Improved fallback: MACD direction + RSI threshold
        rsi        = scalars.get("rsi_norm",  0.0) * 50 + 50
        macd_norm  = scalars.get("macd_norm", 0.0)
        if rsi < config.RSI_OVERSOLD  and macd_norm > 0:
            return "BUY"
        elif rsi > config.RSI_OVERBOUGHT and macd_norm < 0:
            return "SELL"
        return "HOLD"

    obs = build_obs_vector(scalars, lstm_prob, sentiment_score)
    try:
        action, _ = agent.predict(obs, deterministic=True)
        signal = {0: "HOLD", 1: "BUY", 2: "SELL"}.get(int(action), "HOLD")
        return signal
    except Exception as e:
        print(f"[WARN] rl_agent.get_rl_signal({symbol}): {e}")
        return "HOLD"


# Pre-load on import so the first scan has no latency
_load_agent()

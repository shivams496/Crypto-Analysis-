"""
gym_env.py — Custom OpenAI Gym environment for ZORO RL retraining

Observation space : 10 features (matches rl_agent.py build_obs_vector exactly)
Action space      : Discrete(3) — 0=HOLD, 1=BUY, 2=SELL
Reward            : Sharpe-based, not raw P&L

Works for all 5 coins — the environment is coin-agnostic.
The train script feeds it different DataFrames per episode.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class ZoroCryptoEnv(gym.Env):
    """
    Single-asset crypto paper trading environment.

    Episode  = one window of `window_size` hourly candles.
    Obs      = 10-element float32 vector (mirrors rl_agent.build_obs_vector)
    Action   = 0 HOLD | 1 BUY | 2 SELL
    Reward   = Sharpe ratio of trade returns (0 between trades)

    Parameters
    ----------
    df          : DataFrame with columns: close, rsi, macd, bb_upper,
                  bb_lower, volume, sma20, sma50, atr, momentum
    window_size : candles per episode
    initial_cash: starting paper cash
    fee_rate    : commission per trade (0.001 = 0.1%)
    """

    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, window_size: int = 500,
                 initial_cash: float = 1000.0, fee_rate: float = 0.001):
        super().__init__()
        self.df          = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_cash= initial_cash
        self.fee_rate    = fee_rate
        self._vol_mean   = float(df["volume"].mean()) or 1.0

        # Must match build_obs_vector in rl_agent.py — 10 elements
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)   # 0=HOLD 1=BUY 2=SELL
        self._reset_state()

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action: int):
        price  = float(self.df.loc[self.current_step, "close"])
        reward = 0.0

        if action == 1 and self.position == 0:          # BUY → open long
            qty           = (self.cash * (1 - self.fee_rate)) / price
            self.position = qty
            self.entry_price = price
            self.cash     = 0.0

        elif action == 2 and self.position > 0:         # SELL → close long
            proceeds      = self.position * price * (1 - self.fee_rate)
            ret           = (proceeds - self.position * self.entry_price) / \
                            (self.position * self.entry_price)
            self.cash     = proceeds
            self.trade_returns.append(ret)
            reward        = self._sharpe_reward()
            self.position = 0.0
            self.entry_price = 0.0

        equity = self.cash + self.position * price
        self.equity_curve.append(equity)

        self.current_step += 1
        done = self.current_step >= self.start_step + self.window_size - 1

        return self._get_obs(), float(reward), done, False, {"equity": equity}

    # ── Internals ─────────────────────────────────────────────────────────────

    def _reset_state(self):
        max_start = max(0, len(self.df) - self.window_size - 1)
        self.start_step   = int(np.random.randint(0, max_start + 1)) if max_start > 0 else 0
        self.current_step = self.start_step
        self.cash         = self.initial_cash
        self.position     = 0.0
        self.entry_price  = 0.0
        self.trade_returns: list[float] = []
        self.equity_curve : list[float] = [self.initial_cash]

    def _get_obs(self) -> np.ndarray:
        """
        10-element obs — MUST stay in sync with rl_agent.py build_obs_vector().
        Index mapping:
          0  rsi_norm        (RSI-50)/50
          1  macd_norm       MACD/price
          2  bb_position     (price-bb_lower)/(bb_upper-bb_lower)
          3  trend           +1 SMA20>SMA50 else -1
          4  volume_ratio    vol/vol_mean - 1
          5  sentiment       0.0 (no live feed in env)
          6  momentum_norm   momentum/price
          7  lstm_centred    0.0 (not available in env; rl_agent feeds real value)
          8  daily_return    price_now/price_24h_ago - 1
          9  atr_norm        ATR/price
        """
        idx   = min(self.current_step, len(self.df) - 1)
        row   = self.df.loc[idx]
        price = float(row["close"]) or 1.0

        rsi      = float(row.get("rsi",      50.0))
        macd     = float(row.get("macd",      0.0))
        bb_upper = float(row.get("bb_upper", price))
        bb_lower = float(row.get("bb_lower", price))
        volume   = float(row.get("volume",    1.0))
        sma20    = float(row.get("sma20",    price))
        sma50    = float(row.get("sma50",    price))
        atr      = float(row.get("atr",       0.0))
        momentum = float(row.get("momentum",  0.0))

        bb_range   = (bb_upper - bb_lower) or 1.0
        prev_idx   = max(0, idx - 24)
        prev_price = float(self.df.loc[prev_idx, "close"]) or price

        obs = np.array([
            (rsi - 50.0) / 50.0,                      # 0 rsi_norm
            macd / price,                              # 1 macd_norm
            (price - bb_lower) / bb_range,             # 2 bb_position
            1.0 if sma20 > sma50 else -1.0,            # 3 trend
            (volume / self._vol_mean) - 1.0,           # 4 volume_ratio
            0.0,                                       # 5 sentiment (neutral)
            momentum / price,                          # 6 momentum_norm
            0.0,                                       # 7 lstm (not in env)
            (price / prev_price) - 1.0,                # 8 daily_return
            atr / price,                               # 9 atr_norm
        ], dtype=np.float32)

        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def _sharpe_reward(self) -> float:
        """Annualised Sharpe of all trade returns. Returns last return if < 2 trades."""
        if len(self.trade_returns) < 2:
            return float(self.trade_returns[-1]) if self.trade_returns else 0.0
        r = np.array(self.trade_returns)
        sharpe = (r.mean() / (r.std() + 1e-8)) * np.sqrt(252 * 24)
        return float(np.clip(sharpe, -3.0, 3.0))

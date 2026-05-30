"""
train_rl_agent.py — Phase 2 Step 2: Retrain the PPO RL agent
Trains on ALL 5 coins combined: ETH-USD, BTC-USD, SOL-USD, BNB-USD, ADA-USD

What this fixes vs the original zoro_ppo_agent.zip
---------------------------------------------------
- Old agent trained with 7 hardcoded zeros → blind to most signals
- New agent sees real 10-feature obs matching rl_agent.py exactly
- Trained on all 5 coins → generalises across volatility regimes
- Reward = Sharpe ratio, not raw P&L → penalises high-variance strategies
- Walk-forward validation on each coin separately to confirm no overfit

Usage
-----
    python train_rl_agent.py

Outputs (all saved to zoro_v2/ root)
--------------------------------------
    zoro_ppo_agent.zip      ← replaces old agent; trader.py picks up automatically
    rl_train_report.txt     ← honest metrics per coin; share to verify
"""
from __future__ import annotations

import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_OUT  = os.path.join(SCRIPT_DIR, "zoro_ppo_agent")
REPORT_OUT = os.path.join(SCRIPT_DIR, "rl_train_report.txt")

# ── config ────────────────────────────────────────────────────────────────────
SYMBOLS      = ["ETH-USD", "BTC-USD", "SOL-USD", "BNB-USD", "ADA-USD"]
PERIOD       = "360d"
INTERVAL     = "1h"
TOTAL_STEPS  = 300_000    # ~8-15 min on CPU; 5 coins need more steps than 1
WINDOW_SIZE  = 500
LEARNING_RATE= 3e-4
N_ENVS       = 4

# Walk-forward splits (chronological, no leakage)
WF_SPLITS = [
    {"train": (0.00, 0.50), "test": (0.50, 0.65)},
    {"train": (0.00, 0.65), "test": (0.65, 0.80)},
    {"train": (0.00, 0.80), "test": (0.80, 1.00)},
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_and_prepare(symbol: str) -> pd.DataFrame:
    """Download OHLCV for one symbol and compute all indicator columns."""
    import yfinance as yf
    print(f"  Downloading {symbol}...")
    raw = yf.download(symbol, period=PERIOD, interval=INTERVAL,
                      auto_adjust=True, progress=False)
    if raw.empty:
        print(f"  [WARN] No data for {symbol}, skipping")
        return pd.DataFrame()

    df = raw[["Open","High","Low","Close","Volume"]].copy()
    df.columns = ["open","high","low","close","volume"]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    c = df["close"]

    # RSI-14
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-8))

    # MACD
    df["macd"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()

    # Bollinger Bands
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["sma20"]    = sma20
    df["sma50"]    = c.rolling(50).mean()

    # ATR-14
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - c.shift()).abs(),
        (df["low"]  - c.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # Momentum (10-period)
    df["momentum"] = c.diff(10)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["symbol"] = symbol   # tag for multi-coin env
    return df


def fetch_all_coins() -> dict[str, pd.DataFrame]:
    """Fetch and prepare data for all 5 coins."""
    print(f"[INFO] Fetching data for {len(SYMBOLS)} coins ({PERIOD}, {INTERVAL})...")
    data = {}
    for sym in SYMBOLS:
        df = fetch_and_prepare(sym)
        if not df.empty:
            data[sym] = df
            print(f"  {sym}: {len(df)} candles")
    print(f"[INFO] Loaded {len(data)}/{len(SYMBOLS)} coins successfully")
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MULTI-COIN VEC ENV FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def make_multi_env(coin_data: dict[str, pd.DataFrame], n_envs: int):
    """
    Create a vectorised env that randomly samples from all coins each episode.
    This forces the agent to generalise rather than memorise one coin.
    """
    from stable_baselines3.common.env_util import make_vec_env
    sys.path.insert(0, SCRIPT_DIR)
    from gym_env import ZoroCryptoEnv

    symbols = list(coin_data.keys())

    def env_fn():
        # Each reset picks a random coin — agent sees all 5 during training
        sym = symbols[np.random.randint(len(symbols))]
        return ZoroCryptoEnv(coin_data[sym], window_size=WINDOW_SIZE)

    return make_vec_env(env_fn, n_envs=n_envs)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_on_coin(model, df: pd.DataFrame, n_episodes: int = 5) -> dict:
    """Evaluate agent on a single coin's test slice."""
    sys.path.insert(0, SCRIPT_DIR)
    from gym_env import ZoroCryptoEnv

    env = ZoroCryptoEnv(df, window_size=min(WINDOW_SIZE, len(df) - 1))
    all_returns, win_count, trade_count = [], 0, 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(int(action))
            if reward != 0.0:
                all_returns.append(reward)
                win_count   += 1 if reward > 0 else 0
                trade_count += 1

    if not all_returns:
        return {"win_rate": 0.0, "avg_return": 0.0, "sharpe": 0.0, "trades": 0}

    r = np.array(all_returns)
    sharpe = (r.mean() / (r.std() + 1e-8)) * np.sqrt(252 * 24)
    return {
        "win_rate":   win_count / trade_count,
        "avg_return": float(r.mean()),
        "sharpe":     float(sharpe),
        "trades":     trade_count,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def walk_forward_validate(coin_data: dict[str, pd.DataFrame]) -> dict[str, list]:
    """
    For each of 3 time windows:
      - Train the agent on the train slice of ALL coins
      - Test on the test slice of EACH coin individually
    Returns per-coin results.
    """
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("[ERROR] Run: pip install stable-baselines3 gymnasium")
        sys.exit(1)

    all_results: dict[str, list] = {sym: [] for sym in coin_data}

    for i, split in enumerate(WF_SPLITS):
        t0, t1 = split["train"]
        v0, v1 = split["test"]

        # Slice each coin for this window
        train_data, test_data = {}, {}
        for sym, df in coin_data.items():
            n = len(df)
            tr = df.iloc[int(n*t0): int(n*t1)].reset_index(drop=True)
            te = df.iloc[int(n*v0): int(n*v1)].reset_index(drop=True)
            if len(tr) >= WINDOW_SIZE + 50:
                train_data[sym] = tr
            if len(te) >= 50:
                test_data[sym] = te

        if not train_data:
            print(f"[WARN] Window {i+1}: not enough data, skipping")
            continue

        print(f"\n[WF {i+1}/3] Training on {len(train_data)} coins "
              f"({int((t1-t0)*100)}% of data each)...")

        vec_env = make_multi_env(train_data, n_envs=min(N_ENVS, 2))
        wf_model = PPO(
            "MlpPolicy", vec_env,
            learning_rate=LEARNING_RATE,
            n_steps=1024, batch_size=64, n_epochs=5,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            verbose=0,
        )
        wf_model.learn(total_timesteps=TOTAL_STEPS // 3)
        vec_env.close()

        # Test on each coin separately
        print(f"  Results per coin:")
        for sym in coin_data:
            if sym not in test_data:
                continue
            m = evaluate_on_coin(wf_model, test_data[sym])
            m["window"] = i + 1
            all_results[sym].append(m)
            print(f"    {sym:8s}  win={m['win_rate']:.1%}  "
                  f"sharpe={m['sharpe']:.3f}  trades={m['trades']}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FINAL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_final(coin_data: dict[str, pd.DataFrame]):
    """Train on full dataset of all coins and save."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback

    print(f"\n[INFO] Final training on all {len(coin_data)} coins, {TOTAL_STEPS:,} steps...")

    vec_env = make_multi_env(coin_data, n_envs=N_ENVS)

    # Eval env: last 20% of BNB (your strongest signal coin)
    sys.path.insert(0, SCRIPT_DIR)
    from gym_env import ZoroCryptoEnv
    from stable_baselines3.common.env_util import make_vec_env

    df_bnb = coin_data.get("BNB-USD", list(coin_data.values())[0])
    df_eval = df_bnb.iloc[int(len(df_bnb) * 0.8):].reset_index(drop=True)
    eval_env = make_vec_env(
        lambda: ZoroCryptoEnv(df_eval, window_size=min(WINDOW_SIZE, len(df_eval)-1)),
        n_envs=1
    )

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=LEARNING_RATE,
        n_steps=2048, batch_size=128, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
        verbose=1,
        policy_kwargs=dict(net_arch=[dict(pi=[128, 64], vf=[128, 64])]),
    )

    start = time.time()
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=EvalCallback(eval_env, eval_freq=10_000,
                              n_eval_episodes=3, verbose=0, warn=False),
    )
    elapsed = time.time() - start
    vec_env.close(); eval_env.close()

    model.save(AGENT_OUT)
    print(f"\n[INFO] Agent saved → {AGENT_OUT}.zip  ({elapsed:.0f}s)")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 6. REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def write_report(wf_results: dict[str, list], final_per_coin: dict[str, dict]):
    lines = [
        "=" * 65,
        "ZORO RL Agent Retraining Report — Phase 2 Step 2",
        "Trained on: ETH-USD, BTC-USD, SOL-USD, BNB-USD, ADA-USD",
        "=" * 65,
        f"Obs space  : 10 features (rsi, macd, bb, trend, vol, sentiment,",
        f"             momentum, lstm, daily_return, atr)",
        f"Action     : 3 (HOLD / BUY / SELL)",
        f"Reward     : Sharpe ratio",
        f"Steps      : {TOTAL_STEPS:,}",
        "",
        "Walk-Forward Validation (per coin)",
        "-" * 65,
    ]

    all_sharpes = []
    for sym, results in wf_results.items():
        if not results:
            continue
        sharpes = [r["sharpe"] for r in results]
        all_sharpes.extend(sharpes)
        spread = max(sharpes) - min(sharpes)
        verdict = "OK" if spread < 0.6 else "CHECK"
        lines.append(f"  {sym:8s}  sharpes={[f'{s:.2f}' for s in sharpes]}  "
                     f"spread={spread:.2f}  [{verdict}]")

    if all_sharpes:
        lines.append(f"\n  Overall: consistent={'YES' if (max(all_sharpes)-min(all_sharpes)) < 1.5 else 'NO'}")

    lines += ["", "Final Model — Per Coin Evaluation", "-" * 65]
    for sym, m in final_per_coin.items():
        lines.append(f"  {sym:8s}  win={m['win_rate']:.1%}  "
                     f"sharpe={m['sharpe']:.3f}  trades={m['trades']}")

    lines += [
        "",
        "Output",
        "-" * 65,
        "  zoro_ppo_agent.zip — already in zoro_v2/",
        "  Run: python trader.py",
        "=" * 65,
    ]

    report = "\n".join(lines)
    print("\n" + report)
    with open(REPORT_OUT, "w") as f:
        f.write(report)
    print(f"[INFO] Report → {REPORT_OUT}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("ZORO Phase 2 Step 2 — RL Agent Retraining (5 coins)")
    print("=" * 65)

    # 1. Fetch all coins
    coin_data = fetch_all_coins()
    if len(coin_data) < 3:
        print("[ERROR] Fewer than 3 coins loaded — check internet connection")
        sys.exit(1)

    # 2. Walk-forward validation
    print("\n[STEP 1] Walk-forward validation across all coins...")
    wf_results = walk_forward_validate(coin_data)

    # 3. Full training
    print("\n[STEP 2] Full training on all coins...")
    final_model = train_final(coin_data)

    # 4. Evaluate final model per coin
    print("\n[STEP 3] Evaluating final model on each coin...")
    final_per_coin = {}
    for sym, df in coin_data.items():
        df_test = df.iloc[int(len(df) * 0.8):].reset_index(drop=True)
        final_per_coin[sym] = evaluate_on_coin(final_model, df_test, n_episodes=8)
        print(f"  {sym}: sharpe={final_per_coin[sym]['sharpe']:.3f}  "
              f"win={final_per_coin[sym]['win_rate']:.1%}")

    # 5. Report
    write_report(wf_results, final_per_coin)

    print("\n✅ Done!")
    print("   zoro_ppo_agent.zip is saved in zoro_v2/")
    print("   Run: python trader.py")

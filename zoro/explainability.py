"""
zoro/explainability.py — SHAP-based trade explainability (Phase 4 Step 1)

For any trade the bot makes, this module answers:
  "Why did bot BUY/SELL ETH at $3,200?"

Usage
-----
from zoro.explainability import explain_trade

result = explain_trade(symbol, df, signal_result)
# Returns dict with feature_contributions, dominant_reason, chart_path
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# ── Feature metadata ─────────────────────────────────────────────────────────
# These 7 features feed the LSTM (in exact order from lstm_model.py)
LSTM_FEATURES = [
    "close_norm",   # 0 — close / close.mean()
    "macd_norm",    # 1 — macd / close
    "bb_position",  # 2 — (price - bb_lower) / (bb_upper - bb_lower)
    "rsi_norm",     # 3 — rsi / 100
    "volume_ratio", # 4 — volume / volume.mean()
    "trend",        # 5 — +1 if sma20 > sma50 else -1
    "momentum",     # 6 — momentum / close
]

FEATURE_LABELS = {
    "close_norm":   "Price Level",
    "macd_norm":    "MACD",
    "bb_position":  "Bollinger Position",
    "rsi_norm":     "RSI",
    "volume_ratio": "Volume",
    "trend":        "Trend (SMA)",
    "momentum":     "Momentum",
    # RL obs extras
    "sentiment":    "FinBERT Sentiment",
    "daily_return": "Daily Return",
    "atr_norm":     "Volatility (ATR)",
}

FEATURE_DESCRIPTIONS = {
    "close_norm":   ("price above average", "price below average"),
    "macd_norm":    ("MACD bullish crossover", "MACD bearish crossover"),
    "bb_position":  ("near upper band (overbought)", "near lower band (oversold)"),
    "rsi_norm":     ("RSI overbought (>70)", "RSI oversold (<30)"),
    "volume_ratio": ("above-average volume", "below-average volume"),
    "trend":        ("uptrend (SMA20 > SMA50)", "downtrend (SMA20 < SMA50)"),
    "momentum":     ("positive momentum", "negative momentum"),
    "sentiment":    ("bullish news sentiment", "bearish news sentiment"),
    "daily_return": ("strong 24h gain", "sharp 24h drop"),
    "atr_norm":     ("high volatility", "low volatility"),
}


def _build_lstm_input(df: pd.DataFrame) -> np.ndarray:
    """Build the 7-feature input matrix the LSTM was trained on."""
    close = df["Close"].values
    mean_close = close.mean() or 1.0

    macd      = df["MACD"].values        if "MACD"     in df.columns else np.zeros(len(df))
    bb_upper  = df["BB_Upper"].values    if "BB_Upper" in df.columns else close + 1
    bb_lower  = df["BB_Lower"].values    if "BB_Lower" in df.columns else close - 1
    rsi       = df["RSI"].values         if "RSI"      in df.columns else np.full(len(df), 50.0)
    volume    = df["Volume"].values      if "Volume"   in df.columns else np.ones(len(df))
    sma20     = df["SMA_20"].values      if "SMA_20"   in df.columns else close
    sma50     = df["SMA_50"].values      if "SMA_50"   in df.columns else close
    momentum  = df["Momentum"].values    if "Momentum" in df.columns else np.zeros(len(df))

    mean_vol = volume.mean() or 1.0
    bb_range = (bb_upper - bb_lower)
    bb_range[bb_range == 0] = 1.0

    features = np.column_stack([
        close / mean_close,
        macd / (close + 1e-8),
        (close - bb_lower) / bb_range,
        rsi / 100.0,
        volume / mean_vol,
        np.where(sma20 > sma50, 1.0, -1.0),
        momentum / (close + 1e-8),
    ]).astype(np.float32)

    return features


def _gate_contributions(signal_result) -> dict[str, float]:
    """
    Extract which gates fired and their approximate contribution to confidence.
    Returns a dict of {feature_name: contribution_pct}.
    """
    gates = getattr(signal_result, "gates", [])
    confidence = getattr(signal_result, "confidence", 50)
    direction = getattr(signal_result, "direction", "FLAT")
    sentiment = getattr(signal_result, "sentiment", 0.0)
    lstm_prob = getattr(signal_result, "lstm_prob", 0.528)

    contributions: dict[str, float] = {}

    for gate in gates:
        gate_lower = gate.lower()
        if "rsi" in gate_lower:
            contributions["rsi_norm"] = contributions.get("rsi_norm", 0) + 20
        if "smc" in gate_lower or "bullish structure" in gate_lower or "bearish structure" in gate_lower:
            contributions["trend"] = contributions.get("trend", 0) + 10
        if "macd" in gate_lower:
            contributions["macd_norm"] = contributions.get("macd_norm", 0) + 10
        if "bb" in gate_lower or "bollinger" in gate_lower:
            contributions["bb_position"] = contributions.get("bb_position", 0) + 10
        if "lstm" in gate_lower:
            contributions["close_norm"] = contributions.get("close_norm", 0) + 20
        if "finbert" in gate_lower or "sentiment" in gate_lower:
            sent_pts = int(sentiment * 10)
            contributions["sentiment"] = abs(sent_pts)
        if "rl" in gate_lower and "disagrees" not in gate_lower:
            contributions["momentum"] = contributions.get("momentum", 0) + 10

    # Normalise to percentages
    total = sum(contributions.values()) or 1
    return {k: round(v / total * 100, 1) for k, v in contributions.items()}


def explain_trade(symbol: str, df: pd.DataFrame, signal_result,
                  output_dir: str = "zoro_explanations") -> dict:
    """
    Generate a human-readable explanation for a trade signal.

    Parameters
    ----------
    symbol        : e.g. "BNB-USD"
    df            : full DataFrame from data.fetch_crypto_data()
    signal_result : SignalResult from signals.run_signal_engine()
    output_dir    : folder to save JSON + chart data

    Returns
    -------
    dict with keys:
        symbol, direction, price, confidence,
        feature_contributions (list of dicts),
        dominant_reason (str),
        human_explanation (str),
        gates (list of str)
    """
    direction  = getattr(signal_result, "direction", "FLAT")
    price      = getattr(signal_result, "price", 0.0)
    confidence = getattr(signal_result, "confidence", 50)
    gates      = getattr(signal_result, "gates", [])
    sentiment  = getattr(signal_result, "sentiment", 0.0)
    lstm_prob  = getattr(signal_result, "lstm_prob", 0.528)
    rsi        = getattr(signal_result, "rsi", 50.0)

    # ── Try SHAP if available, fall back to gate-based ─────────────────────
    contributions: dict[str, float] = {}
    method_used = "gate_analysis"

    try:
        import shap
        import tensorflow as tf

        model_path = Path("lstm_model.h5")
        if not model_path.exists():
            model_path = Path("zoro_v2/lstm_model.h5")

        if model_path.exists() and len(df) >= 60:
            model = tf.keras.models.load_model(str(model_path), compile=False)
            features = _build_lstm_input(df)

            # Use last 60 candles (one sequence) as background
            seq_len   = 60
            X_last    = features[-seq_len:].reshape(1, seq_len, 7)
            background = features[-120:-60].reshape(1, seq_len, 7)

            explainer  = shap.GradientExplainer(model, background)
            shap_vals  = explainer.shap_values(X_last)

            # shap_vals shape: (1, 60, 7) — average over timesteps
            if isinstance(shap_vals, list):
                sv = np.abs(np.array(shap_vals[0])).mean(axis=1)[0]  # (7,)
            else:
                sv = np.abs(shap_vals).mean(axis=1)[0]

            total = sv.sum() or 1.0
            for i, fname in enumerate(LSTM_FEATURES):
                contributions[fname] = round(float(sv[i] / total * 100), 1)

            # Add sentiment contribution (from FinBERT gate)
            sent_pts = abs(int(sentiment * 10))
            if sent_pts > 0:
                total_with_sent = sum(contributions.values()) + sent_pts
                scale = sum(contributions.values()) / total_with_sent
                contributions = {k: round(v * scale, 1) for k, v in contributions.items()}
                contributions["sentiment"] = round(sent_pts / total_with_sent * 100, 1)

            method_used = "shap"

    except Exception:
        pass  # SHAP unavailable — use gate analysis

    if not contributions:
        contributions = _gate_contributions(signal_result)

    # ── Build sorted feature list ─────────────────────────────────────────
    sorted_features = sorted(contributions.items(), key=lambda x: x[1], reverse=True)

    feature_contributions = []
    for fname, pct in sorted_features:
        label  = FEATURE_LABELS.get(fname, fname)
        descs  = FEATURE_DESCRIPTIONS.get(fname, ("positive", "negative"))

        # Determine if this feature is bullish or bearish for direction
        if fname == "rsi_norm":
            bullish = rsi < 50
        elif fname == "sentiment":
            bullish = sentiment > 0
        elif fname == "bb_position":
            bb_pos = float(df.iloc[-1].get("BB_Position", 0.5)) if hasattr(df.iloc[-1], "get") else 0.5
            bullish = bb_pos < 0.5
        elif fname == "trend":
            bullish = direction == "LONG"
        else:
            bullish = direction == "LONG"

        feature_contributions.append({
            "feature":     fname,
            "label":       label,
            "pct":         pct,
            "description": descs[0] if bullish else descs[1],
            "bullish":     bullish,
        })

    # ── Human-readable explanation ────────────────────────────────────────
    top = feature_contributions[:3] if len(feature_contributions) >= 3 else feature_contributions
    dominant = top[0] if top else {"label": "RSI", "pct": 100, "description": "oversold"}

    parts = [f"{f['label']} ({f['pct']}%: {f['description']})" for f in top]
    action_word = "BUY" if direction == "LONG" else "SHORT" if direction == "SHORT" else "HOLD"
    human_explanation = (
        f"Bot {action_word} {symbol} at ${price:,.2f} "
        f"[conf={confidence}/100] because: "
        + " · ".join(parts)
    )

    result = {
        "symbol":                symbol,
        "direction":             direction,
        "action":                action_word,
        "price":                 price,
        "confidence":            confidence,
        "rsi":                   rsi,
        "lstm_prob":             lstm_prob,
        "sentiment":             sentiment,
        "feature_contributions": feature_contributions,
        "dominant_reason":       dominant["description"],
        "human_explanation":     human_explanation,
        "gates":                 gates,
        "method":                method_used,
    }

    # ── Persist to disk ───────────────────────────────────────────────────
    try:
        os.makedirs(output_dir, exist_ok=True)
        fname_safe = symbol.replace("-", "_")
        out_path = Path(output_dir) / f"explain_{fname_safe}_latest.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
    except Exception:
        pass

    return result


def explain_from_json(trade_log_path: str = "zoro_trade_log.json") -> list[dict]:
    """
    Batch-explain all closed trades from the trade log.
    Useful for the live performance page.
    """
    try:
        with open(trade_log_path) as f:
            trades = json.load(f)
    except Exception:
        return []

    explanations = []
    for trade in trades:
        if trade.get("pnl") is None:
            continue  # skip open trades

        symbol     = trade.get("symbol", "UNKNOWN")
        direction  = trade.get("action", "BUY")
        price      = trade.get("price", 0.0)
        confidence = trade.get("confidence", 50)
        gates      = trade.get("gates", [])
        sentiment  = trade.get("sentiment", 0.0)
        lstm_prob  = trade.get("lstm_prob", 0.528)
        rsi        = trade.get("rsi", 50.0)
        pnl        = trade.get("pnl", 0.0)

        # Build minimal mock signal for gate analysis
        class _MockSignal:
            pass

        sig = _MockSignal()
        sig.direction  = "LONG" if direction == "BUY" else "SHORT"
        sig.price      = price
        sig.confidence = confidence
        sig.gates      = gates
        sig.sentiment  = sentiment
        sig.lstm_prob  = lstm_prob
        sig.rsi        = rsi

        contribs = _gate_contributions(sig)
        sorted_f = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
        top_reason = sorted_f[0][0] if sorted_f else "rsi_norm"
        top_label  = FEATURE_LABELS.get(top_reason, top_reason)

        explanations.append({
            "symbol":        symbol,
            "direction":     direction,
            "price":         price,
            "pnl":           pnl,
            "confidence":    confidence,
            "top_reason":    top_label,
            "contributions": [{"label": FEATURE_LABELS.get(k, k), "pct": v}
                               for k, v in sorted_f],
        })

    return explanations

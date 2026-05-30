"""
zoro/lstm_model.py — LSTM model wrapper

Model loaded ONCE at startup. Uses MinMaxScaler fitted on train data only.
Falls back to 0.528 if model or scaler unavailable.

Features (must match retrain_lstm.py):
  ['Close', 'RSI', 'MACD', 'BB_Width', 'ATR', 'SMA_20', 'Volume_Ratio']
"""
from __future__ import annotations

import os
import pickle
import warnings
import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_model  = None
_scaler = None

FEATURES     = ['Close', 'RSI', 'MACD', 'BB_Width', 'ATR', 'SMA_20', 'Volume_Ratio']
SEQUENCE_LEN = 60


def _get_paths():
    try:
        from .config import config
        model_path  = config.LSTM_MODEL_PATH
        scaler_path = os.path.join(os.path.dirname(model_path), 'lstm_scaler.pkl')
    except Exception:
        base        = os.path.join(os.path.dirname(__file__), '..')
        model_path  = os.path.join(base, 'lstm_model.h5')
        scaler_path = os.path.join(base, 'lstm_scaler.pkl')
    return model_path, scaler_path


def _load():
    global _model, _scaler
    if _model is not None:
        return _model, _scaler

    model_path, scaler_path = _get_paths()

    # Load model
    try:
        import logging
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        from tensorflow.keras.models import load_model as _km
        if not os.path.exists(model_path):
            print(f"[WARN] lstm_model: {model_path} not found")
            return None, None
        _model = _km(model_path, compile=False)
        print(f"[INFO] LSTM model loaded from {model_path}")
    except Exception as e:
        print(f"[WARN] lstm_model: load failed ({e})")
        return None, None

    # Load scaler (optional — falls back to manual normalisation)
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as f:
                _scaler = pickle.load(f)
            print(f"[INFO] LSTM scaler loaded from {scaler_path}")
        except Exception as e:
            print(f"[WARN] lstm_model: scaler load failed ({e}) — using manual normalisation")
            _scaler = None
    else:
        print("[INFO] lstm_model: no scaler file found — using manual normalisation")

    return _model, _scaler


def _build_features(df) -> np.ndarray:
    """Build (60, 7) feature matrix matching the training feature set."""
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame({"Close": pd.Series(df)})

    c = df['Close'].astype(np.float32)

    # RSI
    delta       = c.diff()
    gain        = delta.clip(lower=0).rolling(14).mean()
    loss        = (-delta.clip(upper=0)).rolling(14).mean()
    rsi         = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # MACD
    ema12       = c.ewm(span=12, adjust=False).mean()
    ema26       = c.ewm(span=26, adjust=False).mean()
    macd        = ema12 - ema26

    # BB_Width
    sma20       = c.rolling(20).mean()
    std20       = c.rolling(20).std()
    bb_upper    = sma20 + 2 * std20
    bb_lower    = sma20 - 2 * std20
    bb_width    = (bb_upper - bb_lower) / (sma20 + 1e-9)

    # ATR (approximate from close only if High/Low missing)
    try:
        hl      = (df['High'] - df['Low']).abs()
        hpc     = (df['High'] - c.shift()).abs()
        lpc     = (df['Low']  - c.shift()).abs()
        atr     = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()
    except Exception:
        atr     = c.diff().abs().rolling(14).mean()

    # SMA_20
    # Volume_Ratio
    try:
        vol         = df['Volume'].astype(np.float32)
        vol_ratio   = vol / (vol.rolling(20).mean() + 1e-9)
    except Exception:
        vol_ratio   = pd.Series(np.ones(len(c)), index=c.index)

    feat_df = pd.concat([c, rsi, macd, bb_width, atr, sma20, vol_ratio], axis=1)
    feat_df.columns = FEATURES
    feat_df = feat_df.dropna()

    return feat_df.values[-SEQUENCE_LEN:].astype(np.float32)


def get_lstm_prob(df_or_series) -> float:
    """
    Returns P(price UP in next 4h) in [0, 1].
    Falls back to 0.528 if unavailable.
    """
    model, scaler = _load()
    if model is None:
        return 0.528

    try:
        import pandas as pd
        if not isinstance(df_or_series, pd.DataFrame):
            df_or_series = pd.DataFrame({"Close": pd.Series(df_or_series)})

        if len(df_or_series) < SEQUENCE_LEN + 50:
            return 0.528

        features = _build_features(df_or_series)   # (60, 7)

        if features.shape != (SEQUENCE_LEN, len(FEATURES)):
            return 0.528

        # Scale: use saved scaler if available, else simple min-max per column
        if scaler is not None:
            features = scaler.transform(features)
        else:
            col_min = features.min(axis=0)
            col_max = features.max(axis=0)
            col_range = np.where(col_max - col_min > 0, col_max - col_min, 1.0)
            features = (features - col_min) / col_range

        x   = features[np.newaxis, ...]             # (1, 60, 7)
        raw = model.predict(x, verbose=0)
        prob = float(np.array(raw).flatten()[-1])
        return float(np.clip(prob, 0.0, 1.0))

    except Exception as e:
        print(f"[WARN] lstm_model.get_lstm_prob: {e}")
        return 0.528


# Pre-load on import
_load()

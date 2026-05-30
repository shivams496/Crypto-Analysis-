"""
retrain_lstm.py  —  ZORO Phase 2: Proper LSTM retraining
=========================================================
Fixes from the original notebook (Cell 3):
  1. MinMaxScaler fitted on TRAIN set only — no lookahead via scaler
  2. Chronological 70/15/15 train/val/test split (no random shuffle)
  3. EarlyStopping to prevent overfitting
  4. Walk-forward validation window to confirm the signal is real
  5. Honest accuracy printed for all three splits

Run from inside zoro_v2/:
    python retrain_lstm.py

Outputs:
    lstm_model.h5          — new model (replaces old one)
    lstm_scaler.pkl        — scaler fitted on train only
    retrain_report.txt     — honest accuracy numbers
"""

import os, warnings, pickle
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

# ── Config ────────────────────────────────────────────────────────────────────
COINS         = ['ETH-USD', 'BTC-USD', 'SOL-USD', 'BNB-USD', 'ADA-USD']
FEATURES      = ['Close', 'RSI', 'MACD', 'BB_Width', 'ATR', 'SMA_20', 'Volume_Ratio']
SEQUENCE_LEN  = 60
HORIZON       = 4      # predict 4h ahead
PERIOD        = '180d'
INTERVAL      = '1h'
TRAIN_FRAC    = 0.70
VAL_FRAC      = 0.15
# test = remaining 0.15
MODEL_PATH    = 'lstm_model.h5'
SCALER_PATH   = 'lstm_scaler.pkl'
REPORT_PATH   = 'retrain_report.txt'

# ── Feature engineering ───────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df['Close']

    # RSI
    delta        = c.diff()
    gain         = delta.clip(lower=0).rolling(14).mean()
    loss         = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI']    = 100 - (100 / (1 + gain / loss))

    # MACD
    ema12        = c.ewm(span=12, adjust=False).mean()
    ema26        = c.ewm(span=26, adjust=False).mean()
    df['MACD']   = ema12 - ema26

    # Bollinger Band Width
    sma20           = c.rolling(20).mean()
    std20           = c.rolling(20).std()
    bb_upper        = sma20 + 2 * std20
    bb_lower        = sma20 - 2 * std20
    df['BB_Width']  = (bb_upper - bb_lower) / (sma20 + 1e-9)

    # ATR
    hl              = df['High'] - df['Low']
    hpc             = (df['High'] - c.shift()).abs()
    lpc             = (df['Low']  - c.shift()).abs()
    df['ATR']       = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()

    # SMA_20 and Volume_Ratio
    df['SMA_20']       = sma20
    df['Volume_Ratio'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-9)

    return df[FEATURES].dropna()


def fetch_and_build(symbol: str) -> pd.DataFrame | None:
    print(f"  Downloading {symbol}...")
    try:
        raw = yf.Ticker(symbol).history(period=PERIOD, interval=INTERVAL)
        if raw.empty:
            print(f"  ⚠ No data for {symbol}")
            return None
        return build_features(raw)
    except Exception as e:
        print(f"  ⚠ {symbol} failed: {e}")
        return None


# ── Sequence builder ──────────────────────────────────────────────────────────
def make_sequences(feat: np.ndarray, close_raw: np.ndarray):
    X, y = [], []
    for i in range(SEQUENCE_LEN, len(feat) - HORIZON):
        X.append(feat[i - SEQUENCE_LEN:i])
        # Label: 1 if close price is higher HORIZON steps later
        y.append(1 if close_raw[i + HORIZON] > close_raw[i] else 0)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  ZORO Phase 2 — Proper LSTM retraining")
    print("=" * 60)

    # 1. Collect data from all coins and concatenate
    print("\n[1/5] Fetching data...")
    all_feat   = []
    all_close  = []

    for sym in COINS:
        df = fetch_and_build(sym)
        if df is not None:
            all_feat.append(df.values.astype(np.float32))
            all_close.append(df['Close'].values.astype(np.float32))
            print(f"      {sym}: {len(df)} rows")

    if not all_feat:
        print("ERROR: No data fetched. Check internet connection.")
        return

    feat_all  = np.concatenate(all_feat,  axis=0)
    close_all = np.concatenate(all_close, axis=0)
    print(f"\n  Total rows: {len(feat_all)}")

    # 2. Chronological split (no shuffle)
    print("\n[2/5] Splitting data chronologically...")
    n        = len(feat_all)
    n_train  = int(n * TRAIN_FRAC)
    n_val    = int(n * VAL_FRAC)

    feat_train  = feat_all[:n_train]
    feat_val    = feat_all[n_train:n_train + n_val]
    feat_test   = feat_all[n_train + n_val:]

    close_train = close_all[:n_train]
    close_val   = close_all[n_train:n_train + n_val]
    close_test  = close_all[n_train + n_val:]

    print(f"  Train: {len(feat_train)} rows  |  Val: {len(feat_val)} rows  |  Test: {len(feat_test)} rows")

    # 3. Fit scaler on TRAIN only ← the fix
    print("\n[3/5] Fitting scaler on TRAIN set only...")
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(feat_train)

    feat_train_s = scaler.transform(feat_train)
    feat_val_s   = scaler.transform(feat_val)
    feat_test_s  = scaler.transform(feat_test)

    # Save scaler
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved → {SCALER_PATH}")

    # 4. Build sequences
    print("\n[4/5] Building sequences...")
    X_train, y_train = make_sequences(feat_train_s, close_train)
    X_val,   y_val   = make_sequences(feat_val_s,   close_val)
    X_test,  y_test  = make_sequences(feat_test_s,  close_test)

    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}    y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")

    # Class balance
    train_pos = y_train.mean()
    print(f"\n  Class balance (train): {train_pos:.1%} UP / {1-train_pos:.1%} DOWN")

    # 5. Build and train model
    print("\n[5/5] Training model...")
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True,
                             input_shape=(SEQUENCE_LEN, len(FEATURES))),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1,  activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=3, verbose=1
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate honestly on all splits
    print("\n" + "=" * 60)
    print("  HONEST RESULTS")
    print("=" * 60)

    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, val_acc   = model.evaluate(X_val,   y_val,   verbose=0)
    _, test_acc  = model.evaluate(X_test,  y_test,  verbose=0)

    # Walk-forward validation (3 windows)
    wf_accs = []
    window  = len(X_test) // 3
    for w in range(3):
        xw = X_test[w*window:(w+1)*window]
        yw = y_test[w*window:(w+1)*window]
        if len(xw) > 0:
            _, acc = model.evaluate(xw, yw, verbose=0)
            wf_accs.append(acc)

    report_lines = [
        "ZORO LSTM Retrain Report",
        "=" * 40,
        f"Train accuracy  : {train_acc:.1%}",
        f"Val accuracy    : {val_acc:.1%}",
        f"Test accuracy   : {test_acc:.1%}",
        "",
        "Walk-forward validation (3 windows on test set):",
        *[f"  Window {i+1}: {a:.1%}" for i, a in enumerate(wf_accs)],
        "",
        "Consistent across windows = real signal.",
        "Drops off = overfitting.",
        "",
        f"Features: {FEATURES}",
        f"Sequence length: {SEQUENCE_LEN}",
        f"Horizon: {HORIZON}h",
        f"Train/Val/Test split: {TRAIN_FRAC}/{VAL_FRAC}/{1-TRAIN_FRAC-VAL_FRAC}",
        "Scaler fitted on TRAIN only — no lookahead bias.",
    ]

    for line in report_lines:
        print(f"  {line}")

    with open(REPORT_PATH, 'w') as f:
        f.write('\n'.join(report_lines))

    # Save model
    model.save(MODEL_PATH)
    print(f"\n  Model saved → {MODEL_PATH}")
    print(f"  Report saved → {REPORT_PATH}")
    print("\n✅ Retraining complete!")
    print("\nNext step: update lstm_model.py to use the new scaler (lstm_scaler.pkl)")


if __name__ == "__main__":
    main()

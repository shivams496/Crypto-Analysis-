"""
Run backtests for all coins and both strategies.

Usage:  python backtest_runner.py
"""
from zoro.backtest import run_backtest, print_backtest_summary
from zoro.config import config

STRATEGIES = ["rsi", "buy_hold"]

if __name__ == "__main__":
    for strategy in STRATEGIES:
        for symbol in config.COINS:
            result = run_backtest(symbol, strategy=strategy)
            if result:
                print_backtest_summary(result)
            else:
                print(f"[ERROR] Backtest failed for {symbol} / {strategy}")

-- init.sql — run once when the PostgreSQL container is first created
-- SQLAlchemy creates the tables via metadata.create_all()
-- This file adds indexes for fast dashboard queries

-- Wait for tables to exist (created by the bot on startup)
-- These CREATE INDEX commands are safe to run after tables exist.
-- If tables don't exist yet, they'll be created by SQLAlchemy first.

-- We use DO $$ blocks so errors don't crash the init script
DO $$
BEGIN
  -- Index: fast lookup by symbol + timestamp
  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes
    WHERE tablename = 'trades' AND indexname = 'idx_trades_symbol_ts'
  ) THEN
    CREATE INDEX idx_trades_symbol_ts ON trades (symbol, timestamp DESC);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes
    WHERE tablename = 'trades' AND indexname = 'idx_trades_coin'
  ) THEN
    CREATE INDEX idx_trades_coin ON trades (coin);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes
    WHERE tablename = 'signals' AND indexname = 'idx_signals_symbol_ts'
  ) THEN
    CREATE INDEX idx_signals_symbol_ts ON signals (symbol, timestamp DESC);
  END IF;
EXCEPTION WHEN OTHERS THEN
  -- Tables don't exist yet — SQLAlchemy will create them on bot startup
  NULL;
END $$;

#!/bin/bash
# start.sh — runs FastAPI (background) + trader (foreground) in one container
# FastAPI on port 8000, trader loop in foreground so Docker logs show it

set -e

echo "[START] Launching ZORO Phase 3..."

# Start FastAPI in background
uvicorn api:app --host 0.0.0.0 --port 8000 &
API_PID=$!
echo "[START] FastAPI running (PID $API_PID) on :8000"

# Give API a moment to bind
sleep 2

# Start trader in foreground (Docker logs will show this)
python trader.py

# If trader exits, kill API too
kill $API_PID 2>/dev/null || true

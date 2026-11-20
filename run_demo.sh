#!/usr/bin/env bash
# Start the FHE server and the local client (UI on http://localhost:8501)
set -e
cd "$(dirname "$0")"
mkdir -p logs
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000 > logs/server.log 2>&1 &
SERVER_PID=$!
sleep 4
echo "Server läuft (PID $SERVER_PID). Starte Client + Schlüsselerzeugung ..."
CIPHERCHAT_SERVER=http://127.0.0.1:8000 python -m uvicorn client.app:app --host 127.0.0.1 --port 8501
kill $SERVER_PID

#!/usr/bin/env bash
# Start all three CipherChat generations + the client with the version switch (UI on http://localhost:8501).
#   v1  server/app.py  weights/cipherchat_demo.npz             :8000  (OpenFHE, CPU)
#   v2  server/app.py  weights/cf2_rev_cpl_sqglu_h2048.npz     :8001  (OpenFHE, CPU)
#   v3  server/app_v3.py  weights/v3/<ckpt>.pt                 :8003  (PhantomFHE, GPU; needs build_v3/v3fhe*.so)
# V3_CKPT overrides the v3 checkpoint; without a GPU build the client simply greys out v3.
set -e
cd "$(dirname "$0")"
mkdir -p logs
V3_CKPT=${V3_CKPT:-weights/v3/q05_pos.pt}
CIPHERCHAT_WEIGHTS=weights/cipherchat_demo.npz python -m uvicorn server.app:app --host 127.0.0.1 --port 8000 > logs/server_v1.log 2>&1 &
P1=$!
CIPHERCHAT_WEIGHTS=weights/cf2_rev_cpl_sqglu_h2048.npz python -m uvicorn server.app:app --host 127.0.0.1 --port 8001 > logs/server_v2.log 2>&1 &
P2=$!
P3=
if ls build_v3/v3fhe*.so > /dev/null 2>&1 && [ -f "$V3_CKPT" ]; then
  CIPHERCHAT_V3_CKPT=$V3_CKPT PYTHONPATH=build_v3:$PYTHONPATH \
    python -m uvicorn server.app_v3:app --host 127.0.0.1 --port 8003 > logs/server_v3.log 2>&1 &
  P3=$!
fi
trap 'kill $P1 $P2 $P3 2>/dev/null' EXIT
sleep 6
echo "Server: v1 (PID $P1), v2 (PID $P2), v3 (${P3:-nicht gestartet}). v3 braucht ~1-2 min zum Laden + Schlüsselerzeugung."
CIPHERCHAT_SERVERS="v1=http://127.0.0.1:8000,v2=http://127.0.0.1:8001,v3=http://127.0.0.1:8003" \
  python -m uvicorn client.app:app --host 127.0.0.1 --port 8501

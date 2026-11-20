"""End-to-end test over HTTP: FHE server and client as separate processes, chat via the client API (SSE).

  python scripts/v2_http_e2e.py weights/<name>.npz "was ist 47+38?" "hallo"
Prints each answer, the server-side seconds per token and whether the answer matches the expected sum.
"""
import json
import os
import re
import subprocess
import sys
import time

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
weights = sys.argv[1]
msgs = sys.argv[2:] or ["was ist 47+38?", "hallo"]
env = dict(os.environ, CIPHERCHAT_WEIGHTS=weights, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "8"))
os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
srv_log = open(os.path.join(ROOT, "logs", "e2e_server.log"), "w")
cli_log = open(os.path.join(ROOT, "logs", "e2e_client.log"), "w")
srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "server.app:app", "--port", "8710"], cwd=ROOT, env=env,
                       stdout=srv_log, stderr=subprocess.STDOUT)
cli = None
try:
    for _ in range(120):
        try:
            if httpx.get("http://127.0.0.1:8710/info", timeout=2).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.5)
    info = httpx.get("http://127.0.0.1:8710/info").json()
    print("server info:", {k: info[k] for k in ("model", "arch", "rev", "coupling", "n_layers", "d_ff")})
    cli = subprocess.Popen([sys.executable, "-m", "uvicorn", "client.app:app", "--port", "8711"], cwd=ROOT,
                           env=dict(env, CIPHERCHAT_SERVER="http://127.0.0.1:8710"),
                           stdout=cli_log, stderr=subprocess.STDOUT)
    for _ in range(240):
        try:
            if httpx.get("http://127.0.0.1:8711/api/status", timeout=2).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.5)
    st = httpx.get("http://127.0.0.1:8711/api/status").json()
    print("client status:", {k: st[k] for k in ("ring_dim", "eval_key_mb", "handshake_s", "max_context")})
    results = []
    for m in msgs:
        t0 = time.time()
        answer, server_s, tokens = None, [], []
        with httpx.stream("POST", "http://127.0.0.1:8711/api/chat", json={"message": m}, timeout=None) as r:
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                ev = json.loads(line[6:])
                if ev["kind"] == "upload":
                    server_s.append(ev["server_s"])
                elif ev["kind"] == "token":
                    tokens.append(ev["text"])
                elif ev["kind"] == "done":
                    answer = ev["answer"]
        mm = re.fullmatch(r"was ist (\d+)\+(\d+)\?", m)
        ok = (answer == str(int(mm.group(1)) + int(mm.group(2)))) if mm else None
        row = dict(message=m, answer=answer, streamed="".join(tokens), correct=ok, total_s=round(time.time() - t0, 1),
                   server_s_per_token=round(sum(server_s) / max(1, len(server_s)), 3), tokens=len(server_s))
        results.append(row)
        print(row, flush=True)
    print(json.dumps({"n": len(results), "correct": sum(1 for r in results if r["correct"]),
                      "arith": sum(1 for r in results if r["correct"] is not None)}))
finally:
    for p in (cli, srv):
        if p is not None:
            p.terminate()

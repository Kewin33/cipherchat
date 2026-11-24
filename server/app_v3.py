"""CipherChat v3 server: Qwen2.5-0.5B (FHE-adapted) evaluated non-interactively under CKKS on the GPU.

Run (GPU machine, Python module v3fhe on PYTHONPATH):
  CIPHERCHAT_V3_CKPT=weights/v3/q05_div.pt CIPHERCHAT_V3_BOUNDS=weights/v3/q05_div_bounds.json \\
  PYTHONPATH=build_v3 uvicorn server.app_v3:app --port 8003

Demo limitation (see cipherchat/v3/service.py): the client's key material lives in this process too, because
PhantomFHE cannot export evaluation keys yet. The engine itself only ever sees ciphertexts.
"""
import json
import os
import sys
import threading
from collections import deque

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cipherchat.v3.service import V3Service  # noqa: E402

CKPT = os.environ.get("CIPHERCHAT_V3_CKPT", "weights/v3/q05_div.pt")
BOUNDS = os.environ.get("CIPHERCHAT_V3_BOUNDS", CKPT.replace(".pt", "_bounds.json"))
MAX_NEW = int(os.environ.get("CIPHERCHAT_V3_MAX_NEW", "6"))
app = FastAPI(title="CipherChat v3 (FHE Qwen2.5 on GPU)")
LOCK = threading.Lock()
STATE = {"svc": None, "log": deque(maxlen=400)}


@app.on_event("startup")
def _startup():
    STATE["svc"] = V3Service(CKPT, BOUNDS)


@app.get("/info")
def info():
    return STATE["svc"].info()


class GenIn(BaseModel):
    message: str
    max_new: int = MAX_NEW


@app.post("/v3/generate")
def generate(inp: GenIn):
    def gen():
        with LOCK:                       # one GPU, one conversation at a time
            for ev in STATE["svc"].generate(inp.message, min(inp.max_new, 32)):
                if ev["kind"] == "upload":
                    STATE["log"].append(dict(event="token", pos=ev["pos"], in_kb=ev["kb"], out_kb=ev["back_kb"],
                                             seconds=ev["server_s"], boots=ev["boots"]))
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/server_view")
def server_view(n: int = 50):
    return list(STATE["log"])[-n:]

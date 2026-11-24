"""CipherChat local client ("the HTTPS stack") with a switch between the three generations.

  v1  MVP (OpenFHE, CPU): character model, 2 layers                         server/app.py + weights/cipherchat_demo
  v2  research engine (OpenFHE, CPU): slot-saturated CipherFormer2           server/app.py + weights/cf2_*.npz
  v3  Qwen2.5-0.5B, non-interactive CKKS with bootstrapping on the GPU      server/app_v3.py

For v1/v2 this process generates the CKKS key pair, keeps the SECRET KEY local, encrypts every token and decrypts
the logits. v3 currently runs its client crypto inside the GPU server process (no key export in PhantomFHE yet);
the UI says so.

Run:  CIPHERCHAT_SERVERS="v1=http://localhost:8000,v2=http://localhost:8001,v3=http://localhost:8003" \\
      uvicorn client.app:app --port 8501
      (legacy: CIPHERCHAT_SERVER=http://localhost:8000 -> one server, version detected from /info)
"""
import json
import os
import struct
import sys
import threading
import time

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cipherchat.client_crypto import ClientCrypto  # noqa: E402
from cipherchat.v2.dataset import reverse_digits, significance  # noqa: E402
from cipherchat.vocab import EOS, ITOS, decode, encode_prompt  # noqa: E402

MAX_NEW = int(os.environ.get("CIPHERCHAT_MAX_NEW", "20"))
HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = {"v1": "v1 · MVP", "v2": "v2 · Forschung", "v3": "v3 · Qwen2.5 (GPU)"}


def parse_servers():
    if os.environ.get("CIPHERCHAT_SERVERS"):
        out = {}
        for part in os.environ["CIPHERCHAT_SERVERS"].split(","):
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
        return out
    if os.environ.get("CIPHERCHAT_SERVER"):            # legacy single-server mode
        return {"auto": os.environ["CIPHERCHAT_SERVER"]}
    return {"v1": "http://127.0.0.1:8000", "v2": "http://127.0.0.1:8001", "v3": "http://127.0.0.1:8003"}


SERVERS = parse_servers()
app = FastAPI(title="CipherChat client")
http = httpx.Client(timeout=None)
STATES = {}
LOCK = threading.Lock()


def frames(*parts: bytes) -> bytes:
    return b"".join(struct.pack("<Q", len(p)) + p for p in parts)


def probe(url):
    try:
        return httpx.get(f"{url}/info", timeout=2.5).json()
    except Exception:
        return None


def resolve_auto():
    """legacy mode: name the single server by its /info."""
    if "auto" in SERVERS:
        info = probe(SERVERS["auto"])
        ver = "v3" if info and info.get("arch", "").startswith("qwen") else \
            ("v2" if info and info.get("arch") == "cipherformer2" else "v1")
        SERVERS[ver] = SERVERS.pop("auto")


resolve_auto()


def connect(ver):
    """v1 / v2: key generation here + upload of the evaluation keys (handshake). v3: read the server info."""
    with LOCK:
        if ver in STATES:
            return STATES[ver]
        if ver not in SERVERS:
            raise HTTPException(404, f"no server configured for {ver}")
        url = SERVERS[ver]
        info = http.get(f"{url}/info").json()
        if ver == "v3":
            STATES[ver] = dict(info=info, url=url)
            return STATES[ver]
        t0 = time.time()
        if info.get("arch") == "cipherformer2":
            from cipherchat.v2.crypto import ClientCryptoV2
            crypto = ClientCryptoV2(info["n_layers"])
        else:
            crypto = ClientCrypto(info["n_layers"])
        bundle = crypto.export_server_bundle()
        payload = frames(bundle["cryptocontext"], bundle["eval_mult_key"], bundle["eval_rot_keys"], bundle["public_key"])
        r = http.post(f"{url}/session", content=payload, headers={"Content-Type": "application/octet-stream"})
        r.raise_for_status()
        sess = r.json()
        st = dict(crypto=crypto, info=info, url=url, sid=sess["session_id"], T=sess["max_context"],
                  eval_key_mb=round((len(bundle["eval_mult_key"]) + len(bundle["eval_rot_keys"])) / 1e6, 1),
                  handshake_s=round(time.time() - t0, 1), ring_dim=sess["ring_dim"])
        STATES[ver] = st
        print(f"[client] {ver} connected to {url}: session {st['sid']} | eval keys {st['eval_key_mb']} MB "
              f"| handshake {st['handshake_s']}s", flush=True)
        return st


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/api/versions")
def versions():
    out = []
    for ver in ("v1", "v2", "v3"):
        url = SERVERS.get(ver)
        ok = bool(url) and (ver in STATES or probe(url) is not None)
        out.append(dict(id=ver, label=LABELS[ver], url=url, available=ok, connected=ver in STATES))
    return out


@app.get("/api/status")
def status(v: str = "v1"):
    st = connect(v)
    if v == "v3":
        st["info"] = http.get(f"{st['url']}/info").json()     # the v3 server may have been restarted with another model
        i = st["info"]
        return dict(version=v, server=st["url"], model=i, ring_dim=2 ** i["fhe"]["logN"], slots=2 ** (i["fhe"]["logN"] - 1),
                    max_context=i["max_context"])
    c = st["crypto"]
    return dict(version=v, server=st["url"], session=st["sid"], ring_dim=st["ring_dim"], slots=c.S,
                keygen_s=round(c.keygen_seconds, 2), eval_key_mb=st["eval_key_mb"], handshake_s=st["handshake_s"],
                max_context=st["T"], model=st["info"])


@app.get("/api/server_view")
def server_view(v: str = "v1"):
    url = SERVERS.get(v)
    return http.get(f"{url}/server_view", params={"n": 30}).json() if url else []


class ChatIn(BaseModel):
    message: str
    version: str = "v1"


def sse(kind, **data):
    return f"data: {json.dumps(dict(kind=kind, **data), ensure_ascii=False)}\n\n"


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def chat_v3(st, message):
    with http.stream("POST", f"{st['url']}/v3/generate", json={"message": message}) as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                yield line + "\n\n"


def chat_v12(st, message):
    crypto, sid, T, V = st["crypto"], st["sid"], st["T"], st["info"]["vocab_size"]
    info, url = st["info"], st["url"]
    v2 = info.get("arch") == "cipherformer2"
    rev, coupling = bool(info.get("rev")), bool(info.get("coupling"))
    t_start = time.time()
    # v2 tokenization (client side, invisible to the user): numbers little-endian + digit significance
    ids = encode_prompt(reverse_digits(message) if rev else message)
    if len(ids) > T - 8:
        ids = ids[: T - 9] + [ids[-1]]
    seq = list(ids)
    http.post(f"{url}/session/{sid}/reset").raise_for_status()
    yield sse("status", text=f"{len(ids)} Tokens werden lokal verschlüsselt (CKKS) und gesendet …")

    def send(tok, pos, need):
        t0 = time.time()
        if v2:
            sig = significance(seq, rev)[pos] if coupling else 0   # seq = all tokens known so far
            ctb = crypto.encrypt_token_bytes(tok, sig, V)
        else:
            ctb = crypto.encrypt_token_bytes(tok)
        t_enc = time.time() - t0
        r = http.post(f"{url}/session/{sid}/token", params={"need_logits": int(need)}, content=ctb,
                      headers={"Content-Type": "application/octet-stream"})
        r.raise_for_status()
        ev = dict(pos=pos, kb=len(ctb) // 1024, sha=crypto.fingerprint(ctb), hexprev=ctb[len(ctb) // 2: len(ctb) // 2 + 12].hex(),
                  enc_ms=round(t_enc * 1000), server_s=float(r.headers.get("X-Server-Seconds", 0)),
                  back_kb=len(r.content) // 1024)
        return r, ev

    r = None
    for i, tok in enumerate(ids):
        r, ev = send(tok, i, i == len(ids) - 1)
        yield sse("upload", phase="prompt", **ev)
    out = []
    digit_buf = []          # rev mode: digits arrive least-significant first -> show the run when complete
    pos = len(ids)
    while True:
        t0 = time.time()
        logits = crypto.decrypt_logits_bytes(r.content, V)
        p = _softmax(logits)
        top = np.argsort(-p)[:3]
        nxt = int(top[0])
        show = lambda i: "⏎" if i == EOS else ("␣" if ITOS[i] == " " else ITOS[i])
        yield sse("decrypt", pos=pos, top=[[show(int(i)), round(float(p[i]), 3)] for i in top],
                  dec_ms=round((time.time() - t0) * 1000))
        if nxt == EOS or len(out) >= MAX_NEW or pos >= T:
            break
        out.append(nxt)
        seq.append(nxt)
        ch = decode([nxt])
        if rev and ch.isdigit():
            digit_buf.append(ch)
        else:
            if digit_buf:
                yield sse("token", text="".join(reversed(digit_buf)))
                digit_buf = []
            yield sse("token", text=ch)
        r, ev = send(nxt, pos, True)
        pos += 1
        yield sse("upload", phase="gen", **ev)
    if digit_buf:
        yield sse("token", text="".join(reversed(digit_buf)))
    answer = reverse_digits(decode(out)) if rev else decode(out)
    yield sse("done", answer=answer, seconds=round(time.time() - t_start, 1), tokens=pos)


@app.post("/api/chat")
def chat(inp: ChatIn):
    st = connect(inp.version)
    gen = chat_v3(st, inp.message) if inp.version == "v3" else chat_v12(st, inp.message)
    return StreamingResponse(gen, media_type="text/event-stream")

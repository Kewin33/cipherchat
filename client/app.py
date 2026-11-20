"""CipherChat local client ("the HTTPS stack").

Runs on the user's machine. Generates the CKKS key pair, keeps the SECRET KEY local,
encrypts every token before it leaves the machine and decrypts the returned logits.
Also serves the chat UI on http://localhost:8501

Run:  CIPHERCHAT_SERVER=http://localhost:8000 uvicorn client.app:app --port 8501
"""
import json
import os
import struct
import sys
import time

import httpx
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cipherchat.client_crypto import ClientCrypto  # noqa: E402
from cipherchat.v2.dataset import reverse_digits, significance  # noqa: E402
from cipherchat.vocab import EOS, ITOS, decode, encode_prompt  # noqa: E402

SERVER = os.environ.get("CIPHERCHAT_SERVER", "http://localhost:8000")
MAX_NEW = int(os.environ.get("CIPHERCHAT_MAX_NEW", "20"))
HERE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="CipherChat client")
http = httpx.Client(timeout=None)
STATE = {}


def frames(*parts: bytes) -> bytes:
    return b"".join(struct.pack("<Q", len(p)) + p for p in parts)


def connect():
    info = http.get(f"{SERVER}/info").json()
    t0 = time.time()
    if info.get("arch") == "cipherformer2":
        from cipherchat.v2.crypto import ClientCryptoV2
        crypto = ClientCryptoV2(info["n_layers"])
    else:
        crypto = ClientCrypto(info["n_layers"])
    bundle = crypto.export_server_bundle()
    payload = frames(bundle["cryptocontext"], bundle["eval_mult_key"], bundle["eval_rot_keys"], bundle["public_key"])
    r = http.post(f"{SERVER}/session", content=payload, headers={"Content-Type": "application/octet-stream"})
    r.raise_for_status()
    sess = r.json()
    STATE.update(crypto=crypto, info=info, sid=sess["session_id"], T=sess["max_context"],
                 eval_key_mb=round((len(bundle["eval_mult_key"]) + len(bundle["eval_rot_keys"])) / 1e6, 1),
                 handshake_s=round(time.time() - t0, 1), ring_dim=sess["ring_dim"])
    print(f"[client] connected to {SERVER}: session {STATE['sid']} | eval keys {STATE['eval_key_mb']} MB "
          f"| handshake {STATE['handshake_s']}s", flush=True)


@app.on_event("startup")
def _startup():
    connect()


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/api/status")
def status():
    c = STATE["crypto"]
    return dict(server=SERVER, session=STATE["sid"], ring_dim=STATE["ring_dim"], slots=c.S,
                keygen_s=round(c.keygen_seconds, 2), eval_key_mb=STATE["eval_key_mb"],
                handshake_s=STATE["handshake_s"], max_context=STATE["T"], model=STATE["info"])


@app.get("/api/server_view")
def server_view():
    return http.get(f"{SERVER}/server_view", params={"n": 30}).json()


class ChatIn(BaseModel):
    message: str


def sse(kind, **data):
    return f"data: {json.dumps(dict(kind=kind, **data), ensure_ascii=False)}\n\n"


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


@app.post("/api/chat")
def chat(inp: ChatIn):
    def gen():
        crypto, sid, T, V = STATE["crypto"], STATE["sid"], STATE["T"], STATE["info"]["vocab_size"]
        info = STATE["info"]
        v2 = info.get("arch") == "cipherformer2"
        rev, coupling = bool(info.get("rev")), bool(info.get("coupling"))
        t_start = time.time()
        # v2 tokenization (client side, invisible to the user): numbers little-endian + digit significance
        ids = encode_prompt(reverse_digits(inp.message) if rev else inp.message)
        if len(ids) > T - 8:
            ids = ids[: T - 9] + [ids[-1]]
        seq = list(ids)
        http.post(f"{SERVER}/session/{sid}/reset").raise_for_status()
        yield sse("status", text=f"{len(ids)} Tokens werden lokal verschlüsselt (CKKS) und gesendet …")

        def send(tok, pos, need):
            t0 = time.time()
            if v2:
                sig = significance(seq, rev)[pos] if coupling else 0   # seq = all tokens known so far
                ctb = crypto.encrypt_token_bytes(tok, sig, V)
            else:
                ctb = crypto.encrypt_token_bytes(tok)
            t_enc = time.time() - t0
            r = http.post(f"{SERVER}/session/{sid}/token", params={"need_logits": int(need)}, content=ctb,
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

    return StreamingResponse(gen(), media_type="text/event-stream")

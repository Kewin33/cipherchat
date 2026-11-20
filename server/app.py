"""CipherChat inference server.

Receives ONLY: public CKKS parameters, evaluation keys and ciphertexts.
Has NO access to: the secret key, the user's tokens, the answer.

Run:  uvicorn server.app:app --port 8000
"""
import hashlib
import json
import os
import struct
import sys
import threading
import time
import uuid
from collections import deque

import openfhe as ofhe
from fastapi import FastAPI, HTTPException, Request, Response
from openfhe import BINARY

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cipherchat.fhe_common import FIRST_MOD_BITS, SCALE_BITS, depth_for_layers, ser  # noqa: E402
from cipherchat.fhe_engine import EncryptedSession, FHEServerModel  # noqa: E402
from cipherchat.reference import FoldedModel  # noqa: E402

WEIGHTS = os.environ.get("CIPHERCHAT_WEIGHTS", "weights/cipherchat_demo.npz")
app = FastAPI(title="CipherChat FHE inference server")
LOCK = threading.Lock()
META = json.load(open(WEIGHTS.replace(".npz", ".json"), encoding="utf-8"))
V2 = META.get("arch") == "cipherformer2"
if V2:   # v2: slot-saturated engine + CipherFormer2 (see cipherchat/v2)
    from cipherchat.v2.engine import EngineV2, OpenFHEBackend, Session2  # noqa: E402
    from cipherchat.v2.reference import Folded2  # noqa: E402
    FM = Folded2(WEIGHTS)
    FM.d, FM.d_ff = 64, FM.H
else:
    FM = FoldedModel(WEIGHTS)
STATE = {"model": None, "sessions": {}, "log": deque(maxlen=400)}


def new_session():
    return Session2(FM.L) if V2 else EncryptedSession(FM.L)


def new_model(cc):
    return EngineV2(FM, OpenFHEBackend(cc)) if V2 else FHEServerModel(FM, cc)


def set_cc(model, cc):
    if V2:
        model.be.cc = cc
    else:
        model.cc = cc


def unpack_frames(body: bytes):
    out, off = [], 0
    while off < len(body):
        (n,) = struct.unpack_from("<Q", body, off)
        off += 8
        out.append(body[off: off + n])
        off += n
    return out


def log(event, **kw):
    STATE["log"].append(dict(ts=time.time(), event=event, **kw))


@app.get("/info")
def info():
    meta = META
    return dict(model=os.path.basename(WEIGHTS), vocab_size=FM.V, n_layers=FM.L, d_model=FM.d, d_ff=FM.d_ff,
                max_context=FM.T, train_exact_match=meta.get("train_exact_match"),
                arch=meta.get("arch", "cipherformer"), rev=bool(meta.get("rev")), coupling=bool(meta.get("coupling")),
                n_sig=meta.get("n_sig", 0), heldout_addition_exact=meta.get("heldout_addition_exact"),
                fhe=dict(scheme="CKKS (OpenFHE)", depth=depth_for_layers(FM.L), scale_bits=SCALE_BITS,
                         first_mod_bits=FIRST_MOD_BITS, security="128-bit classic (HE standard)"),
                sessions=len(STATE["sessions"]))


@app.post("/session")
async def create_session(request: Request):
    body = await request.body()
    t0 = time.time()
    frames = unpack_frames(body)
    cc_b, mult_b, rot_b = frames[:3]
    pk_b = frames[3] if len(frames) > 3 else None
    with LOCK:
        # MVP policy: one active client at a time -> drop previous keys
        STATE["sessions"].clear()
        ofhe.ClearEvalMultKeys()
        ofhe.CryptoContext.ClearEvalAutomorphismKeys()
        cc = ofhe.DeserializeCryptoContextString(cc_b, BINARY)
        ofhe.DeserializeEvalMultKeyString(mult_b, BINARY)
        ofhe.DeserializeEvalAutomorphismKeyString(rot_b, BINARY)
        m = STATE["model"]
        old_cc = None if m is None else (m.be.cc if V2 else m.cc)
        if m is None or old_cc.GetRingDimension() != cc.GetRingDimension():
            STATE["model"] = new_model(cc)     # static plaintexts depend only on the CKKS parameters
        else:
            set_cc(m, cc)
        sid = uuid.uuid4().hex[:12]
        STATE["sessions"][sid] = new_session()
        # OpenFHE quirk: plaintext multiplications on *deserialized* ciphertexts are ~5x slower.
        # Adding a server-side fresh encryption of zero (first operand) yields a "native" ciphertext.
        STATE["zero"] = None
        if pk_b:
            pk = ofhe.DeserializePublicKeyString(pk_b, BINARY)
            STATE["zero"] = cc.Encrypt(pk, cc.MakeCKKSPackedPlaintext([0.0] * (cc.GetRingDimension() // 2)))
    log("session", sid=sid, eval_key_mb=round((len(mult_b) + len(rot_b)) / 1e6, 1),
        ring_dim=cc.GetRingDimension(), seconds=round(time.time() - t0, 2))
    return dict(session_id=sid, max_context=STATE["model"].T, ring_dim=cc.GetRingDimension())


@app.post("/session/{sid}/reset")
def reset(sid: str):
    with LOCK:
        if sid not in STATE["sessions"]:
            raise HTTPException(404, "unknown session")
        STATE["sessions"][sid] = new_session()
    log("reset", sid=sid)
    return {"ok": True}


@app.post("/session/{sid}/token")
async def token(sid: str, request: Request, need_logits: int = 1):
    body = await request.body()
    with LOCK:
        sess = STATE["sessions"].get(sid)
        if sess is None:
            raise HTTPException(404, "unknown session")
        model = STATE["model"]
        t0 = time.time()
        ct = ofhe.DeserializeCiphertextString(body, BINARY)
        if STATE.get("zero") is not None:
            ct = (model.be.cc if V2 else model.cc).EvalAdd(STATE["zero"], ct)
        try:
            out = model.step(sess, ct, need_logits=bool(need_logits))
        except ValueError as e:
            raise HTTPException(400, str(e))
        dt = time.time() - t0
        pos = sess.t - 1
    h_in = hashlib.sha256(body).hexdigest()[:16]
    if out is None:
        log("token", sid=sid, pos=pos, in_sha=h_in, in_kb=len(body) // 1024, out_sha=None, out_kb=0,
            seconds=round(dt, 2), preview=body[len(body) // 2: len(body) // 2 + 16].hex())
        return Response(status_code=204, headers={"X-Server-Seconds": f"{dt:.3f}"})
    ob = ser(out)
    h_out = hashlib.sha256(ob).hexdigest()[:16]
    log("token", sid=sid, pos=pos, in_sha=h_in, in_kb=len(body) // 1024, out_sha=h_out, out_kb=len(ob) // 1024,
        seconds=round(dt, 2), preview=body[len(body) // 2: len(body) // 2 + 16].hex())
    return Response(content=ob, media_type="application/octet-stream",
                    headers={"X-Server-Seconds": f"{dt:.3f}", "X-Out-Sha": h_out})


@app.get("/server_view")
def server_view(n: int = 50):
    """Everything the server knows about the conversation: sizes, hashes, timings. No plaintext."""
    return list(STATE["log"])[-n:]

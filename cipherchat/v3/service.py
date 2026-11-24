"""CipherChat v3 – chat service around the encrypted Qwen2.5 engine (used by server/app_v3.py and the benchmarks).

Roles (kept in separate classes):
  ClientV3  tokenizer, embedding lookup, encrypt / decrypt, final RMSNorm + LM head, greedy choice  (holds the key)
  Engine3   all 24 transformer blocks on ciphertexts                                          (never decrypts)
Demo limitation: PhantomFHE has no key serialisation yet, so both roles run in ONE GPU process (the ~7 GB of
evaluation keys are not transferred). The server code path never calls decrypt; this is an organisational
separation, not a cryptographic one, until key export is implemented.
"""
import hashlib
import json
import os
import time

import numpy as np

from . import engine as E
from . import fold

HOTSPOTS = {
    "linear": ("qkv", "attn.out", "ffn.up", "ffn.down"),
    "attention": ("attn.scores", "attn.av", "attn.cache", "attn.pack"),
    "nonlinear": ("norm1", "norm2", "attn.softmax", "attn.inverse", "ffn.act"),
}


def hotspot_table(profile):
    """profile: {"section|op": (ms, count)} -> the five hotspot groups of the spec (ms per token)."""
    out = {"linear_ptmul_ms": 0.0, "linear_rot_ms": 0.0, "attention_ms": 0.0, "attention_rot_ms": 0.0,
           "nonlinear_ms": 0.0, "bootstrap_ms": 0.0, "other_ms": 0.0}
    for key, (ms, _) in profile.items():
        sec, op = key.split("|")
        if op == "bootstrap":
            out["bootstrap_ms"] += ms
        elif sec in HOTSPOTS["linear"]:
            out["linear_rot_ms" if op == "rot" else "linear_ptmul_ms"] += ms
        elif sec in HOTSPOTS["attention"]:
            out["attention_rot_ms" if op == "rot" else "attention_ms"] += ms
        elif sec in HOTSPOTS["nonlinear"]:
            out["nonlinear_ms"] += ms
        else:
            out["other_ms"] += ms
    return out


class ClientV3:
    def __init__(self, be, tok, prefix, emb, g_fin, eps, W_lm):
        self.be, self.tok, self.prefix = be, tok, prefix
        self.emb, self.g_fin, self.eps, self.W_lm = emb, g_fin, eps, W_lm

    def encrypt_token(self, tid, level):
        v = np.zeros(E.D)
        v[: self.emb.shape[1]] = self.emb[tid]
        return self.be.encrypt(np.tile(v, self.be.n // E.D), level)

    def logits(self, ct):
        y = self.be.decrypt(ct)[: self.emb.shape[1]]
        h = y / np.sqrt(np.mean(y ** 2) + self.eps) * self.g_fin
        return self.W_lm @ h


class V3Service:
    def __init__(self, ckpt, bounds=None, profile=True, device="cpu"):
        t0 = time.time()
        fm, student, tok, prefix = fold.build(ckpt, device=device)
        if bounds and os.path.exists(bounds):
            fm.boot_bounds = json.load(open(bounds))
        emb = student.model.embed_tokens.weight.detach().double().cpu().numpy()
        W_lm = student.lm_head.weight.detach().float().cpu().numpy()
        g_fin = student.model.norm.weight.detach().double().cpu().numpy()
        eps = student.model.norm.variance_epsilon
        del student
        self.fold_s = time.time() - t0
        t0 = time.time()
        self.be = E.PhantomBackend(fm.cfg, profile=profile, extra_steps=E.engine_rotation_steps())
        self.keygen_s = time.time() - t0
        self.fm = fm
        self.eng = E.Engine3(fm, self.be)
        self.client = ClientV3(self.be, tok, prefix, emb, g_fin, eps, W_lm)
        self.log = []

    def info(self):
        inf = dict(self.be.ctx.info())
        mem = dict(self.be.ctx.mem_info())
        ck = self.fm.cfg["ckks"]
        return dict(arch="qwen2.5-0.5b-fhe", model=self.fm.cfg["model"]["base"], n_layers=self.fm.L, d_model=self.fm.d,
                    max_context=E.T_MAX, attn_norm=self.fm.attn_norm, p=self.fm.p,
                    fhe=dict(scheme="CKKS (PhantomFHE + NEXUS bootstrapping, GPU)", logN=ck["logN"], log_qp=inf["log_qp"],
                             levels=inf["compute_levels"], hamming_weight=ck["hamming_weight"],
                             security="128-bit (log QP <= 1546 for h = 192)", rotation_keys=self.be.n_keys,
                             key_gb=round(self.be.n_keys * 0.162, 1)),
                    vram_mb=mem, fold_s=round(self.fold_s, 1), keygen_s=round(self.keygen_s, 1),
                    mode="demo: client keys and server engine share one GPU process (no key export yet)")

    def generate(self, message, max_new=8, max_prompt=24):
        """Yields events (dicts) while the token loop runs."""
        cl, eng, be = self.client, self.eng, self.be
        ids = cl.tok(message)["input_ids"][:max_prompt]
        sess = eng.new_session()
        eng.prefill_public(sess)
        L = be.L
        t_start = time.time()
        yield dict(kind="status", text=f"{len(ids)} Prompt-Token, je eines verschlüsselt durch alle {self.fm.L} Schichten …")
        out = []
        logits = None
        pos = 0
        for i, tid in enumerate(list(ids) + [None] * max_new):
            if tid is None:
                if logits is None:
                    break
                p = np.exp(logits - logits.max())
                p /= p.sum()
                top = np.argsort(-p)[:3]
                nxt = int(top[0])
                yield dict(kind="decrypt", pos=pos, top=[[cl.tok.decode([int(j)]), round(float(p[j]), 3)] for j in top],
                           dec_ms=0)
                if nxt == cl.tok.eos_token_id or len(out) >= max_new:
                    break
                out.append(nxt)
                yield dict(kind="token", text=cl.tok.decode([nxt]))
                tid = nxt
            t0 = time.time()
            ct = cl.encrypt_token(tid, L)
            enc_ms = (time.time() - t0) * 1e3
            be.ctx.reset_profile()
            b0 = eng.stats["bootstraps"]
            t1 = time.time()
            res = eng.step(sess, ct)
            server_s = time.time() - t1
            prof = {k: tuple(v) for k, v in be.ctx.profile().items()}
            hs = hotspot_table(prof)
            pos = sess.t - 1
            kb = be.ctx.ct_bytes(ct) // 1024
            sha = hashlib.sha256(f"{pos}-{time.time()}".encode()).hexdigest()[:16]
            ev = dict(kind="upload", phase="prompt" if i < len(ids) else "gen", pos=pos, kb=kb, sha=sha,
                      hexprev=sha[:12], enc_ms=round(enc_ms), server_s=round(server_s, 2),
                      back_kb=be.ctx.ct_bytes(res) // 1024, boots=eng.stats["bootstraps"] - b0,
                      hotspots={k: round(v, 1) for k, v in hs.items()})
            self.log.append(dict(ts=time.time(), event="token", pos=pos, in_kb=kb, out_kb=ev["back_kb"],
                                 seconds=round(server_s, 2), boots=ev["boots"]))
            yield ev
            logits = cl.logits(res) if (i >= len(ids) - 1) else None
            if sess.t >= E.T_MAX:
                break
        answer = cl.tok.decode(out)
        yield dict(kind="done", answer=answer, seconds=round(time.time() - t_start, 1), tokens=sess.t - self.fm.n_pub)

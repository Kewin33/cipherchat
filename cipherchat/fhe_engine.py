"""Server-side homomorphic inference for CipherFormer (CKKS / OpenFHE).

The server holds: the model weights (plaintext) + the client's *evaluation* keys
(relinearisation + rotation keys). It never sees the secret key, the tokens, the
attention pattern or the output distribution - only ciphertexts.

Per generated token the client sends ONE ciphertext (an encrypted one-hot vector)
and receives ONE ciphertext (encrypted logits). Keys and values of previous tokens
stay on the server as an *encrypted KV cache* (2 ciphertexts per layer).
"""
import math
import time

import numpy as np

from .fhe_common import B, BABY, BLK, GIANT, bsgs_vectors, kmask, max_context, n_slots, tile, vmask
from .reference import FoldedModel


class EncryptedSession:
    def __init__(self, n_layers):
        self.K = [None] * n_layers
        self.V = [None] * n_layers
        self.t = 0
        self.stats = []


class FHEServerModel:
    def __init__(self, fm: FoldedModel, cc, cache_plaintexts: bool = True):
        self.fm, self.cc = fm, cc
        self.S = n_slots(cc)
        self.T = min(max_context(cc), fm.T)
        self.m = 2 * cc.GetRingDimension()
        self.cache_plaintexts = cache_plaintexts
        self._pt = {}
        self._vec = {}
        self._ones = {}
        self.counters = dict(rot=0, ctct=0, ptmul=0, encode=0)

    # ---------- primitives ----------
    def _lvl(self, ct):
        return ct.GetLevel() + (1 if ct.GetNoiseScaleDeg() > 1 else 0)

    def _encode(self, vec, level):
        self.counters["encode"] += 1
        return self.cc.MakeCKKSPackedPlaintext(vec.tolist(), 1, level)

    def _rot(self, ct, k):
        self.counters["rot"] += 1
        return self.cc.EvalRotate(ct, k)

    def _rescale(self, ct):
        """Force the pending (lazy) FLEXIBLEAUTO rescale *in place* once, so that the many
        plaintext multiplications that follow do not each rescale a fresh copy."""
        if ct.GetNoiseScaleDeg() > 1:
            lvl = ct.GetLevel() + 1
            if lvl not in self._ones:
                self._ones[lvl] = self.cc.MakeCKKSPackedPlaintext([1.0] * self.S, 1, lvl)
            self.cc.EvalMultMutable(ct, self._ones[lvl])
        return ct

    def _baby(self, ct):
        self._rescale(ct)
        pre = self.cc.EvalFastRotationPrecompute(ct)
        self.counters["rot"] += BABY - 1
        return [ct] + [self.cc.EvalFastRotation(ct, j, self.m, pre) for j in range(1, BABY)]

    def _static_pts(self, name, M, level):
        key = (name, level)
        if key in self._pt:
            return self._pt[key]
        vecs = self._vec.get(name)
        if vecs is None:
            vecs = bsgs_vectors(M, self.S)
            self._vec[name] = vecs
        pts = [[self._encode(v, level) for v in row] for row in vecs]
        if self.cache_plaintexts:
            self._pt[key] = pts
        return pts

    def _matvec(self, rots, pts):
        """BSGS diagonal method: y = sum_i rot( sum_j rot(x, j) * P[i][j], BABY*i )   (1 level)."""
        cc = self.cc
        acc = None
        for i in range(GIANT):
            prods = []
            for j in range(BABY):
                prods.append(cc.EvalMult(rots[j], pts[i][j]))
                self.counters["ptmul"] += 1
            inner = cc.EvalAddMany(prods)
            if i:
                inner = self._rot(inner, BABY * i)
            acc = inner if acc is None else cc.EvalAdd(acc, inner)
        return acc

    def _matvec_dynamic(self, rots, M, level, mask=None, scale=1.0):
        vecs = bsgs_vectors(M, self.S, mask=mask, scale=scale)
        pts = [[self._encode(v, level) for v in row] for row in vecs]
        return self._matvec(rots, pts)

    def _add_vec(self, ct, vec):
        # additions do not trigger a rescale -> encode at the ciphertext's current level
        return self.cc.EvalAdd(ct, self._encode(vec, ct.GetLevel()))

    def _sum_window(self, ct, steps):
        for s in steps:
            ct = self.cc.EvalAdd(ct, self._rot(ct, s))
        return ct

    # ---------- one decoding step ----------
    def step(self, sess: EncryptedSession, ct_onehot, need_logits: bool = True):
        """Process one encrypted token. If need_logits is False (prompt tokens before the last
        one) the last layer only appends to the encrypted KV cache and returns None."""
        cc, fm, S = self.cc, self.fm, self.S
        t = sess.t
        if t >= self.T:
            raise ValueError(f"context window of {self.T} tokens exhausted")
        sc = 1.0 / math.sqrt(t + 1)
        km, vm = kmask(t, S), vmask(t, S)
        block_steps = [BLK << k for k in range(int(math.log2(S // BLK)))]
        inp = ct_onehot
        logits = None
        for l, Lr in enumerate(fm.layers):
            bx, bq, bk, bv, bz = fm.pos_biases(l, t)
            lvl = self._lvl(inp)
            rots = self._baby(inp)
            # --- projections (1 level) ---
            k = self._matvec_dynamic(rots, Lr["Mk"], lvl, mask=km)
            k = self._add_vec(k, tile(bk, S) * km)
            v = self._matvec_dynamic(rots, Lr["Mv"], lvl, mask=vm)
            v = self._add_vec(v, tile(bv, S) * vm)
            sess.K[l] = k if sess.K[l] is None else cc.EvalAdd(sess.K[l], k)
            sess.V[l] = v if sess.V[l] is None else cc.EvalAdd(sess.V[l], v)
            if Lr["last"] and not need_logits:
                break                       # KV cache updated; nothing else needed for this token
            q = self._matvec_dynamic(rots, Lr["Mq"], lvl, scale=sc)
            q = self._add_vec(q, tile(bq * sc, S))
            z = [self._add_vec(self._matvec(rots, self._static_pts(f"L{l}.Mz{b}", Lr["Mz"][b], lvl)), tile(bz[b], S))
                 for b in range(fm.n_ff)]
            # --- polynomial attention over the encrypted KV cache ---
            s = cc.EvalMultMutable(q, sess.K[l]); self.counters["ctct"] += 1   # (2) q (.) [k_j, k_j]
            s = self._sum_window(s, [1, 2, 4, 8, 16, 32])                      # s_j in block j
            s = cc.EvalAdd(s, Lr["c"] * sc)
            w = cc.EvalSquare(s); self.counters["ctct"] += 1                   # (3) w_j = (s_j + c)^2/(t+1)
            av = cc.EvalMultMutable(w, sess.V[l]); self.counters["ctct"] += 1  # (4) w_j * [v_j, 0]
            av = self._sum_window(av, block_steps)                             # sum over positions j
            # --- MLP with square activation (runs in parallel levels) ---
            mw = None
            for b in range(fm.n_ff):
                mb = cc.EvalSquare(z[b]); self.counters["ctct"] += 1
                r2 = self._baby(mb)
                term = self._matvec(r2, self._static_pts(f"L{l}.W2{b}", Lr["W2"][b], self._lvl(mb)))
                mw = term if mw is None else cc.EvalAdd(mw, term)
            if Lr["last"]:
                xu = self._matvec(rots, self._static_pts(f"L{l}.Mxu", Lr["Mxu"], lvl))
                logits = cc.EvalAdd(cc.EvalAdd(xu, av), mw)
                logits = self._add_vec(logits, tile(Lr["bfinal"] + bx @ fm.Wu, S))
            else:
                if l == 0:
                    x = self._matvec(rots, self._static_pts("L0.Mx", Lr["Mx"], lvl))
                    x = self._add_vec(x, tile(bx, S))
                else:
                    x = inp
                a = cc.EvalAdd(av, self._rot(av, -B))                           # [a, 0] -> [a, a]
                inp = cc.EvalAdd(cc.EvalAdd(x, a), mw)
                inp = self._add_vec(inp, tile(Lr["bres"], S))
        sess.t += 1
        return logits

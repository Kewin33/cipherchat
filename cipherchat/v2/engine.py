"""v2 homomorphic inference engine for CipherFormer2.

The same engine code runs on two backends:
  NumpyBackend   - exact slot-level simulation (validates every layout/rotation, no noise)
  OpenFHEBackend - CKKS via OpenFHE (server side; only evaluation keys, never the secret key)

Per generated token (2 layers): 45 hoisted + 51 full rotations, 512 plaintext mults, 7 ct x ct
mults, 0 dynamic encodes (v1: 90 hoisted + 73 rotations, 1024 pt mults, 10 ct x ct, 384 dynamic encodes).

Level schedule (depth 4 per layer, unchanged):
  layer 0:  u(0) -> k^,v,x0 (1) -> s = u*K^ (2) -> w = (s+cr)^2 (3) -> av = w*V (4) -> x1 = x0 + cross_sum(av)
  layer 1:  x1(4) -> k~,v,z (5) -> s = x1*K~ (6) | m = z^2 (6) -> w (7) | m W2 (7) -> av (8)
            logits = cross_sum(W2-part + x1 Wu-part + av) + bias       (level 8)
"""
import numpy as np

from .layout import (BABY, BLK, GIANT, WINDOW_STEPS, cache_add_vector, cross_steps, from_rep_plaintexts,
                     full_block0, full_cache_k, full_cache_v, full_from_hidden, full_hidden, full_rep,
                     tile_rep, to_rep_plaintexts)


# ----------------------------------------------------------------------------- backends

class NumpyBackend:
    name = "numpy"

    def __init__(self, S=8192):
        self.S = S
        self.counters = dict(rot=0, rot_hoisted=0, ptmul=0, ctmul=0, encode=0)

    def level(self, ct):
        return 0

    def baby(self, ct):
        self.counters["rot_hoisted"] += BABY - 1
        return [ct] + [np.roll(ct, -j) for j in range(1, BABY)]

    def rot(self, ct, k):
        self.counters["rot"] += 1
        return np.roll(ct, -k)

    def add(self, a, b):
        return a + b

    def add_many(self, xs):
        acc = xs[0]
        for v in xs[1:]:
            acc = acc + v
        return acc

    def mul_pt(self, ct, pt):
        self.counters["ptmul"] += 1
        return ct * pt

    def mul_ct(self, a, b):
        self.counters["ctmul"] += 1
        return a * b

    def square(self, a):
        self.counters["ctmul"] += 1
        return a * a

    def encode(self, vec, level):
        self.counters["encode"] += 1
        return np.asarray(vec, dtype=np.float64)

    def add_level(self, ct):
        return 0

    def add_pt(self, ct, pt):
        return ct + pt


class OpenFHEBackend:
    name = "openfhe"

    def __init__(self, cc):
        self.cc = cc
        self.S = cc.GetRingDimension() // 2
        self.m = 2 * cc.GetRingDimension()
        self._ones = {}
        self.counters = dict(rot=0, rot_hoisted=0, ptmul=0, ctmul=0, encode=0)

    def level(self, ct):
        # level after the pending (lazy) FLEXIBLEAUTO rescale
        return ct.GetLevel() + (1 if ct.GetNoiseScaleDeg() > 1 else 0)

    def _prepare(self, ct):
        """Force a pending rescale in place, so that many plaintext products on this ciphertext
        do not each rescale a fresh copy (OpenFHE FLEXIBLEAUTO quirk, see docs/HANDOVER.md 5)."""
        if ct.GetNoiseScaleDeg() > 1:
            lvl = ct.GetLevel() + 1
            if lvl not in self._ones:
                self._ones[lvl] = self.cc.MakeCKKSPackedPlaintext([1.0] * self.S, 1, lvl)
            self.cc.EvalMultMutable(ct, self._ones[lvl])
        return ct

    def baby(self, ct):
        self._prepare(ct)
        pre = self.cc.EvalFastRotationPrecompute(ct)
        self.counters["rot_hoisted"] += BABY - 1
        return [ct] + [self.cc.EvalFastRotation(ct, j, self.m, pre) for j in range(1, BABY)]

    def rot(self, ct, k):
        self.counters["rot"] += 1
        return self.cc.EvalRotate(ct, k)

    def add(self, a, b):
        return self.cc.EvalAdd(a, b)

    def add_many(self, xs):
        return self.cc.EvalAddMany(xs)

    def mul_pt(self, ct, pt):
        self.counters["ptmul"] += 1
        return self.cc.EvalMult(ct, pt)

    def mul_ct(self, a, b):
        self.counters["ctmul"] += 1
        return self.cc.EvalMult(a, b)

    def square(self, a):
        self.counters["ctmul"] += 1
        return self.cc.EvalSquare(a)

    def encode(self, vec, level):
        self.counters["encode"] += 1
        return self.cc.MakeCKKSPackedPlaintext(np.asarray(vec, dtype=np.float64).tolist(), 1, level)

    def add_level(self, ct):
        return ct.GetLevel()

    def add_pt(self, ct, pt):
        return self.cc.EvalAdd(ct, pt)


# ----------------------------------------------------------------------------- engine

class Session2:
    def __init__(self, n_layers):
        self.K = [None] * n_layers
        self.V = [None] * n_layers
        self.t = 0


class EngineV2:
    def __init__(self, fm, backend, cache_plaintexts=True):
        """fm: reference.Folded2 (exported, folded weights)."""
        self.fm, self.be = fm, backend
        self.S = backend.S
        self.nblk = self.S // BLK
        self.T = min(self.nblk, fm.T)
        assert fm.T <= self.nblk, "context longer than the number of cache blocks"
        self.cache_plaintexts = cache_plaintexts
        self.vec = self._build_vectors()
        self._pt = {}

    # ---- static plaintext vectors (numpy); encoded lazily once per (name, level) ----
    def _build_vectors(self):
        W, S, L = self.fm.W, self.S, self.fm.L
        V = {
            "L0.K": from_rep_plaintexts(full_cache_k(W["L0.K"], S)),
            "L0.V": from_rep_plaintexts(full_cache_v(W["L0.V"], S)),
            "L0.X": from_rep_plaintexts(full_rep(W["L0.X"], S)),
            "L0.CR": cache_add_vector(W["L0.CR"], S),
        }
        for l in range(1, L):
            last = l == L - 1
            V[f"L{l}.K"] = from_rep_plaintexts(full_cache_k(W[f"L{l}.K"], S))
            V[f"L{l}.V"] = from_rep_plaintexts(full_cache_v(W[f"L{l}.V"], S))
            V[f"L{l}.Z"] = from_rep_plaintexts(full_hidden(W[f"L{l}.Z"], S))
            if f"L{l}.G" in W:
                V[f"L{l}.G"] = from_rep_plaintexts(full_hidden(W[f"L{l}.G"], S))
            V[f"L{l}.W2"] = to_rep_plaintexts(full_from_hidden(W[f"L{l}.W2"], S))
            V[f"L{l}.CR"] = cache_add_vector(W[f"L{l}.CR"], S)
            if last:
                V[f"L{l}.XU"] = from_rep_plaintexts(full_block0(W[f"L{l}.XU"], S))
                V["BIAS"] = tile_rep(W["BIAS"], S)
            else:
                V[f"L{l}.B2"] = tile_rep(W[f"L{l}.B2"], S)
        return V

    def _pts(self, name, level):
        key = (name, level)
        pts = self._pt.get(key)
        if pts is None:
            pts = [[self.be.encode(v, level) for v in row] for row in self.vec[name]]
            if self.cache_plaintexts:
                self._pt[key] = pts
        return pts

    def _add_static(self, ct, name):
        lvl = self.be.add_level(ct)
        key = (name, "add", lvl)
        pt = self._pt.get(key)
        if pt is None:
            pt = self.be.encode(self.vec[name], lvl)
            self._pt[key] = pt
        return self.be.add_pt(ct, pt)

    def warmup(self, levels):
        """Encode all static plaintexts up front. levels: dict name -> level."""
        for name, lvl in levels.items():
            self._pts(name, lvl)

    # ---- building blocks ----
    def _inner(self, rots, P, i):
        be = self.be
        return be.add_many([be.mul_pt(rots[j], P[i][j]) for j in range(BABY)])

    def _matvec(self, rots, name, level):
        be = self.be
        P = self._pts(name, level)
        acc = None
        for i in range(GIANT):
            inner = self._inner(rots, P, i)
            if i:
                inner = be.rot(inner, BABY * i)
            acc = inner if acc is None else be.add(acc, inner)
        return acc

    def _window(self, ct):
        for s in WINDOW_STEPS:
            ct = self.be.add(ct, self.be.rot(ct, s))
        return ct

    def _cross(self, ct):
        for s in cross_steps(self.S):
            ct = self.be.add(ct, self.be.rot(ct, s))
        return ct

    def _insert(self, sess, l, k, v):
        be = self.be
        if sess.K[l] is None:
            sess.K[l], sess.V[l] = k, v
        else:
            sess.K[l] = be.add(be.rot(sess.K[l], BLK), k)
            sess.V[l] = be.add(be.rot(sess.V[l], BLK), v)

    def _attention(self, sess, l, query):
        """av_raw = (window_sum(query * K) + cr)^2 * V   (V layout; cross_sum turns it into rep)."""
        be = self.be
        s = self._window(be.mul_ct(query, sess.K[l]))
        s = self._add_static(s, f"L{l}.CR")
        return be.mul_ct(be.square(s), sess.V[l])

    # ---- one decoding step ----
    def step(self, sess, ct_onehot, need_logits=True):
        be, fm = self.be, self.fm
        if sess.t >= self.T:
            raise ValueError(f"context window of {self.T} tokens exhausted")
        # layer 0: input = encrypted one-hot (rep layout)
        lvl = be.level(ct_onehot)
        rots = be.baby(ct_onehot)
        self._insert(sess, 0, self._matvec(rots, "L0.K", lvl), self._matvec(rots, "L0.V", lvl))
        x0 = self._matvec(rots, "L0.X", lvl)
        x = be.add(x0, self._cross(self._attention(sess, 0, ct_onehot)))
        logits = None
        for l in range(1, fm.L):
            last = l == fm.L - 1
            lvl = be.level(x)
            rots = be.baby(x)
            self._insert(sess, l, self._matvec(rots, f"L{l}.K", lvl), self._matvec(rots, f"L{l}.V", lvl))
            if last and not need_logits:
                break
            z = self._matvec(rots, f"L{l}.Z", lvl)
            kind = fm.cfg.get("mlp", "sq")
            gate = self._matvec(rots, f"L{l}.G", lvl) if kind != "sq" else None
            av = self._attention(sess, l, x)
            if kind == "sq":
                m = be.square(z)
            elif kind == "bil":
                m = be.mul_ct(z, gate)
            else:                                    # degree 4, uses the spare level of the MLP path
                m = be.square(be.mul_ct(z, gate))
            lvl_m = be.level(m)
            rm = be.baby(m)
            P2 = self._pts(f"L{l}.W2", lvl_m)
            PX = self._pts(f"L{l}.XU", lvl) if last else None
            acc = None
            for i in range(GIANT):
                inner = self._inner(rm, P2, i)
                if last:   # x1 @ Wu shares the giant-step rotations (merged BSGS)
                    inner = be.add(inner, self._inner(rots, PX, i))
                if i:
                    inner = be.rot(inner, BABY * i)
                acc = inner if acc is None else be.add(acc, inner)
            tot = self._cross(be.add(acc, av))      # MLP cross-sum == attention block-sum
            if last:
                logits = self._add_static(tot, "BIAS")
            else:
                x = self._add_static(be.add(x, tot), f"L{l}.B2")
        sess.t += 1
        return logits

"""CipherChat v3 – non-interactive homomorphic decode step for the FHE-adapted Qwen2.5 (cipherchat/v3/model.py).

The client encrypts the embedding of one token; the server runs ALL transformer blocks on ciphertexts (norms,
attention incl. power-softmax and its division, polynomial activations, bootstrapping) and returns the encrypted
final residual vector. The client decrypts it and applies the final RMSNorm + LM head (public open weights).

The same engine runs on two backends:
  NumpyBackend    exact slot simulation with level accounting (validates layouts, rotations, the level schedule)
  PhantomBackend  CKKS on the GPU (pybind module v3fhe: PhantomFHE + NEXUS-CUDA bootstrapping)

Slot layouts (n = 32768 slots, block D = 1024, R = 32 replicas, S = D / R = 32):
  PER  periodic:   slot r*D + j = x[j]                       (= a sparse CKKS vector with 1024 slots)
  OFF  offset:     slot r*D + j = x[(j + r*S) mod D]          (input layout of every matrix-vector product)
  KV cache:        slot j*128 + g*64 + d   (position j < 256, kv head g, dim d)
  q rows are permuted to o = m*128 + g*64 + d with head h = 7 g + m (m < 7) so that head pairs line up with
  the cache; the attention output comes back in the same order (W_o columns permuted accordingly).
Every bootstrapped ciphertext is PER (period 1024) or a replicated scalar, so one sparse bootstrapper (logn = 10)
serves everything.
"""
import json
import math
from collections import deque

import numpy as np

N_SLOTS = 32768
D = 1024
R = N_SLOTS // D
S = D // R
T_MAX = 256
BABY = 8          # matvec baby steps (S = BABY * GIANT)
BABY_CONV = 8     # PER -> OFF baby steps (R = BABY_CONV * GIANT_CONV)
# bootstrapping input bounds (powers of two) per site; calibrated by scripts/v3_bounds.py on the trained model
DEFAULT_BOUNDS = {"x": 128.0, "x2": 128.0, "o": 16.0, "num": 256.0, "den": 256.0, "h": 64.0, "q": 16.0, "kv": 16.0}


# ----------------------------------------------------------------------------- backends
class NCt:
    __slots__ = ("v", "level")

    def __init__(self, v, level):
        self.v, self.level = v, level


class NumpyBackend:
    """Exact slot simulation. Levels are tracked like CKKS (every multiplication costs one) and checked."""
    name = "numpy"

    def __init__(self, n=N_SLOTS, levels=10, boot_noise=0.0, seed=0):
        self.n, self.L = n, levels
        self.boot_noise = boot_noise
        self.rng = np.random.default_rng(seed)
        self.counters = {}
        self.boot_max_abs = 0.0
        self.section_name = "-"
        self.obs = {}

    def _c(self, k, v=1):
        key = self.section_name + "|" + k
        self.counters[key] = self.counters.get(key, 0) + v

    def section(self, name):
        self.section_name = name

    def encrypt(self, v, level):
        x = np.zeros(self.n)
        x[: len(v)] = v
        return NCt(x, level)

    def decrypt(self, ct):
        return ct.v.copy()

    def level(self, ct):
        return ct.level

    def rot(self, ct, k):
        self._c("rot")
        return NCt(np.roll(ct.v, -k), ct.level)

    def add(self, a, b):
        return NCt(a.v + b.v, min(a.level, b.level))

    def sub(self, a, b):
        return NCt(a.v - b.v, min(a.level, b.level))

    def add_pt(self, a, vec):
        return NCt(a.v + np.asarray(vec, dtype=np.float64), a.level)

    def mul_int(self, a, k):
        return NCt(a.v * k, a.level)

    def mul_pt_sum(self, cts, vecs):
        lvl = min(c.level for c in cts) - 1
        assert lvl >= 0, "out of levels (pt mult)"
        self._c("ptmul", len(cts))
        acc = np.zeros(self.n)
        for c, v in zip(cts, vecs):
            acc += c.v * v
        return NCt(acc, lvl)

    def mul(self, a, b):
        lvl = min(a.level, b.level) - 1
        assert lvl >= 0, "out of levels (ct mult)"
        self._c("ctmul")
        return NCt(a.v * b.v, lvl)

    def mod_down(self, ct, level):
        return NCt(ct.v, min(ct.level, level))

    def bootstrap_scalar(self, ct):
        self._c("bootstrap_scalar")
        v = ct.v
        assert np.allclose(v, v[0], atol=1e-9 * max(1.0, np.abs(v).max())), "scalar bootstrap input not constant"
        self.boot_max_abs = max(self.boot_max_abs, float(np.abs(v).max()))
        return NCt(v.copy(), self.L)

    def observe(self, site, ct, B):
        """max |true value| (= max |ct| * B) seen at a bootstrapping site, and whether |ct| <= 1 held."""
        m = float(np.abs(ct.v).max())
        o = self.obs.setdefault(site, {"max_true": 0.0, "max_scaled": 0.0, "n": 0})
        o["max_true"] = max(o["max_true"], m * B)
        o["max_scaled"] = max(o["max_scaled"], m)
        o["n"] += 1

    def bootstrap(self, ct):
        self._c("bootstrap")
        v = ct.v
        assert np.allclose(v, np.tile(v[:D], self.n // D), atol=1e-9 * max(1.0, np.abs(v).max())), \
            "bootstrap input is not 1024-periodic"
        self.boot_max_abs = max(self.boot_max_abs, float(np.abs(v).max()))
        if self.boot_noise:
            v = v + np.tile(self.rng.normal(0, self.boot_noise, D), self.n // D)
        return NCt(v.copy(), self.L)


class PhantomBackend:
    """CKKS on the GPU through the pybind module v3fhe (cipherchat/v3/fhe/src/pyv3fhe.cu)."""
    name = "phantom"

    def __init__(self, cfg, logn_boot=10, profile=True, extra_steps=()):
        import v3fhe
        self.ctx = v3fhe.Ctx(json.dumps(cfg), logn_boot)
        info = self.ctx.info()
        self.n, self.L = info["slots"], info["compute_levels"]
        self.boot_steps = list(self.ctx.boot_steps())
        self.n_keys = self.ctx.gen_rotation_keys(list(extra_steps))
        self.ctx.set_profiling(profile)
        self.boot_max_abs = 0.0

    def section(self, name):
        self.ctx.set_section(name)

    def encrypt(self, v, level):
        return self.ctx.encrypt(np.asarray(v, dtype=np.float64), level)

    def decrypt(self, ct):
        return self.ctx.decrypt(ct)

    def level(self, ct):
        return self.ctx.level(ct)

    def rot(self, ct, k):
        return self.ctx.rot(ct, int(k))

    def add(self, a, b):
        return self.ctx.add(a, b)

    def sub(self, a, b):
        return self.ctx.sub(a, b)

    def add_pt(self, a, vec):
        return self.ctx.add_pt(a, np.asarray(vec, dtype=np.float64))

    def mul_int(self, a, k):
        return self.ctx.mul_int(a, int(k))

    def mul_pt_sum(self, cts, vecs):
        return self.ctx.mul_pt_sum(list(cts), [np.asarray(v, dtype=np.float64) for v in vecs])

    def mul(self, a, b):
        return self.ctx.mul(a, b)

    def mod_down(self, ct, level):
        return self.ctx.mod_down(ct, int(level))

    def bootstrap(self, ct):
        return self.ctx.bootstrap(ct)

    def bootstrap_scalar(self, ct):
        return self.ctx.bootstrap_scalar(ct)


# ----------------------------------------------------------------------------- rotation planning
class RotPlan:
    """Rotation keys cost ~160 MB each at N = 2^16, so only a small set exists (bootstrapping keys + a few extra);
    any other rotation is composed of several keyed rotations (shortest path over Z_n, BFS)."""

    def __init__(self, n, steps):
        self.n = n
        self.steps = sorted({s % n for s in steps if s % n})
        dist = np.full(n, -1, dtype=np.int64)
        prev = np.full(n, -1, dtype=np.int64)
        dist[0] = 0
        frontier = np.array([0])
        while frontier.size:
            nxt = []
            for s in self.steps:
                cand = (frontier + s) % n
                new = cand[dist[cand] < 0]
                if new.size:
                    new, idx = np.unique(new, return_index=True)
                    dist[new] = dist[frontier[0]] + 1
                    prev[new] = s
                    nxt.append(new)
            frontier = np.unique(np.concatenate(nxt)) if nxt else np.array([], dtype=np.int64)
        self.dist, self.prev = dist, prev

    def path(self, k):
        k %= self.n
        out = []
        while k:
            s = int(self.prev[k])
            out.append(s)
            k = (k - s) % self.n
        return out


def engine_rotation_steps():
    """Extra keys requested besides the bootstrapping set (which already holds 2^i, i < 15)."""
    n = N_SLOTS
    ex = [3, 5, 7, 24, 96, 160, 224, 768]                        # matvec babies / giants, PER->OFF steps
    ex += [n - 1, n - 2, n - 4, n - 8, n - 16, n - 32]          # dd broadcast
    ex += [n - 128, n - 256, n - 512]                            # q replication / cache insertion
    return ex


# ----------------------------------------------------------------------------- engine
class Session3:
    def __init__(self, n_layers):
        self.K = [None] * n_layers
        self.V = [None] * n_layers
        self.t = 0


class Engine3:
    def __init__(self, fm, be, boot_steps=None):
        """fm: FoldedModel (cipherchat/v3/fold.py); be: backend."""
        assert fm.p in (1, 2), "power attention: 2p must be a power of two (p = 1 or 2)"
        self.fm, self.be = fm, be
        self.n = be.n
        boot_steps = boot_steps or getattr(be, "boot_steps", None) or [1 << i for i in range(15)]
        steps = list(boot_steps) + engine_rotation_steps()
        self.rp = RotPlan(self.n, steps)
        self._idx_cache = {}
        self._diag_cache = {}
        self._masks()
        self.stats = {"bootstraps": 0}
        self.bounds = dict(DEFAULT_BOUNDS)
        self.bounds.update(getattr(fm, "boot_bounds", None) or {})
        self.cur_layer = 0
        # NEXUS logn = 0 bootstrapping for replicated scalars is ~40 % faster (97 vs 160 ms, 14 levels left) but its
        # error is ~5e-3 at |x| = 1 (vs 7e-6), too coarse for the RMSNorm factor -> off by default
        self.scalar_boot = False

    # ---------------- helpers
    def rot(self, ct, k):
        for s in self.rp.path(k):
            ct = self.be.rot(ct, s)
        return ct

    def need(self, ct, k, site, prescaled=1.0, scalar=False):
        """Make sure ct has at least k levels left; bootstrap if not. ct must be PER / a replicated scalar.
        CKKS bootstrapping (EvalMod) is accurate only for |x| <~ 1 (scripts/v3_boot_range.py: rel. error 7e-6 at 1,
        2e-3 at 16), so the input is scaled down by the site's bound B (power of two) first:
          prescaled = 1: ct holds the true values; multiply by 1/B here (costs the one level ct must still have)
          prescaled = B: the caller already folded 1/B into an earlier plaintext product (free)
        After bootstrapping the values are scaled back up by B through B/2 doublings (additions, no level)."""
        if self.be.level(ct) >= k:
            return ct if prescaled == 1.0 else self.scale_up(ct, prescaled)
        B = prescaled if prescaled != 1.0 else self.bound(site)
        if prescaled == 1.0 and B > 1.0:
            ct = self.be.mul_pt_sum([ct], [self.const(1.0 / B)])
        self.stats["bootstraps"] += 1
        self.stats.setdefault("boot_sites", {}).setdefault(site, 0)
        self.stats["boot_sites"][site] += 1
        if hasattr(self.be, "observe"):
            self.be.observe(f"{site}@{self.cur_layer}", ct, B)
        ct = self.be.bootstrap_scalar(ct) if (scalar and self.scalar_boot) else self.be.bootstrap(ct)
        return self.scale_up(ct, B)

    def bound(self, site):
        """bootstrapping bound of `site` in the current layer (per-layer calibration, else the global default)."""
        return max(1.0, float(self.bounds.get(f"{site}@{self.cur_layer}", self.bounds.get(site, 1.0))))

    def scale_up(self, ct, B):
        assert B >= 1 and abs(math.log2(B) - round(math.log2(B))) < 1e-9, "bounds are powers of two"
        for _ in range(int(round(math.log2(B)))):
            ct = self.be.add(ct, ct)
        return ct

    def _masks(self):
        n = self.n
        s = np.arange(n)
        self.mask_block = [((s // D) == r).astype(float) for r in range(R)]
        self.mask_m = [(((s % D) // 128) == m).astype(float) for m in range(8)]     # j = m mod 8 rows

    # ---------------- layouts
    def to_off(self, ct):
        """PER -> OFF: out = sum_r mask_r * rot(x, r S), BSGS over r (BABY_CONV x GIANT_CONV), 1 level."""
        G = R // BABY_CONV
        babies = [ct] + [self.rot(ct, b * S) for b in range(1, BABY_CONV)]
        acc = None
        for g in range(G):
            vecs = [np.roll(self.mask_block[g * BABY_CONV + b], g * BABY_CONV * S) for b in range(BABY_CONV)]
            inner = self.be.mul_pt_sum(babies, vecs)
            inner = self.rot(inner, g * BABY_CONV * S)
            acc = inner if acc is None else self.be.add(acc, inner)
        return acc

    def _diag_index(self):
        """index maps of the offset->periodic matvec: P_t[slot] = M[row(slot), col(slot, t)]."""
        if "diag" not in self._idx_cache:
            n = self.n
            sl = np.arange(n)
            r, j = sl // D, sl % D
            rows, cols = [], []
            for t in range(S):
                rp = (r + (j + t >= D)) % R
                k = (rp * S + t) % D
                rows.append(j)
                cols.append((j + k) % D)
            self._idx_cache["diag"] = (np.array(rows), np.array(cols))
        return self._idx_cache["diag"]

    def diagonals(self, M, static=True):
        """the 32 pre-rotated plaintext diagonals of a 1024 x 1024 block (BABY x GIANT order). Static weight blocks
        are cached (float32, ~4 MB each, ~1.7 GB for the whole model) so the per-token cost is only the encode."""
        key = id(M)
        if static and key in self._diag_cache:
            return self._diag_cache[key][1]
        rows, cols = self._diag_index()
        flat = (rows * D + cols)
        Mf = np.ascontiguousarray(M).ravel()
        G = S // BABY
        out = [[np.roll(Mf[flat[g * BABY + b]], g * BABY) for b in range(BABY)] for g in range(G)]
        if static:
            if self.be.name != "numpy":                # float32 cache for the GPU path (encode precision ~1e-7 rel.)
                out = [[v.astype(np.float32) for v in row] for row in out]
            self._diag_cache[key] = (M, out)          # keep M alive so id(M) stays unique
        return out

    def matvec_multi(self, x_off, mats, static=True):
        """y_i = M_i x for several 1024 x 1024 blocks sharing the input (baby rotations computed once).
        x_off: OFF ciphertext; mats: list of (D, D) arrays; returns list of PER ciphertexts (1 level)."""
        G = S // BABY
        babies = [x_off] + [self.rot(x_off, b) for b in range(1, BABY)]
        outs = []
        for M in mats:
            dg = self.diagonals(M, static)
            acc = None
            for g in range(G):
                inner = self.be.mul_pt_sum(babies, dg[g])
                inner = self.rot(inner, g * BABY)
                acc = inner if acc is None else self.be.add(acc, inner)
            outs.append(acc)
        return [self.fold(a) for a in outs]

    def matvec_sum(self, x_offs, mats):
        """y = sum_c M_c x_c (inputs in OFF layout) with one shared fold; returns PER (1 level)."""
        G = S // BABY
        acc = None
        for x_off, M in zip(x_offs, mats):
            babies = [x_off] + [self.rot(x_off, b) for b in range(1, BABY)]
            dg = self.diagonals(M)
            for g in range(G):
                inner = self.rot(self.be.mul_pt_sum(babies, dg[g]), g * BABY)
                acc = inner if acc is None else self.be.add(acc, inner)
        return self.fold(acc)

    def fold(self, ct):
        s = D
        while s < self.n:
            ct = self.be.add(ct, self.rot(ct, s))
            s *= 2
        return ct

    def block_sum(self, ct):
        """every slot := sum of its 1024-block (all-reduce inside the block)."""
        s = 1
        while s < D:
            ct = self.be.add(ct, self.rot(ct, s))
            s *= 2
        return ct

    def tile(self, v):
        out = np.zeros(D)
        out[: len(v)] = v
        return np.tile(out, self.n // D)

    def const(self, c):
        return np.full(self.n, float(c))

    # ---------------- Newton approximations (exactly approx.NewtonApprox)
    def _start(self, w_ct, ap, w_scale):
        """start polynomial p(u), u = a (w_scale w) + b, Horner; costs `degree` levels."""
        be = self.be
        coef, a, b = ap["coef"], ap["a"], ap["b"]
        d = len(coef) - 1
        assert d >= 1, "degree 0 start"
        u = be.add_pt(be.mul_pt_sum([w_ct], [self.const(a * w_scale)]), self.const(b)) if d >= 2 else None
        y = be.add_pt(be.mul_pt_sum([w_ct], [self.const(coef[d] * a * w_scale)]), self.const(coef[d] * b + coef[d - 1]))
        for k in range(d - 2, -1, -1):
            y = be.add_pt(be.mul(y, u), self.const(coef[k]))
        return y

    def inv_sqrt(self, s_ct, ap, z_scale, out_scale=1.0):
        """out_scale * (z_scale * s)^(-1/2), depth degree + 2 iters.
        Intermediate iterates are kept as Y = kappa y (kappa = 2^k public) so that the Newton step
        Y' = 3Y - (z / kappa^2) Y^3 needs no non-integer constant; the LAST step returns out_scale * y directly:
        y' = (1.5 out_scale / kappa) Y - (0.5 out_scale z / kappa^3) Y^3, same depth."""
        be = self.be
        y = self._start(s_ct, ap, z_scale)
        it = ap["iters"]
        if it == 0:
            return be.mul_pt_sum([y], [self.const(out_scale)])
        kappa = 1.0
        for k in range(it):
            t1 = be.mul(y, y)
            if k < it - 1:
                zk = be.mul_pt_sum([s_ct], [self.const(z_scale / kappa ** 2)])
                t3 = be.mul(t1, be.mul(zk, y))
                y = be.sub(be.mul_int(y, 3), t3)
                kappa *= 2.0
            else:
                zk = be.mul_pt_sum([s_ct], [self.const(0.5 * out_scale * z_scale / kappa ** 3)])
                t3 = be.mul(t1, be.mul(zk, y))
                lin = be.mul_pt_sum([y], [self.const(1.5 * out_scale / kappa)])
                y = be.sub(lin, t3)
        return y

    def inv(self, w_ct, ap, B=1.0):
        """1 / w for w = B * w_ct (w_ct holds the normalised denominator divided by the bound B, power of two).
        Newton y' = 2y - w y^2 = 2y - B (w_ct y^2), B through doublings."""
        be = self.be
        y = self._start(w_ct, ap, B)
        for _ in range(ap["iters"]):
            t1 = be.mul(y, y)
            t2 = self.scale_up(be.mul(w_ct, t1), B)
            y = be.sub(be.mul_int(y, 2), t2)
        return y

    # ---------------- one decode step
    def new_session(self):
        return Session3(self.fm.L)

    def prefill_public(self, sess):
        """Insert the public prefix K/V (computed in plaintext with exact ops) into the encrypted caches."""
        fm = self.fm
        for i in range(fm.L):
            K = np.zeros(self.n)
            V = np.zeros(self.n)
            for j in range(fm.n_pub):
                K[j * 128: (j + 1) * 128] = fm.k_pub[i][j]
                V[j * 128: (j + 1) * 128] = fm.v_pub[i][j]
            sess.K[i] = self.be.encrypt(K, fm.cache_level)
            sess.V[i] = self.be.encrypt(V, fm.cache_level)
        sess.t = fm.n_pub

    def step(self, sess, x_ct):
        """x_ct: encrypted embedding (PER) of the token at position sess.t. Returns encrypted final residual (PER)."""
        for i in range(self.fm.L):
            x_ct = self.layer(i, sess, x_ct)
        sess.t += 1
        return x_ct

    def layer(self, i, sess, x):
        be, fm, t = self.be, self.fm, sess.t
        self.cur_layer = i
        lw = fm.layer(i, t)
        isq_depth = len(fm.norm_approx["coef"]) - 1 + 2 * fm.norm_approx["iters"]
        # ---- RMSNorm 1 (scalar y1, returned as y1/2 so that it bootstraps inside [-1, 1]) + q / kv projections
        be.section("norm1")
        x = self.need(x, max(10, 1 + isq_depth), "x")   # q reaches the attention with 7 levels: x->OFF->matvec->*y1
        ssum = self.block_sum(be.mul(x, x))
        y1 = self.inv_sqrt(ssum, fm.norm_approx, 1.0 / (fm.d * lw["m1"]), out_scale=0.5)
        y1 = self.need(y1, 8, "y1", prescaled=2.0, scalar=True)
        be.section("qkv")
        xo = self.to_off(x)
        q_raw, kv_raw = self.matvec_multi(xo, [lw["Aq"], lw["Akv"]], static=False)   # RoPE: new per position
        q = be.add_pt(be.mul(y1, q_raw), self.tile(lw["bq"]))
        kv = be.add_pt(be.mul(y1, kv_raw), self.tile(lw["bkv"]))
        # ---- cache insertion at position t
        be.section("attn.cache")
        r = t % 8
        mt = np.zeros(self.n)
        mt[t * 128: (t + 1) * 128] = 1.0
        kv = self.need(kv, 2, "kv")
        sess.K[i] = be.add(sess.K[i], be.mul_pt_sum([self.rot(kv, -128 * r)], [mt]))
        sess.V[i] = be.add(sess.V[i], be.mul_pt_sum([self.rot(kv, 128 - 128 * r)], [mt]))
        # ---- attention per query-head slot m (heads m and m + 7)
        out_terms, den_terms = [], []
        q = self.need(q, 7, "q")
        valid = np.zeros(self.n)
        valid[: (t + 1) * 128] = 1.0
        sl = np.arange(self.n)
        g_of = (sl % 128) // 64
        d0 = (sl % 64) == 0
        for m in range(7):
            be.section("attn.scores")
            qm = be.mul_pt_sum([q], [self.mask_m[m]])
            for sh in (128, 256, 512):
                qm = be.add(qm, self.rot(qm, -sh))
            prod = be.mul(qm, sess.K[i])
            for sh in (1, 2, 4, 8, 16, 32):
                prod = be.add(prod, self.rot(prod, sh))
            be.section("attn.softmax")
            heads = g_of * 7 + m
            amask = valid * d0
            e = be.add_pt(be.mul_pt_sum([prod], [lw["alpha"][heads] * amask]), lw["c"][heads] * amask)
            for _ in range(fm.p):
                e = be.mul(e, e)
            for sh in (1, 2, 4, 8, 16, 32):
                e = be.add(e, self.rot(e, -sh))
            be.section("attn.av")
            nm = be.mul(e, sess.V[i])
            sh = 128
            while sh < self.n:
                nm = be.add(nm, self.rot(nm, sh))
                sh *= 2
            out_terms.append(nm)
            if fm.attn_norm == "div":
                be.section("attn.softmax")
                dm = e
                sh = 128
                while sh < self.n:
                    dm = be.add(dm, self.rot(dm, sh))
                    sh *= 2
                den_terms.append(dm)
        # pack the 7 head pairs into one PER ciphertext (order o = m*128 + g*64 + d); bounds folded into the masks
        be.section("attn.pack")
        o_idx = np.arange(D)
        heads_o = np.where(o_idx < 896, ((o_idx % 128) // 64) * 7 + o_idx // 128, 0)   # head of row o
        npos = t + 1
        Bo = self.bound("o")
        if fm.attn_norm == "pos":
            wt = np.tile(np.where(o_idx < 896, np.exp(-lw["logk"][heads_o]) / npos, 0.0), R)
            o = be.mul_pt_sum(out_terms, [self.mask_m[m] * wt / Bo for m in range(7)])
        else:
            # o = (sum_j e_j v_j / ((t+1) d_ref)) * inv(sum_j e_j / ((t+1) d_ref))
            wt = np.tile(np.where(o_idx < 896, 1.0 / (npos * lw["d_ref"][heads_o]), 0.0), R)
            Bd = self.bound("den")
            Bo = self.bound("num")                    # num and o = num * inv share the bound of num
            num = be.mul_pt_sum(out_terms, [self.mask_m[m] * wt / Bo for m in range(7)])
            den = be.mul_pt_sum(den_terms, [self.mask_m[m] * wt / Bd for m in range(7)])
            # padding rows (o >= 896): a denominator inside the range keeps the inverse finite there
            pad = 0.5 * (fm.attn_approx["lo"] + fm.attn_approx["hi"]) / Bd
            den = be.add_pt(den, np.tile((o_idx >= 896) * pad, R))
            den_l = be.level(den)
            if den_l < fm.inv_depth + 1:                 # +1: the product num * inv
                self.stats["bootstraps"] += 1
                if hasattr(be, "observe"):
                    be.observe(f"den@{i}", den, Bd)
                den = be.bootstrap(den)
            be.section("attn.inverse")
            inv = self.inv(den, fm.attn_approx, Bd)
            if be.level(num) < 1:
                self.stats["bootstraps"] += 1
                if hasattr(be, "observe"):
                    be.observe(f"num@{i}", num, Bo)
                num = be.bootstrap(num)
            o = be.mul(num, inv)                          # = o / Bo
        be.section("attn.out")
        o = self.need(o, 3, "o", prescaled=Bo)
        attn = self.matvec_multi(self.to_off(o), [lw["Wo"]])[0]
        x2 = be.add(x, attn)
        # ---- RMSNorm 2 + MLP
        be.section("norm2")
        x2 = self.need(x2, max(9, 1 + isq_depth), "x2")    # raw (x2 -> OFF -> matvec) needs 7 levels, the norm 1 + depth
        s2 = self.block_sum(be.mul(x2, x2))
        y2 = self.inv_sqrt(s2, fm.norm_approx, 1.0 / (fm.d * lw["m2"]), out_scale=0.5)
        y2 = self.need(y2, 7, "y2", prescaled=2.0, scalar=True)
        be.section("ffn.up")
        x2o = self.to_off(x2)
        nc = len(lw["Ag"])
        raw = self.matvec_multi(x2o, list(lw["Ag"]) + list(lw["Au"]))
        hs = []
        be.section("ffn.act")
        for c in range(nc):
            g = be.mul(y2, raw[c])
            u = be.mul(y2, raw[nc + c])
            gsq = be.mul(g, g)
            c0, c1, c2 = (self.tile(v) for v in lw["act"][c])
            poly = be.add_pt(be.mul_pt_sum([gsq, g], [c2, c1]), c0)
            hs.append(be.mul(poly, u))
        be.section("ffn.down")
        hs = [self.need(h, 3, "h") for h in hs]
        d_out = self.matvec_sum([self.to_off(h) for h in hs], lw["Wd"])
        be.section("residual")
        return be.add(x2, d_out)

"""CipherChat v5 – the v3 decode step (Qwen2.5-0.5B, all 24 layers on ciphertexts) WITHOUT bootstrapping.

    *** NOT SECURE. v5 runs on CKKS with RLWE error e = 0 (cipherchat/v5/fhe). It is an upper bound that shows what
    *** bootstrapping and the security parameters cost; the secret key follows from the public key.

Without e there is no lattice-security bound log QP <= f(N), so the modulus chain can be as long as the whole model
is deep (~36 levels per layer, ~870 in total) and N only has to be as large as the slot layout needs. Everything else
is the v3 engine (cipherchat/v3/engine.py) with the same model, weights folding and approximations:

  - the slot layout is parameterised by n = N/2 (v3: n = 32768 fixed): block D = 1024, R = n / D replicas,
    S = D / R diagonals per 1024 x 1024 matvec, KV cache slot j*128 + g*64 + d for j < n / 128 positions
  - no bootstrapping: no `need()`, no scaling to |x| <= 1 (the norm factors are returned at scale 1, the attention
    output is packed without the 1/B factor)
  - the public prefix K/V are added as plaintexts to the first encrypted cache entry (v3 encrypts them at the cache
    level). Same values, no ciphertext in memory before the first token.
  - only the divisionless attention (`norm = pos`, the model the v3 server runs) is implemented, with the per-head
    factor exp(-logk_h) folded into the power (numerically better, mathematically identical; see layer())

Backends: NumpyBackend from v3 (exact slot simulation with level accounting, n configurable) and PhantomBackend5
(pybind module v5fhe).
"""
import json
import math

import numpy as np

from cipherchat.v3.engine import NumpyBackend, RotPlan  # noqa: F401  (NumpyBackend re-exported for the scripts)


class Layout:
    """Slot layout for n slots (see module docstring)."""

    def __init__(self, n, D=1024, baby=None, baby_conv=None):
        assert n >= D and n % D == 0 and (n & (n - 1)) == 0, "n must be a power of two >= D"
        self.n, self.D = n, D
        self.R = n // D
        self.S = D // self.R
        self.BABY = baby or 2 ** int(math.ceil(math.log2(math.sqrt(self.S))))
        assert self.S % self.BABY == 0
        self.GIANT = self.S // self.BABY
        self.BABY_CONV = baby_conv or max(1, 2 ** int(math.floor(math.log2(math.sqrt(self.R)))))
        assert self.R % self.BABY_CONV == 0
        self.G_CONV = self.R // self.BABY_CONV
        self.T_MAX = n // 128

    def as_dict(self):
        return dict(n=self.n, D=self.D, R=self.R, S=self.S, BABY=self.BABY, GIANT=self.GIANT,
                    BABY_CONV=self.BABY_CONV, G_CONV=self.G_CONV, T_MAX=self.T_MAX)


def used_rotation_steps(lay):
    """every rotation step the engine performs (before composition by RotPlan)."""
    n, D, S = lay.n, lay.D, lay.S
    st = set(range(1, lay.BABY)) | {g * lay.BABY for g in range(1, lay.GIANT)}           # matvec
    st |= {b * S for b in range(1, lay.BABY_CONV)} | {g * lay.BABY_CONV * S for g in range(1, lay.G_CONV)}  # PER->OFF
    st |= {D << k for k in range(int(math.log2(n // D)))}                                # fold
    st |= {1 << k for k in range(int(math.log2(D)))}                                      # block sum
    st |= {1 << k for k in range(6)} | {-(1 << k) for k in range(6)}                      # attention d sums / bcast
    st |= {-128, -256, -512} | {128 << k for k in range(int(math.log2(n // 128)))}         # q replicate, j sum
    st |= {-128 * r for r in range(1, 8)} | {128 - 128 * r for r in range(2, 8)}           # cache insertion
    return sorted({s % n for s in st} - {0})


def rotation_keys(lay, budget="small"):
    """keyed steps. Each key is one top-level ciphertext big (~0.1 GB at N = 2^13 with ~870 primes), so 'small'
    keys only half of the baby / giant steps, the powers of two and the negative attention steps; every other
    rotation is composed by RotPlan (BFS shortest path). 'all' keys every used step."""
    n = lay.n
    if budget == "all":
        return used_rotation_steps(lay)
    st = set(range(1, lay.BABY // 2 + 1)) | {g * lay.BABY for g in range(1, lay.GIANT // 2 + 1)}
    st |= {1 << k for k in range(int(math.log2(n)))}
    st |= {-(1 << k) for k in range(6)} | {-128, -256, -512}
    return sorted({s % n for s in st} - {0})


class NumpyBackendLPP(NumpyBackend):
    """Exact slot simulation with the level accounting of two levels per prime (v5fhe, levels_per_prime = 2).
    Virtual level v = 2 l + full. Everything costs one level as usual, except a ciphertext product in which neither
    operand can enter "full" at the common prime level (both half): that one costs two."""

    def mul(self, a, b):
        l = min(a.level // 2, b.level // 2)
        nfull = int(a.level >= 2 * l + 1) + int(b.level >= 2 * l + 1)
        lvl = 2 * l if nfull == 2 else (2 * l - 1 if nfull == 1 else 2 * l - 2)
        assert lvl >= 0, "out of levels (ct mult)"
        self._c("ctmul")
        if nfull == 0:
            self._c("ctmul_half_half")
        return type(a)(a.v * b.v, lvl)


def chain_primes(logN, bits, count):
    """the `count` largest primes q = 1 mod 2N below 2^bits, largest first (PhantomFHE's CoeffModulus::Create order)."""
    def is_prime(n):
        if n < 2:
            return False
        d, s = n - 1, 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            x = pow(a, d, n)
            if x in (1, n - 1):
                continue
            for _ in range(s - 1):
                x = x * x % n
                if x == n - 1:
                    break
            else:
                return False
        return True
    step = 2 << logN
    q = (1 << bits) - step + 1
    out = []
    while len(out) < count:
        if is_prime(q):
            out.append(q)
        q -= step
    return out


class NumpyBackendScaleDev(NumpyBackend):
    """Exact slot simulation plus ONLY the scale-bookkeeping error of the GPU backend (one level per prime): a
    ciphertext product is rescaled by the real prime q_l but labelled with the nominal scale Delta, i.e. its value is
    multiplied by Delta / q_l. Plaintext products are exact (plaintexts are encoded at q_l). No rounding, no noise:
    comparing this with the GPU run separates the prime deviation from the CKKS rounding errors."""

    def __init__(self, n, levels, logN, scale_bits):
        super().__init__(n=n, levels=levels)
        # Phantom/SEAL CoeffModulus::Create hands out the primes of one bit size smallest first, so chain position
        # lvl (1 = right after q0) holds the lvl-th smallest of the `levels` largest primes
        primes = chain_primes(logN, scale_bits, levels)       # largest first
        self.ratio = {lvl: float(2 ** scale_bits / primes[levels - lvl]) for lvl in range(1, levels + 1)}

    def mul(self, a, b):
        out = super().mul(a, b)
        out.v = out.v * self.ratio[out.level + 1]           # rescaled by the prime of the level it came from
        return out


class PhantomBackend5:
    """CKKS with e = 0 on the GPU through the pybind module v5fhe (cipherchat/v5/fhe/src/pyv5fhe.cu)."""
    name = "phantom5"

    def __init__(self, cfg, rot_steps, profile=True):
        import v5fhe
        self.ctx = v5fhe.Ctx(json.dumps(cfg))
        info = self.ctx.info()
        self.n, self.L = info["slots"], info["levels"]
        self.info = dict(info)
        self.info["timing"] = json.loads(info["timing"])
        self.n_keys = self.ctx.gen_rotation_keys(list(rot_steps))
        self.info["timing"].update(json.loads(self.ctx.info()["timing"]))
        self.ctx.set_profiling(profile)

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

    def trim(self):
        self.ctx.trim()


class Session5:
    def __init__(self, n_layers):
        self.K = [None] * n_layers
        self.V = [None] * n_layers
        self.t = 0


class Engine5:
    def __init__(self, fm, be, lay, key_steps, diag_dtype=None):
        """fm: FoldedModel (cipherchat/v3/fold.py, unchanged); be: backend with be.n == lay.n; key_steps: keyed
        rotation steps (rotation_keys()); diag_dtype: storage of the cached static weight diagonals (default as in
        v3: float32 on the GPU backend, ~1.6 GB host RAM; float64 doubles that but removes the ~6e-8 relative
        weight rounding)."""
        self.diag_dtype = diag_dtype or (np.float64 if be.name == "numpy" else np.float32)
        assert fm.p in (1, 2), "power attention: 2p must be a power of two (p = 1 or 2)"
        assert fm.attn_norm == "pos", "v5 implements the divisionless attention (norm = pos) only"
        assert be.n == lay.n, f"backend has {be.n} slots, layout {lay.n}"
        self.fm, self.be, self.lay = fm, be, lay
        self.n = lay.n
        self.rp = RotPlan(self.n, key_steps)
        self._idx_cache = {}
        self._diag_cache = {}
        s = np.arange(self.n)
        self.mask_block = [((s // lay.D) == r).astype(float) for r in range(lay.R)]
        self.mask_m = [(((s % lay.D) // 128) == m).astype(float) for m in range(8)]
        self.stats = {"bootstraps": 0, "rotations_keyed": 0}
        self.levels_seen = {}

    # ---------------- helpers
    def rot(self, ct, k):
        for s in self.rp.path(k):
            ct = self.be.rot(ct, s)
            self.stats["rotations_keyed"] += 1
        return ct

    def note(self, name, ct):
        """level of a named intermediate (first token only) – documents the level schedule."""
        key = f"{name}@{self.cur_layer}"
        if key not in self.levels_seen:
            self.levels_seen[key] = self.be.level(ct)

    def tile(self, v):
        out = np.zeros(self.lay.D)
        out[: len(v)] = v
        return np.tile(out, self.n // self.lay.D)

    def const(self, c):
        return np.full(self.n, float(c))

    # ---------------- layouts (v3 with n, R, S from the layout)
    def to_off(self, ct):
        lay = self.lay
        babies = [ct] + [self.rot(ct, b * lay.S) for b in range(1, lay.BABY_CONV)]
        acc = None
        for g in range(lay.G_CONV):
            vecs = [np.roll(self.mask_block[g * lay.BABY_CONV + b], g * lay.BABY_CONV * lay.S)
                    for b in range(lay.BABY_CONV)]
            inner = self.be.mul_pt_sum(babies, vecs)
            inner = self.rot(inner, g * lay.BABY_CONV * lay.S)
            acc = inner if acc is None else self.be.add(acc, inner)
        return acc

    def _diag_index(self):
        if "diag" not in self._idx_cache:
            lay = self.lay
            sl = np.arange(self.n)
            r, j = sl // lay.D, sl % lay.D
            rows, cols = [], []
            for t in range(lay.S):
                rp = (r + (j + t >= lay.D)) % lay.R
                k = (rp * lay.S + t) % lay.D
                rows.append(j)
                cols.append((j + k) % lay.D)
            self._idx_cache["diag"] = (np.array(rows), np.array(cols))
        return self._idx_cache["diag"]

    def diagonals(self, M, static=True):
        key = id(M)
        if static and key in self._diag_cache:
            return self._diag_cache[key][1]
        lay = self.lay
        rows, cols = self._diag_index()
        flat = rows * lay.D + cols
        Mf = np.ascontiguousarray(M).ravel()
        out = [[np.roll(Mf[flat[g * lay.BABY + b]], g * lay.BABY) for b in range(lay.BABY)] for g in range(lay.GIANT)]
        if static:
            if self.diag_dtype != np.float64:
                out = [[v.astype(self.diag_dtype) for v in row] for row in out]
            self._diag_cache[key] = (M, out)
        return out

    def matvec_multi(self, x_off, mats, static=True):
        lay = self.lay
        babies = [x_off] + [self.rot(x_off, b) for b in range(1, lay.BABY)]
        outs = []
        for M in mats:
            dg = self.diagonals(M, static)
            acc = None
            for g in range(lay.GIANT):
                inner = self.rot(self.be.mul_pt_sum(babies, dg[g]), g * lay.BABY)
                acc = inner if acc is None else self.be.add(acc, inner)
            outs.append(acc)
        return [self.fold(a) for a in outs]

    def matvec_sum(self, x_offs, mats):
        lay = self.lay
        acc = None
        for x_off, M in zip(x_offs, mats):
            babies = [x_off] + [self.rot(x_off, b) for b in range(1, lay.BABY)]
            dg = self.diagonals(M)
            for g in range(lay.GIANT):
                inner = self.rot(self.be.mul_pt_sum(babies, dg[g]), g * lay.BABY)
                acc = inner if acc is None else self.be.add(acc, inner)
        return self.fold(acc)

    def fold(self, ct):
        s = self.lay.D
        while s < self.n:
            ct = self.be.add(ct, self.rot(ct, s))
            s *= 2
        return ct

    def block_sum(self, ct):
        s = 1
        while s < self.lay.D:
            ct = self.be.add(ct, self.rot(ct, s))
            s *= 2
        return ct

    # ---------------- Newton 1/sqrt (exactly approx.NewtonApprox, as in v3)
    def _start(self, w_ct, ap, w_scale):
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

    # ---------------- one decode step
    def new_session(self):
        sess = Session5(self.fm.L)
        sess.t = self.fm.n_pub
        return sess

    def prefix_vectors(self, i):
        fm = self.fm
        K = np.zeros(self.n)
        V = np.zeros(self.n)
        for j in range(fm.n_pub):
            K[j * 128: (j + 1) * 128] = fm.k_pub[i][j]
            V[j * 128: (j + 1) * 128] = fm.v_pub[i][j]
        return K, V

    def step(self, sess, x_ct):
        assert sess.t < self.lay.T_MAX, f"context full: the layout holds {self.lay.T_MAX} positions"
        for i in range(self.fm.L):
            x_ct = self.layer(i, sess, x_ct)
        sess.t += 1
        return x_ct

    def layer(self, i, sess, x):
        be, fm, t, lay = self.be, self.fm, sess.t, self.lay
        self.cur_layer = i
        lw = fm.layer(i, t)
        R = lay.R
        self.note("x", x)
        # ---- RMSNorm 1 (scalar y1) + q / kv projections
        be.section("norm1")
        ssum = self.block_sum(be.mul(x, x))
        y1 = self.inv_sqrt(ssum, fm.norm_approx, 1.0 / (fm.d * lw["m1"]))
        be.section("qkv")
        xo = self.to_off(x)
        q_raw, kv_raw = self.matvec_multi(xo, [lw["Aq"], lw["Akv"]], static=False)   # RoPE: new per position
        q = be.add_pt(be.mul(y1, q_raw), self.tile(lw["bq"]))
        kv = be.add_pt(be.mul(y1, kv_raw), self.tile(lw["bkv"]))
        del xo, q_raw, kv_raw, ssum, y1          # free early: ~0.1 GB per ciphertext at the top of the chain
        self.note("q", q)
        # ---- cache insertion at position t (public prefix added as plaintext on the first insertion)
        be.section("attn.cache")
        r = t % 8
        mt = np.zeros(self.n)
        mt[t * 128: (t + 1) * 128] = 1.0
        newK = be.mul_pt_sum([self.rot(kv, -128 * r)], [mt])
        newV = be.mul_pt_sum([self.rot(kv, 128 - 128 * r)], [mt])
        if sess.K[i] is None:
            Kp, Vp = self.prefix_vectors(i)
            sess.K[i], sess.V[i] = be.add_pt(newK, Kp), be.add_pt(newV, Vp)
        else:
            sess.K[i], sess.V[i] = be.add(sess.K[i], newK), be.add(sess.V[i], newV)
        del kv, newK, newV
        # ---- attention per query-head slot m (heads m and m + 7)
        out_terms = []
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
            # exp(-logk_h) of the divisionless normalisation is folded INTO the power: (k (a s + c))^(2p) with
            # k = exp(-logk_h / 2p). Same value as v3's e^(2p) * exp(-logk_h) in the pack step, but e stays O(1):
            # v3 multiplies e ~ 1e5..1e10 with plaintexts ~ 1e-10, which costs ~1e-5 relative plaintext precision
            kh = np.exp(-lw["logk"][heads] / (2 * fm.p))
            e = be.add_pt(be.mul_pt_sum([prod], [lw["alpha"][heads] * kh * amask]), lw["c"][heads] * kh * amask)
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
        be.section("attn.pack")
        o_idx = np.arange(lay.D)
        heads_o = np.where(o_idx < 896, ((o_idx % 128) // 64) * 7 + o_idx // 128, 0)
        wt = np.tile(np.where(o_idx < 896, 1.0 / (t + 1), 0.0), R)          # exp(-logk) already inside e
        o = be.mul_pt_sum(out_terms, [self.mask_m[m] * wt for m in range(7)])
        del out_terms, q, qm, prod, e, nm
        be.section("attn.out")
        attn = self.matvec_multi(self.to_off(o), [lw["Wo"]])[0]
        x2 = be.add(x, attn)
        del x, o, attn
        self.note("x2", x2)
        # ---- RMSNorm 2 + MLP
        be.section("norm2")
        s2 = self.block_sum(be.mul(x2, x2))
        y2 = self.inv_sqrt(s2, fm.norm_approx, 1.0 / (fm.d * lw["m2"]))
        be.section("ffn.up")
        x2o = self.to_off(x2)
        nc = len(lw["Ag"])
        raw = self.matvec_multi(x2o, list(lw["Ag"]) + list(lw["Au"]))
        del x2o, s2
        hs = []
        be.section("ffn.act")
        for c in range(nc):
            g = be.mul(y2, raw[c])
            u = be.mul(y2, raw[nc + c])
            gsq = be.mul(g, g)
            c0, c1, c2 = (self.tile(v) for v in lw["act"][c])
            poly = be.add_pt(be.mul_pt_sum([gsq, g], [c2, c1]), c0)
            hs.append(be.mul(poly, u))
        del raw, y2, g, u, gsq, poly
        be.section("ffn.down")
        d_out = self.matvec_sum([self.to_off(h) for h in hs], lw["Wd"])
        del hs
        be.section("residual")
        out = be.add(x2, d_out)
        del x2, d_out
        if hasattr(be, "trim"):
            be.trim()                            # give fully free pool chunks back (ciphertext sizes shrink per layer)
        return out

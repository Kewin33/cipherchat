"""CipherChat v5 – debug: run the engine on the GPU backend and the numpy slot simulation in lockstep and report the
first operation whose decrypted result deviates (value or level). Slow (decrypts after every op); 1 layer is enough.

  PYTHONPATH=build_v5 python scripts/v5_tee_debug.py --ckpt weights/v3/q05_pos.pt --lpp 2 --levels 60
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cipherchat.v3 import fold  # noqa: E402
from cipherchat.v3 import model as v3m  # noqa: E402
from cipherchat.v5 import engine as E5  # noqa: E402


class Tee:
    name = "tee"

    def __init__(self, gpu, ref, tol):
        self.g, self.r, self.tol = gpu, ref, tol
        self.n, self.L = gpu.n, gpu.L
        self.k = 0
        self.sec = "-"

    def section(self, s):
        self.sec = s
        self.g.section(s)

    def _wrap(self, op, gres, rres, args):
        self.k += 1
        y = self.g.decrypt(gres)
        err = float(np.abs(y - rres.v).max())
        scale = max(1.0, float(np.abs(rres.v).max()))
        lg, lr = self.g.level(gres), rres.level
        if err > self.tol * scale or lg != lr:
            ins = [(self.g.level(a[0]), a[1].level) for a in args]
            print(f"DIVERGED at op #{self.k} {self.sec}|{op}: err {err:.3e} (max |ref| {scale:.3g}) level gpu {lg} ref {lr}"
                  f" | input levels (gpu, ref) {ins}", flush=True)
            raise SystemExit(1)
        return (gres, rres)

    def encrypt(self, v, level):
        return (self.g.encrypt(v, level), self.r.encrypt(v, level))

    def decrypt(self, ct):
        return self.g.decrypt(ct[0])

    def level(self, ct):
        return self.g.level(ct[0])

    def rot(self, ct, k):
        return self._wrap(f"rot{k}", self.g.rot(ct[0], k), self.r.rot(ct[1], k), [ct])

    def add(self, a, b):
        return self._wrap("add", self.g.add(a[0], b[0]), self.r.add(a[1], b[1]), [a, b])

    def sub(self, a, b):
        return self._wrap("sub", self.g.sub(a[0], b[0]), self.r.sub(a[1], b[1]), [a, b])

    def add_pt(self, a, v):
        return self._wrap("add_pt", self.g.add_pt(a[0], v), self.r.add_pt(a[1], v), [a])

    def mul_int(self, a, k):
        return self._wrap("mul_int", self.g.mul_int(a[0], k), self.r.mul_int(a[1], k), [a])

    def mul_pt_sum(self, cts, vecs):
        return self._wrap(f"mul_pt_sum[{len(cts)}]", self.g.mul_pt_sum([c[0] for c in cts], vecs),
                          self.r.mul_pt_sum([c[1] for c in cts], vecs), list(cts))

    def mul(self, a, b):
        return self._wrap("mul", self.g.mul(a[0], b[0]), self.r.mul(a[1], b[1]), [a, b])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lpp", type=int, default=2)
    ap.add_argument("--levels", type=int, default=60)
    ap.add_argument("--scale_bits", type=int, default=30)
    ap.add_argument("--prime_bits", type=int, default=60)
    ap.add_argument("--special_bits", type=int, default=50)
    ap.add_argument("--logN", type=int, default=13)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()
    cfg = v3m.load_config()
    cfg["ckks"] = {"logN": args.logN, "levels": args.levels, "scale_bits": args.scale_bits, "base_bits": 60,
                   "special_bits": args.special_bits, "special_count": 1, "hamming_weight": 192,
                   "levels_per_prime": args.lpp, "prime_bits": args.prime_bits}
    lay = E5.Layout(2 ** (args.logN - 1), baby=16, baby_conv=2)
    fm, student, tok, prefix = fold.build(args.ckpt, cfg, device="cpu")
    fm.L = 1
    keys = E5.rotation_keys(lay)
    g = E5.PhantomBackend5(cfg, keys, profile=False)
    r = (E5.NumpyBackendLPP if args.lpp == 2 else E5.NumpyBackend)(n=lay.n, levels=g.L)
    be = Tee(g, r, args.tol)
    eng = E5.Engine5(fm, be, lay, keys)
    sess = eng.new_session()
    emb = student.model.embed_tokens.weight.detach().double().cpu().numpy()
    ids = prefix + tok("The capital")["input_ids"]
    x = eng.tile(emb[ids[2]])
    eng.layer(0, sess, be.encrypt(x, g.L))
    print(f"layer 0 OK: {be.k} ops identical within tol {args.tol}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()

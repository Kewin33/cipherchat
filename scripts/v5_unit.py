"""CipherChat v5 – unit test of the v5fhe operations against numpy, incl. the two-levels-per-prime state machine
(full = scale Delta, half = scale Delta^2): products of every state combination, plaintext products from both states,
additions across states and prime levels, rotations, and the virtual level after each op (must match
cipherchat/v5/engine.py NumpyBackendLPP).

  PYTHONPATH=build_v5 python scripts/v5_unit.py --lpp 2 --scale_bits 30 --prime_bits 60 --special_bits 50
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cipherchat.v5.engine import NumpyBackend, NumpyBackendLPP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logN", type=int, default=12)
    ap.add_argument("--levels", type=int, default=41)
    ap.add_argument("--lpp", type=int, default=2)
    ap.add_argument("--scale_bits", type=int, default=30)
    ap.add_argument("--prime_bits", type=int, default=60)
    ap.add_argument("--special_bits", type=int, default=50)
    args = ap.parse_args()
    import v5fhe
    cfg = {"ckks": {"logN": args.logN, "levels": args.levels, "scale_bits": args.scale_bits, "base_bits": 60,
                    "special_bits": args.special_bits, "special_count": 1, "hamming_weight": 192,
                    "levels_per_prime": args.lpp, "prime_bits": args.prime_bits}}
    ctx = v5fhe.Ctx(json.dumps(cfg))
    n = ctx.info()["slots"]
    L = ctx.info()["levels"]
    ctx.gen_rotation_keys([1, -3])
    nb = (NumpyBackendLPP if args.lpp == 2 else NumpyBackend)(n=n, levels=L)
    rng = np.random.default_rng(1)
    worst = 0.0
    fails = 0

    def check(name, ct, nct):
        nonlocal worst, fails
        err = float(np.abs(ctx.decrypt(ct) - nct.v).max())
        lv, nlv = ctx.level(ct), nct.level
        ok = err < 1e-5 * max(1.0, float(np.abs(nct.v).max())) and lv == nlv
        worst = max(worst, err)
        fails += not ok
        print(f"{'ok ' if ok else 'BAD'} {name:28s} level {lv:3d} (numpy {nlv:3d})  max err {err:.2e}", flush=True)

    a, b, w = (rng.uniform(-1, 1, n) for _ in range(3))
    A, B = ctx.encrypt(a, L), ctx.encrypt(b, L)
    nA, nB = nb.encrypt(a, L), nb.encrypt(b, L)
    check("fresh", A, nA)
    F2, nF2 = ctx.mul(A, B), nb.mul(nA, nB)                      # full x full -> half
    check("full*full", F2, nF2)
    H1, nH1 = ctx.mul(F2, A), nb.mul(nF2, nA)                    # half x full
    check("half*full", H1, nH1)
    HH, nHH = ctx.mul(F2, F2), nb.mul(nF2, nF2)                  # half x half
    check("half*half", HH, nHH)
    P1, nP1 = ctx.mul_pt_sum([A, B], [w, w]), nb.mul_pt_sum([nA, nB], [w, w])
    check("ptsum(full)", P1, nP1)
    P2, nP2 = ctx.mul_pt_sum([F2], [w]), nb.mul_pt_sum([nF2], [w])
    check("ptsum(half)", P2, nP2)
    S1, nS1 = ctx.add(A, F2), nb.add(nA, nF2)
    check("add(full, half)", S1, nS1)
    S2, nS2 = ctx.sub(HH, A), nb.sub(nHH, nA)
    check("sub(deep, fresh)", S2, nS2)
    S3, nS3 = ctx.add(F2, H1), nb.add(nF2, nH1)
    check("add(half l, full l-1)", S3, nS3)
    M1, nM1 = ctx.mul(HH, A), nb.mul(nHH, nA)
    check("mul(deep, fresh)", M1, nM1)
    R1, nR1 = ctx.rot(H1, 1), nb.rot(nH1, 1)
    check("rot +1 (full)", R1, nR1)
    R2, nR2 = ctx.rot(F2, -3), nb.rot(nF2, -3)
    check("rot -3 (half)", R2, nR2)
    T, nT = ctx.add_pt(F2, w), nb.add_pt(nF2, w)
    check("add_pt(half)", T, nT)
    I3, nI3 = ctx.mul_int(F2, 3), nb.mul_int(nF2, 3)
    check("mul_int(half)", I3, nI3)
    # a chain of squarings down to the bottom of the modulus chain
    X, nX = ctx.encrypt(0.9 * a, L), nb.encrypt(0.9 * a, L)
    k = 0
    while nX.level >= 5:
        if k % 3 == 2:
            X, nX = ctx.mul_pt_sum([X], [w]), nb.mul_pt_sum([nX], [w])
        else:
            X, nX = ctx.mul(X, X), nb.mul(nX, nX)
        k += 1
    check(f"chain of {k} ops", X, nX)
    print(f"{'PASS' if not fails else 'FAIL'}: {fails} failures, worst abs error {worst:.2e}", flush=True)
    os._exit(1 if fails else 0)


if __name__ == "__main__":
    main()

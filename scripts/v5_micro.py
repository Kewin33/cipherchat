"""CipherChat v5 – micro benchmark + correctness of the INSECURE e = 0 backend (v5fhe) on long modulus chains.

For a chain of `levels` primes at N = 2^logN: setup time (primes, context, keys), VRAM, and per operation and level the
GPU time and the error against numpy (fresh encryption, ct x ct + relin + rescale, rotation = key switching,
encode + pt x ct + rescale).

  PYTHONPATH=build_v5 python scripts/v5_micro.py --logN 13 --levels 866 --out results/v5/micro_N13_L866.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logN", type=int, default=13)
    ap.add_argument("--levels", type=int, default=866)
    ap.add_argument("--scale_bits", type=int, default=50)
    ap.add_argument("--base_bits", type=int, default=60)
    ap.add_argument("--special_bits", type=int, default=60)
    ap.add_argument("--at", default="", help="comma separated levels to measure (default: top, 3/4, 1/2, 1/4, 8, 1)")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    import v5fhe
    cfg = {"ckks": {"logN": args.logN, "levels": args.levels, "scale_bits": args.scale_bits,
                    "base_bits": args.base_bits, "special_bits": args.special_bits, "special_count": 1,
                    "hamming_weight": 192}}
    t0 = time.time()
    ctx = v5fhe.Ctx(json.dumps(cfg))
    t_ctx = time.time() - t0
    n = ctx.info()["slots"]
    t0 = time.time()
    nk = ctx.gen_rotation_keys([1, -1, 16])
    t_keys = time.time() - t0
    info = dict(ctx.info())
    info["timing"] = json.loads(info["timing"])
    mem = dict(ctx.mem_info())
    print(f"N=2^{args.logN} levels {args.levels}: {info} | ctx {t_ctx:.1f} s, {nk} rot keys {t_keys:.1f} s | {mem}",
          flush=True)
    L = args.levels
    lv = [int(x) for x in args.at.split(",")] if args.at else sorted({L, 3 * L // 4, L // 2, L // 4, 8, 1}, reverse=True)
    rng = np.random.default_rng(0)
    rows = []
    ctx.set_profiling(True)
    for lvl in lv:
        a = rng.uniform(-1, 1, n)
        b = rng.uniform(-1, 1, n)
        w = rng.uniform(-1, 1, n)
        ca, cb = ctx.encrypt(a, lvl), ctx.encrypt(b, lvl)
        r = {"level": lvl, "primes": lvl + 1, "ct_mb": ctx.ct_bytes(ca) / 2 ** 20}
        r["err_fresh"] = float(np.abs(ctx.decrypt(ca) - a).max())
        if lvl >= 1:
            r["err_mul"] = float(np.abs(ctx.decrypt(ctx.mul(ca, cb)) - a * b).max())
            r["err_ptmul"] = float(np.abs(ctx.decrypt(ctx.mul_pt_sum([ca], [w])) - a * w).max())
        r["err_rot1"] = float(np.abs(ctx.decrypt(ctx.rot(ca, 1)) - np.roll(a, -1)).max())
        r["err_rot-1"] = float(np.abs(ctx.decrypt(ctx.rot(ca, -1)) - np.roll(a, 1)).max())
        ctx.reset_profile()
        for _ in range(args.reps):
            ctx.rot(ca, 16)
            ctx.add(ca, cb)
            if lvl >= 1:
                ctx.mul(ca, cb)
                ctx.mul_pt_sum([ca], [w])
        prof = ctx.profile()
        for key, (ms, cnt) in prof.items():
            r["ms_" + key.split("|")[1]] = ms / cnt
        rows.append(r)
        print("  " + ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()),
              flush=True)
    res = {"cfg": cfg, "info": info, "ctx_s": t_ctx, "keys_s": t_keys, "mem": mem, "rows": rows,
           "insecure_zero_error": info["zero_error"]}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1)
    os._exit(0)


if __name__ == "__main__":
    main()

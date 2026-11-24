"""Bootstrapping error vs. input amplitude (GPU, v3fhe). Decides how far a ciphertext must be scaled down
before bootstrapping (EvalMod approximates x mod q0 by a sine, accurate only for small |x|)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cipherchat.v3 import engine as E  # noqa: E402
from cipherchat.v3.model import load_config  # noqa: E402


def main():
    cfg = load_config()
    be = E.PhantomBackend(cfg, profile=False)
    rng = np.random.default_rng(0)
    out = {}
    for amp in (0.25, 0.5, 1, 2, 4, 8, 16, 64):
        base = rng.uniform(-amp, amp, E.D)
        v = np.tile(base, be.n // E.D)
        ct = be.encrypt(v, 1)
        ct = be.mod_down(ct, 0)
        res = be.bootstrap(ct)
        y = be.decrypt(res)
        err = float(np.abs(y - v).max())
        out[amp] = err
        print(f"amplitude {amp:6.2f}: max abs err {err:.3e}  rel {err / amp:.3e}  level after {be.level(res)}", flush=True)
    # replicated scalar (logn = 0 bootstrapper): every slot holds the same value
    for val in (0.3, -0.9, 1.0):
        ct = be.mod_down(be.encrypt(np.full(be.n, val), 1), 0)
        import time
        t0 = time.time()
        res = be.bootstrap_scalar(ct)
        dt = time.time() - t0
        y = be.decrypt(res)
        err = float(np.abs(y - val).max())
        out[f"scalar_{val}"] = dict(err=err, ms=dt * 1e3)
        print(f"scalar {val:5.2f}: max abs err {err:.3e}, {dt * 1e3:.0f} ms, level after {be.level(res)}", flush=True)
    os.makedirs("results/v3", exist_ok=True)
    json.dump(out, open("results/v3/boot_range.json", "w"), indent=1)
    os._exit(0)


if __name__ == "__main__":
    main()

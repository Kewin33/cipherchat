"""v2 exactness chain without OpenFHE:
   torch CipherFormer2 (float64)  ==  PlainReference2 (token by token)  ==  EngineV2 on NumpyBackend (slot level).

Usage: python scripts/v2_check.py [--weights weights/<name>.npz] [--hidden 256] [--tokens 20]
Without --weights a random model is used (also checks the export/folding)."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cipherchat.v2.engine import EngineV2, NumpyBackend, Session2  # noqa: E402
from cipherchat.v2.reference import Folded2, PlainReference2, input_vector  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default=None)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--tokens", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n_sig", type=int, default=5)
ap.add_argument("--mlp", default="sq")
args = ap.parse_args()
rng = np.random.default_rng(args.seed)

torch_logits = None
if args.weights:
    fm = Folded2(args.weights)
    seq = [1] + list(rng.integers(5, fm.V, size=args.tokens - 1))
    sig = list(rng.integers(0, fm.cfg.get("n_sig", 0) + 1, size=len(seq)))
else:
    import torch
    from cipherchat.v2.model import CipherFormer2
    torch.manual_seed(args.seed)
    V = 59
    model = CipherFormer2(V, n_layers=args.layers, hidden=args.hidden, max_len=64, n_sig=args.n_sig, mlp=args.mlp).double()
    with torch.no_grad():                       # random relative biases so that the rho path is exercised
        model.rho.normal_(0, 0.5)
        model.c.normal_(1, 0.3)
    W = model.export()
    fm = Folded2.from_dict(W, model.cfg)
    seq = [1] + list(rng.integers(5, V, size=args.tokens - 1))
    sig = list(rng.integers(0, args.n_sig + 1, size=len(seq)))
    with torch.no_grad():
        torch_logits = model(torch.tensor([seq]), torch.tensor([sig]) if args.n_sig else None)[0].numpy()

ref = PlainReference2(fm)
ref_logits = np.array([ref.step(int(t), sig=int(g)) for t, g in zip(seq, sig)])
if torch_logits is not None:
    d = np.abs(torch_logits - ref_logits).max()
    print(f"torch vs reference: max|diff| = {d:.3e}  (max|logit| {np.abs(ref_logits).max():.2f})")
    assert d < 1e-6 * max(1.0, np.abs(ref_logits).max()), "torch and reference disagree"

be = NumpyBackend(8192)
eng = EngineV2(fm, be)
sess = Session2(fm.L)
sim = []
per_token = []
for t, g in zip(seq, sig):
    before = dict(be.counters)
    out = eng.step(sess, np.tile(input_vector(int(t), int(g), fm.V), 8192 // 64))
    sim.append(out[: fm.V])
    per_token.append({k: be.counters[k] - before[k] for k in be.counters})
sim = np.array(sim)
d2 = np.abs(sim - ref_logits).max()
print(f"numpy slot simulation vs reference: max|diff| = {d2:.3e}")
assert d2 < 1e-8 * max(1.0, np.abs(ref_logits).max()), "slot simulation disagrees with the reference"
steady = per_token[-1]
print("ops per generated token (last step):", json.dumps(steady))
print("encodes (static plaintexts, once):", be.counters["encode"])
print("OK")

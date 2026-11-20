"""v2 engine under real CKKS (OpenFHE): correctness vs. the plaintext reference, timing, memory.

  python scripts/v2_fhe_test.py --weights weights/<name>.npz --prompts "was ist 23+45?" "hallo"
  python scripts/v2_fhe_test.py                     # random model (hidden 1024), random tokens
Writes a JSON summary with --out.
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cipherchat.v2.crypto import ClientCryptoV2  # noqa: E402
from cipherchat.v2.engine import EngineV2, OpenFHEBackend, Session2  # noqa: E402
from cipherchat.v2.reference import Folded2, PlainReference2  # noqa: E402


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return float("nan")


ap = argparse.ArgumentParser()
ap.add_argument("--weights", default=None)
ap.add_argument("--prompts", nargs="*", default=None)
ap.add_argument("--hidden", type=int, default=1024)
ap.add_argument("--mlp", default="sq")
ap.add_argument("--tokens", type=int, default=16)
ap.add_argument("--max_new", type=int, default=12)
ap.add_argument("--out", default=None)
ap.add_argument("--heldout", type=int, default=0, help="add N random held-out addition prompts (trained model)")
ap.add_argument("--ring_dim", type=int, default=0)
ap.add_argument("--scale_bits", type=int, default=33)
ap.add_argument("--first_mod_bits", type=int, default=45)
ap.add_argument("--dnum", type=int, default=0)
args = ap.parse_args()

if args.weights:
    fm = Folded2(args.weights)
else:
    import torch
    from cipherchat.v2.model import CipherFormer2
    torch.manual_seed(0)
    model = CipherFormer2(59, n_layers=2, hidden=args.hidden, mlp=args.mlp).double()
    with torch.no_grad():
        model.rho.normal_(0, 0.5)
    fm = Folded2.from_dict(model.export(), model.cfg)

rss0 = rss_mb()
t0 = time.time()
client = ClientCryptoV2(fm.L, args.ring_dim, args.scale_bits, args.first_mod_bits, args.dnum)
bundle = client.export_server_bundle()
key_mb = (len(bundle["eval_mult_key"]) + len(bundle["eval_rot_keys"])) / 1e6
print(f"keygen {client.keygen_seconds:.2f}s | ring {client.cc.GetRingDimension()} | {len(client.rot_idx)} rotation keys "
      f"| eval keys {key_mb:.1f} MB | RSS {rss_mb() - rss0:.0f} MB")
be = OpenFHEBackend(client.cc)
if client.S // 128 < fm.T:          # smaller ring: shorter context (cache blocks = slots/128)
    print(f"context limited to {client.S // 128} tokens by the ring dimension")
    fm.T = client.S // 128
eng = EngineV2(fm, be)

# warm-up: one throw-away step encodes every static plaintext at its level
rss1 = rss_mb()
tw = time.time()
eng.step(Session2(fm.L), client.encrypt_token(1))
warm_s = time.time() - tw
pt_mb = rss_mb() - rss1
print(f"warm-up (encode {len(eng._pt)} static plaintext sets) {warm_s:.1f}s | plaintext cache ~{pt_mb:.0f} MB RSS")

REV = bool(fm.cfg.get("rev")) if args.weights else False
CPL = bool(fm.cfg.get("coupling")) if args.weights else False
from cipherchat.v2.dataset import Task, reverse_digits, significance  # noqa: E402
sig_fn = (lambda seq: significance(seq, REV)) if CPL else None
expected = {}
if args.heldout:
    import random as _r
    task = Task(rev=REV, coupling=CPL)
    for a, b in _r.Random(123).sample(task.test_pairs, args.heldout):
        if _r.Random(a * 100 + b).random() < 0.5:
            a, b = b, a
        expected[f"was ist {a}+{b}?"] = str(a + b)
    args.prompts = (args.prompts or []) + list(expected)
if args.prompts:
    from cipherchat.vocab import EOS, decode, encode_prompt
    seqs = [(p, encode_prompt(reverse_digits(p) if REV else p)) for p in args.prompts]
else:
    rng = np.random.default_rng(1)
    EOS, decode = 3, None
    seqs = [("random", [1] + list(rng.integers(5, fm.V, size=args.tokens - 1)))]

rows, t_prompt, t_gen, errs = [], [], [], []
for name, ids in seqs:
    sess = Session2(fm.L)
    ref = PlainReference2(fm)
    out_fhe, my_errs = [], []
    seq = list(ids)
    i = 0
    while i < len(seq):
        tok = int(seq[i])
        # random mode: logits at every position; prompt mode: realistic protocol (prefill without logits)
        need = (not args.prompts) or i >= len(ids) - 1
        sg = sig_fn(seq)[i] if sig_fn else 0
        ct = client.encrypt_token(tok, sg, fm.V)
        ts = time.time()
        ct_l = eng.step(sess, ct, need_logits=need)
        dt = time.time() - ts
        lr = ref.step(tok, need_logits=need, sig=sg)
        i += 1
        if not need:
            t_prompt.append(dt)
            continue
        t_gen.append(dt)
        lg = client.decrypt_logits(ct_l, fm.V)
        my_errs.append(float(np.abs(lg - lr).max()))
        nxt = int(np.argmax(lg))
        if nxt != int(np.argmax(lr)):
            print(f"   !! argmax mismatch at step {i - 1}: fhe {nxt} ref {int(np.argmax(lr))}")
        if args.prompts and i == len(seq):          # generation phase: feed back the argmax
            if nxt == EOS or len(out_fhe) >= args.max_new or sess.t >= eng.T:
                break
            out_fhe.append(nxt)
            seq.append(nxt)
    errs += my_errs
    row = dict(prompt=name, tokens=sess.t, max_err=max(my_errs))
    if args.prompts:
        fix = (lambda txt: reverse_digits(txt)) if REV else (lambda txt: txt)
        row["fhe"] = fix(decode(out_fhe))
        row["plain"] = fix(decode(PlainReference2(fm).generate(ids, max_new=args.max_new, sig_fn=sig_fn)))
        row["match"] = row["fhe"] == row["plain"]
        if name in expected:
            row["correct"] = row["fhe"] == expected[name]
    rows.append(row)
    print(row, flush=True)

n_exp = sum(1 for r in rows if "correct" in r)
if n_exp:
    print(f"held-out additions under FHE: {sum(r['correct'] for r in rows if 'correct' in r)}/{n_exp} correct, "
          f"{sum(r['match'] for r in rows)}/{len(rows)} identical to plaintext reference")
res = dict(rows=rows, sec_per_prompt_token=float(np.mean(t_prompt)) if t_prompt else None,
           sec_per_generated_token=float(np.mean(t_gen)), median_generated=float(np.median(t_gen)),
           max_logit_err=float(max(errs)), eval_key_mb=key_mb, n_rot_keys=len(client.rot_idx),
           warmup_s=warm_s, plaintext_cache_mb=pt_mb, rss_total_mb=rss_mb(),
           maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
           ring_dim=client.cc.GetRingDimension(), scale_bits=args.scale_bits, first_mod_bits=args.first_mod_bits,
           counters=be.counters, omp=os.environ.get("OMP_NUM_THREADS"), cpus=os.sched_getaffinity(0).__len__(),
           ct_up_mb=len(client.encrypt_token_bytes(1)) / 1e6)
print({k: v for k, v in res.items() if k != "rows"})
if args.out:
    json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)

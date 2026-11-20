"""Decision margins of a trained v2 model (plaintext reference, no OpenFHE needed).

For every greedy decoding step: margin = top1 - top2 logit. Compared with the CKKS logit error measured by
scripts/v2_fhe_test.py this gives the safety factor of encrypted greedy decoding.

  python scripts/v2_margins.py weights/cf2_rev_cpl_sqglu_h2048.npz
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cipherchat.v2.dataset import Task, reverse_digits, significance  # noqa: E402
from cipherchat.v2.reference import Folded2, PlainReference2  # noqa: E402
from cipherchat.vocab import EOS, encode_prompt  # noqa: E402

fm = Folded2(sys.argv[1])
rev, cpl = bool(fm.cfg.get("rev")), bool(fm.cfg.get("coupling"))
task = Task(rev=rev, coupling=cpl)
prompts = [p for _, p, _ in task.eval_others()] + [p for _, p, _ in task.eval_addition(task.test_pairs[::10])]
margins = []
for p in prompts:
    ids = encode_prompt(reverse_digits(p) if rev else p)
    ref = PlainReference2(fm)
    seq = list(ids)
    sig = (lambda s: significance(s, rev)) if cpl else (lambda s: [0] * len(s))
    lg = None
    for i, tok in enumerate(ids):
        lg = ref.step(tok, need_logits=(i == len(ids) - 1), sig=sig(seq)[i])
    for _ in range(24):
        top = np.sort(lg)[::-1]
        margins.append(top[0] - top[1])
        nxt = int(np.argmax(lg))
        if nxt == EOS or ref.t >= fm.T:
            break
        seq.append(nxt)
        lg = ref.step(nxt, sig=sig(seq)[-1])
m = np.array(margins)
print(f"{len(prompts)} prompts, {len(m)} decoding steps: min margin {m.min():.3f}, 1% quantile {np.quantile(m, 0.01):.3f}, "
      f"median {np.median(m):.2f}")

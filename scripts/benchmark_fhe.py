"""Benchmark + correctness check of encrypted generation on a set of prompts.

Uses the realistic protocol (prompt tokens without logits, then generation) and compares
every answer with the plaintext reference. Writes docs/fhe_benchmark.json.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cipherchat.client_crypto import ClientCrypto  # noqa: E402
from cipherchat.fhe_common import ser  # noqa: E402
from cipherchat.fhe_engine import EncryptedSession, FHEServerModel  # noqa: E402
from cipherchat.reference import FoldedModel, PlainReference  # noqa: E402
from cipherchat.vocab import EOS, decode, encode_prompt  # noqa: E402

PROMPTS = sys.argv[1:] or ["hallo", "was ist 5+3?", "ist 4 gerade?", "hauptstadt von polen?",
                           "gegenteil von laut?", "danke", "zähl bis 3", "ist 2 größer als 9?"]
fm = FoldedModel("weights/cipherchat_demo.npz")
client = ClientCrypto(fm.L)
server = FHEServerModel(fm, client.cc)
rows = []
t_prompt, t_gen = [], []
for p in PROMPTS:
    ids = encode_prompt(p)
    sess = EncryptedSession(fm.L)
    out = []
    t0 = time.time()
    for i, tok in enumerate(ids):
        ts = time.time()
        ct_l = server.step(sess, client.encrypt_token(tok), need_logits=(i == len(ids) - 1))
        (t_gen if i == len(ids) - 1 else t_prompt).append(time.time() - ts)
    while True:
        lg = client.decrypt_logits(ct_l, fm.V)
        nxt = int(np.argmax(lg))
        if nxt == EOS or len(out) >= 20 or sess.t >= server.T:
            break
        out.append(nxt)
        ts = time.time()
        ct_l = server.step(sess, client.encrypt_token(nxt))
        t_gen.append(time.time() - ts)
    fhe_ans = decode(out)
    ref_ans = decode(PlainReference(fm).generate(ids))
    rows.append(dict(prompt=p, fhe=fhe_ans, plain=ref_ans, match=fhe_ans == ref_ans, seconds=round(time.time() - t0, 1),
                     tokens=sess.t))
    print(rows[-1], flush=True)
res = dict(rows=rows, all_match=all(r["match"] for r in rows),
           sec_per_prompt_token=float(np.mean(t_prompt)), sec_per_generated_token=float(np.mean(t_gen)),
           ciphertext_up_mb=len(client.encrypt_token_bytes(1)) / 1e6,
           logits_down_mb=len(ser(ct_l)) / 1e6, cpu_count=os.cpu_count())
print({k: v for k, v in res.items() if k != "rows"})
os.makedirs("docs", exist_ok=True)
json.dump(res, open("docs/fhe_benchmark.json", "w"), indent=2, ensure_ascii=False)

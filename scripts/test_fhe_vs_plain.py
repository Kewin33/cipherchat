"""Compare encrypted (CKKS) inference against the plaintext reference, token by token."""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cipherchat.client_crypto import ClientCrypto  # noqa: E402
from cipherchat.fhe_engine import EncryptedSession, FHEServerModel  # noqa: E402
from cipherchat.reference import FoldedModel, PlainReference  # noqa: E402
from cipherchat.vocab import EOS, decode, encode_prompt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="weights/cipherchat_demo.npz")
ap.add_argument("--prompts", nargs="*", default=["wie geht's?", "was ist 3+4?", "hauptstadt von italien?"])
ap.add_argument("--max_new", type=int, default=16)
args = ap.parse_args()

fm = FoldedModel(args.weights)
ref = PlainReference(fm)
t0 = time.time()
client = ClientCrypto(fm.L)
print(f"keygen: {client.keygen_seconds:.1f}s  ring dim {client.cc.GetRingDimension()}  slots {client.S}")
server = FHEServerModel(fm, client.cc)

all_err = []
for prompt in args.prompts:
    sess = EncryptedSession(fm.L)
    ref.reset()
    ids = encode_prompt(prompt)
    out_fhe = []
    step_times = []
    tok = None
    seq = list(ids)
    i = 0
    while True:
        tok = seq[i]
        ct = client.encrypt_token(tok)
        ts = time.time()
        ct_logits = server.step(sess, ct)
        step_times.append(time.time() - ts)
        lg = client.decrypt_logits(ct_logits, fm.V)
        lr = ref.step(tok)
        err = np.abs(lg - lr).max()
        all_err.append(err)
        if i == len(seq) - 1:  # generation phase
            nxt = int(np.argmax(lg))
            agree = nxt == int(np.argmax(lr))
            if not agree:
                print(f"   !! argmax mismatch at step {i}: fhe {nxt} ref {int(np.argmax(lr))}")
            if nxt == EOS or len(out_fhe) >= args.max_new or sess.t >= server.T:
                break
            out_fhe.append(nxt)
            seq.append(nxt)
        i += 1
    ref_out = decode(PlainReference(fm).generate(ids))
    print(f"{prompt!r:32} -> FHE: {decode(out_fhe)!r:24} plain: {ref_out!r:24} | steps {len(step_times)} "
          f"avg {np.mean(step_times):.2f}s/token | max|err| {max(all_err[-len(step_times):]):.2e}")
print("counters:", server.counters)
print(f"overall max logit error {max(all_err):.3e}, mean {np.mean(all_err):.3e}; total {time.time()-t0:.0f}s")

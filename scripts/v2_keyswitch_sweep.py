"""Memory/speed trade-off of the key-switching parameter dnum (OpenFHE NumLargeDigits) for the v2 engine.

For each dnum: ring dimension chosen by OpenFHE for 128-bit security, eval-key size, keygen time and the
v2 per-token latency (random model, logits compared with the plaintext reference)."""
import json
import os
import sys
import time

import numpy as np
import openfhe as ofhe
from openfhe import BINARY

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cipherchat.fhe_common import FIRST_MOD_BITS, SCALE_BITS, n_slots  # noqa: E402
from cipherchat.v2.engine import EngineV2, OpenFHEBackend, Session2  # noqa: E402
from cipherchat.v2.layout import rotation_indices  # noqa: E402
from cipherchat.v2.reference import Folded2, PlainReference2, input_vector  # noqa: E402


def context(depth, dnum):
    p = ofhe.CCParamsCKKSRNS()
    p.SetMultiplicativeDepth(depth)
    p.SetScalingModSize(SCALE_BITS)
    p.SetFirstModSize(FIRST_MOD_BITS)
    p.SetSecurityLevel(ofhe.SecurityLevel.HEStd_128_classic)
    p.SetScalingTechnique(ofhe.ScalingTechnique.FLEXIBLEAUTO)
    p.SetKeySwitchTechnique(ofhe.KeySwitchTechnique.HYBRID)
    if dnum:
        p.SetNumLargeDigits(dnum)
    cc = ofhe.GenCryptoContext(p)
    for f in (ofhe.PKESchemeFeature.PKE, ofhe.PKESchemeFeature.KEYSWITCH, ofhe.PKESchemeFeature.LEVELEDSHE,
              ofhe.PKESchemeFeature.ADVANCEDSHE):
        cc.Enable(f)
    return cc


import torch  # noqa: E402
from cipherchat.v2.model import CipherFormer2  # noqa: E402
torch.manual_seed(0)
model = CipherFormer2(59, n_layers=2, hidden=1024).double()
fm = Folded2.from_dict(model.export(), model.cfg)
rows = []
for dnum in [int(a) for a in (sys.argv[1:] or ["0", "2", "3", "4", "9"])]:
    ofhe.ClearEvalMultKeys()
    ofhe.CryptoContext.ClearEvalAutomorphismKeys()
    try:
        cc = context(8, dnum)
    except Exception as e:  # noqa: BLE001
        print(dnum, "failed:", e)
        continue
    t0 = time.time()
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    S = n_slots(cc)
    cc.EvalRotateKeyGen(keys.secretKey, rotation_indices(S))
    kg = time.time() - t0
    key_mb = (len(ofhe.SerializeEvalMultKeyString(BINARY, "")) + len(ofhe.SerializeEvalAutomorphismKeyString(BINARY, ""))) / 1e6
    eng = EngineV2(fm, OpenFHEBackend(cc))
    enc = lambda tok: cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(np.tile(input_vector(tok, 0, 59), S // 64).tolist()))
    eng.step(Session2(2), enc(1))                     # warm-up (plaintext encoding)
    sess, ref = Session2(2), PlainReference2(fm)
    ts, err = [], 0.0
    for tok in [1, 10, 20, 30, 40, 12, 7, 33, 50, 9]:
        t1 = time.time()
        ct = eng.step(sess, enc(tok))
        ts.append(time.time() - t1)
        pt = cc.Decrypt(keys.secretKey, ct)
        pt.SetLength(64)
        err = max(err, float(np.abs(np.array(pt.GetRealPackedValue()[:59]) - ref.step(tok)).max()))
    row = dict(dnum=dnum or "default", ring_dim=cc.GetRingDimension(), eval_key_mb=round(key_mb, 1),
               keygen_s=round(kg, 2), sec_per_token=round(float(np.median(ts[2:])), 4), max_err=err)
    print(row, flush=True)
    rows.append(row)
json.dump(rows, open("logs/v2_keyswitch_sweep.json", "w"), indent=2)

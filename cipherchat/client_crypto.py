"""Client-side cryptography: key generation, token encryption, logit decryption.

The secret key is created here and never leaves this object (= never leaves the
user's machine). Only the crypto context (public parameters) and the evaluation
keys are exported to the server.
"""
import hashlib
import time

import numpy as np
import openfhe as ofhe
from openfhe import BINARY

from .fhe_common import B, make_context, n_slots, onehot_rep, rotation_indices, ser


class ClientCrypto:
    def __init__(self, n_layers: int):
        t0 = time.time()
        self.cc = make_context(n_layers)
        self.S = n_slots(self.cc)
        self.keys = self.cc.KeyGen()
        self.cc.EvalMultKeyGen(self.keys.secretKey)
        self.cc.EvalRotateKeyGen(self.keys.secretKey, rotation_indices(self.cc))
        self.keygen_seconds = time.time() - t0

    def export_server_bundle(self) -> dict:
        """Everything the server needs - and nothing that allows decryption."""
        return {
            "cryptocontext": ser(self.cc),
            "eval_mult_key": ofhe.SerializeEvalMultKeyString(BINARY, ""),
            "eval_rot_keys": ofhe.SerializeEvalAutomorphismKeyString(BINARY, ""),
            # the public key is public; the server only uses it to create an encryption of zero
            # that normalises incoming ciphertexts (OpenFHE performance quirk, see server/app.py)
            "public_key": ser(self.keys.publicKey),
        }

    def encrypt_token(self, token: int):
        pt = self.cc.MakeCKKSPackedPlaintext(onehot_rep(token, self.S).tolist())
        return self.cc.Encrypt(self.keys.publicKey, pt)

    def encrypt_token_bytes(self, token: int) -> bytes:
        return ser(self.encrypt_token(token))

    def decrypt_logits(self, ct, vocab_size: int) -> np.ndarray:
        pt = self.cc.Decrypt(self.keys.secretKey, ct)
        pt.SetLength(B)
        return np.array(pt.GetRealPackedValue()[:vocab_size])

    def decrypt_logits_bytes(self, b: bytes, vocab_size: int) -> np.ndarray:
        return self.decrypt_logits(ofhe.DeserializeCiphertextString(b, BINARY), vocab_size)

    @staticmethod
    def fingerprint(b: bytes) -> str:
        return hashlib.sha256(b).hexdigest()[:16]

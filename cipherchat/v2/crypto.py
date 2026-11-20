"""Client-side keys for the v2 engine (same CKKS parameters as v1, v2 rotation-key set)."""
import hashlib
import time

import numpy as np
import openfhe as ofhe
from openfhe import BINARY

import openfhe as _o

from ..fhe_common import B, FIRST_MOD_BITS, SCALE_BITS, depth_for_layers, n_slots, ser
from .layout import rotation_indices
from .reference import input_vector


def make_context_v2(n_layers, ring_dim=0, scale_bits=SCALE_BITS, first_mod_bits=FIRST_MOD_BITS, dnum=0):
    """Like fhe_common.make_context, but with tunable precision (the CKKS analogue of LLM quantisation) and dnum.
    OpenFHE refuses parameter sets below 128-bit security (HE standard) for the requested ring dimension."""
    p = _o.CCParamsCKKSRNS()
    p.SetMultiplicativeDepth(depth_for_layers(n_layers))
    p.SetScalingModSize(scale_bits)
    p.SetFirstModSize(first_mod_bits)
    p.SetSecurityLevel(_o.SecurityLevel.HEStd_128_classic)
    p.SetScalingTechnique(_o.ScalingTechnique.FLEXIBLEAUTO)
    p.SetKeySwitchTechnique(_o.KeySwitchTechnique.HYBRID)
    if ring_dim:
        p.SetRingDim(ring_dim)
    if dnum:
        p.SetNumLargeDigits(dnum)
    cc = _o.GenCryptoContext(p)
    for f in (_o.PKESchemeFeature.PKE, _o.PKESchemeFeature.KEYSWITCH, _o.PKESchemeFeature.LEVELEDSHE,
              _o.PKESchemeFeature.ADVANCEDSHE):
        cc.Enable(f)
    return cc


class ClientCryptoV2:
    def __init__(self, n_layers: int, ring_dim: int = 0, scale_bits=SCALE_BITS, first_mod_bits=FIRST_MOD_BITS, dnum=0):
        t0 = time.time()
        self.cc = make_context_v2(n_layers, ring_dim, scale_bits, first_mod_bits, dnum)
        self.S = n_slots(self.cc)
        self.keys = self.cc.KeyGen()
        self.cc.EvalMultKeyGen(self.keys.secretKey)
        self.rot_idx = rotation_indices(self.S)
        self.cc.EvalRotateKeyGen(self.keys.secretKey, self.rot_idx)
        self.keygen_seconds = time.time() - t0

    def export_server_bundle(self) -> dict:
        return {
            "cryptocontext": ser(self.cc),
            "eval_mult_key": ofhe.SerializeEvalMultKeyString(BINARY, ""),
            "eval_rot_keys": ofhe.SerializeEvalAutomorphismKeyString(BINARY, ""),
            "public_key": ser(self.keys.publicKey),
        }

    def encrypt_token(self, token: int, sig: int = 0, vocab_size: int = 59):
        """Encrypt the (two-hot) input vector of one token, replicated over all slots (rep layout)."""
        vec = np.tile(input_vector(token, sig, vocab_size), self.S // B)
        pt = self.cc.MakeCKKSPackedPlaintext(vec.tolist())
        return self.cc.Encrypt(self.keys.publicKey, pt)

    def encrypt_token_bytes(self, token: int, sig: int = 0, vocab_size: int = 59) -> bytes:
        return ser(self.encrypt_token(token, sig, vocab_size))

    def decrypt_logits(self, ct, vocab_size: int) -> np.ndarray:
        pt = self.cc.Decrypt(self.keys.secretKey, ct)
        pt.SetLength(B)
        return np.array(pt.GetRealPackedValue()[:vocab_size])

    def decrypt_logits_bytes(self, b: bytes, vocab_size: int) -> np.ndarray:
        return self.decrypt_logits(ofhe.DeserializeCiphertextString(b, BINARY), vocab_size)

    @staticmethod
    def fingerprint(b: bytes) -> str:
        return hashlib.sha256(b).hexdigest()[:16]

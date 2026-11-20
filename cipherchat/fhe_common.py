"""Shared CKKS parameters, slot layouts and (de)serialisation helpers.

Slot layouts (S = 8192 slots for ring dimension N = 16384):
  rep     : a 64-vector tiled 128x over all slots              -> used for x, q, z, logits
  K-cache : block j (slots 128j..128j+127) holds [k_j, k_j]     -> all keys in ONE ciphertext
  V-cache : block j holds [v_j, 0]                              -> all values in ONE ciphertext
With these layouts one ciphertext-ciphertext product + log2 rotations gives the
attention scores against *all* cached positions at once (encrypted KV cache).
"""
import numpy as np
from openfhe import (BINARY, CCParamsCKKSRNS, GenCryptoContext, KeySwitchTechnique, PKESchemeFeature,
                     ScalingTechnique, SecurityLevel)

B = 64            # block size: d_model = vocab_pad = d_ff block = 64
BLK = 2 * B       # KV-cache block (doubled layout)
BABY = 16         # BSGS baby steps (hoisted rotations, shared by all matrices on the same input)
GIANT = 4         # BSGS giant steps (one full rotation each, per matrix); B = BABY * GIANT
G = BABY          # backwards-compatible alias

SCALE_BITS = 33
FIRST_MOD_BITS = 45


def depth_for_layers(n_layers: int) -> int:
    # per parallel block: projection, q.K product, square, w*V product
    return 4 * n_layers


def make_context(n_layers: int, ring_dim: int = 0):
    p = CCParamsCKKSRNS()
    p.SetMultiplicativeDepth(depth_for_layers(n_layers))
    p.SetScalingModSize(SCALE_BITS)
    p.SetFirstModSize(FIRST_MOD_BITS)
    p.SetSecurityLevel(SecurityLevel.HEStd_128_classic)   # standard 128-bit security
    p.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO)
    p.SetKeySwitchTechnique(KeySwitchTechnique.HYBRID)
    if ring_dim:
        p.SetRingDim(ring_dim)
    cc = GenCryptoContext(p)
    for f in (PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH, PKESchemeFeature.LEVELEDSHE, PKESchemeFeature.ADVANCEDSHE):
        cc.Enable(f)
    return cc


def n_slots(cc) -> int:
    return cc.GetRingDimension() // 2


def max_context(cc) -> int:
    return n_slots(cc) // BLK


def rotation_indices(cc):
    S = n_slots(cc)
    idx = set(range(1, BABY))                    # baby steps
    idx |= {BABY * i for i in range(1, GIANT)}   # giant steps
    idx |= {1 << k for k in range(6)}            # in-block sum (window 64)
    k = BLK
    while k < S:                                 # sum over cache blocks
        idx.add(k)
        k *= 2
    idx.add(-B)                                  # re-double [a, 0] -> [a, a]
    return sorted(idx)


# ---------- layouts ----------

def tile(v, S):
    v = np.asarray(v, dtype=np.float64)
    out = np.zeros(B)
    out[: len(v)] = v
    return np.tile(out, S // B)


def kmask(t, S):
    m = np.zeros(S)
    m[BLK * t: BLK * (t + 1)] = 1.0
    return m


def vmask(t, S):
    m = np.zeros(S)
    m[BLK * t: BLK * t + B] = 1.0
    return m


def diagonals(M):
    """d_k[c] = M[(c + k) % B, c]  (row-vector convention y = x @ M)."""
    M = np.asarray(M, dtype=np.float64)
    assert M.shape == (B, B), M.shape
    c = np.arange(B)
    return np.stack([M[(c + k) % B, c] for k in range(B)])


def bsgs_vectors(M, S, mask=None, scale=1.0):
    """Plaintext vectors P[i][j] for y = sum_i rot( sum_j rot(x, j) * P[i][j], BABY*i )."""
    D = diagonals(M) * scale
    out = []
    for i in range(GIANT):
        row = []
        for j in range(BABY):
            v = tile(D[BABY * i + j], S)
            if mask is not None:
                v = v * mask
            row.append(np.roll(v, BABY * i))
        out.append(row)
    return out


def onehot_rep(token: int, S: int):
    v = np.zeros(B)
    v[token] = 1.0
    return tile(v, S)


# ---------- (de)serialisation ----------
import openfhe as ofhe


def ser(obj) -> bytes:
    s = ofhe.Serialize(obj, BINARY)
    return s if isinstance(s, (bytes, bytearray)) else s.encode("latin1")


def de_ct(b: bytes):
    return ofhe.DeserializeCiphertextString(b, BINARY)


def de_cc(b: bytes):
    return ofhe.DeserializeCryptoContextString(b, BINARY)

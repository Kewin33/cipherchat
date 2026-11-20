"""Slot layouts and generalized BSGS plaintexts for the v2 engine (pure numpy, no OpenFHE).

Conventions (S = number of CKKS slots, 8192 for N = 16384):
  rot(v, k)[i] = v[(i + k) mod S]            (OpenFHE EvalRotate semantics, = np.roll(v, -k))
  rep layout   : a 64-vector tiled S/64 times (period 64)
  cache layout : S/128 blocks of 128 slots; block j holds one cached position.
                 K-cache block = [k, k] (doubled), V-cache block = [v, 0].
                 The cache SHIFTS by one block per token (rot by 128), the newest entry is always
                 written into the last block INS = S/128 - 1, so block j holds the token at relative
                 distance r = INS - j. => all plaintexts are static (no per-token re-encoding).

Two generalized diagonal (BSGS) matrix-vector products, both 1 multiplicative level:

  from_rep:  x in rep layout -> ANY S-slot output  y[s] = sum_r x[r] W[r, s]          (W: 64 x S)
             y = sum_i rot( sum_j rot(x, j) * P[i][j], BABY*i )
             P[i][j][tau] = W[(tau + j) mod 64, (tau - BABY*i) mod S]

  to_rep:    m arbitrary S-slot vector -> rep output  y[c] = sum_rho m[rho] W[rho, c]   (W: S x 64)
             T = cross_sum( sum_i rot( sum_j rot(m, j) * P[i][j], BABY*i ) )
             cross_sum(v) = sum_{g < S/64} rot(v, 64 g)      (log2(S/64) rotations)
             P[i][j][tau] = W[(tau + j) mod S, (tau - BABY*i) mod 64]

`to_rep` is what lets the MLP hidden layer fill all S slots (d_ff up to S) at the cost of a
64x64 matvec, and cross_sum also performs the attention's block-sum when the attention output
is added before it (cross_sum of a V-layout vector = attention output in rep layout).
"""
import numpy as np

B = 64            # model width / block of the rep layout
BLK = 128         # cache block (doubled)
BABY = 16
GIANT = 4
WINDOW_STEPS = (1, 2, 4, 8, 16, 32)          # in-block sum of 64 slots (scores)


def cross_steps(S):
    out, k = [], B
    while k < S:
        out.append(k)
        k *= 2
    return out


def rotation_indices(S):
    idx = set(range(1, BABY)) | {BABY * i for i in range(1, GIANT)} | set(WINDOW_STEPS)
    idx |= set(cross_steps(S))        # 64 .. S/2 (includes the cache shift 128)
    return sorted(idx)


def rot(v, k):
    return np.roll(v, -k)


def from_rep_plaintexts(W):
    """W: (64, S) -> P[GIANT][BABY] numpy vectors of length S."""
    W = np.asarray(W, dtype=np.float64)
    S = W.shape[1]
    tau = np.arange(S)
    P = []
    for i in range(GIANT):
        cols = (tau - BABY * i) % S
        P.append([W[(tau + j) % B, cols] for j in range(BABY)])
    return P


def to_rep_plaintexts(W):
    """W: (S, 64) -> P[GIANT][BABY] numpy vectors of length S."""
    W = np.asarray(W, dtype=np.float64)
    S = W.shape[0]
    tau = np.arange(S)
    P = []
    for i in range(GIANT):
        cols = (tau - BABY * i) % B
        P.append([W[(tau + j) % S, cols] for j in range(BABY)])
    return P


# ---------- output layouts for from_rep products (W_full: 64 x S) ----------

def full_rep(M, S):
    """y = x @ M replicated over all slots (M: 64 x 64)."""
    return np.tile(np.asarray(M, dtype=np.float64), (1, S // B))


def full_block0(M, S):
    """y = x @ M in slots 0..63 only (cross_sum of it = rep layout of x @ M)."""
    W = np.zeros((B, S))
    W[:, :B] = M
    return W


def full_cache_k(M, S):
    """[x @ M, x @ M] in the insert block (last 128-slot block), zeros elsewhere."""
    W = np.zeros((B, S))
    ins = S - BLK
    W[:, ins:ins + B] = M
    W[:, ins + B:ins + BLK] = M
    return W


def full_cache_v(M, S):
    """[x @ M, 0] in the insert block, zeros elsewhere."""
    W = np.zeros((B, S))
    ins = S - BLK
    W[:, ins:ins + B] = M
    return W


def full_hidden(W1, S):
    """hidden layer z = x @ W1 (W1: 64 x H, H | S) replicated with period H over all S slots."""
    W1 = np.asarray(W1, dtype=np.float64)
    H = W1.shape[1]
    assert S % H == 0, (S, H)
    return np.tile(W1, (1, S // H))


def full_from_hidden(W2, S):
    """to_rep matrix for y = m_hidden @ W2 when the hidden layer (H units) is replicated S/H times:
    every unit appears S/H times in the cross-sum, so the weights are divided by S/H."""
    W2 = np.asarray(W2, dtype=np.float64)
    H = W2.shape[0]
    return np.tile(W2, (S // H, 1)) * (H / S)


def cache_add_vector(cr, S):
    """Static additive vector for the scores: block j (first half) gets cr[r] with r = INS - j
    (relative distance of the token stored in block j). cr: length >= S/128."""
    nblk = S // BLK
    v = np.zeros(S)
    for j in range(nblk):
        r = nblk - 1 - j
        v[j * BLK: j * BLK + B] = cr[r] if r < len(cr) else 0.0   # blocks beyond the context stay empty
    return v


def tile_rep(vec, S):
    out = np.zeros(B)
    out[: len(vec)] = vec
    return np.tile(out, S // B)

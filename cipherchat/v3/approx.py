"""CipherChat v3 – the FHE approximations, written once and used everywhere.

Training (torch), the plaintext reference and the CKKS engine evaluate exactly these functions, so the distilled
model is trained on the operations the server really computes (same rule as v1/v2: "training uses the FHE ops").

All functions are polynomials in their inputs (CKKS can only add and multiply); every constant is public.

  inv_sqrt(z)  ~ z^-1/2 on [lo, hi]: degree-d least-squares start value, then Newton y <- y (3 - z y^2) / 2
  inv(w)       ~ 1/w   on [lo, hi]: degree-d start value, then Newton y <- y (2 - w y)
  power attention: e_j = (alpha s_j + c)^(2p), out = sum_j e_j v_j * inv(den / n_ref) / n_ref, den = sum_j e_j
  activation:  silu(g) -> a g^2 + b g + c  (per neuron, coefficients trained)

Multiplicative depth (levels) of each piece is reported by `depth_*` and checked against the engine.
"""
import math

import numpy as np
import torch


# ------------------------------------------------------------------ start values
def fit_start(fn, lo, hi, degree, n=4000):
    """Least-squares fit (relative error, log-spaced points) of a degree-`degree` polynomial to fn on [lo, hi].
    Returns coefficients c[0..degree] in the scaled variable u = (2x - (lo+hi)) / (hi - lo) in [-1, 1]."""
    x = np.exp(np.linspace(math.log(lo), math.log(hi), n))
    y = fn(x)
    u = (2 * x - (lo + hi)) / (hi - lo)
    V = np.stack([u ** k for k in range(degree + 1)], 1) / y[:, None]     # relative error weighting
    c, *_ = np.linalg.lstsq(V, np.ones_like(x), rcond=None)
    return c


class NewtonApprox:
    """Start polynomial (in u = a x + b) + Newton iterations. kind = 'inv_sqrt' or 'inv'."""

    def __init__(self, kind, lo, hi, degree, iters):
        assert kind in ("inv_sqrt", "inv") and 0 < lo < hi
        self.kind, self.lo, self.hi, self.degree, self.iters = kind, float(lo), float(hi), int(degree), int(iters)
        fn = (lambda x: x ** -0.5) if kind == "inv_sqrt" else (lambda x: 1.0 / x)
        self.coef = fit_start(fn, self.lo, self.hi, self.degree)
        self.a = 2.0 / (self.hi - self.lo)
        self.b = -(self.hi + self.lo) / (self.hi - self.lo)

    def to_dict(self):
        return dict(kind=self.kind, lo=self.lo, hi=self.hi, degree=self.degree, iters=self.iters,
                    coef=[float(c) for c in self.coef], a=self.a, b=self.b)

    def start(self, x):
        u = x * self.a + self.b
        y = torch.full_like(x, float(self.coef[-1])) if torch.is_tensor(x) else np.full_like(x, self.coef[-1])
        for c in self.coef[-2::-1]:
            y = y * u + float(c)
        return y

    def __call__(self, x):
        y = self.start(x)
        for _ in range(self.iters):
            if self.kind == "inv_sqrt":
                y = 1.5 * y - 0.5 * (x * y) * (y * y)
            else:
                y = y * (2.0 - x * y)
        return y

    def max_rel_err(self):
        x = np.exp(np.linspace(math.log(self.lo), math.log(self.hi), 20000))
        ref = x ** -0.5 if self.kind == "inv_sqrt" else 1.0 / x
        return float(np.abs(self(x) / ref - 1).max())

    def depth(self):
        # engine: the start polynomial costs `degree` levels (affine step folded into the first Horner step),
        # every Newton iteration 2 levels (inv_sqrt uses the 1/kappa trick, see engine.Engine3.inv_sqrt)
        return self.degree + 2 * self.iters


def search_newton(kind, lo, hi, target=1e-3, max_depth=12):
    """Cheapest (depth) start degree / iteration count reaching max relative error <= target on [lo, hi]."""
    best = None
    for degree in range(1, 6):
        for iters in range(0, 6):
            ap = NewtonApprox(kind, lo, hi, degree, iters)
            if ap.depth() > max_depth:
                continue
            err = ap.max_rel_err()
            if err <= target and (best is None or ap.depth() < best.depth()):
                best = ap
    return best


# ------------------------------------------------------------------ activation
def silu(x):
    return x * torch.sigmoid(x)


def fit_poly_act(x, degree=2):
    """Per-channel LS fit of silu on samples x (n, C) -> coefficients (degree+1, C) in x (unscaled)."""
    x = x.double()
    X = torch.stack([x ** k for k in range(degree + 1)], -1)               # (n, C, K)
    y = silu(x)
    A = torch.einsum("nck,ncl->ckl", X, X) / len(x)
    A = A + 1e-6 * torch.diagonal(A, dim1=-2, dim2=-1).mean(-1)[:, None, None] * torch.eye(degree + 1, dtype=A.dtype)
    rhs = torch.einsum("nck,nc->ck", X, y) / len(x)
    return torch.linalg.solve(A, rhs).T.float()                            # (K, C)


def poly_eval(coef, x):
    y = coef[-1].to(x.dtype)
    for k in range(coef.shape[0] - 2, -1, -1):
        y = y * x + coef[k].to(x.dtype)
    return y

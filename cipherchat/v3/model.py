"""CipherChat v3 – FHE-friendly surgery on a pretrained Qwen2/Qwen2.5 causal LM (HF transformers).

Replaced (positions >= 1; see BOS note):
  RMSNorm   -> x * inv_sqrt(mean(x^2) / m_ref) / sqrt(m_ref) * g          (approx.NewtonApprox, calibrated range)
  softmax   -> power attention e = (alpha s + c)^(2p), divided by den via approx inverse (calibrated range)
  SiLU gate -> per-neuron quadratic a g^2 + b g + c                          (trained)

Public prefix: positions 0..n_pub-1 hold the fixed public tokens "<|endoftext|>" + newline (config model.public_prefix).
Their forward pass is input independent, so the server computes them in plaintext (exact ops) and inserts their K/V
into the encrypted cache. Training mirrors that: prefix positions use the exact ops, later positions the
approximations. With a one-token prefix Qwen2.5 still puts massive activations (mean(x^2) ~ 1e3) on ~20 % of the
first user tokens; with the two-token prefix they only occur at position 0 (scripts/v3_stats.py).

`mix` (curriculum) blends approximation and exact op during the first distillation steps; mix = 1 is the FHE model.
"""
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .approx import NewtonApprox, fit_poly_act, poly_eval, search_newton, silu

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = {"mix": 1.0, "calib": False, "n_pub": 2, "clamp": False}
# clamp=True (training only) = range guard: an input outside the calibrated range of an approximation (Newton
# inverse / inverse sqrt, activation polynomial) uses the EXACT op instead, so rare outliers cannot start the
# polynomial error cascade that diverges after a few layers; the range penalty pulls such inputs inside.
# Evaluation uses clamp=False = exactly what the server computes.   # n_pub: public prefix positions (exact ops)


def load_config(path=None):
    return json.load(open(path or os.path.join(HERE, "config.json")))


def acc_dtype(dt):
    """float32 accumulation for bf16 / fp32 training, float64 when the reference runs in float64."""
    return torch.float64 if dt == torch.float64 else torch.float32


def range_excess(x, lo, hi):
    """Squared relative distance outside [lo, hi] (0 inside): keeps encrypted values where the approximation holds."""
    return (F.relu(lo - x) / lo + F.relu(x - hi) / hi).pow(2).mean()


def range_penalty(mods):
    norms, acts, attns = mods
    ex = [m.excess for m in list(norms) + list(attns) + list(acts) if m.excess is not None]
    return torch.stack(ex).sum() if ex else None


# ------------------------------------------------------------------ RMSNorm
class ApproxRMSNorm(nn.Module):
    def __init__(self, orig, name):
        super().__init__()
        self.weight = orig.weight
        self.eps = orig.variance_epsilon
        self.name = name
        self.approx = None                     # NewtonApprox on z = m / m_ref
        self.register_buffer("m_ref", torch.ones(()))
        self.samples = []
        self.excess = None

    def forward(self, x):
        dt = x.dtype
        x = x.to(acc_dtype(x.dtype))
        m = x.pow(2).mean(-1, keepdim=True)
        exact = x * torch.rsqrt(m + self.eps)
        if STATE["calib"]:
            self.samples.append(m[:, STATE['n_pub']:].detach().flatten().cpu())
        if self.approx is None:
            y = exact
        else:
            z = m / self.m_ref
            if STATE["clamp"]:
                inside = (z >= self.approx.lo) & (z <= self.approx.hi)
                ap = torch.where(inside, x * self.approx(z.clamp(self.approx.lo, self.approx.hi)) / torch.sqrt(self.m_ref),
                                 exact)
            else:
                ap = x * self.approx(z) / torch.sqrt(self.m_ref)
            if self.training:
                self.excess = range_excess(z[:, STATE['n_pub']:], self.approx.lo, self.approx.hi)
            mix = STATE["mix"]
            ap = ap if mix >= 1 else mix * ap + (1 - mix) * exact
            k = STATE['n_pub']
            y = torch.cat([exact[:, :k], ap[:, k:]], 1)
        return (self.weight * y).to(dt)


# ------------------------------------------------------------------ activation
class PolyAct(nn.Module):
    def __init__(self, channels, degree=2):
        super().__init__()
        self.coef = nn.Parameter(torch.zeros(degree + 1, channels))
        self.register_buffer("R", torch.ones(channels))      # calibrated range |g| <= R per neuron
        self.ready = False
        self.samples = []
        self.excess = None

    def forward(self, g):
        if STATE["calib"]:
            rows = g[:, STATE['n_pub']:].detach().reshape(-1, g.shape[-1])
            idx = torch.randint(0, rows.shape[0], (256,), device=rows.device)
            self.samples.append(rows[idx].float().cpu())
        if not self.ready:
            return silu(g)
        y = poly_eval(self.coef, g)
        Rg = self.R.to(g.dtype)
        if STATE["clamp"]:
            y = torch.where(g.abs() <= Rg, y, silu(g))
        if self.training:
            self.excess = F.relu(g[:, STATE["n_pub"]:].abs().float() / self.R - 1).pow(2).mean()
        mix = STATE["mix"]
        if mix < 1:
            y = mix * y + (1 - mix) * silu(g)
        k = STATE['n_pub']
        return torch.cat([silu(g[:, :k]), y[:, k:]], 1)


# ------------------------------------------------------------------ attention
def allowed_mask(attention_mask, T, S, dev):
    if attention_mask is None:
        return torch.ones(T, S, device=dev, dtype=torch.bool).tril(S - T)[None, None]
    if attention_mask.dtype == torch.bool:
        return attention_mask
    return attention_mask > -1.0


def power_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """query/key/value: (B, H, T, D) after RoPE. Returns (B, T, H, D)."""
    st = module.v3                              # AttnState attached by convert()
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)
    B, H, T, D = query.shape
    S = key.shape[2]
    scaling = scaling if scaling is not None else 1.0 / math.sqrt(D)
    adt = acc_dtype(query.dtype)
    q32, k32, v32 = query.to(adt), key.to(adt), value.to(adt)
    s = torch.matmul(q32, k32.transpose(-1, -2)) * scaling
    allow = allowed_mask(attention_mask, T, S, query.device)
    exact = s.masked_fill(~allow, float("-inf")).softmax(-1)
    if not st.ready:
        w = exact
    else:
        e = (st.alpha.view(1, H, 1, 1) * s + st.c.view(1, H, 1, 1)).pow(2 * st.p) * allow
        npos = allow.sum(-1, keepdim=True).clamp(min=1).to(adt)          # t + 1 (public)
        den = e.sum(-1, keepdim=True)
        dn = den / (npos * st.d_ref.view(1, H, 1, 1))
        if STATE["calib"]:
            st.samples.append(dn[:, :, STATE['n_pub']:].detach().flatten(2).cpu())
        if STATE.get("diag") is not None and st.approx is not None:     # scripts/v3_range_diag.py
            STATE["diag"].append((dn[:, :, STATE['n_pub']:].detach().float(), st.approx.lo, st.approx.hi))
        if st.norm == "pos":                                              # no division: public 1 / ((t+1) kappa_h)
            w = e * torch.exp(-st.logk).view(1, H, 1, 1) / npos
        elif st.approx is None:
            w = e / den
        else:
            if STATE["clamp"]:
                inside = (dn >= st.approx.lo) & (dn <= st.approx.hi)
                inv = torch.where(inside, st.approx(dn.clamp(st.approx.lo, st.approx.hi)), 1.0 / dn.clamp(min=1e-30))
            else:
                inv = st.approx(dn)
            w = e * (inv / (npos * st.d_ref.view(1, H, 1, 1)))
            if module.training:
                st.excess = range_excess(dn[:, :, STATE['n_pub']:], st.approx.lo, st.approx.hi)
        mix = STATE["mix"]
        if mix < 1:
            w = mix * w + (1 - mix) * exact
        k = STATE['n_pub']
        w = torch.cat([exact[:, :, :k], w[:, :, k:]], 2)                  # public prefix queries exact
    out = torch.matmul(w, v32).to(query.dtype)
    return out.transpose(1, 2).contiguous(), None


class AttnState(nn.Module):
    """norm = "div": w = e / sum(e) with the sum inverted by approx.NewtonApprox (the spec's power-softmax);
       norm = "pos": w = e / ((t + 1) kappa_h), kappa_h trained - no encrypted division at all."""

    def __init__(self, H, p, norm="div"):
        super().__init__()
        self.p, self.norm = p, norm
        self.logk = nn.Parameter(torch.zeros(H))
        # (1 + s / 2p)^(2p) ~ exp(s) near s = 0: start close to softmax
        self.alpha = nn.Parameter(torch.full((H,), 1.0 / (2 * p)))
        self.c = nn.Parameter(torch.ones(H))
        self.excess = None
        self.register_buffer("d_ref", torch.ones(H))
        self.approx = None
        self.ready = False
        self.samples = []


def register():
    from transformers import AttentionInterface
    AttentionInterface.register("v3power", power_attention)


# ------------------------------------------------------------------ conversion + calibration
def convert(model, cfg):
    """Swap norms / activation / attention of a Qwen2 model in place. Returns lists of the new modules."""
    register()
    ap = cfg["approx"]
    norms, acts, attns = [], [], []
    for i, layer in enumerate(model.model.layers):
        layer.input_layernorm = ApproxRMSNorm(layer.input_layernorm, f"L{i}.ln1")
        layer.post_attention_layernorm = ApproxRMSNorm(layer.post_attention_layernorm, f"L{i}.ln2")
        norms += [layer.input_layernorm, layer.post_attention_layernorm]
        dev = layer.mlp.gate_proj.weight.device
        layer.mlp.act_fn = PolyAct(layer.mlp.gate_proj.out_features, ap["act"]["degree"]).to(dev)
        acts.append(layer.mlp.act_fn)
        H = model.config.num_attention_heads
        layer.self_attn.v3 = AttnState(H, ap["softmax"]["p"], ap["softmax"].get("norm", "div")).to(dev)
        attns.append(layer.self_attn.v3)
    model.config._attn_implementation = "v3power"
    for layer in model.model.layers:
        layer.self_attn.config._attn_implementation = "v3power"
    return norms, acts, attns


@torch.no_grad()
def run_calib(model, batches):
    STATE["calib"] = True
    try:
        for b in batches:
            model(input_ids=b.to(model.device))
    finally:
        STATE["calib"] = False


@torch.no_grad()
def qrange(x, q):
    x = x.flatten().double()
    if x.numel() > 2_000_000:
        x = x[torch.randperm(x.numel())[:2_000_000]]
    return float(torch.quantile(x, q)), float(torch.quantile(x, 1 - q))


def _dn_cat(st):
    return torch.cat([x.reshape(x.shape[0], x.shape[1], -1).permute(1, 0, 2).reshape(x.shape[1], -1)
                      for x in st.samples], 1)                                    # (H, n)


def calibrate(model, mods, batches, cfg, margin=2.0, target=2e-3, max_depth=None, log=print, q=1e-4, act_margin=1.5):
    """Calibration on the EXACT model's activations (mix = 0: approximations are evaluated and sampled, but the
    forward pass uses the exact ops). Sets: activation polynomials (per-neuron LS fit of silu), attention
    d_ref (geometric mean of den / (t+1)) and kappa (pos), RMSNorm m_ref and the inverse-sqrt range.
    The attention inverse (norm = "div") is fitted later by calibrate_inverse() once alpha / c are trained."""
    norms, acts, attns = mods
    info = {}
    old_mix = STATE["mix"]
    STATE["mix"] = 0.0
    try:
        run_calib(model, batches)                                  # activations
        for a in acts:
            x = torch.cat(a.samples)
            a.coef.data = fit_poly_act(x, a.coef.shape[0] - 1).to(a.coef.device)
            a.R.data = (x.abs().amax(0) * act_margin + 1e-3).to(a.R.device)
            a.samples, a.ready = [], True
        for st in attns:
            st.ready = True
        run_calib(model, batches)                                  # attention + norms
        for st in attns:
            dn = _dn_cat(st)
            st.d_ref.data = dn.clamp(min=1e-30).log().mean(1).exp().to(st.d_ref.device)
            st.logk.data = st.d_ref.data.log().clone()
            st.samples = []
        zs = []
        for nm in norms:
            m = torch.cat(nm.samples)
            nm.m_ref.data = m.clamp(min=1e-30).log().mean().exp().to(nm.m_ref.device)
            zs.append(m / nm.m_ref.cpu())
            nm.samples = []
    finally:
        STATE["mix"] = old_mix
    z = torch.cat(zs)
    lo, hi = qrange(z, q)
    lo, hi = lo / margin, hi * margin
    md = max_depth or cfg["approx"]["rmsnorm"].get("max_depth", 12)
    isq = search_newton("inv_sqrt", lo, hi, target, md) or NewtonApprox(
        "inv_sqrt", lo, hi, 3, cfg["approx"]["rmsnorm"]["newton_iters"])
    for nm in norms:
        nm.approx = isq
    info["rmsnorm"] = dict(isq.to_dict(), max_rel_err=isq.max_rel_err(), depth=isq.depth(),
                           z_min=float(z.min()), z_max=float(z.max()))
    log(f"rmsnorm z = m / m_ref: min {float(z.min()):.3g} max {float(z.max()):.3g}; inv-sqrt on [{lo:.3g}, {hi:.3g}]: "
        f"degree {isq.degree}, {isq.iters} Newton, depth {isq.depth()}, rel err {isq.max_rel_err():.2e}")
    info["m_ref"] = {nm.name: float(nm.m_ref) for nm in norms}
    return info


@torch.no_grad()
def calibrate_inverse(model, mods, batches, cfg, margin=2.0, target=2e-3, max_depth=None, log=print, q=1e-4, mix=None,
                      ratio=None):
    """norm = "div": range of the normalised denominators dn = sum(e) / ((t+1) d_ref) on the current model ->
    Newton inverse. Re-estimates d_ref per head first (geometric mean)."""
    norms, acts, attns = mods
    for st in attns:
        st.approx = None
    old_mix = STATE["mix"]
    if mix is not None:
        STATE["mix"] = mix
    run_calib(model, batches)
    for st in attns:
        dn = _dn_cat(st)
        st.d_ref.data *= dn.clamp(min=1e-30).log().mean(1).exp().to(st.d_ref.device)
        st.samples = []
    run_calib(model, batches)
    STATE["mix"] = old_mix
    dns = torch.cat([_dn_cat(st).flatten() for st in attns])
    dns = dns[torch.isfinite(dns)]
    for st in attns:
        st.samples = []
    if ratio:
        # forced target range: hi / lo = ratio around the median. The natural spread of the power-softmax
        # denominators is ~4-5 decades, far beyond what a shallow Newton inverse (and ~17-bit CKKS precision after
        # bootstrapping) can handle; the range penalty + guard train the model into this window instead.
        center = float(dns.clamp(min=1e-30).log().median().exp())
        lo, hi = center / math.sqrt(ratio), center * math.sqrt(ratio)
    else:
        lo, hi = qrange(dns, q)
        lo, hi = max(lo, 1e-4 * hi) / margin, hi * margin  # an all-zero row would give lo = 0
    md = max_depth or cfg["approx"]["attn_inverse"].get("max_depth", 12)
    inv = search_newton("inv", lo, hi, target, md) or NewtonApprox(
        "inv", lo, hi, 3, cfg["approx"]["attn_inverse"]["newton_iters"])
    for st in attns:
        st.approx = inv
    log(f"attention denominators (normalised): min {float(dns.min()):.3g} max {float(dns.max()):.3g}; inverse on "
        f"[{lo:.3g}, {hi:.3g}]: degree {inv.degree}, {inv.iters} Newton, depth {inv.depth()}, rel err {inv.max_rel_err():.2e}")
    return dict(inv.to_dict(), max_rel_err=inv.max_rel_err(), depth=inv.depth(), dn_min=float(dns.min()),
                dn_max=float(dns.max()))


def fhe_state(model, mods):
    """Everything besides the HF weights that defines the FHE model (saved next to the checkpoint)."""
    norms, acts, attns = mods
    return dict(
        norm_approx=norms[0].approx.to_dict() if norms[0].approx else None,
        attn_approx=attns[0].approx.to_dict() if attns[0].approx else None,
        m_ref=[float(n.m_ref) for n in norms],
        d_ref=[a.d_ref.tolist() for a in attns],
        p=attns[0].p, attn_norm=attns[0].norm,
    )


def load_fhe_state(model, mods, st):
    norms, acts, attns = mods
    na = NewtonApprox(**{k: st["norm_approx"][k] for k in ("kind", "lo", "hi", "degree", "iters")})
    aa = NewtonApprox(**{k: st["attn_approx"][k] for k in ("kind", "lo", "hi", "degree", "iters")}) \
        if st.get("attn_approx") else None
    for n, m in zip(norms, st["m_ref"]):
        n.approx, n.m_ref.data = na, torch.tensor(m, device=n.m_ref.device)
    for a, d in zip(attns, st["d_ref"]):
        a.approx, a.ready = aa, True
        a.d_ref.data = torch.tensor(d, device=a.d_ref.device)
    for a in acts:
        a.ready = True

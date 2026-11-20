"""CipherFormer2 - a decoder co-designed with the v2 CKKS engine.

Differences to v1 (cipherchat/model.py), each one removes FHE work:
  * NoPE input + learned RELATIVE position bias rho[t-j] inside the polynomial attention
    -> with the shifting KV cache the bias is a static plaintext; no per-position re-encoding.
  * fixed attention normalisation kappa (instead of 1/(t+1)) -> static query plaintexts.
  * bilinear scores s_tj = x_t M x_j with a constant channel (x[63] = 1) carrying all biases
    -> no query projection at all: the cache stores k~_j = x_j M^T, the score is x_t . k~_j.
  * layer 0 has no MLP: on a one-hot input an MLP is just a lookup table, so two embedding
    tables (E_attn for the attention input, E_res for the residual) express the same function.
  * MLPs of layers >= 1 can be very wide (hidden up to 8192): in the engine the hidden layer fills
    the otherwise unused CKKS slots, so its cost is that of a single 64x64 matvec.
Everything stays a polynomial: w = kappa*(s + c + rho)^2, m = (x W1)^2, depth 4 per layer.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 64
CONST = D_MODEL - 1          # index of the constant-1 channel


class CipherFormer2(nn.Module):
    def __init__(self, vocab_size, n_layers=2, hidden=1024, max_len=64, kappa=1.0 / 64, init_scale=1.0, n_sig=0,
                 mlp="sq"):
        """n_sig > 0: the client adds a second hot entry (dims vocab_size .. vocab_size+n_sig-1 of the input
        one-hot) carrying the digit significance (position coupling); free for the engine.
        mlp: "sq"    m = (x W1)^2                     (depth 2, as v1)
             "bil"   m = (x W1) * (x Wg)              (bilinear GLU, depth 2, square is a special case)
             "sqglu" m = ((x W1) * (x Wg))^2          (degree 4; uses the MLP path's spare level - free in depth)"""
        assert mlp in ("sq", "bil", "sqglu")
        super().__init__()
        assert vocab_size + n_sig <= D_MODEL and n_layers >= 2
        d, D = D_MODEL, D_MODEL - 1
        self.cfg = dict(arch="cipherformer2", vocab_size=vocab_size, n_layers=n_layers, hidden=hidden,
                        max_len=max_len, kappa=kappa, d=d, n_sig=n_sig, mlp=mlp)
        self.kappa = kappa
        self.n_sig = n_sig
        self.Ea = nn.Parameter(torch.randn(vocab_size, D) * 0.5 * init_scale)
        self.Er = nn.Parameter(torch.randn(vocab_size, D) * 0.5 * init_scale)
        self.Pa = nn.Parameter(torch.randn(max(n_sig, 1), D) * 0.5 * init_scale)
        self.Pr = nn.Parameter(torch.randn(max(n_sig, 1), D) * 0.5 * init_scale)
        self.Wq = nn.ParameterList([nn.Parameter(torch.randn(d, d) * 0.3 / math.sqrt(d)) for _ in range(n_layers)])
        self.Wk = nn.ParameterList([nn.Parameter(torch.randn(d, d) * 0.3 / math.sqrt(d)) for _ in range(n_layers)])
        self.Wv = nn.ParameterList([nn.Parameter(torch.randn(d, D) / math.sqrt(d)) for _ in range(n_layers)])
        self.c = nn.Parameter(torch.ones(n_layers))
        self.rho = nn.Parameter(torch.zeros(n_layers, max_len))
        self.alpha_a = nn.Parameter(torch.full((n_layers,), 0.3))
        self.alpha_m = nn.Parameter(torch.full((n_layers,), 0.3))
        W1, W2, b2, Wg = [], [], [], []
        for _ in range(n_layers - 1):
            w1 = torch.randn(d, hidden) / math.sqrt(d)
            w1[CONST] = 0.5                       # b1 = 0.5 through the constant channel
            W1.append(nn.Parameter(w1))
            wg = torch.randn(d, hidden) / math.sqrt(d)
            wg[CONST] = 1.0
            Wg.append(nn.Parameter(wg))
            W2.append(nn.Parameter(torch.randn(hidden, D) / math.sqrt(hidden)))
            b2.append(nn.Parameter(torch.zeros(D)))
        self.W1, self.W2, self.b2 = nn.ParameterList(W1), nn.ParameterList(W2), nn.ParameterList(b2)
        self.Wg = nn.ParameterList(Wg) if mlp != "sq" else None
        self.mlp = mlp
        self.Wu = nn.Parameter(torch.randn(d, vocab_size) / math.sqrt(d))
        self.bu = nn.Parameter(torch.zeros(vocab_size))
        self.last_stats = None
        self.track_stats = True

    def _embed(self, E, P, ids, sig):
        e = E[ids]
        if self.n_sig and sig is not None:
            Pz = torch.cat([torch.zeros(1, P.shape[1], device=P.device, dtype=P.dtype), P], 0)   # sig 0 -> nothing
            e = e + Pz[sig]
        return torch.cat([e, torch.ones(*ids.shape, 1, device=ids.device, dtype=e.dtype)], -1)

    def forward(self, ids, sig=None):
        Bsz, T = ids.shape
        dev = ids.device
        ea = self._embed(self.Ea, self.Pa, ids, sig)
        x = self._embed(self.Er, self.Pr, ids, sig)
        t = torch.arange(T, device=dev)
        rel = t[:, None] - t[None, :]
        causal = (rel >= 0).float()
        rel_idx = rel.clamp(min=0)
        stats = {"s": 0.0, "w": 0.0, "z": 0.0, "x": 0.0}
        for l in range(self.cfg["n_layers"]):
            inp = ea if l == 0 else x
            q = inp @ self.Wq[l]
            k = inp @ self.Wk[l]
            s = q @ k.transpose(-1, -2) + self.c[l] + self.rho[l][rel_idx]
            w = self.kappa * s.pow(2) * causal
            upd = self.alpha_a[l] * (w @ (inp @ self.Wv[l]))
            if l >= 1:
                z = inp @ self.W1[l - 1]
                if self.mlp == "sq":
                    m = z.pow(2)
                elif self.mlp == "bil":
                    m = z * (inp @ self.Wg[l - 1])
                else:
                    m = (z * (inp @ self.Wg[l - 1])).pow(2)
                upd = upd + self.alpha_m[l] * (m @ self.W2[l - 1] + self.b2[l - 1])
            x = x + F.pad(upd, (0, 1))
            if self.track_stats:
                stats["s"] = max(stats["s"], float((s.detach() * causal).abs().max() * math.sqrt(self.kappa)))
                stats["w"] = max(stats["w"], float(w.detach().abs().max()))
                stats["x"] = max(stats["x"], float(x.detach().abs().max()))
                if l >= 1:
                    stats["z"] = max(stats["z"], float(z.detach().abs().max()))
        logits = x @ self.Wu + self.bu
        if self.track_stats:
            self.last_stats = stats
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids, max_new=24, eos=3, sig_fn=None):
        return self.generate_batch([prompt_ids], max_new=max_new, eos=eos, sig_fn=sig_fn)[0]

    @torch.no_grad()
    def generate_batch(self, prompts, max_new=8, eos=3, pad=0, sig_fn=None):
        """Greedy decoding for many prompts at once (left-aligned, per-row lengths).
        sig_fn(token_list) -> significance list (client-side position coupling), recomputed every step."""
        dev = self.Wu.device
        n = len(prompts)
        seqs = [list(p) for p in prompts]
        outs = [[] for _ in range(n)]
        done = [False] * n
        for _ in range(max_new):
            T = max(len(s) for s in seqs)
            if T >= self.cfg["max_len"]:
                break
            ids = torch.full((n, T), pad, dtype=torch.long, device=dev)
            sig = torch.zeros((n, T), dtype=torch.long, device=dev) if sig_fn else None
            for i, s in enumerate(seqs):
                ids[i, : len(s)] = torch.tensor(s, device=dev)
                if sig_fn:
                    sig[i, : len(s)] = torch.tensor(sig_fn(s), device=dev)
            logits = self(ids, sig)
            lens = torch.tensor([len(s) - 1 for s in seqs], device=dev)
            nxt = logits[torch.arange(n, device=dev), lens].argmax(-1).tolist()
            for i in range(n):
                if done[i]:
                    continue
                if nxt[i] == eos:
                    done[i] = True
                    continue
                outs[i].append(nxt[i])
                seqs[i].append(nxt[i])
            if all(done):
                break
        return outs

    def export(self):
        """Fold everything into the matrices the v2 engine multiplies with (row convention y = x @ W)."""
        g = lambda p: p.detach().cpu().double().numpy()
        cfg = self.cfg
        d, V, L, H = D_MODEL, cfg["vocab_size"], cfg["n_layers"], cfg["hidden"]
        sk = math.sqrt(cfg["kappa"])
        pad_cols = lambda M: np.concatenate([M, np.zeros((M.shape[0], d - M.shape[1]))], 1)
        ns = cfg.get("n_sig", 0)
        ea = np.zeros((d, d)); ea[:V, :d - 1] = g(self.Ea); ea[:V, CONST] = 1.0
        er = np.zeros((d, d)); er[:V, :d - 1] = g(self.Er); er[:V, CONST] = 1.0
        if ns:   # input rows V .. V+ns-1 = significance 1..ns (second hot entry, no constant channel)
            ea[V:V + ns, :d - 1] = g(self.Pa)[:ns]
            er[V:V + ns, :d - 1] = g(self.Pr)[:ns]
        Wu = np.zeros((d, d)); Wu[:, :V] = g(self.Wu)
        bu = np.zeros(d); bu[:V] = g(self.bu)
        aa, am, c, rho = g(self.alpha_a), g(self.alpha_m), g(self.c), g(self.rho)
        W = {"cfg_json": np.array(str(cfg))}
        # layer 0: everything is a lookup of the current token
        G = ea @ g(self.Wq[0]) @ g(self.Wk[0]).T @ ea.T                       # G[a, b] = q(a) . k(b)
        W["L0.K"] = sk * G.T                                                  # k^_j = onehot_j @ K  (column of G)
        W["L0.V"] = aa[0] * pad_cols(ea @ g(self.Wv[0]))
        W["L0.X"] = er
        W["L0.CR"] = sk * (c[0] + rho[0])
        for l in range(1, L):
            last = l == L - 1
            W[f"L{l}.K"] = sk * (g(self.Wk[l]) @ g(self.Wq[l]).T)            # k~_j = x_j @ K, score = x_t . k~_j
            Vl = aa[l] * pad_cols(g(self.Wv[l]))
            W2 = am[l] * pad_cols(g(self.W2[l - 1]))
            b2 = am[l] * np.concatenate([g(self.b2[l - 1]), [0.0]])
            W[f"L{l}.Z"] = g(self.W1[l - 1])
            if self.mlp != "sq":
                W[f"L{l}.G"] = g(self.Wg[l - 1])
            W[f"L{l}.CR"] = sk * (c[l] + rho[l])
            if last:
                W[f"L{l}.V"] = Vl @ Wu
                W[f"L{l}.W2"] = W2 @ Wu
                W[f"L{l}.XU"] = Wu
                W["BIAS"] = bu + b2 @ Wu
            else:
                W[f"L{l}.V"] = Vl
                W[f"L{l}.W2"] = W2
                W[f"L{l}.B2"] = b2
        return W


def masked_ce(logits, ids, mask):
    logits = logits[:, :-1]
    tgt = ids[:, 1:]
    m = mask[:, :-1].float()
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none")
    return (loss * m.reshape(-1)).sum() / m.sum()

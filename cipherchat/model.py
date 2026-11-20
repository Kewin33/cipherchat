"""CipherFormer: a tiny, FHE-friendly decoder-only transformer.

Design choices (all driven by what is cheap under CKKS):
  * parallel blocks (GPT-J/PaLM style): x <- x + Attn(x) + MLP(x)
    -> attention and MLP share multiplicative levels (depth 4 per block instead of 7)
  * polynomial attention: w_tj = (q_t.k_j + c)^2 / (t+1)   (no softmax, no max, no division
    by an encrypted value - the 1/(t+1) factor is public and folded into the weights)
  * square activation in the MLP: m = z^2
  * no LayerNorm (needs an encrypted inverse square root); stability via ReZero-style
    learnable residual gains alpha, which are folded into the weights at export time
Everything is a polynomial of degree 2^k in the encrypted input -> evaluable with CKKS.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.w1 = nn.Linear(d, d_ff)
        self.w2 = nn.Linear(d_ff, d)
        self.c = nn.Parameter(torch.tensor(1.0))        # attention shift
        self.alpha_a = nn.Parameter(torch.tensor(0.1))  # ReZero gains
        self.alpha_m = nn.Parameter(torch.tensor(0.1))
        for lin in (self.q, self.k):
            nn.init.normal_(lin.weight, std=0.3 / math.sqrt(d))
            nn.init.zeros_(lin.bias)
        nn.init.constant_(self.w1.bias, 0.5)

    def forward(self, x, causal_norm):
        q, k = self.q(x), self.k(x)
        s = q @ k.transpose(-1, -2)                       # (B, T, T)
        w = (s + self.c) ** 2 * causal_norm               # polynomial attention
        a = self.o(w @ self.v(x))
        m = self.w1(x) ** 2                               # square activation
        mlp = self.w2(m)
        stats = (s.detach().abs().max(), w.detach().abs().max(), m.detach().abs().max())
        return x + self.alpha_a * a + self.alpha_m * mlp, stats


class CipherFormer(nn.Module):
    def __init__(self, vocab_size=64, d=64, d_ff=64, n_layers=2, max_len=64):
        super().__init__()
        self.cfg = dict(vocab_size=vocab_size, d=d, d_ff=d_ff, n_layers=n_layers, max_len=max_len)
        self.emb = nn.Embedding(vocab_size, d)
        self.pos = nn.Parameter(torch.randn(max_len, d) * 0.3)
        nn.init.normal_(self.emb.weight, std=0.5)
        self.blocks = nn.ModuleList([Block(d, d_ff) for _ in range(n_layers)])
        self.head = nn.Linear(d, vocab_size)
        self.last_stats = None

    def forward(self, ids):
        B, T = ids.shape
        x = self.emb(ids) + self.pos[:T]
        t = torch.arange(T, device=ids.device)
        causal_norm = torch.tril(torch.ones(T, T, device=ids.device)) / (t[:, None] + 1.0)
        stats = []
        for blk in self.blocks:
            x, st = blk(x, causal_norm)
            stats.append(st)
        self.last_stats = stats + [x.detach().abs().max()]
        return self.head(x)

    @torch.no_grad()
    def generate(self, prompt_ids, max_new=24, eos=3):
        ids = list(prompt_ids)
        out = []
        for _ in range(max_new):
            if len(ids) >= self.cfg["max_len"]:
                break
            logits = self(torch.tensor([ids], device=self.head.weight.device))[0, -1]
            nxt = int(logits.argmax())
            if nxt == eos:
                break
            out.append(nxt)
            ids.append(nxt)
        return out

    def export(self):
        """Fold ReZero gains + output projections and return plain numpy weights (FHE layout).

        Row-vector convention: y = x @ W + b  with W of shape (in, out).
        """
        import numpy as np
        g = lambda t: t.detach().cpu().double().numpy()
        W = {"E": g(self.emb.weight), "P": g(self.pos), "Wu": g(self.head.weight).T, "bu": g(self.head.bias)}
        for i, b in enumerate(self.blocks):
            aa, am = float(b.alpha_a.detach()), float(b.alpha_m.detach())
            Wv, bv = g(b.v.weight).T, g(b.v.bias)
            Wo, bo = g(b.o.weight).T, g(b.o.bias)
            W[f"L{i}.Wq"], W[f"L{i}.bq"] = g(b.q.weight).T, g(b.q.bias)
            W[f"L{i}.Wk"], W[f"L{i}.bk"] = g(b.k.weight).T, g(b.k.bias)
            W[f"L{i}.Wvo"] = aa * (Wv @ Wo)        # v'_j = x_j Wvo + bvo   (cached, encrypted)
            W[f"L{i}.bvo"] = aa * (bv @ Wo)
            W[f"L{i}.bo"] = aa * bo                 # added once after aggregation
            W[f"L{i}.c"] = np.array(float(b.c.detach()))
            W[f"L{i}.W1"], W[f"L{i}.b1"] = g(b.w1.weight).T, g(b.w1.bias)
            W[f"L{i}.W2"], W[f"L{i}.b2"] = am * g(b.w2.weight).T, am * g(b.w2.bias)
        return W


def masked_ce(logits, ids, mask):
    """Next-token cross-entropy on answer positions only."""
    logits = logits[:, :-1]
    tgt = ids[:, 1:]
    m = mask[:, :-1].float()
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none")
    return (loss * m.reshape(-1)).sum() / m.sum()

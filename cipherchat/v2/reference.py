"""Plaintext (numpy) reference for the v2 engine: token-by-token with KV cache, folded weights,
exactly the arithmetic the encrypted engine performs (no slot layouts, see engine.py for those)."""
import ast
import json

import numpy as np

B = 64


class Folded2:
    def __init__(self, npz_path, json_path=None):
        W = dict(np.load(npz_path, allow_pickle=False))
        self.W = W
        self.cfg = json.load(open(json_path or npz_path.replace(".npz", ".json"), encoding="utf-8"))
        c = self.cfg
        self.V, self.L, self.H, self.T = c["vocab_size"], c["n_layers"], c["hidden"], c["max_len"]

    @classmethod
    def from_dict(cls, W, cfg):
        self = cls.__new__(cls)
        self.W = {k: np.asarray(v) for k, v in W.items()}
        self.cfg = cfg
        self.V, self.L, self.H, self.T = cfg["vocab_size"], cfg["n_layers"], cfg["hidden"], cfg["max_len"]
        return self


class PlainReference2:
    def __init__(self, fm: Folded2):
        self.fm = fm
        self.reset()

    def reset(self):
        self.K = [[] for _ in range(self.fm.L)]
        self.Vc = [[] for _ in range(self.fm.L)]
        self.t = 0

    def _attend(self, l, x, cr):
        t = self.t
        K = np.array(self.K[l])                       # (t+1, 64), position j = row j
        s = K @ x                                     # s_j = x . k_j
        r = t - np.arange(t + 1)                      # relative distance
        w = (s + cr[r]) ** 2
        return w @ np.array(self.Vc[l])

    def step(self, token, need_logits=True, sig=0):
        W, fm = self.fm.W, self.fm
        u = input_vector(token, sig, fm.V)
        self.K[0].append(u @ W["L0.K"])
        self.Vc[0].append(u @ W["L0.V"])
        x = u @ W["L0.X"] + self._attend(0, u, W["L0.CR"])
        logits = None
        for l in range(1, fm.L):
            last = l == fm.L - 1
            self.K[l].append(x @ W[f"L{l}.K"])
            self.Vc[l].append(x @ W[f"L{l}.V"])
            if last and not need_logits:
                break
            a = self._attend(l, x, W[f"L{l}.CR"])
            mlp = mlp_act(fm.cfg.get("mlp", "sq"), x @ W[f"L{l}.Z"], (x @ W[f"L{l}.G"]) if f"L{l}.G" in W else None) @ W[f"L{l}.W2"]
            if last:
                logits = x @ W[f"L{l}.XU"] + a + mlp + W["BIAS"]
            else:
                x = x + a + mlp + W[f"L{l}.B2"]
        self.t += 1
        return None if logits is None else logits[: fm.V]

    def generate(self, prompt_ids, max_new=24, eos=3, sig_fn=None):
        self.reset()
        seq = list(prompt_ids)
        sig = lambda: (sig_fn(seq)[len(seq) - 1] if sig_fn else 0)
        logits = None
        for i in range(len(prompt_ids)):
            s = sig_fn(seq)[i] if sig_fn else 0
            logits = self.step(seq[i], need_logits=(i == len(prompt_ids) - 1), sig=s)
        out = []
        for _ in range(max_new):
            nxt = int(np.argmax(logits))
            if nxt == eos or self.t >= self.fm.T:
                break
            out.append(nxt)
            seq.append(nxt)
            logits = self.step(nxt, sig=sig())
        return out


def mlp_act(kind, z, g):
    if kind == "sq":
        return z ** 2
    if kind == "bil":
        return z * g
    return (z * g) ** 2


def input_vector(token, sig, vocab_size):
    """Client input: one-hot of the token (+ second hot entry vocab_size+sig-1 for digit significance)."""
    u = np.zeros(B)
    u[token] = 1.0
    if sig:
        u[vocab_size + sig - 1] = 1.0
    return u

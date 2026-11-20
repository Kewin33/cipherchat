"""Plaintext (numpy) reference that performs *exactly* the computation the FHE server performs:
token-by-token decoding with a KV cache, folded weights, polynomial attention.

Used to (1) check the weight folding against PyTorch and (2) measure the CKKS error.
"""
import json

import numpy as np

B = 64  # FHE block size


class FoldedModel:
    """Fuses the exported weights into the matrices the FHE engine multiplies with.

    For layer 0 the token embedding E is folded into every projection, so the server
    starts directly from the encrypted one-hot vector (no separate embedding level).
    For the last layer the unembedding Wu is folded into the value cache (u_j = v'_j Wu),
    so the logits come out one multiplicative level earlier.
    """

    def __init__(self, npz_path, json_path=None):
        W = dict(np.load(npz_path))
        cfg = json.load(open(json_path or npz_path.replace(".npz", ".json")))
        self.cfg = cfg
        self.V, self.d, self.d_ff, self.L, self.T = cfg["vocab_size"], cfg["d"], cfg["d_ff"], cfg["n_layers"], cfg["max_len"]
        assert self.d == B and self.d_ff % B == 0 and self.V <= B
        self.n_ff = self.d_ff // B
        E = np.zeros((B, B)); E[: self.V] = W["E"]            # one-hot (64) -> d
        Wu = np.zeros((B, B)); Wu[:, : self.V] = W["Wu"]      # d -> vocab (padded to 64)
        bu = np.zeros(B); bu[: self.V] = W["bu"]
        self.P = W["P"]
        self.Wu = Wu
        self.layers = []
        for l in range(self.L):
            A = E if l == 0 else np.eye(B)                     # input transform of this layer
            last = l == self.L - 1
            Wq, bq, Wk, bk = W[f"L{l}.Wq"], W[f"L{l}.bq"], W[f"L{l}.Wk"], W[f"L{l}.bk"]
            Wvo, bvo, bo = W[f"L{l}.Wvo"], W[f"L{l}.bvo"], W[f"L{l}.bo"]
            W1, b1, W2, b2 = W[f"L{l}.W1"], W[f"L{l}.b1"], W[f"L{l}.W2"], W[f"L{l}.b2"]
            Lr = dict(last=last, c=float(W[f"L{l}.c"]))
            Lr["Mx"] = A                                        # x_l = inp @ A + beta
            Lr["Mq"], Lr["bq"] = A @ Wq, bq
            Lr["Mk"], Lr["bk"] = A @ Wk, bk
            if last:
                Lr["Mv"], Lr["bv"] = A @ Wvo @ Wu, bvo @ Wu     # value cache holds u_j = v'_j Wu
                Lr["Mxu"] = A @ Wu
                Lr["W2"] = [W2[b * B:(b + 1) * B] @ Wu for b in range(self.n_ff)]
                Lr["bfinal"] = bu + (bo + b2) @ Wu
            else:
                Lr["Mv"], Lr["bv"] = A @ Wvo, bvo
                Lr["W2"] = [W2[b * B:(b + 1) * B] for b in range(self.n_ff)]
                Lr["bres"] = bo + b2
            Lr["Mz"] = [A @ W1[:, b * B:(b + 1) * B] for b in range(self.n_ff)]
            Lr["bz"] = [b1[b * B:(b + 1) * B] for b in range(self.n_ff)]
            # raw weights needed for position-dependent biases of layer 0
            Lr["Wq"], Lr["Wk"], Lr["Wvo"], Lr["W1"] = Wq, Wk, (Wvo @ Wu if last else Wvo), W1
            self.layers.append(Lr)

    def pos_biases(self, l, t):
        """Biases of layer l at position t (layer 0 absorbs the position embedding P[t])."""
        Lr = self.layers[l]
        beta = self.P[t] if l == 0 else np.zeros(B)
        bx = beta
        bq = beta @ Lr["Wq"] + Lr["bq"]
        bk = beta @ Lr["Wk"] + Lr["bk"]
        bv = beta @ Lr["Wvo"] + Lr["bv"]
        bz = [beta @ Lr["W1"][:, b * B:(b + 1) * B] + Lr["bz"][b] for b in range(self.n_ff)]
        return bx, bq, bk, bv, bz


class PlainReference:
    def __init__(self, fm: FoldedModel):
        self.fm = fm
        self.reset()

    def reset(self):
        self.K = [[] for _ in range(self.fm.L)]
        self.Vc = [[] for _ in range(self.fm.L)]
        self.t = 0

    def step(self, token: int):
        fm, t = self.fm, self.t
        inp = np.zeros(B); inp[token] = 1.0
        sc = 1.0 / np.sqrt(t + 1)
        logits = None
        for l, Lr in enumerate(fm.layers):
            bx, bq, bk, bv, bz = fm.pos_biases(l, t)
            x = inp @ Lr["Mx"] + bx
            q = (inp @ Lr["Mq"] + bq) * sc
            self.K[l].append(inp @ Lr["Mk"] + bk)
            self.Vc[l].append(inp @ Lr["Mv"] + bv)
            s = np.array([q @ k for k in self.K[l]])
            w = (s + Lr["c"] * sc) ** 2
            av = (w[:, None] * np.array(self.Vc[l])).sum(0)
            m = [(inp @ Lr["Mz"][b] + bz[b]) ** 2 for b in range(fm.n_ff)]
            mw = sum(m[b] @ Lr["W2"][b] for b in range(fm.n_ff))
            if Lr["last"]:
                # x_last @ Wu = inp @ (A Wu) + bx @ Wu   (bx = P[t] only for a 1-layer model)
                logits = inp @ Lr["Mxu"] + bx @ fm.Wu + av + mw + Lr["bfinal"]
            else:
                inp = x + av + mw + Lr["bres"]
        self.t += 1
        return logits[: fm.V]

    def generate(self, prompt_ids, max_new=24, eos=3):
        self.reset()
        logits = None
        for tok in prompt_ids:
            logits = self.step(tok)
        out = []
        for _ in range(max_new):
            nxt = int(np.argmax(logits))
            if nxt == eos or self.t >= self.fm.T:
                break
            out.append(nxt)
            logits = self.step(nxt)
        return out

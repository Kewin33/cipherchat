"""CipherChat v3 – fold the trained FHE-friendly Qwen2 model into the matrices the CKKS engine multiplies with.

Per layer (float64):
  Aq = R_t (W_q diag(g1)) / sqrt(m1) / 8   rows permuted to o = m*128 + g*64 + d (head h = 7g + m), padded 1024 x 1024
  Akv = [R_t W_k ; W_v] diag(g1) / sqrt(m1)  (256 rows: k(g, d) then v(g, d))
  W_o columns permuted to the same order; W_gate / W_up (gain g2 / sqrt(m2) folded) in five 1024-row chunks,
  W_down in five 1024-column chunks; activation coefficients per chunk.
R_t is the RoPE rotation of position t (folded per token; the engine encodes weights on the fly anyway) and 1/8 the
attention scale 1/sqrt(64). The public prefix K/V (positions 0..n_pub-1) are computed here with the exact ops.
"""
import numpy as np
import torch

from . import model as v3m

D = 1024


def rope_matrix(t, dim=64, theta=1e6):
    inv = 1.0 / (theta ** (np.arange(0, dim, 2) / dim))
    ang = t * inv
    cos = np.concatenate([np.cos(ang), np.cos(ang)])
    sin = np.concatenate([np.sin(ang), np.sin(ang)])
    h = dim // 2
    Rh = np.zeros((dim, dim))
    Rh[np.arange(h), np.arange(h) + h] = -1.0          # rotate_half: (-x2, x1)
    Rh[np.arange(h) + h, np.arange(h)] = 1.0
    return np.diag(cos) + np.diag(sin) @ Rh


def perm_q(n_heads=14, n_kv=2, hd=64):
    """original row index (h*64 + d) for every permuted row o = m*128 + g*64 + d."""
    per = n_heads // n_kv
    rows = []
    for m in range(per):
        for g in range(n_kv):
            h = g * per + m
            rows += list(range(h * hd, (h + 1) * hd))
    return np.array(rows)


def pad(M, rows=D, cols=D):
    out = np.zeros((rows, cols))
    out[: M.shape[0], : M.shape[1]] = M
    return out


class FoldedModel:
    def __init__(self, student, mods, cfg, prefix_ids):
        mc = cfg["model"]
        self.cfg = cfg
        self.L, self.d, self.H, self.Hkv, self.hd = mc["n_layers"], mc["d_model"], mc["n_heads"], mc["n_kv_heads"], mc["head_dim"]
        self.theta = mc["rope_theta"]
        self.n_pub = len(prefix_ids)
        norms, acts, attns = mods
        st = v3m.fhe_state(student, mods)
        self.p, self.attn_norm = st["p"], st["attn_norm"]
        self.norm_approx = st["norm_approx"]
        self.attn_approx = st["attn_approx"]
        self.inv_depth = (len(self.attn_approx["coef"]) - 1 + 2 * self.attn_approx["iters"]) if self.attn_approx else 0
        self.cache_level = cfg["ckks"]["compute_levels"]
        self.perm = perm_q(self.H, self.Hkv, self.hd)
        self.static = []
        f64 = lambda t: t.detach().double().cpu().numpy()
        for i, layer in enumerate(student.model.layers):
            a = layer.self_attn
            g1 = f64(layer.input_layernorm.weight)
            g2 = f64(layer.post_attention_layernorm.weight)
            m1, m2 = float(norms[2 * i].m_ref), float(norms[2 * i + 1].m_ref)
            s1, s2 = g1 / np.sqrt(m1), g2 / np.sqrt(m2)
            Wg = f64(layer.mlp.gate_proj.weight) * s2
            Wu = f64(layer.mlp.up_proj.weight) * s2
            Wd = f64(layer.mlp.down_proj.weight)
            ff = Wg.shape[0]
            nc = (ff + D - 1) // D
            coef = f64(layer.mlp.act_fn.coef)                                  # (3, ff): c0, c1, c2
            act = []
            for c in range(nc):
                cc = np.zeros((coef.shape[0], D))
                cc[:, : min(D, ff - c * D)] = coef[:, c * D: (c + 1) * D]
                act.append(cc)
            Wo = f64(a.o_proj.weight)[:, self.perm]
            self.static.append(dict(
                m1=m1, m2=m2,
                Wq=f64(a.q_proj.weight) * s1, bq=f64(a.q_proj.bias),
                Wk=f64(a.k_proj.weight) * s1, bk=f64(a.k_proj.bias),
                Wv=f64(a.v_proj.weight) * s1, bv=f64(a.v_proj.bias),
                Wo=pad(Wo),
                Ag=[pad(Wg[c * D: (c + 1) * D]) for c in range(nc)],
                Au=[pad(Wu[c * D: (c + 1) * D]) for c in range(nc)],
                Wd=[pad(Wd[:, c * D: (c + 1) * D]) for c in range(nc)],
                act=act,
                alpha=f64(a.v3.alpha), c=f64(a.v3.c), d_ref=f64(a.v3.d_ref), logk=f64(a.v3.logk)))
        self._prefix_kv(student, prefix_ids)
        self._cache = {}

    @torch.no_grad()
    def _prefix_kv(self, student, prefix_ids):
        x = torch.tensor([prefix_ids], device=student.device)
        out = student(input_ids=x, use_cache=True)
        pkv = out.past_key_values
        self.k_pub, self.v_pub = [], []
        for i in range(self.L):
            if hasattr(pkv, "layers"):
                k, v = pkv.layers[i].keys, pkv.layers[i].values
            elif hasattr(pkv, "key_cache"):
                k, v = pkv.key_cache[i], pkv.value_cache[i]
            else:
                k, v = pkv[i]
            k = k[0].double().cpu().numpy()      # (Hkv, n_pub, hd)
            v = v[0].double().cpu().numpy()
            self.k_pub.append([k[:, j, :].reshape(-1) for j in range(self.n_pub)])
            self.v_pub.append([v[:, j, :].reshape(-1) for j in range(self.n_pub)])

    META = ("cfg", "L", "d", "H", "Hkv", "hd", "theta", "n_pub", "p", "attn_norm", "norm_approx", "attn_approx",
            "inv_depth", "cache_level")

    def save(self, path):
        """Export for C++/other runtimes: one .npz (float32 weights, float64 small vectors) + the json meta inside.
        Keys: L{i}.{name}[.{chunk}], k_pub / v_pub (L, n_pub, 128), perm."""
        import json
        arr = {"perm": self.perm, "k_pub": np.array(self.k_pub), "v_pub": np.array(self.v_pub),
               "meta": np.frombuffer(json.dumps({k: getattr(self, k) for k in self.META}).encode(), dtype=np.uint8)}
        for i, s in enumerate(self.static):
            for k, v in s.items():
                if isinstance(v, list):
                    for c, m in enumerate(v):
                        arr[f"L{i}.{k}.{c}"] = m.astype(np.float32) if m.ndim == 2 and m.shape[0] == D else m
                else:
                    v = np.asarray(v)
                    arr[f"L{i}.{k}"] = v.astype(np.float32) if v.ndim == 2 else v
        np.savez(path, **arr)

    @classmethod
    def load(cls, path):
        import json
        z = np.load(path)
        self = cls.__new__(cls)
        for k, v in json.loads(bytes(z["meta"]).decode()).items():
            setattr(self, k, v)
        self.perm, self.k_pub, self.v_pub = z["perm"], list(z["k_pub"]), list(z["v_pub"])
        self.static = []
        for i in range(self.L):
            s = {}
            for key in z.files:
                if not key.startswith(f"L{i}."):
                    continue
                parts = key.split(".")
                val = z[key].astype(np.float64) if z[key].dtype == np.float32 else z[key]
                if len(parts) == 3:
                    s.setdefault(parts[1], {})[int(parts[2])] = val
                else:
                    s[parts[1]] = float(val) if val.ndim == 0 else val
            for k, v in list(s.items()):
                if isinstance(v, dict):
                    s[k] = [v[c] for c in sorted(v)]
            self.static.append(s)
        self._cache = {}
        return self

    def layer(self, i, t):
        """Weights of layer i for the token at position t (RoPE folded)."""
        key = (i, t)
        if key in self._cache:
            return self._cache[key]
        s = self.static[i]
        Rt = rope_matrix(t, self.hd, self.theta)
        Rq = np.kron(np.eye(self.H), Rt)
        Rk = np.kron(np.eye(self.Hkv), Rt)
        Aq = (Rq @ s["Wq"]) / np.sqrt(self.hd)
        bq = (Rq @ s["bq"]) / np.sqrt(self.hd)
        Akv = np.concatenate([Rk @ s["Wk"], s["Wv"]], 0)
        bkv = np.concatenate([Rk @ s["bk"], s["bv"]])
        lw = dict(s)
        lw.update(Aq=pad(Aq[self.perm]), bq=bq[self.perm], Akv=pad(Akv), bkv=bkv)
        self._cache = {key: lw}            # one position at a time
        return lw


def wikitext_batches(tok, prefix, n=2, seq=128):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ids = tok("\n\n".join(ds["text"][:2000]))["input_ids"]
    L = seq - len(prefix)
    return [torch.tensor([prefix + ids[(b * 8 + r) * L: (b * 8 + r + 1) * L] for r in range(8)]) for b in range(n)]


def build(ckpt=None, cfg=None, device="cuda", dtype=torch.float64, calib_batches=None, log=print):
    """Load the student (trained checkpoint, or freshly calibrated when ckpt is None) and fold it."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = cfg or v3m.load_config()
    base = cfg["model"]["base"]
    tok = AutoTokenizer.from_pretrained(base)
    prefix = [tok.convert_tokens_to_ids(t) if t.startswith("<|") else tok(t)["input_ids"][0]
              for t in cfg["model"]["public_prefix"]]
    v3m.STATE["n_pub"] = len(prefix)
    student = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device).eval()
    if ckpt:
        sd = torch.load(ckpt, map_location=device)
        cfg = sd.get("cfg", cfg)
        mods = v3m.convert(student, cfg)
        student.load_state_dict({k: v.to(dtype) for k, v in sd["model"].items()}, strict=False)
        v3m.load_fhe_state(student, mods, sd["fhe"])
        if cfg["approx"]["softmax"].get("norm", "div") == "div" and mods[2][0].approx is None:
            # checkpoint from the exact-division phase: fit the Newton inverse now so the model is FHE-computable
            log("checkpoint has no attention inverse yet -> calibrating it on WikiText-2 train")
            v3m.STATE["mix"] = 1.0
            v3m.calibrate_inverse(student, mods, calib_batches or wikitext_batches(tok, prefix), cfg, log=log)
    else:
        mods = v3m.convert(student, cfg)
        v3m.calibrate(student, mods, calib_batches, cfg, log=log)
        if cfg["approx"]["softmax"].get("norm", "div") == "div":
            v3m.calibrate_inverse(student, mods, calib_batches, cfg, log=log, mix=0.0)
    student = student.to(dtype)
    v3m.STATE["mix"], v3m.STATE["clamp"] = 1.0, False
    return FoldedModel(student, mods, cfg, prefix), student, tok, prefix


if __name__ == "__main__":
    # export: python -m cipherchat.v3.fold --ckpt weights/v3/q05_pos.pt --out weights/v3/q05_pos_folded.npz
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fm, *_ = build(args.ckpt, device="cpu")
    fm.save(args.out)
    fm2 = FoldedModel.load(args.out)
    a, b = fm.layer(3, 7), fm2.layer(3, 7)
    err = max(float(np.abs(np.asarray(a[k]) - np.asarray(b[k])).max()) for k in ("Aq", "Akv", "Wo", "bq"))
    print(f"exported {args.out}; reload max abs diff (float32 storage) {err:.2e}")

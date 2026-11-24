"""CipherChat v3 – correctness of the encrypted decode step.

Reference: the FHE-adapted student in float64 (PyTorch). Engine: cipherchat/v3/engine.py on
  --backend numpy    exact slot simulation (expected error ~1e-10, validates layouts / level schedule)
  --backend phantom  CKKS on the GPU (expected cosine > 0.99, MSE < 1e-3 per the spec)
For every generated position the final residual vector (before the client's final norm) is compared, plus the
top-1 token of the logits the client computes from it. With --per_layer the residual is decrypted after every
layer (debug only; a real server never can).

  python scripts/v3_check.py --ckpt weights/v3/q05_div.pt --backend numpy --tokens 4
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cipherchat.v3 import engine as E  # noqa: E402
from cipherchat.v3 import fold  # noqa: E402
from cipherchat.v3 import model as v3m  # noqa: E402


def calib_from_wikitext(tok, prefix, n=2, seq=128):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ids = tok("\n\n".join(ds["text"][:2000]))["input_ids"]
    L = seq - len(prefix)
    out = []
    for b in range(n):
        rows = [prefix + ids[(b * 8 + r) * L: (b * 8 + r + 1) * L] for r in range(8)]
        out.append(torch.tensor(rows))
    return out


def stats(a, b):
    err = a - b
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))
    return dict(max_abs=float(np.abs(err).max()), rel=float(np.linalg.norm(err) / np.linalg.norm(b)),
                mse=float(np.mean(err ** 2)), cos=cos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--backend", default="numpy", choices=["numpy", "phantom"])
    ap.add_argument("--tokens", type=int, default=3)
    ap.add_argument("--text", default="The capital of France is Paris. The capital of Germany is")
    ap.add_argument("--per_layer", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--norm", default="", choices=["", "div", "pos"])
    ap.add_argument("--layers", type=int, default=0, help="only the first N layers (debug)")
    ap.add_argument("--bounds", default="", help="json with bootstrapping bounds per site (written by a numpy run)")
    ap.add_argument("--save_bounds", default="", help="numpy backend: write bounds = next pow2 >= 1.5 x max |value|")
    args = ap.parse_args()
    cfg = v3m.load_config()
    if args.norm:
        cfg["approx"]["softmax"]["norm"] = args.norm
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]["base"])
    prefix = fold.FoldedModel and [tok.convert_tokens_to_ids(t) if t.startswith("<|") else tok(t)["input_ids"][0]
                                   for t in cfg["model"]["public_prefix"]]
    calib = None if args.ckpt else calib_from_wikitext(tok, prefix)
    t0 = time.time()
    fm, student, tok, prefix = fold.build(args.ckpt or None, cfg, calib_batches=calib)
    print(f"folded in {time.time() - t0:.1f} s (n_pub {fm.n_pub}, attn norm {fm.attn_norm}, p {fm.p}, "
          f"inv depth {fm.inv_depth})", flush=True)
    if args.layers:
        fm.L = args.layers
    ids = prefix + tok(args.text)["input_ids"]
    ids = ids[: fm.n_pub + args.tokens]
    # reference: outputs of every layer (residual), float64
    outs = {}
    hooks = [layer.register_forward_hook(lambda m, i, o, k=k: outs.__setitem__(k, (o[0] if isinstance(o, tuple) else o)[0].double().cpu().numpy()))
             for k, layer in enumerate(student.model.layers)]
    with torch.no_grad():
        ref = student(input_ids=torch.tensor([ids], device=student.device), output_hidden_states=True)
    for h in hooks:
        h.remove()
    ref_logits = ref.logits[0].double().cpu().numpy()
    emb = student.model.embed_tokens.weight.detach().double().cpu().numpy()
    W_lm = student.lm_head.weight.detach().double().cpu().numpy()
    g_fin = student.model.norm.weight.detach().double().cpu().numpy()
    eps = student.model.norm.variance_epsilon
    student.to("cpu")                      # free the GPU for the CKKS keys (~7 GB)
    torch.cuda.empty_cache()

    res = {"backend": args.backend, "tokens": [], "ckpt": args.ckpt, "layers": fm.L}
    if args.backend == "numpy":
        be = E.NumpyBackend(levels=cfg["ckks"]["compute_levels"])
    else:
        from cipherchat.v3.service import hotspot_table
        t_k = time.time()
        be = E.PhantomBackend(cfg, extra_steps=E.engine_rotation_steps())
        res["keygen_s"] = time.time() - t_k
        res["mem_after_keys"] = dict(be.ctx.mem_info())
        res["n_rotation_keys"] = be.n_keys
        print(f"phantom: {be.n_keys} rotation keys, VRAM {be.ctx.mem_info()}", flush=True)
    if args.bounds:
        fm.boot_bounds = json.load(open(args.bounds))
    eng = E.Engine3(fm, be)
    print("bootstrapping bounds:", eng.bounds, flush=True)
    sess = eng.new_session()
    eng.prefill_public(sess)
    if args.backend == "phantom":
        res["mem_after_prefill"] = dict(be.ctx.mem_info())
    for t in range(fm.n_pub, len(ids)):
        x = eng.tile(emb[ids[t]])
        ct = be.encrypt(x, be.L)
        if args.backend == "phantom":
            be.ctx.reset_profile()
        b0 = eng.stats["bootstraps"]
        t1 = time.time()
        if args.per_layer:
            for i in range(fm.L):
                ct = eng.layer(i, sess, ct)
                y = be.decrypt(ct)[:fm.d]
                st = stats(y, outs[i][t])
                print(f"  t={t} layer {i:2d}: level {be.level(ct)} max_abs {st['max_abs']:.2e} rel {st['rel']:.2e} "
                      f"cos {st['cos']:.6f}", flush=True)
            sess.t += 1
            out = ct
        else:
            out = eng.step(sess, ct)
        dt = time.time() - t1
        y = be.decrypt(out)[:fm.d]
        st = stats(y, outs[fm.L - 1][t])
        hn = y / np.sqrt(np.mean(y ** 2) + eps) * g_fin
        logits = W_lm @ hn
        top_e, top_r = int(np.argmax(logits)), int(np.argmax(ref_logits[t]))
        st.update(t=t, seconds=dt, top1_engine=tok.decode([top_e]), top1_ref=tok.decode([top_r]),
                  logit_max_abs=float(np.abs(logits - ref_logits[t]).max()))
        st["bootstraps"] = eng.stats["bootstraps"] - b0
        if args.backend == "phantom":
            prof = {k: tuple(v) for k, v in be.ctx.profile().items()}
            st["profile"] = {k: list(v) for k, v in prof.items()}
            st["hotspots"] = hotspot_table(prof)
            st["boot_phases"] = json.loads(be.ctx.boot_phases())
            st["gpu_ms"] = sum(v[0] for v in prof.values())
            hs = st["hotspots"]
            print("   hotspots (ms): " + ", ".join(f"{k} {v:.0f}" for k, v in hs.items()) +
                  f" | boot phases {st['boot_phases']}", flush=True)
        res["tokens"].append(st)
        print(f"t={t} ({tok.decode([ids[t]])!r}): {dt:.1f} s, final residual max_abs {st['max_abs']:.2e} rel "
              f"{st['rel']:.2e} cos {st['cos']:.6f} mse {st['mse']:.2e} | top1 {st['top1_engine']!r} vs "
              f"{st['top1_ref']!r}", flush=True)
    if args.backend == "numpy":
        tot = {}
        for k, v in be.counters.items():
            op = k.split("|")[1]
            tot[op] = tot.get(op, 0) + v
        res["counters_total"] = tot
        res["counters"] = be.counters
        res["boot_max_abs"] = be.boot_max_abs
        res["boot_obs"] = be.obs
        print("ops (all tokens):", tot, "| max |bootstrap input| after scaling", round(be.boot_max_abs, 3), flush=True)
        sites = {}
        for key, o in be.obs.items():
            site = key.split("@")[0]
            g = sites.setdefault(site, {"n": 0, "max_true": 0.0, "max_scaled": 0.0})
            g["n"] += o["n"]
            g["max_true"] = max(g["max_true"], o["max_true"])
            g["max_scaled"] = max(g["max_scaled"], o["max_scaled"])
        for site, o in sites.items():
            print(f"  bootstrap site {site:4s}: n={o['n']:3d} max |true| {o['max_true']:.3g} max |scaled| {o['max_scaled']:.3g}",
                  flush=True)
        if args.save_bounds:
            # per layer and site: smallest power of two >= 1.5 x the largest |value| seen (bootstrapping is
            # accurate for |x| <= 1, the absolute error scales with the bound, so tight per-layer bounds matter)
            bounds = {}
            glob = {}
            for key, o in be.obs.items():
                site = key.split("@")[0]
                if site in ("y1", "y2"):
                    continue
                bounds[key] = float(2.0 ** np.ceil(np.log2(max(1.0, 1.5 * o["max_true"]))))
                glob[site] = max(glob.get(site, 0.0), bounds[key])
            bounds.update(glob)
            json.dump(bounds, open(args.save_bounds, "w"), indent=1, sort_keys=True)
            print("saved bounds (global max per site)", glob, flush=True)
    else:
        res["profile"] = {k: list(v) for k, v in be.ctx.profile().items()}
        res["boot_phases"] = json.loads(be.ctx.boot_phases())
        res["n_boot"] = be.ctx.n_boot()
        res["mem"] = dict(be.ctx.mem_info())
    res["bootstraps"] = eng.stats["bootstraps"]
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1)
    os._exit(0)


if __name__ == "__main__":
    main()

"""CipherChat v5 – correctness, speed and memory of the INSECURE (e = 0) decode step without bootstrapping.

Reference: the FHE-adapted student (v3 checkpoint) in float64. Engine: cipherchat/v5/engine.py on
  --backend numpy    exact slot simulation with level accounting (layout, rotations, depth of the whole model)
  --backend phantom  CKKS with e = 0 on the GPU (pybind module v5fhe, build_v5/)
Per generated position: final residual vs. the reference (max abs, rel, MSE, cosine), top-1 token of the client's
logits, time, GPU hotspots, VRAM. --per_layer decrypts after every layer (debug only).

  python scripts/v5_check.py --ckpt weights/v3/q05_pos.pt --backend numpy --tokens 1 --levels 2000   # depth
  PYTHONPATH=build_v5 python scripts/v5_check.py --ckpt weights/v3/q05_pos.pt --backend phantom --tokens 4 \
      --out results/v5/check_phantom.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cipherchat.v3 import fold  # noqa: E402
from cipherchat.v3 import model as v3m  # noqa: E402
from cipherchat.v5 import engine as E5  # noqa: E402

V5_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cipherchat", "v5", "config.json")


def stats(a, b):
    err = a - b
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))
    return dict(max_abs=float(np.abs(err).max()), rel=float(np.linalg.norm(err) / np.linalg.norm(b)),
                mse=float(np.mean(err ** 2)), cos=cos)


def hotspot_table(prof):
    """GPU ms per hotspot from the (section|op) profile of v5fhe (same grouping as the v3 report)."""
    out = {"1 linear: encode + ptmul": 0.0, "1 linear: rotations": 0.0, "2 attention: ctmul/cache/pack": 0.0,
           "2 attention: rotations": 0.0, "3 nonlinear": 0.0, "4 bootstrapping": 0.0, "other": 0.0}
    for key, (ms, _cnt) in prof.items():
        sec, op = key.split("|")
        if sec in ("qkv", "attn.out", "ffn.up", "ffn.down", "residual"):
            out["1 linear: rotations" if op == "rot" else "1 linear: encode + ptmul"] += ms
        elif sec.startswith("attn"):
            out["2 attention: rotations" if op == "rot" else "2 attention: ctmul/cache/pack"] += ms
        elif sec in ("norm1", "norm2", "ffn.act", "attn.softmax"):
            out["3 nonlinear"] += ms
        else:
            out["other"] += ms
    return out


def op_table(prof):
    out = {}
    for key, (ms, cnt) in prof.items():
        op = key.split("|")[1]
        o = out.setdefault(op, [0.0, 0])
        o[0] += ms
        o[1] += cnt
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backend", default="numpy", choices=["numpy", "numpy_dev", "phantom"],
                    help="numpy_dev: exact simulation + only the prime-vs-scale deviation of the GPU bookkeeping")
    ap.add_argument("--tokens", type=int, default=3)
    ap.add_argument("--text", default="The capital of France is Paris. The capital of Germany is")
    ap.add_argument("--per_layer", action="store_true")
    ap.add_argument("--layers", type=int, default=0, help="only the first N layers")
    ap.add_argument("--levels", type=int, default=0, help="override ckks.levels (chain length)")
    ap.add_argument("--logN", type=int, default=0, help="override ckks.logN (layout n = 2^(logN-1))")
    ap.add_argument("--scale_bits", type=int, default=0)
    ap.add_argument("--lpp", type=int, default=0, choices=[0, 1, 2], help="levels per prime (2: Delta = 2^30, 60-bit primes)")
    ap.add_argument("--prime_bits", type=int, default=0, help="lpp = 2: bit size of the scale primes")
    ap.add_argument("--special_bits", type=int, default=0, help="bits of the special prime (lpp = 2 with 60-bit primes: 50)")
    ap.add_argument("--diag", default="", choices=["", "f32", "f64"], help="storage of cached weight diagonals")
    ap.add_argument("--unfused", action="store_true", help="plain multiply_plain + add instead of the fused kernel")
    ap.add_argument("--keys", default="", choices=["", "small", "all"])
    ap.add_argument("--no_profile", action="store_true", help="no per-op device syncs (pure wall time)")
    ap.add_argument("--device", default="cuda", help="device of the float64 reference model (cpu for numpy runs)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = v3m.load_config()
    cfg5 = json.load(open(V5_CFG))
    cfg["ckks"] = dict(cfg5["ckks"])
    cfg["layout"] = dict(cfg5["layout"])
    for k, v in (("levels", args.levels), ("logN", args.logN), ("scale_bits", args.scale_bits),
                 ("levels_per_prime", args.lpp), ("prime_bits", args.prime_bits),
                 ("special_bits", args.special_bits)):
        if v:
            cfg["ckks"][k] = v
    if args.keys:
        cfg["layout"]["rotation_keys"] = args.keys
    lay = E5.Layout(2 ** (cfg["ckks"]["logN"] - 1), baby=cfg["layout"].get("baby"),
                    baby_conv=cfg["layout"].get("baby_conv"))
    print("layout", lay.as_dict(), "| ckks", {k: v for k, v in cfg["ckks"].items() if k != "security_note"},
          flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]["base"])
    t0 = time.time()
    fm, student, tok, prefix = fold.build(args.ckpt, cfg, device=args.device)
    print(f"folded in {time.time() - t0:.1f} s (n_pub {fm.n_pub}, attn norm {fm.attn_norm}, p {fm.p})", flush=True)
    if args.layers:
        fm.L = args.layers
    ids = prefix + tok(args.text)["input_ids"]
    ids = ids[: fm.n_pub + args.tokens]
    assert len(ids) <= lay.T_MAX, f"{len(ids)} positions > layout capacity {lay.T_MAX}"
    outs = {}
    hooks = [layer.register_forward_hook(
        lambda m, i, o, k=k: outs.__setitem__(k, (o[0] if isinstance(o, tuple) else o)[0].double().cpu().numpy()))
        for k, layer in enumerate(student.model.layers)]
    with torch.no_grad():
        ref = student(input_ids=torch.tensor([ids], device=student.device))
    for h in hooks:
        h.remove()
    ref_logits = ref.logits[0].double().cpu().numpy()
    emb = student.model.embed_tokens.weight.detach().double().cpu().numpy()
    W_lm = student.lm_head.weight.detach().double().cpu().numpy()
    g_fin = student.model.norm.weight.detach().double().cpu().numpy()
    eps = student.model.norm.variance_epsilon
    student.to("cpu")
    torch.cuda.empty_cache()

    keys = E5.rotation_keys(lay, cfg["layout"].get("rotation_keys", "small"))
    res = {"backend": args.backend, "insecure_zero_error": True, "tokens": [], "ckpt": args.ckpt, "layers": fm.L,
           "layout": lay.as_dict(), "ckks": cfg["ckks"], "n_rotation_keys": len(keys)}
    L = cfg["ckks"]["levels"]
    lpp = cfg["ckks"].get("levels_per_prime", 1)
    if args.backend == "numpy":
        be = (E5.NumpyBackendLPP if lpp == 2 else E5.NumpyBackend)(n=lay.n, levels=L)
    elif args.backend == "numpy_dev":
        assert lpp == 1, "numpy_dev models one level per prime"
        be = E5.NumpyBackendScaleDev(lay.n, L, cfg["ckks"]["logN"], cfg["ckks"]["scale_bits"])
    else:
        t_k = time.time()
        be = E5.PhantomBackend5(cfg, keys, profile=not args.no_profile)
        be.ctx.set_fused(not args.unfused)
        res["fused"] = not args.unfused
        L = be.L                                   # virtual level of a fresh ciphertext
        res["setup_s"] = time.time() - t_k
        res["setup_timing_ms"] = be.info["timing"]
        res["phantom_info"] = {k: v for k, v in be.info.items() if k != "timing"}
        res["mem_after_keys"] = dict(be.ctx.mem_info())
        print(f"phantom v5: {be.info}, {be.n_keys} rotation keys, setup {res['setup_s']:.1f} s, "
              f"VRAM {res['mem_after_keys']}", flush=True)
        assert be.info["zero_error"], "v5fhe was not built with PHANTOM_ZERO_ERROR"
    eng = E5.Engine5(fm, be, lay, keys, diag_dtype={"f32": np.float32, "f64": np.float64}.get(args.diag))
    res["diag_dtype"] = np.dtype(eng.diag_dtype).name
    sess = eng.new_session()
    for t in range(fm.n_pub, len(ids)):
        x = eng.tile(emb[ids[t]])
        ct = be.encrypt(x, L)
        if args.backend == "phantom":
            be.ctx.reset_profile()
        r0 = eng.stats["rotations_keyed"]
        t1 = time.time()
        if args.per_layer:
            for i in range(fm.L):
                ct = eng.layer(i, sess, ct)
                y = be.decrypt(ct)[:fm.d]
                st = stats(y, outs[i][t])
                print(f"  t={t} layer {i:2d}: level {be.level(ct)} max_abs {st['max_abs']:.2e} rel {st['rel']:.2e} "
                      f"cos {st['cos']:.8f}", flush=True)
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
                  logit_max_abs=float(np.abs(logits - ref_logits[t]).max()), level_left=be.level(out),
                  rotations_keyed=eng.stats["rotations_keyed"] - r0)
        if fm.L == len(outs):
            st["logits_ref_top2_margin"] = float(np.diff(np.sort(ref_logits[t])[-2:])[0])
        if args.backend == "phantom":
            prof = {k: tuple(v) for k, v in be.ctx.profile().items()}
            st["profile"] = {k: list(v) for k, v in prof.items()}
            st["hotspots"] = hotspot_table(prof)
            st["ops"] = op_table(prof)
            st["gpu_ms"] = sum(v[0] for v in prof.values())
            st["mem"] = dict(be.ctx.mem_info())
            print("   hotspots (ms): " + ", ".join(f"{k} {v:.0f}" for k, v in st["hotspots"].items()) +
                  " | ops " + ", ".join(f"{k} {v[0]:.0f} ms/{v[1]}" for k, v in st["ops"].items()) +
                  f" | VRAM {st['mem']}", flush=True)
        res["tokens"].append(st)
        if args.out:                                   # write after every token (long runs may die, e.g. OOM)
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            json.dump(res, open(args.out, "w"), indent=1)
        print(f"t={t} ({tok.decode([ids[t]])!r}): {dt:.1f} s, level left {st['level_left']}, final residual max_abs "
              f"{st['max_abs']:.2e} rel {st['rel']:.2e} cos {st['cos']:.8f} mse {st['mse']:.2e} | top1 "
              f"{st['top1_engine']!r} vs {st['top1_ref']!r} | logit max err {st['logit_max_abs']:.2e}", flush=True)
    res["levels_seen"] = eng.levels_seen
    res["depth_used"] = L - res["tokens"][0]["level_left"]
    if args.backend != "phantom":
        tot = {}
        for k, v in be.counters.items():
            op = k.split("|")[1]
            tot[op] = tot.get(op, 0) + v
        res["counters_total"] = tot
        res["counters"] = be.counters
        print("ops (all tokens):", tot, flush=True)
    print(f"depth used by {fm.L} layers: {res['depth_used']} levels (chain {L})", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1)
    os._exit(0)


if __name__ == "__main__":
    main()

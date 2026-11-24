"""Which FHE approximation leaves its calibrated range, and how often? (checkpoint, WikiText-2 test chunks)
Runs with the range guard ON (exact op outside the range) so the statistics are not destroyed by the cascade, and
reports per approximation type the fraction of out-of-range evaluations and the worst relative excess.
Then evaluates the perplexity without guard (= what the server computes) per chunk and counts broken chunks."""
import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cipherchat.v3 import model as v3m  # noqa: E402
from train_v3 import prefix_ids  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--chunks", type=int, default=64)
    args = ap.parse_args()
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sd = torch.load(args.ckpt, map_location="cuda")
    cfg = sd["cfg"]
    tok = AutoTokenizer.from_pretrained(cfg["model"]["base"])
    prefix = prefix_ids(tok, cfg)
    v3m.STATE["n_pub"] = P = len(prefix)
    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["base"], dtype=torch.float32).cuda().eval()
    mods = v3m.convert(model, cfg)
    model.load_state_dict(sd["model"], strict=False)
    v3m.load_fhe_state(model, mods, sd["fhe"])
    norms, acts, attns = mods
    stats = {"norm": [0, 0, 0.0], "act": [0, 0, 0.0], "attn": [0, 0, 0.0]}

    def rec(kind, x, lo, hi):
        x = x.detach().float()
        out = (x < lo) | (x > hi)
        stats[kind][0] += int(out.sum())
        stats[kind][1] += x.numel()
        exc = torch.maximum(lo / x.clamp(min=1e-30), x / hi).max().item()
        stats[kind][2] = max(stats[kind][2], exc)

    hooks = []
    for nm in norms:
        def h(mod, inp):
            x = inp[0].to(torch.float32)
            z = x.pow(2).mean(-1)[:, P:] / mod.m_ref
            rec("norm", z, mod.approx.lo, mod.approx.hi)
        hooks.append(nm.register_forward_pre_hook(h))
    for a in acts:
        def h2(mod, inp):
            g = inp[0][:, P:].detach().float().abs()
            r = g / mod.R
            rec("act", r, 0.0 + 1e-30, 1.0)
        hooks.append(a.register_forward_pre_hook(h2))
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]))["input_ids"]
    L = 128 - P
    chunks = [ids[i: i + L] for i in range(0, len(ids) - L, L)][: args.chunks]
    v3m.STATE["mix"] = 1.0
    v3m.STATE["diag"] = []
    res = {}
    for guard in (True, False):
        v3m.STATE["clamp"] = guard
        nll, n, broken = 0.0, 0, 0
        for c in chunks:
            x = torch.tensor([prefix + c]).cuda()
            with torch.no_grad():
                lg = model(input_ids=x).logits.float()
            if guard and v3m.STATE["diag"] is not None and len(v3m.STATE["diag"]) > 2000:
                for dn, lo, hi in v3m.STATE["diag"]:
                    rec("attn", dn, lo, hi)
                v3m.STATE["diag"] = []
            lp = F.log_softmax(lg[:, P - 1:-1], -1).gather(-1, x[:, P:, None])
            if not torch.isfinite(lp).all():
                broken += 1
                continue
            nll -= lp.sum().item()
            n += lp.numel()
        res["guard" if guard else "fhe"] = dict(ppl=math.exp(nll / max(1, n)), broken_chunks=broken, chunks=len(chunks))
        if guard:
            for h in hooks:
                h.remove()
            for dn, lo, hi in v3m.STATE["diag"]:
                rec("attn", dn, lo, hi)
            v3m.STATE["diag"] = None
            res["out_of_range"] = {k: dict(frac=v[0] / max(1, v[1]), worst_excess=v[2]) for k, v in stats.items()}
    print(json.dumps(res, indent=1), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()

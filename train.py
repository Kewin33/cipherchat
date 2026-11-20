"""Train the FHE-friendly CipherFormer on the synthetic German mini-chat dataset.

CPU:   python train.py --steps 6000
GPU:   python train.py --device cuda --d 64 --d_ff 128 --steps 20000 --batch 512
Colab: see notebooks/train_colab.ipynb

The exported weights (weights/<name>.npz + .json) are what the FHE server loads.
NOTE: the FHE engine packs everything into blocks of B=64 slots, so d must be 64,
vocab <= 64 and d_ff a multiple of 64 (64, 128, 192, ...).
"""
import argparse
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch

from cipherchat.dataset import MAX_LEN, build_pairs, eval_prompts, sample_batch
from cipherchat.model import CipherFormer, masked_ce
from cipherchat.vocab import VOCAB_SIZE, decode, encode_prompt, encode_text


def evaluate(model, items, limit=None):
    model.eval()
    per = defaultdict(lambda: [0, 0])
    examples = []
    prompts = eval_prompts(items)
    if limit:
        prompts = prompts[:limit]
    for cat, p, a in prompts:
        out = decode(model.generate(encode_prompt(p)))
        ok = out == a
        per[cat][0] += ok
        per[cat][1] += 1
        if len(examples) < 12 and random.random() < 0.05:
            examples.append((p, out, a))
    model.train()
    tot = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
    return {k: v[0] / v[1] for k, v in per.items()}, tot[0] / max(tot[1], 1), examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--d_ff", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--act_limit", type=float, default=200.0, help="soft bound on activations (CKKS range)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="cipherformer")
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--holdout", type=float, default=0.15,
                    help="fraction of arithmetic/comparison pairs held out (0 = train on everything, for the demo model)")
    args = ap.parse_args()

    assert args.d == 64 and args.d_ff % 64 == 0, "FHE engine uses blocks of 64 slots"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)
    train_items, test_items = build_pairs(seed=0, holdout_frac=args.holdout)
    print(f"train items: {len(train_items)}  held-out items: {len(test_items)}  vocab: {VOCAB_SIZE}")

    model = CipherFormer(VOCAB_SIZE, args.d, args.d_ff, args.layers, MAX_LEN).to(args.device)
    print("params:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 300) * (0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)) * 0.95 + 0.05))

    t0 = time.time()
    ema = None
    for step in range(1, args.steps + 1):
        ids, msk = sample_batch(train_items, args.batch, rng)
        ids = torch.tensor(ids, device=args.device)
        msk = torch.tensor(msk, device=args.device)
        logits = model(ids)
        loss = masked_ce(logits, ids, msk)
        stats = model.last_stats
        # small logit penalty keeps the final CKKS values comfortably inside the modulus
        total = loss + (1e-3 * logits.pow(2).mean())
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % 200 == 0:
            s_max = max(float(st[0]) for st in stats[:-1])
            w_max = max(float(st[1]) for st in stats[:-1])
            m_max = max(float(st[2]) for st in stats[:-1])
            print(f"step {step:6d} loss {ema:.4f} | max|s| {s_max:.1f} max|w| {w_max:.1f} max|m| {m_max:.1f} "
                  f"max|x| {float(stats[-1]):.1f} | {time.time()-t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            acc_tr, tot_tr, ex = evaluate(model, train_items)
            acc_te, tot_te, _ = evaluate(model, test_items) if test_items else ({}, float("nan"), [])
            print(f"  [eval] train exact-match {tot_tr:.3f}  held-out {tot_te:.3f}")
            print("   per category (train):", {k: round(v, 2) for k, v in sorted(acc_tr.items())})
            print("   per category (held-out):", {k: round(v, 2) for k, v in sorted(acc_te.items())})
            for p, o, a in ex[:6]:
                print(f"     {p!r} -> {o!r} (want {a!r})")

    os.makedirs("weights", exist_ok=True)
    W = model.export()
    np.savez(f"weights/{args.name}.npz", **W)
    acc_tr, tot_tr, _ = evaluate(model, train_items)
    acc_te, tot_te, _ = evaluate(model, test_items) if test_items else ({}, None, [])
    meta = dict(model.cfg, train_exact_match=tot_tr, heldout_exact_match=tot_te,
                per_category_train=acc_tr, per_category_heldout=acc_te, steps=args.steps)
    with open(f"weights/{args.name}.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    torch.save(model.state_dict(), f"weights/{args.name}.pt")
    print("saved", f"weights/{args.name}.npz", meta)


if __name__ == "__main__":
    main()

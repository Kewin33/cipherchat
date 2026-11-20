"""Train CipherFormer2 (v2) on the German mini-chat + 2-digit addition with held-out operand pairs.

  python train_v2.py --device cuda --hidden 1024 --steps 30000 --name cf2_add
  python train_v2.py --rev ...      # little-endian digit format (client reverses digit runs)

Writes weights/<name>.npz (folded for the v2 engine), .json (config + metrics) and .pt.
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

from cipherchat.v2.dataset import ADD_SUFFIXES, ADD_TEMPLATES, N_SIG, Task
from cipherchat.v2.model import CipherFormer2, masked_ce
from cipherchat.vocab import VOCAB_SIZE, decode


@torch.no_grad()
def evaluate(model, task, items, bs=2048, max_new=24):
    was_training = model.training
    model.eval()
    model.track_stats = False
    prompts = [task.prompt_ids(p) for _, p, _ in items]
    outs = []
    sig_fn = task.sig if task.coupling else None
    for i in range(0, len(prompts), bs):
        outs += model.generate_batch(prompts[i:i + bs], max_new=max_new, sig_fn=sig_fn)
    model.train(was_training)
    per = defaultdict(lambda: [0, 0])
    wrong = []
    for (cat, p, a), o in zip(items, outs):
        ans = task.decode_answer(decode(o))
        ok = ans == a
        per[cat][0] += ok
        per[cat][1] += 1
        if not ok and len(wrong) < 12:
            wrong.append((p, ans, a))
    acc = {k: v[0] / v[1] for k, v in per.items()}
    tot = sum(v[0] for v in per.values()) / max(1, sum(v[1] for v in per.values()))
    return acc, tot, wrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--kappa", type=float, default=1 / 64)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--logit_pen", type=float, default=1e-4)
    ap.add_argument("--max_n", type=int, default=99)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--p_add", type=float, default=0.5)
    ap.add_argument("--rev", action="store_true", help="little-endian digits (tokenization option)")
    ap.add_argument("--coupling", action="store_true", help="client-side digit significance (position coupling)")
    ap.add_argument("--mlp", default="sq", choices=["sq", "bil", "sqglu"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--pool", type=int, default=400000, help="pre-generated training sequences")
    ap.add_argument("--name", default="cf2")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)
    task = Task(max_n=args.max_n, test_frac=args.test_frac, seed=0, rev=args.rev, p_add=args.p_add, coupling=args.coupling)
    print(f"train pairs {len(task.train_pairs)} | held-out pairs {len(task.test_pairs)} | other items {len(task.others)} "
          f"| rev={args.rev} | device {args.device}", flush=True)
    model = CipherFormer2(VOCAB_SIZE, n_layers=args.layers, hidden=args.hidden, kappa=args.kappa,
                          n_sig=N_SIG if args.coupling else 0, mlp=args.mlp).to(args.device)
    print("params:", sum(p.numel() for p in model.parameters()), flush=True)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * (0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)) * 0.97 + 0.03))

    held = task.eval_addition(task.test_pairs)
    held_tpl = []
    r2 = random.Random(1)
    for a, b in task.test_pairs:
        tpl = r2.choice(ADD_TEMPLATES) + r2.choice(ADD_SUFFIXES)
        held_tpl.append(("add_tpl", tpl.format(a=a, b=b), str(a + b)))
    train_probe = task.eval_addition(r2.sample(task.train_pairs, 1000))
    others = task.eval_others()

    # pre-generate a large pool of training sequences once (python sampling was the bottleneck per step)
    tp = time.time()
    pool_ids, pool_msk, pool_sig = [], [], []
    while len(pool_ids) < args.pool:
        a, b, c = task.batch(4096, rng)
        T = len(a[0])
        pad = lambda rows: [r + [0] * (64 - T) for r in rows]
        pool_ids += pad(a); pool_msk += pad(b); pool_sig += pad(c)
    pool_ids = torch.tensor(pool_ids, device=args.device)
    pool_msk = torch.tensor(pool_msk, device=args.device)
    pool_sig = torch.tensor(pool_sig, device=args.device)
    pool_len = (pool_ids != 0).sum(1)
    print(f"pool of {len(pool_ids)} sequences built in {time.time() - tp:.0f}s", flush=True)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    t0 = time.time()
    ema = None
    hist = []
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(pool_ids), (args.batch,), device=args.device, generator=gen)
        T = int(pool_len[idx].max())
        ids, msk, sig = pool_ids[idx, :T], pool_msk[idx, :T], pool_sig[idx, :T]
        model.track_stats = step % 500 == 0
        logits = model(ids, sig if args.coupling else None)
        loss = masked_ce(logits, ids, msk)
        total = loss + args.logit_pen * logits.pow(2).mean()
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % 500 == 0:
            st = model.last_stats
            print(f"step {step:6d} loss {ema:.4f} | max s {st['s']:.1f} w {st['w']:.1f} z {st['z']:.1f} x {st['x']:.1f} "
                  f"| {time.time() - t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            acc_h, tot_h, wrong = evaluate(model, task, held, max_new=6)
            _, tot_tr, _ = evaluate(model, task, train_probe, max_new=6)
            hist.append(dict(step=step, loss=ema, heldout=tot_h, train=tot_tr))
            print(f"  [eval {step}] held-out addition {tot_h:.4f} | train-pair addition {tot_tr:.4f} | wrong: {wrong[:5]}",
                  flush=True)

    # no checkpoint selection on the held-out pairs (that would leak the test set): the final model is reported
    acc_h, tot_h, wrong_h = evaluate(model, task, held, max_new=6)
    acc_t, tot_t, wrong_t = evaluate(model, task, held_tpl, max_new=6)
    all_pairs = task.eval_addition(task.train_pairs + task.test_pairs)
    _, tot_all, wrong_all = evaluate(model, task, all_pairs, max_new=6)
    acc_o, tot_o, wrong_o = evaluate(model, task, others, max_new=24)
    model.eval()
    model.track_stats = True
    with torch.no_grad():
        ids, _, sig = task.batch(1024, random.Random(7))
        model(torch.tensor(ids, device=args.device), torch.tensor(sig, device=args.device) if args.coupling else None)
    stats = model.last_stats
    os.makedirs("weights", exist_ok=True)
    W = model.export()
    np.savez(f"weights/{args.name}.npz", **W)
    meta = dict(model.cfg, rev=args.rev, coupling=args.coupling, steps=args.steps, lr=args.lr, wd=args.wd, batch=args.batch,
                heldout_pairs=len(task.test_pairs), train_pairs=len(task.train_pairs),
                heldout_addition_exact=tot_h, heldout_addition_random_template=tot_t,
                all_pairs_addition_exact=tot_all, others_exact=tot_o, others_per_category=acc_o,
                wrong_heldout=wrong_h, wrong_all=wrong_all[:12], wrong_others=wrong_o,
                act_stats=stats, history=hist, train_seconds=time.time() - t0,
                params=sum(p.numel() for p in model.parameters()))
    json.dump(meta, open(f"weights/{args.name}.json", "w"), indent=2, ensure_ascii=False)
    torch.save(model.state_dict(), f"weights/{args.name}.pt")
    print(json.dumps({k: v for k, v in meta.items() if k not in ("history",)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

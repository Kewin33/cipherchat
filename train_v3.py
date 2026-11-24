"""CipherChat v3 – Milestone 2: distil Qwen2.5-0.5B into its FHE-friendly variant (cipherchat/v3/model.py).

  python train_v3.py --name q05_p2 --steps 6000            # calibrate, distil, evaluate, save
  python train_v3.py --name q05_p2 --eval_only             # evaluate a saved checkpoint

Loss: KL(teacher || student) on positions >= 1 + hid_w * relative MSE of every layer's output
      + range_w * range penalty (keeps encrypted values inside the calibrated approximation ranges).
Curriculum: approximations are blended in over the first --ramp steps (model.STATE["mix"]: 0 -> 1).
Positions 0..1 are the fixed public prefix "<|endoftext|>" + newline (computed in plaintext by the server, exact ops).
Embeddings / LM head / final norm stay frozen and exact: they run on the client.
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cipherchat.v3 import model as v3m  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "v3")


def prefix_ids(tok, cfg):
    return [tok.convert_tokens_to_ids(t) if t.startswith("<|") else tok(t)["input_ids"][0]
            for t in cfg["model"]["public_prefix"]]


def token_stream(tok, seq_len, batch, prefix, seed=0):
    """Batches of (batch, seq_len): the public prefix followed by fineweb-edu text (documents joined by EOS)."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)
    buf = []
    while True:
        for ex in ds:
            buf.extend(tok(ex["text"])["input_ids"] + [tok.eos_token_id or prefix[0]])
            L = seq_len - len(prefix)
            while len(buf) >= L * batch:
                chunk = torch.tensor(buf[: L * batch]).view(batch, L)
                buf = buf[L * batch:]
                yield torch.cat([torch.tensor(prefix).expand(batch, -1), chunk], 1)


@torch.no_grad()
def wikitext_ppl(model, tok, prefix, seq_len=128, max_chunks=None, batch=4):
    """Perplexity on WikiText-2 test, non-overlapping chunks of seq_len - len(prefix) tokens after the public prefix;
    every chunk token is scored."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]))["input_ids"]
    P = len(prefix)
    L = seq_len - P
    chunks = [ids[i: i + L] for i in range(0, len(ids) - L, L)]
    if max_chunks:
        chunks = chunks[:max_chunks]
    nll, n = 0.0, 0
    model.eval()
    torch.cuda.empty_cache()
    for i in range(0, len(chunks), batch):
        x = torch.tensor(chunks[i: i + batch])
        x = torch.cat([torch.tensor(prefix).expand(len(x), -1), x], 1).to(model.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x).logits.float()
        lp = F.log_softmax(logits[:, P - 1:-1], -1)
        nll -= lp.gather(-1, x[:, P:, None]).sum().item()
        n += x[:, P:].numel()
    return math.exp(nll / n)


def kl_div(t_logits, s_logits, chunk=256):
    """KL(teacher || student) per token, mean; computed in token chunks to bound fp32 memory."""
    t = t_logits.reshape(-1, t_logits.shape[-1])
    s = s_logits.reshape(-1, s_logits.shape[-1])
    tot = 0.0
    for i in range(0, t.shape[0], chunk):
        tl = F.log_softmax(t[i: i + chunk].float(), -1)
        sl = F.log_softmax(s[i: i + chunk].float(), -1)
        tot = tot + (tl.exp() * (tl - sl)).sum()
    return tot / t.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr_fhe", type=float, default=1e-3)
    ap.add_argument("--ramp", type=int, default=1000)
    ap.add_argument("--hid_w", type=float, default=0.1)
    ap.add_argument("--range_w", type=float, default=1.0)
    ap.add_argument("--calib", type=int, default=8)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--target", type=float, default=2e-3, help="max rel. error of the Newton approximations")
    ap.add_argument("--max_depth", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_chunks", type=int, default=400)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--p", type=int, default=0, help="override approx.softmax.p")
    ap.add_argument("--norm", default="", choices=["", "div", "pos"], help="override approx.softmax.norm")
    ap.add_argument("--inv_ratio", type=float, default=0,
                    help="norm=div: force the inverse range to hi/lo = ratio around the median (0: quantiles)")
    ap.add_argument("--div_switch", type=float, default=0.4,
                    help="norm=div: fraction of the steps trained with exact division before the Newton inverse")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = v3m.load_config()
    if args.p:
        cfg["approx"]["softmax"]["p"] = args.p
    if args.norm:
        cfg["approx"]["softmax"]["norm"] = args.norm
    base = cfg["model"]["base"]
    os.makedirs(OUT, exist_ok=True)
    log_f = open(os.path.join(OUT, f"{args.name}.log"), "a")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log_f.write(s + "\n")
        log_f.flush()

    tok = AutoTokenizer.from_pretrained(base)
    prefix = prefix_ids(tok, cfg)
    v3m.STATE["n_pub"] = P = len(prefix)
    dev = "cuda"
    teacher = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).to(dev)
    mods = v3m.convert(student, cfg)
    res = {"name": args.name, "args": vars(args)}
    ck = os.path.join(OUT, f"{args.name}.pt")

    if args.eval_only:
        sd = torch.load(ck, map_location=dev)
        student.load_state_dict(sd["model"], strict=False)
        v3m.load_fhe_state(student, mods, sd["fhe"])
    else:
        res["teacher_ppl"] = wikitext_ppl(teacher, tok, prefix, args.seq, args.eval_chunks)
        log(f"teacher wikitext-2 ppl (ctx {args.seq}, {args.eval_chunks} chunks): {res['teacher_ppl']:.3f}")
        stream = token_stream(tok, args.seq, args.batch, prefix)
        calib = [next(stream) for _ in range(args.calib)]
        info = v3m.calibrate(student, mods, calib, cfg, margin=args.margin, target=args.target,
                             max_depth=args.max_depth or None, log=log)
        res["calib"] = info
        res["ppl_after_calib"] = wikitext_ppl(student, tok, prefix, args.seq, args.eval_chunks)
        log(f"student ppl right after calibration (no training): {res['ppl_after_calib']:.3f}")

        for n, p in student.named_parameters():
            p.requires_grad_(".layers." in n)
        fhe_params = [p for n, p in student.named_parameters() if p.requires_grad and ("v3." in n or "act_fn" in n)]
        ids = {id(p) for p in fhe_params}
        main_params = [p for p in student.parameters() if p.requires_grad and id(p) not in ids]
        opt = torch.optim.AdamW([{"params": main_params, "lr": args.lr}, {"params": fhe_params, "lr": args.lr_fhe}],
                                weight_decay=0.0, betas=(0.9, 0.95))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
        t0 = time.time()
        ema = None
        div = cfg["approx"]["softmax"].get("norm", "div") == "div"
        switch = int(args.div_switch * args.steps) if div else -1
        for step in range(args.steps):
            if step == switch:
                v3m.STATE["mix"] = 1.0
                res["attn_inverse"] = v3m.calibrate_inverse(student, mods, calib, cfg, margin=args.margin,
                                                            target=args.target, max_depth=args.max_depth or None,
                                                            log=log, ratio=args.inv_ratio or None)
            v3m.STATE["mix"] = min(1.0, (step + 1) / max(1, args.ramp))
            v3m.STATE["clamp"] = True
            student.train()
            opt.zero_grad(set_to_none=True)
            for _ in range(args.accum):
                x = next(stream).to(dev)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    to = teacher(input_ids=x, output_hidden_states=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    so = student(input_ids=x, output_hidden_states=True)
                kl = kl_div(to.logits[:, P:], so.logits[:, P:])
                hid = 0.0
                for th, sh in zip(to.hidden_states[1:], so.hidden_states[1:]):
                    th, sh = th[:, P:].float(), sh[:, P:].float()
                    hid = hid + ((sh - th).pow(2).mean(-1) / th.pow(2).mean(-1).clamp(min=1e-6)).mean()
                hid = hid / len(to.hidden_states[1:])
                rp = v3m.range_penalty(mods)
                loss = kl + args.hid_w * hid + (args.range_w * rp if rp is not None else 0.0)
                if not torch.isfinite(loss):
                    break
                (loss / args.accum).backward()
            if not torch.isfinite(loss):          # rare blow-up while the approximations ramp in: skip the step
                res["skipped_steps"] = res.get("skipped_steps", 0) + 1
                opt.zero_grad(set_to_none=True)
                sched.step()
                log(f"step {step}: non-finite loss, skipped")
                continue
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            sched.step()
            ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
            if step < 5 or step % 50 == 0:
                log(f"step {step} mix {v3m.STATE['mix']:.2f} loss {loss.item():.4f} (ema {ema:.4f}) kl {kl.item():.4f} "
                    f"hid {float(hid):.4f} range {float(rp) if rp is not None else 0:.2e} "
                    f"{(time.time() - t0) / (step + 1):.2f} s/step")
            if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
                v3m.STATE["clamp"] = False
                ppl = wikitext_ppl(student, tok, prefix, args.seq, args.eval_chunks)
                res.setdefault("ppl_curve", []).append([step + 1, ppl])
                log(f"== step {step + 1}: student ppl {ppl:.3f} (teacher {res['teacher_ppl']:.3f})")
                torch.save({"model": {k: v for k, v in student.state_dict().items() if ".layers." in k},
                            "fhe": v3m.fhe_state(student, mods), "cfg": cfg}, ck)
    v3m.STATE["mix"], v3m.STATE["clamp"] = 1.0, True
    res["student_ppl_full_clamped"] = wikitext_ppl(student, tok, prefix, args.seq)
    v3m.STATE["clamp"] = False
    res["student_ppl_full"] = wikitext_ppl(student, tok, prefix, args.seq)
    if args.eval_only or "teacher_ppl_full" not in res:
        res["teacher_ppl_full"] = wikitext_ppl(teacher, tok, prefix, args.seq)
    log(f"FINAL wikitext-2 ppl ctx {args.seq}: student {res['student_ppl_full']:.3f} teacher {res['teacher_ppl_full']:.3f} "
        f"(+{100 * (res['student_ppl_full'] / res['teacher_ppl_full'] - 1):.1f} %), with range clamping "
        f"{res['student_ppl_full_clamped']:.3f}")
    json.dump(res, open(os.path.join(OUT, f"{args.name}{'_eval' if args.eval_only else ''}.json"), "w"), indent=1)
    log_f.close()
    os._exit(0)   # streaming-dataset threads otherwise hang the interpreter at shutdown


if __name__ == "__main__":
    main()

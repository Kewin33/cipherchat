"""Statistics of the exact Qwen2.5 model that decide the v3 approximation ranges (positions after the public prefix):
per RMSNorm input: quantiles of m = mean(x^2); which tokens / channels carry the outliers.
Writes results/v3/stats_<model>.json."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from train_v3 import prefix_ids, token_stream  # noqa: E402
from cipherchat.v3.model import load_config  # noqa: E402


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
    nb = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    tok = AutoTokenizer.from_pretrained(base)
    prefix = prefix_ids(tok, load_config())
    P = len(prefix)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).cuda().eval()
    stream = token_stream(tok, 128, 8, prefix)
    rec = {}
    hooks = []
    for i, layer in enumerate(model.model.layers):
        for nm in ("input_layernorm", "post_attention_layernorm"):
            key = f"L{i}.{nm[:4]}"

            def hook(mod, inp, key=key):
                x = inp[0].float()
                rec.setdefault(key, []).append((x[:, P:].pow(2).mean(-1).flatten().cpu(),
                                                x[:, P:].abs().amax((0, 1)).cpu()))
            hooks.append(getattr(layer, nm).register_forward_pre_hook(hook))
    toks = []
    with torch.no_grad():
        for _ in range(nb):
            x = next(stream)
            toks.append(x[:, P:].flatten())
            model(input_ids=x.cuda())
    toks = torch.cat(toks)
    out = {}
    qs = torch.tensor([0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0])
    for key, lst in rec.items():
        m = torch.cat([a for a, _ in lst])
        chmax = torch.stack([b for _, b in lst]).amax(0)
        q = torch.quantile(m, qs).tolist()
        top = torch.topk(m, 5)
        top_tok = [tok.decode([int(toks[i])]) for i in top.indices]
        big_ch = torch.topk(chmax, 4)
        out[key] = dict(q=q, ratio_max_min=q[-1] / q[0], ratio_999_001=q[5] / q[1], top_tokens=top_tok,
                        top_m=top.values.tolist(), big_channels=big_ch.indices.tolist(), big_vals=big_ch.values.tolist())
        print(f"{key:10s} min {q[0]:9.3g} p0.1 {q[1]:9.3g} med {q[3]:9.3g} p99.9 {q[5]:9.3g} max {q[6]:9.3g} "
              f"ratio {q[6] / q[0]:9.3g} (p99.9/p0.1 {q[5] / q[1]:7.3g}) top {top_tok[:3]} ch {big_ch.indices.tolist()[:3]}",
              flush=True)
    os.makedirs("results/v3", exist_ok=True)
    json.dump(out, open(f"results/v3/stats_{base.split('/')[-1]}.json", "w"), indent=1)
    os._exit(0)


if __name__ == "__main__":
    main()

"""CipherChat v5 – comparison table v3 vs. v3 (e = 0) vs. v5 (e = 0, no bootstrapping) from the result JSONs.

  python scripts/v5_report.py [--dir results/v5] > results/v5/report.md
"""
import argparse
import glob
import json
import os

import numpy as np

RUNS = [
    ("v3 (e ≠ 0, 128 Bit, Bootstrapping)", "v3_check_phantom_pos.json"),
    ("v3-Parameter, e = 0 (Bootstrapping)", "v3e0_check_phantom_pos.json"),
    ("v5 lpp1: e = 0, kein Bootstrapping, 866 × 50-Bit-Primzahlen", "check_phantom_lpp1.json"),
    ("v5 lpp2: e = 0, kein Bootstrapping, 576 × 60-Bit-Primzahlen (Δ = 2³⁰)", "check_phantom_lpp2.json"),
]


def tok_stats(r, skip_first=True):
    toks = r["tokens"][1:] if skip_first and len(r["tokens"]) > 1 else r["tokens"]
    g = lambda k: [t[k] for t in r["tokens"] if k in t]
    return dict(
        n=len(r["tokens"]),
        s=float(np.mean([t["seconds"] for t in toks])),
        gpu=float(np.mean([t.get("gpu_ms", np.nan) for t in toks])) / 1e3,
        rel=max(g("rel")), cos=min(g("cos")), mse=max(g("mse")), logit=max(g("logit_max_abs")),
        top1=all(t["top1_engine"] == t["top1_ref"] for t in r["tokens"]),
        boots=float(np.mean([t.get("bootstraps", 0) for t in toks])),
        mem=max([t.get("mem", {}).get("pool_high_mb", 0) for t in r["tokens"]] +
                [r.get("mem", {}).get("pool_high_mb", 0)]),
        used=max([t.get("mem", {}).get("used_mb", 0) for t in r["tokens"]] + [r.get("mem", {}).get("used_mb", 0)]),
    )


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/v5")
    args = ap.parse_args()
    print("| Variante | Token | s/Token (wall) | GPU s/Token | Token/s | Bootstraps | rel. Fehler (max) | Kosinus (min) "
          "| Logit-Fehler (max) | Top-1 = Ref. | VRAM-Pool-Spitze |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, f in RUNS:
        p = os.path.join(args.dir, f)
        if not os.path.exists(p):
            print(f"| {name} | – | (fehlt: {f}) |||||||||")
            continue
        st = tok_stats(json.load(open(p)))
        print(f"| {name} | {st['n']} | {st['s']:.1f} | {st['gpu']:.1f} | {1 / st['s']:.3f} | {st['boots']:.0f} | "
              f"{st['rel']:.1e} | {st['cos']:.8f} | {st['logit']:.1e} | {'ja' if st['top1'] else 'NEIN'} | "
              f"{st['mem'] / 1024:.1f} GB |")
    print()
    print("Rescale-Rundung (erste 4 Schichten, Fehler nach jeder Schicht):")
    print()
    print("| Variante | Fehler nach Schicht 4 (rel.) | max abs |")
    print("|---|---|---|")
    for p in sorted(glob.glob(os.path.join(args.dir, "rounding_*.json"))):
        r = json.load(open(p))
        t = r["tokens"][0]
        print(f"| {os.path.basename(p)[:-5]} (Δ = 2^{r['ckks']['scale_bits']}) | {t['rel']:.1e} | {t['max_abs']:.1e} |")
    for name, f in RUNS[2:]:
        p = os.path.join(args.dir, f)
        if os.path.exists(p):
            r = json.load(open(p))
            hs = r["tokens"][-1].get("hotspots")
            ops = r["tokens"][-1].get("ops")
            if hs:
                print(f"\n{name}, letztes Token – Hotspots (ms): " + ", ".join(f"{k} {v:.0f}" for k, v in hs.items()))
                print("Operationen (ms / Anzahl): " + ", ".join(f"{k} {v[0]:.0f}/{v[1]}" for k, v in ops.items()))


if __name__ == "__main__":
    main()

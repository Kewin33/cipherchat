"""Markdown tables for docs/V3.md from the v3 result files (M3 block profile, M4 full model, accuracy, quality).

  python scripts/v3_report.py pos        # reads results/v3/{m3_block,m4_full,check_numpy,range_diag}_pos.json
"""
import json
import os
import sys

R = "results/v3"


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def hot_rows(tokens, per):
    keys = ["linear_ptmul_ms", "linear_rot_ms", "attention_ms", "attention_rot_ms", "nonlinear_ms", "bootstrap_ms",
            "other_ms"]
    names = {"linear_ptmul_ms": "1 Lineare Projektionen: Kodieren + Ct×Pt (+ Rescale/Add)",
             "linear_rot_ms": "1 Lineare Projektionen: Rotationen (Key-Switching)",
             "attention_ms": "2 Attention: Ct×Ct (QKᵀ, A·V), Cache-Einfügen, Packen",
             "attention_rot_ms": "2 Attention: Rotationen",
             "nonlinear_ms": "3 Nichtlinear: RMSNorm-Newton, Power-Softmax (+Inverse), Aktivierung",
             "bootstrap_ms": "4 Bootstrapping", "other_ms": "Sonstiges"}
    tot = {k: sum(t["hotspots"][k] for t in tokens) / len(tokens) / per for k in keys}
    s = sum(tot.values())
    out = ["| Hotspot | ms | Anteil |", "|---|---|---|"]
    for k in keys:
        out.append(f"| {names[k]} | {tot[k]:.0f} | {100 * tot[k] / s:.1f} % |")
    out.append(f"| **Summe (GPU)** | **{s:.0f}** | |")
    ph = {}
    for t in tokens:
        for k, v in t.get("boot_phases", {}).items():
            ph[k] = ph.get(k, 0) + v / len(tokens) / per
    if ph:
        out.append("")
        out.append("Bootstrapping-Phasen (ms): " + ", ".join(f"{k} {v:.0f}" for k, v in ph.items()) +
                   f"; Bootstraps: {sum(t['bootstraps'] for t in tokens) / len(tokens) / per:.1f}")
    return "\n".join(out)


def main():
    v = sys.argv[1] if len(sys.argv) > 1 else "pos"
    m3, m4 = load(f"m3_block_{v}.json"), load(f"m4_full_{v}.json")
    num, rd = load(f"check_numpy_{v}.json"), load(f"range_diag_{v}.json")
    if rd:
        print(f"### Qualität ({v})")
        print(f"- mit Bereichsschutz: ppl {rd['guard']['ppl']:.2f}; ohne (= Server): ppl {rd['fhe']['ppl']:.2f}, "
              f"kaputte Abschnitte {rd['fhe']['broken_chunks']}/{rd['fhe']['chunks']}")
        print(f"- außerhalb des Bereichs: " + ", ".join(f"{k} {o['frac']:.1e} (max ×{o['worst_excess']:.2f})"
                                                       for k, o in rd["out_of_range"].items()))
    if num:
        print(f"\n### numpy-Simulation ({v}, {len(num['tokens'])} Token, {num['layers']} Schichten)")
        for t in num["tokens"]:
            print(f"- t={t['t']}: rel {t['rel']:.1e}, cos {t['cos']:.8f}, top1 {t['top1_engine']!r} = {t['top1_ref']!r}")
        print(f"- Operationen pro Token: " + ", ".join(f"{k} {n / len(num['tokens']):.0f}" for k, n in num["counters_total"].items()))
    if m3:
        print(f"\n### Meilenstein 3: ein Block ({v}) – ms pro Schicht und Token")
        print(hot_rows(m3["tokens"], m3["layers"]))
        print(f"\nGenauigkeit Schicht 0: " + "; ".join(f"t={t['t']} cos {t['cos']:.6f}, MSE {t['mse']:.1e}" for t in m3["tokens"]))
        print(f"VRAM: nach Schlüsseln {m3['mem_after_keys']['used_mb']:.0f} MB, Spitze {m3['mem_after_keys']['pool_high_mb']:.0f} MB")
    if m4:
        print(f"\n### Meilenstein 4: ganzes Modell ({v}) – pro Token")
        print(hot_rows(m4["tokens"], 1))
        for t in m4["tokens"]:
            print(f"- t={t['t']}: {t['seconds']:.1f} s wall, GPU {t['gpu_ms'] / 1e3:.1f} s, {t['bootstraps']} Bootstraps, "
                  f"cos {t['cos']:.6f}, rel {t['rel']:.1e}, MSE {t['mse']:.1e}, top1 {t['top1_engine']!r} vs {t['top1_ref']!r}")
        print(f"- Schlüsselerzeugung {m4.get('keygen_s', 0):.0f} s, {m4.get('n_rotation_keys')} Rotationsschlüssel, "
              f"VRAM nach Schlüsseln {m4['mem_after_keys']['used_mb']:.0f} MB")


if __name__ == "__main__":
    main()

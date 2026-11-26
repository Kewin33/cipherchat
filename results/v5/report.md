| Variante | Token | s/Token (wall) | GPU s/Token | Token/s | Bootstraps | rel. Fehler (max) | Kosinus (min) | Logit-Fehler (max) | Top-1 = Ref. | VRAM-Pool-Spitze |
|---|---|---|---|---|---|---|---|---|---|---|
| v3 (e ≠ 0, 128 Bit, Bootstrapping) | 3 | 32.2 | 30.4 | 0.031 | 119 | 2.7e-03 | 0.99999796 | 3.3e-02 | ja | 7.9 GB |
| v3-Parameter, e = 0 (Bootstrapping) | 3 | 30.8 | 29.2 | 0.032 | 119 | 2.7e-03 | 0.99999796 | 3.3e-02 | ja | 8.3 GB |
| v5 lpp1: e = 0, kein Bootstrapping, 866 × 50-Bit-Primzahlen | 3 | 105.8 | 83.5 | 0.009 | 0 | 1.8e-07 | 1.00000000 | 2.8e-06 | ja | 12.1 GB |
| v5 lpp2: e = 0, kein Bootstrapping, 576 × 60-Bit-Primzahlen (Δ = 2³⁰) | 3 | 66.8 | 51.9 | 0.015 | 0 | 7.0e-04 | 0.99999976 | 1.3e-02 | ja | 7.9 GB |

Rescale-Rundung (erste 4 Schichten, Fehler nach jeder Schicht):

| Variante | Fehler nach Schicht 4 (rel.) | max abs |
|---|---|---|
| rounding_lpp1_sb50 (Δ = 2^50) | 6.0e-08 | 2.2e-07 |
| rounding_lpp2_sb20 (Δ = 2^20) | 6.5e-02 | 1.1e-01 |
| rounding_lpp2_sb25 (Δ = 2^25) | 6.5e-04 | 1.7e-03 |
| rounding_lpp2_sb30 (Δ = 2^30) | 1.8e-05 | 4.7e-05 |

v5 lpp1: e = 0, kein Bootstrapping, 866 × 50-Bit-Primzahlen, letztes Token – Hotspots (ms): 1 linear: encode + ptmul 49744, 1 linear: rotations 21890, 2 attention: ctmul/cache/pack 2761, 2 attention: rotations 5201, 3 nonlinear 3969, 4 bootstrapping 0, other 0
Operationen (ms / Anzahl): add 2348/12096, ctmul 4137/1824, rot 27826/18696, encode 27572/29256, ptmac 16781/28872, rescale 4900/8112, decrypt 0/1

v5 lpp2: e = 0, kein Bootstrapping, 576 × 60-Bit-Primzahlen (Δ = 2³⁰), letztes Token – Hotspots (ms): 1 linear: encode + ptmul 31629, 1 linear: rotations 13224, 2 attention: ctmul/cache/pack 1576, 2 attention: rotations 3122, 3 nonlinear 2419, 4 bootstrapping 0, other 0
Operationen (ms / Anzahl): add 1430/12096, ctmul 2178/1824, rot 16798/18696, encode 20313/29256, ptmac 10591/28872, rescale 465/1206, to_half 7/97, lift 190/454, decrypt 0/1

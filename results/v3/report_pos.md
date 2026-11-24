### Qualität (pos)
- mit Bereichsschutz: ppl 42.19; ohne (= Server): ppl 44.58, kaputte Abschnitte 65/2000
- außerhalb des Bereichs: norm 5.8e-07 (max ×1.26), act 1.1e-05 (max ×3.03), attn 0.0e+00 (max ×0.00)

### numpy-Simulation (pos, 6 Token, 24 Schichten)
- t=2: rel 1.3e-08, cos 1.00000000, top1 ' answer' = ' answer'
- t=3: rel 1.5e-08, cos 1.00000000, top1 ' of' = ' of'
- t=4: rel 1.6e-08, cos 1.00000000, top1 ' the' = ' the'
- t=5: rel 1.4e-08, cos 1.00000000, top1 ' is' = ' is'
- t=6: rel 1.2e-08, cos 1.00000000, top1 ' Paris' = ' Paris'
- t=7: rel 1.7e-08, cos 1.00000000, top1 ',' = ','
- Operationen pro Token: ctmul 1824, rot 11048, ptmul 21095, bootstrap 119

### Meilenstein 3: ein Block (pos) – ms pro Schicht und Token
| Hotspot | ms | Anteil |
|---|---|---|
| 1 Lineare Projektionen: Kodieren + Ct×Pt (+ Rescale/Add) | 279 | 25.9 % |
| 1 Lineare Projektionen: Rotationen (Key-Switching) | 68 | 6.4 % |
| 2 Attention: Ct×Ct (QKᵀ, A·V), Cache-Einfügen, Packen | 12 | 1.1 % |
| 2 Attention: Rotationen | 22 | 2.1 % |
| 3 Nichtlinear: RMSNorm-Newton, Power-Softmax (+Inverse), Aktivierung | 48 | 4.5 % |
| 4 Bootstrapping | 647 | 60.1 % |
| Sonstiges | 1 | 0.0 % |
| **Summe (GPU)** | **1077** | |

Bootstrapping-Phasen (ms): Coeff-to-slot 180, Modular reduction 290, Modulus Raising 0, Slot-to-coeff 146, Subsum 20; Bootstraps: 4.0

Genauigkeit Schicht 0: t=2 cos 1.000000, MSE 1.0e-06; t=3 cos 1.000000, MSE 8.1e-07
VRAM: nach Schlüsseln 7516 MB, Spitze 6688 MB

### Meilenstein 4: ganzes Modell (pos) – pro Token
| Hotspot | ms | Anteil |
|---|---|---|
| 1 Lineare Projektionen: Kodieren + Ct×Pt (+ Rescale/Add) | 6510 | 22.3 % |
| 1 Lineare Projektionen: Rotationen (Key-Switching) | 1642 | 5.6 % |
| 2 Attention: Ct×Ct (QKᵀ, A·V), Cache-Einfügen, Packen | 282 | 1.0 % |
| 2 Attention: Rotationen | 542 | 1.9 % |
| 3 Nichtlinear: RMSNorm-Newton, Power-Softmax (+Inverse), Aktivierung | 1164 | 4.0 % |
| 4 Bootstrapping | 19072 | 65.3 % |
| Sonstiges | 2 | 0.0 % |
| **Summe (GPU)** | **29213** | |

Bootstrapping-Phasen (ms): Coeff-to-slot 5339, Modular reduction 8566, Modulus Raising 0, Slot-to-coeff 4289, Subsum 596; Bootstraps: 119.0
- t=2: 33.0 s wall, GPU 29.2 s, 119 Bootstraps, cos 0.999999, rel 1.6e-03, MSE 2.6e-05, top1 ' answer' vs ' answer'
- t=3: 30.8 s wall, GPU 29.2 s, 119 Bootstraps, cos 0.999998, rel 2.7e-03, MSE 5.6e-05, top1 ' of' vs ' of'
- t=4: 30.9 s wall, GPU 29.2 s, 119 Bootstraps, cos 0.999999, rel 1.5e-03, MSE 2.1e-05, top1 ' the' vs ' the'
- t=5: 31.0 s wall, GPU 29.2 s, 119 Bootstraps, cos 0.999997, rel 2.5e-03, MSE 3.6e-05, top1 ' is' vs ' is'
- Schlüsselerzeugung 21 s, 42 Rotationsschlüssel, VRAM nach Schlüsseln 7484 MB

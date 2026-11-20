# CipherChat v2 – Forschungslog

Stand: 2026-11-19. Ziel dieser Runde: (1) ≤ 1 s pro Token und weniger Speicher, (2) Addition fehlerfrei
**durch Generalisierung** (zurückgehaltene Operandenpaare, kein Auswendiglernen), (3) Konzepte aus der LLM-Literatur
(MoE/GLU, RL, Speculative Decoding, Arithmetik-Embeddings) auf FHE übertragen. Das ausführliche Paper liegt in Notion
(Unterseite der SSOT-Seite „HTTPS für LLMs …“).

Messmaschine: `at-10-1-2` (Intel Core Ultra 9 285K, 8 P- + 16 E-Kerne, 62 GB RAM, RTX 5080 16 GB, Ubuntu 24.04,
Python 3.12, OpenFHE 1.6.1 Python-Wheel). Die Maschine wird mit anderen Nutzern geteilt; alle CPU-Messungen
wurden mit `taskset` auf den P-Kernen und bei niedriger Grundlast (Load < 1) gemacht.

## 1. Baseline (v1-Engine, unverändert)

`scripts/benchmark_fhe.py`, Prompts „hallo“, „was ist 5+3?“; alle Antworten identisch zur Klartext-Referenz.

| OMP-Threads (Kerne) | s/Prompt-Token | s/generiertes Token |
|---|---|---|
| 1 (P-Kern 0) | 1,70 | 2,13 |
| 2 | 1,27 | 1,60 |
| 4 | 1,00 | 1,23 |
| 8 (P-Kerne) | **0,95** | **1,20** |
| 16 | 1,01 | 1,25 |
| 24 | 0,98 | 1,25 |

Befund: Ab ~4 Threads skaliert OpenMP nicht mehr (Parallelität nur über die ~9 RNS-Limbs; Python-Overhead und
384 dynamische Encodes pro Token sind seriell).

## 2. v2-Engine: Ideen und warum sie funktionieren

Kernbeobachtung: v1 nutzt von 8192 CKKS-Slots nur 64 sinnvoll (der 64-Vektor wird 128× repliziert). Die Kosten
einer Plaintext-Multiplikation oder Rotation sind aber unabhängig davon, wie viele Slots Nutzdaten tragen. v2 nutzt die
Redundanz gezielt aus. Alles ist in `cipherchat/v2/layout.py` (numpy, ohne OpenFHE) formuliert und durch eine exakte
Slot-Simulation geprüft (`NumpyBackend`, gleiche Engine wie mit OpenFHE).

1. **Generalisiertes BSGS (beliebiges Ausgabe-Layout).** Aus einem rep-Vektor x berechnet *eine* BSGS-Matvec
   (64 PtMults, 15 gehoistete + 3 Giant-Rotationen, 1 Level) jede lineare Abbildung R^64 → R^8192:
   `P[i][j][τ] = W[(τ+j) mod 64, (τ−16i) mod S]`. Damit landen k/v direkt im richtigen Cache-Block, die
   MLP-Hidden-Schicht füllt alle 8192 Slots, und Biases/Masken sind in die Plaintexts eingebaut.
2. **Rückrichtung „to_rep“ mit Cross-Sum.** Eine beliebige 8192-Slot-Eingabe m wird mit 64 PtMults + 3 Giant +
   7 Rotationen (Cross-Sum über 128 Blöcke) auf einen replizierten 64-Vektor abgebildet:
   `P[i][j][τ] = W[(τ+j) mod S, (τ−16i) mod 64]`. ⇒ **MLP mit d_ff bis 8192 kostet so viel wie eine 64×64-Matvec**
   (v1: d_ff = 128 kostete 2 × 2 Matvecs).
3. **Cross-Sum = Attention-Blocksumme.** Die Cross-Sum-Rotationen (64 … 4096) enthalten die Blocksummen-Rotationen
   der Attention (128 … 4096). Die Attention-Ausgabe im V-Layout wird *vor* der Cross-Sum addiert: Blocksumme +
   Re-Doubling (7 Rotationen) entfallen.
4. **Schiebender KV-Cache.** Statt Masken pro Position (dynamisch encodiert) wird der Cache pro Token um einen
   128-Slot-Block rotiert (1 Rotation pro Cache und Layer); der neue Eintrag landet immer im letzten Block.
   ⇒ alle Plaintexts sind statisch, **0 dynamische Encodes** (v1: 384 pro Token).
5. **Relative Positions-Biases gratis.** Im schiebenden Cache entspricht Block j dem Abstand r = 63 − j. Ein gelernter
   Bias ρ[t−j] in der polynomiellen Attention ist damit ein statischer Plaintext (eine Addition). Das Modell braucht
   keine absoluten Positionen mehr (NoPE + ALiBi/T5-artiger Bias).
6. **Feste Attention-Normierung κ** statt 1/(t+1) (in Wq/c/ρ gefaltet) ⇒ keine q-Skalierung pro Token.
7. **Bilineare Scores ohne q-Projektion.** Mit konstantem Kanal x[63] = 1 (trägt alle Biases) ist
   s_tj = x_t M x_jᵀ; der Cache speichert k̃_j = x_j Mᵀ, der Score ist x_t · k̃_j. Die q-Matvec entfällt komplett.
8. **Layer 0 ist ein Lookup.** Bei One-Hot-Eingabe ist jedes MLP eine Tabelle; Layer 0 hat daher kein MLP, sondern zwei
   Embeddings (Attention-Eingabe, Residual). Scores in Layer 0: s = u ⊙ K̂ mit K̂-Einträgen = Spalten von E M Eᵀ.
9. **Gated MLP im Reserve-Level.** Der MLP-Pfad verbraucht pro Layer nur 3 der 4 Level. GLU-Varianten
   (`bil`: (xW1)⊙(xWg), `sqglu`: ((xW1)⊙(xWg))², Grad 4) passen ohne zusätzliche Tiefe (+1 Matvec, +0/1 ct×ct).

Operationen pro generiertem Token (2 Layer, gezählt in der Slot-Simulation):

| | v1 | v2 (`sq`) | v2 (`sqglu`) |
|---|---|---|---|
| Rotationen (davon gehoistet) | ~163 (90) | **96 (45)** | 99 (45) |
| Plaintext-Multiplikationen | 1024 | **512** | 576 |
| ct×ct-Multiplikationen | 10 | **7** | 8 |
| dynamische Encodes | 384 | **0** | 0 |
| Tiefe / Ringdimension | 8 / 2^14 | 8 / 2^14 | 8 / 2^14 |
| Rotations-Keys | 25 | 25 | 25 |

Exaktheit: torch (float64) = Referenz = Slot-Simulation bis auf 1e-15 (`scripts/v2_check.py`, auch für 3 Layer,
Hidden 8192, alle MLP-Varianten, mit Signifikanz-Eingabe).

## 3. v2 unter echtem CKKS (OpenFHE)

`scripts/v2_fhe_test.py`, zufälliges Modell, Hidden 8192 (= alle Slots), Logits an jeder Position mit der Referenz verglichen.

| OMP-Threads (Kerne) | v1 s/Token | **v2 s/Token** | Speedup |
|---|---|---|---|
| 1 | 2,13 | **0,75** | 2,8× |
| 2 | 1,60 | 0,55 | 2,9× |
| 4 | 1,23 | 0,42 | 2,9× |
| 8 (P-Kerne) | 1,20 | **0,365** | 3,3× |
| 16 | 1,25 | 0,40 | 3,1× |
| 24 | 1,25 | 0,42 | 3,0× |

Max. Logit-Fehler 0,9–1,4·10⁻⁴; Eval-Keys 225 MB (25 Keys, unverändert); statischer Plaintext-Cache ≈ 300 MB
(einmal pro Server, nicht pro Session; v1 ≈ 640 MB bei Hidden 128); Warm-up (alle Plaintexts encodieren) 1,1 s.
Hidden 1024 statt 8192: 0,325 s/Token. Die Breite des MLP ist also praktisch kostenlos.

## 4. Addition, die generalisiert

Aufgabe: a + b mit 0 ≤ a, b ≤ 99 in natürlichen Formulierungen („was ist 23+45?“, „23 plus 45“, „rechne …“, Präfixe,
Suffixe). **20 % aller ungeordneten Paare {a,b} (1010 Paare, 2020 geordnete Aufgaben) werden vollständig
zurückgehalten**, in keiner Reihenfolge und keinem Template im Training. Gemessen: Exact Match auf allen
zurückgehaltenen Paaren (kanonisches Template) und mit zufälligen Templates. Daneben laufen alle v1-Chat-Kategorien.

Alle Runs: 30k Schritte, Batch 512, AdamW (lr 2e-3, wd 0,1), RTX 5080; Endmodell ohne Checkpoint-Auswahl.

| Run | Zahlenformat | MLP | Params | Held-out exakt (2.020) | Held-out, zufällige Templates | alle 10.000 Paare | Chat-Kategorien |
|---|---|---|---|---|---|---|---|
| `cf2_msb_h1024` | big-endian (normal) | `sq`, 1024 | 166k | 0,389 | 0,383 | 0,468 | 1,000 |
| `cf2_rev_cpl_h1024` | little-endian + Position Coupling | `sq`, 1024 | 167k | 0,9975 | 0,9960 | 0,9995 | 1,000 |
| **`cf2_rev_cpl_sqglu_h2048`** | little-endian + Position Coupling | **`sqglu`, 2048** | 428k | **1,000** | **1,000** | **1,000** | **1,000** |

Lernkurven (held-out exakt): `sq` 0,88 (2k) → 0,96 (4k) → 0,99 (16k) → 0,9975 (30k, 5 Fehler: 40+70, 48+48, 48+49 …);
`sqglu` 0,9885 (2k) → 0,9995 (4k) → **1,000 ab 6k bis 30k** stabil. Die Baseline lernt die Stellenzuordnung nie (0,16 → 0,39).
Aktivierungsbereiche des `sqglu`-Modells: max |s| 3,2, |w| 10,2, |z| 4,9, |x| 19,1 – deutlich kleiner als v1 (|x| ≈ 30–70).

Entscheidungsabstände (`scripts/v2_margins.py`, 646 Prompts, 3.329 Greedy-Schritte): min. Top-1 − Top-2 = **12,9** Logits
(Median 16,1). Größter gemessener CKKS-Logit-Fehler 0,15 ⇒ ~85-fache Reserve.

Position Coupling (Cho et al., NeurIPS 2024) passt perfekt zu FHE: Der *Client* kennt den Klartext und markiert jede
Ziffer mit ihrer Stellenwertigkeit; die Markierung steckt in den 5 freien Dimensionen des One-Hot-Inputs (Vokabular 59 von
64). Für die Engine ändert sich nichts (die Eingabe ist ein „Two-Hot“-Vektor, Layer 0 ist linear in der Eingabe).
Little-Endian (Lee et al. 2023) ist eine Tokenisierungs-Entscheidung des Clients, für den Nutzer unsichtbar.

### Unter echter Verschlüsselung (`cf2_rev_cpl_sqglu_h2048`)

`scripts/v2_fhe_test.py --weights weights/cf2_rev_cpl_sqglu_h2048.npz --heldout 50` (8 P-Kerne, Load 0):
**50/50 zurückgehaltene Additionen richtig, 58/58 Antworten identisch zur Klartext-Referenz**, 0,342 s pro generiertem
Token, 0,231 s pro Prompt-Token, Logit-Fehler Median 0,0035 / max. 0,15, RSS 1,16 GB.

`scripts/v2_http_e2e.py` (Server + Client als Prozesse, HTTP): „was ist 47+38?“ → 85 (5,7 s), „was ist 99+1?“ → 100 (5,0 s),
„23 plus 45“ → 68 (3,8 s), „hauptstadt von polen?“ → warschau. (9,2 s; v1 auf 2 vCPU: 72 s), Server 0,26–0,30 s/Token.
Rohdaten: `docs/v2_results/`.

## 5. Offene Punkte

- GPU-Backend (FIDESlib/Phantom): Projektion ~10–20 ms/Token auf RTX-4090/5080-Klasse (aus FIDESlib-Primitiven
  hochgerechnet, nicht gemessen).
- Speculative Decoding / Multi-Token-Prediction mit clientseitigem Drafting (Leakage der Akzeptanzlänge beachten).
- Mehrere Sessions gleichzeitig (Key-Tags), Batched Prefill, Rotation-Key-Kompression.

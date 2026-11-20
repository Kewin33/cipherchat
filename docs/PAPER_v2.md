# CipherChat v2: Slot-gesättigte CKKS-Inferenz und generalisierende Arithmetik für einen verschlüsselten Chatbot

Stand 20.11.2026 · Code: `cipherchat/v2/`, Messlog: `docs/RESEARCH_LOG.md` · Messrechner: Intel Core Ultra 9 285K (8 P- + 16 E-Kerne), RTX 5080, OpenFHE 1.6.1

## Kurzfassung

CipherChat ist ein Chatbot, dessen Server einen kleinen Transformer ausschließlich auf CKKS-Ciphertexts auswertet („HTTPS für LLMs“). Der MVP (v1) brauchte 2,6 s pro generiertem Token auf 2 vCPUs (1,20 s auf 8 Kernen) und rechnete auf zurückgehaltenen Aufgaben nur zu 39 % richtig – er hatte eher auswendig gelernt. Wir zeigen, dass beide Probleme dieselbe Wurzel haben: Das Modell ist nicht auf die Struktur von CKKS und der Aufgabe zugeschnitten. **v2** nutzt die in v1 zu 99 % redundanten SIMD-Slots aus (generalisiertes Baby-Step-Giant-Step, schiebender verschlüsselter KV-Cache, bilineare Attention ohne Query-Projektion, Fusion von Attention- und MLP-Summen) und überträgt Konzepte aus der LLM-Forschung (Position Coupling, Little-Endian-Zahlen, ALiBi-artige relative Positionen, Parallel Blocks, GLU/Soft-MoE-Gating) so, dass sie unter CKKS **keine zusätzliche multiplikative Tiefe** kosten. Ergebnis: **0,365 s pro Token** auf 8 P-Kernen (0,75 s auf einem Kern; 3,3× bzw. 2,8× schneller als v1), **0 statt 384 dynamische Encodes** pro Token, 41 % weniger Rotationen, 50 % weniger Plaintext-Multiplikationen, ein MLP mit bis zu 8192 Hidden-Units ohne Mehrkosten, und **100 % (2.020 von 2.020)** auf zurückgehaltenen zweistelligen Additionsaufgaben, deren Operandenpaare das Modell im Training nie gesehen hat. Unter echter Verschlüsselung liefert das Modell **bei 50 zufällig gezogenen zurückgehaltenen Additionen 50/50 richtige Antworten und bei 58/58 Prompts dieselbe Antwort wie die Klartext-Rechnung, mit 0,34 s pro generiertem und 0,23 s pro Prompt-Token**.

## 1. Motivation und Ziele

Die SSOT-Seite „HTTPS für LLMs“ beschreibt die Vision: Der Prompt bleibt auch während der Berechnung verschlüsselt. Der MVP hat gezeigt, dass das für ein kleines Modell technisch funktioniert, aber zwei Schwächen offengelegt:

1. **Latenz:** 2,6 s pro Token (2 vCPU) sind für einen Chat zu langsam. Ziel dieser Arbeit: **≤ 1 s pro Token** auf einem Laptop-/Workstation-CPU, bei weniger Speicher.
2. **Fähigkeiten:** Das 78k-Parameter-Modell beantwortete trainierte Prompts perfekt, generalisierte aber kaum (Rechnen 39 %, Vergleiche 82 % auf zurückgehaltenen Paaren). Ziel: **Addition fehlerfrei durch Generalisierung**, nachgewiesen auf Operandenpaaren, die im Training nie vorkamen.

Langfristig soll das System chatfähig wie ein großes LLM werden; Abschnitt 9 schätzt ab, was das unter FHE kostet und welche der hier entwickelten Techniken skalieren.

## 2. Ausgangslage: wohin die Zeit in v1 geht

CKKS-Kostenmodell (N = 2¹⁴, 8 multiplikative Level, HYBRID-Key-Switching, OpenFHE):

| Operation | Kosten (1 Kern) | Bemerkung |
| --- | --- | --- |
| Rotation (Key-Switch) | ~8–12 ms | teuerste Operation; gehoistet ≈ halb so teuer |
| ct×ct-Multiplikation + Relinearisierung | ~10 ms | enthält einen Key-Switch |
| Plaintext-Multiplikation | ~0,3–0,6 ms | elementweise im NTT-Raum |
| Encode eines Plaintexts | ~1–2 ms | FFT + NTT über alle Limbs |

Pro generiertem Token rechnete v1 ~163 Rotationen (90 gehoistet), 1.024 Plaintext-Multiplikationen, 10 ct×ct-Multiplikationen und **384 dynamische Encodes** (Masken für das Einfügen in den KV-Cache und die positionsabhängige Query-Skalierung 1/√(t+1)). Auf der Testmaschine skaliert v1 nur bis ~4 Threads (OpenMP parallelisiert in OpenFHE nur über die ~9 RNS-Limbs):

| Threads | 1 | 2 | 4 | 8 | 16 | 24 |
| --- | --- | --- | --- | --- | --- | --- |
| v1 s/generiertes Token | 2,13 | 1,60 | 1,23 | **1,20** | 1,25 | 1,25 |

Die zentrale Beobachtung: v1 packt einen 64-dimensionalen Vektor 128-fach repliziert in 8.192 Slots. **99,2 % der SIMD-Kapazität tragen Kopien**, aber jede Rotation und Multiplikation kostet trotzdem den vollen Ciphertext.

## 3. Verwandte Arbeiten

**FHE-Inferenz für Transformer.** NEXUS (NDSS 2025) war das erste nicht-interaktive Protokoll; THOR (CCS 2025) und MOAI (2025) optimieren Packing und rotationsfreie Nichtlinearitäten; Powerformer (ACL 2025) und Rho et al. (ICLR 2025) ersetzen Softmax durch FHE-freundliche Kerne. Park, Stehlé, Cheon et al. (2026) bringen Llama-3-8B vollständig unter CKKS (18 s/Token auf 8 GPUs). Am nächsten an unserer Arbeit ist **Cachemir** (2026): Es identifiziert ebenfalls die Slot-Unterauslastung als Hauptkostentreiber (77,8 % der Decoding-Latenz sind ct-pt-Multiplikationen), löst sie mit „Interleaved Replicated Packing“ und führt einen verschlüsselten KV-Cache ein, nutzt dabei aber Masken (Level) und Bootstrapping. Unser Design zielt auf kleine Modelle ohne Bootstrapping und kommt ohne jede Masken-Multiplikation aus. Grundlagen: Diagonalmethode und BSGS (Halevi & Shoup 2014/2018), Hoisting (Bossuat et al. 2021), GAZELLE (2018).

**GPU-FHE.** FIDESlib (2025) und Phantom erreichen auf einer RTX 4090 ~1,1 ms pro Rotation bei N = 2¹⁶ und 30 Limbs (≈ 330× schneller als ein CPU-Kern).

**LLM-Konzepte, die wir übertragen.** Parallel Blocks (GPT-J, PaLM), ReZero, ALiBi/T5-Relativ-Biases und NoPE, polynomielle Attention (PolySketchFormer), quadratische Aktivierungen (Primer), GLU-Varianten (Shazeer 2020), Mixture-of-Experts (Switch, Mixtral, DeepSeekMoE, Soft MoE; privat: CryptoMoE, NeurIPS 2025), Speculative Decoding (Leviathan et al. 2023; privat: POST, ICML 2025), Multi-Token-Prediction (Gloeckle et al. 2024), RL mit verifizierbaren Belohnungen (GRPO/DeepSeekMath, DeepSeek-R1, L1/LCPO) sowie Arithmetik in kleinen Transformern: umgekehrte Ziffernreihenfolge (Lee et al. 2023), Abacus-Embeddings (McLeish et al. 2024) und **Position Coupling** (Cho et al., NeurIPS 2024: ein 1-Layer-Transformer mit gekoppelten Positionen kann addieren, ohne Positionsinformation nicht).

## 4. Engine v2

Notation: S = 8.192 Slots, `rot(v,k)[i] = v[i+k]`, rep-Layout = 64-Vektor S/64-fach kachelt. Alle Formeln sind in `cipherchat/v2/layout.py` implementiert und durch eine **exakte Slot-Simulation** (numpy-Backend der gleichen Engine) gegen die Klartext-Referenz bis 10⁻¹⁵ geprüft.

### 4.1 Generalisiertes BSGS: eine Matvec, beliebiges Ausgabe-Layout

Für x im rep-Layout und eine beliebige Matrix W ∈ ℝ^{64×S} gilt mit 16 Baby- und 4 Giant-Steps

$$
y = \sum_{i<4} \mathrm{rot}\Big(\sum_{j<16} \mathrm{rot}(x, j) \odot P_{ij},\ 16i\Big),\qquad P_{ij}[\tau] = W\big[(\tau + j) \bmod 64,\ (\tau - 16i) \bmod S\big]
$$

und damit y[s] = Σ_r x[r]·W[r,s] für **alle** S Ausgabe-Slots – mit denselben Kosten wie v1s 64×64-Matvec (64 PtMults, 15 gehoistete + 3 Rotationen, 1 Level). Weil x repliziert ist, ist jede Rotation um < 64 zyklisch korrekt; die Plaintexts „wissen“, welche Zeile in welchem Slot ankommt. Damit lassen sich in *einer* Matvec Masken (Cache-Block), Verdopplungen ([k,k]) oder eine 8.192-breite Hidden-Schicht erzeugen.

Die Gegenrichtung (beliebige S-Slot-Eingabe m → replizierter 64-Vektor) gelingt mit einer abschließenden **Cross-Sum** über alle 128 Blöcke (7 Rotationen):

$$
T = \sum_{g<128}\mathrm{rot}\Big(\sum_{i<4}\mathrm{rot}\big(\textstyle\sum_{j<16}\mathrm{rot}(m,j)\odot P_{ij}, 16i\big), 64g\Big),\qquad P_{ij}[\tau] = W\big[(\tau + j)\bmod S,\ (\tau - 16i)\bmod 64\big]
$$

**Folge: Ein MLP mit d_ff bis 8.192 kostet so viel wie eine 64×64-Matvec** (v1: d_ff = 128 kostete vier Matvecs). Die Breite des Modells ist unter CKKS bis zur Slot-Grenze praktisch gratis – das unter FHE vorhandene Gegenstück zur „Parameter ohne FLOPs“-Idee von Mixture-of-Experts.

### 4.2 Schiebender KV-Cache und relative Positionen

v1 fügte k_t, v_t über positionsabhängige Masken ein (384 Encodes pro Token). v2 rotiert den Cache pro Token um einen 128-Slot-Block (`K ← rot(K,128) + k_new`) und schreibt den neuen Eintrag **immer in den letzten Block** – alle Plaintexts werden statisch und einmal pro Server encodiert. Weil Block j dann immer den Token im Abstand r = 63 − j enthält, wird ein gelernter **relativer Positions-Bias** ρ_r (ALiBi/T5-artig, aber innerhalb der polynomiellen Attention) zu einem einzigen statischen Plaintext, der zu den Scores addiert wird. Das Modell braucht keine absoluten Positionen mehr (NoPE-Eingabe). Die Normierung 1/(t+1) ersetzen wir durch eine feste Konstante κ, die in die Gewichte gefaltet wird.

### 4.3 Bilineare Scores ohne Query-Projektion

Mit einem konstanten Kanal x[63] = 1, der alle Biases trägt, ist der Score s_tj = x_t M x_jᵀ (M = W_q W_kᵀ) bilinear. Der Cache speichert k̃_j = x_j Mᵀ, und der Score ist das Skalarprodukt x_t · k̃_j – **die Query-Matvec entfällt**. Die polynomielle Attention lautet w_tj = (√κ·x_t·k̃_j + √κ(c + ρ_{t−j}))², a_t = Σ_j w_tj v_j.

### 4.4 Layer 0 als Lookup und Fusion der Summen

Bei One-Hot-Eingabe ist jede Funktion der Eingabe eine Tabelle: Layer 0 hat daher kein MLP, sondern zwei Embeddings (Attention-Eingabe, Residual), und seine Scores sind u ⊙ K̂ mit K̂-Einträgen aus E M Eᵀ. Die 7 Cross-Sum-Rotationen des MLP enthalten außerdem genau die Blocksummen-Rotationen der Attention: Addiert man die Attention-Ausgabe im V-Layout **vor** der Cross-Sum, entfallen Blocksumme und Re-Doubling vollständig. Im letzten Layer teilt sich die Unembedding-Matvec x·W_u die Giant-Steps mit W₂.

### 4.5 Gated MLP im Reserve-Level (GLU / Soft-MoE)

Pro Layer verbraucht der Attention-Pfad 4 Level (Projektion → Score → Quadrat → ×V), der MLP-Pfad nur 3 (Projektion → Quadrat → W₂). Ein multiplikatives Gate passt daher **ohne zusätzliche Tiefe** hinein: `bil` m = (xW₁)⊙(xW_g) (bilineares GLU, Shazeer 2020; das Quadrat ist ein Spezialfall) oder `sqglu` m = ((xW₁)⊙(xW_g))² (Grad 4). Ist W_g blockweise konstant, ist das ein Soft-MoE mit bis zu 128 Experten in den freien Slots – ohne Routing-Leakage, weil alle Experten dicht ausgewertet werden (vgl. CryptoMoE, wo dichte Auswertung 8–15× teurer ist; unter Slot-Sättigung kostet sie nichts).

### 4.6 Operationsbilanz pro generiertem Token (2 Layer, gezählt)

| | v1 | v2 (`sq`) | v2 (`sqglu`) |
| --- | --- | --- | --- |
| Rotationen (davon gehoistet) | ~163 (90) | **96 (45)** | 99 (45) |
| Plaintext-Multiplikationen | 1.024 | **512** | 576 |
| ct×ct-Multiplikationen | 10 | **7** | 8 |
| dynamische Encodes | 384 | **0** | 0 |
| Tiefe / Ringdimension / Rotations-Keys | 8 / 2¹⁴ / 25 | 8 / 2¹⁴ / 25 | 8 / 2¹⁴ / 25 |

## 5. Addition, die generalisiert

**Aufgabe und Protokoll.** a + b mit 0 ≤ a, b ≤ 99 in natürlichen Formulierungen („was ist 23+45?“, „23 plus 45“, „rechne …“, mit Präfixen und Suffixen), gemischt mit allen v1-Chat-Kategorien. **20 % aller 5.050 ungeordneten Paare {a,b} werden vollständig zurückgehalten** (in keiner Reihenfolge, keinem Template); bewertet wird Exact Match auf allen 2.020 zurückgehaltenen geordneten Aufgaben. Es gibt keine Checkpoint-Auswahl auf den Testpaaren.

**Warum v1 scheitert.** Mit einem Head und zwei Layern muss das Modell bei der ersten Ausgabeziffer (höchstwertig) die ganze Übertragskette kennen und gleichzeitig herausfinden, welche Ziffer welche Stellenwertigkeit hat – bei variabler Zahlenlänge und Formulierung. Eine Baseline mit normaler Schreibweise erreicht nach 30k Schritten nur 38,9 % (auf Trainingspaaren selbst nur 46,8 %).

**Zwei Client-seitige Tokenisierungsentscheidungen** (für den Nutzer unsichtbar, für den Server verschlüsselt):

1. **Little-Endian-Zahlen** (Lee et al. 2023): Der Client schreibt jede Ziffernfolge umgekehrt („23+45“ → „32+54“) und dreht die Antwort zurück. Jede Ausgabeziffer hängt nur noch von den Ziffern gleicher Stellenwertigkeit und dem Übertrag ab.
2. **Position Coupling** (Cho et al. 2024) über einen **Two-Hot-Input**: Das Vokabular belegt 59 der 64 Input-Dimensionen; die 5 freien Dimensionen tragen die Stellenwertigkeit jeder Ziffer (1 = Einer …), \<sep> und erzeugte Ziffern tragen die Stellenwertigkeit der *nächsten* Ausgabeziffer. Weil Layer 0 linear in der Eingabe ist, ändert sich an der Engine **nichts** – nur der Vektor, den der Client verschlüsselt.

## 6. Experimente

### 6.1 Exaktheit

torch (float64) = Token-für-Token-Referenz = Slot-Simulation bis auf 1,6·10⁻¹⁵ (zufällige Modelle: alle MLP-Varianten, 2 und 3 Layer, Hidden bis 8.192, mit Two-Hot-Input) bzw. 4·10⁻¹² (trainiertes Modell). Unter CKKS: maximaler Logit-Fehler 0,9–1,5·10⁻⁴ bei zufälligen Modellen; beim trainierten Modell Median 0,0035 und Maximum 0,15 bei Entscheidungsabständen ≥ 12,9 (v1: < 3·10⁻³ bei Abständen ~10).

### 6.2 Laufzeit und Skalierung (v2, Hidden 8.192)

| Threads (Kerne) | v1 s/Token | v2 s/Token | Speedup |
| --- | --- | --- | --- |
| 1 | 2,13 | **0,75** | 2,8× |
| 2 | 1,60 | 0,55 | 2,9× |
| 4 | 1,23 | 0,42 | 2,9× |
| 8 (P-Kerne) | 1,20 | **0,365** | 3,3× |
| 16 | 1,25 | 0,40 | 3,1× |
| 24 | 1,25 | 0,42 | 3,0× |

Hidden 1.024 statt 8.192: 0,325 s/Token – die volle Slot-Breite kostet nur ~10 %.

Die gegatete Variante (`sqglu`, Hidden 2.048), die für die fehlerfreie Addition nötig ist (§6.4), kostet unter CKKS **0,404 s/Token** (8 P-Kerne, gemessen während paralleler GPU-Trainings) bei einem Logit-Fehler von 1,5·10⁻⁴.

### 6.3 Speicher

| Größe | v1 | v2 | Bemerkung |
| --- | --- | --- | --- |
| Eval-Keys pro Session | 225 MB (25 Rotations-Keys) | 225 MB (25 Keys) | dnum = 3 ist optimal: dnum = 2 erzwingt N = 2¹⁵ (368 MB, 0,99 s/Token), dnum = 9 → 614 MB, 0,63 s/Token |
| Plaintext-Cache (einmal pro Server) | ~800 MB (640 Plaintexts, RSS-Zuwachs) | **~300 MB** (`sq`), 346 MB (`sqglu`) | v2: tiefere Level für Layer 1 (weniger Limbs), keine Masken-Plaintexts |
| Dynamische Encodes pro Token | 384 (~0,8 s auf 2 vCPU) | **0** | |
| KV-Cache pro Session | 4 Ciphertexts | 4 Ciphertexts | ~6 MB |
| Upload / Download pro Token | 2,36 MB / 0,53 MB | 2,36 MB / 0,53 MB | Two-Hot-Input ändert die Größe nicht |
| CKKS-Präzision 2²⁶ statt 2³³ | – | gleiche Zeit/Keys, Fehler 1,6·10⁻² | RNS-Limbs sind immer 64-Bit-Worte |

### 6.4 Addition

Alle Runs: 30k Schritte, Batch 512, AdamW (lr 2·10⁻³, Weight Decay 0,1), RTX 5080 (17–47 min); berichtet wird das Endmodell.

| Run | Zahlenformat | MLP | Parameter | Held-out exakt (2.020) | Held-out, zufällige Formulierung | alle 10.000 Paare | Chat-Kategorien |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1-Evaluationsmodell (zum Vergleich) | big-endian | Quadrat, 128 | 78k | 0,39 (nur einstellig) | – | – | 1,00 (Train) |
| Baseline v2 | big-endian | `sq`, 1.024 | 166k | 0,389 | 0,383 | 0,468 | 1,000 |
| v2 mit Little-Endian + Position Coupling | Client-Tokenisierung | `sq`, 1.024 | 167k | 0,9975 | 0,9960 | 0,9995 | 1,000 |
| **v2 mit Position Coupling + Gated MLP (Reserve-Level)** | Client-Tokenisierung | **`sqglu`, 2.048** | 428k | **1,000** | **1,000** | **1,000** | **1,000** |

Lernkurven (held-out exakt): Die Baseline steigt von 0,16 (2k) nur bis 0,39 (30k) – sie findet die Stellenzuordnung nie. Mit Position Coupling: 0,88 (2k) → 0,99 (16k) → 0,9975 (30k), Restfehler bei Überträgen in eine fehlende Stelle (40+70, 48+49). Mit Gate: 0,9885 (2k) → 0,9995 (4k) → **1,000 ab 6k bis 30k stabil**. Das Grad-4-Gate kann Übertrag und Stellen-Ziffern in *einem* MLP multiplikativ verknüpfen; ohne Gate muss das quadratische MLP das über zwei Layer verteilen.

Die gelernten Aktivierungen bleiben klein (max |x| = 19, |w| = 10), und der kleinste Entscheidungsabstand (Top-1 − Top-2) über 3.329 Greedy-Schritte beträgt **12,9 Logits** (Median 16,1) – ein komfortabler Puffer für das CKKS-Rauschen.

### 6.5 Ende-zu-Ende unter FHE

Trainiertes Modell (`sqglu`, Hidden 2.048), echte CKKS-Verschlüsselung, realistisches Protokoll (Prefill ohne Logit-Pfad der letzten Schicht), 8 P-Kerne, unbelastete Maschine:

| Messung | Ergebnis |
| --- | --- |
| 50 zufällige **zurückgehaltene** Additionen | **50/50 richtig** |
| alle 58 Prompts (Additionen + Chat) | **58/58 identisch** zur Klartext-Referenz |
| Zeit pro generiertem Token / Prompt-Token | **0,342 s / 0,231 s** |
| Logit-Fehler (Median / Maximum) | 0,0035 / 0,15 (kleinster Entscheidungsabstand 12,9) |
| Server-Speicher (RSS inkl. Keys und Plaintext-Cache) | 1,16 GB |

Über HTTP (Server und Client als getrennte Prozesse, Handshake 0,8 s, Ciphertexts serialisiert):

| Nachricht | Antwort | Gesamtzeit | Server-Zeit/Token |
| --- | --- | --- | --- |
| was ist 47+38? | 85 | 5,7 s | 0,30 s |
| was ist 7+5? | 12 | 4,3 s | 0,26 s |
| was ist 99+1? | 100 | 5,0 s | 0,27 s |
| 23 plus 45 | 68 | 3,8 s | 0,26 s |
| hallo | hallo! | 4,1 s (v1, 2 vCPU: 32 s) | 0,30 s |
| hauptstadt von polen? | warschau. | 9,2 s (v1, 2 vCPU: 72 s) | 0,27 s |
| ist 4 gerade? | ja. | 5,0 s (v1, 2 vCPU: 39 s) | 0,26 s |

## 7. LLM-Konzepte auf FHE übertragen – Analyse

Leitfrage für jedes Konzept: Was ist im Klartext-LLM der eigentliche Gewinn, was davon überlebt die Regeln von CKKS (keine datenabhängige Verzweigung, Kosten = Rotationen + Level, SIMD-Slots), und wie sieht die übertragene Form aus? ✅ = umgesetzt und gemessen, 📐 = Design mit Kostenmodell (nicht implementiert).

| Konzept (Klartext) | Problem unter FHE | Übertragene Form | Kosten in CKKS | Status |
| --- | --- | --- | --- | --- |
| **Mixture-of-Experts** (Switch, Mixtral, DeepSeekMoE): mehr Parameter pro FLOP durch Top-k-Routing | Routing ist eine datenabhängige Verzweigung; offengelegtes Routing leakt (bis 91 % Token-Rekonstruktion, arXiv 2602.04105); dichte Auswertung aller Experten kostet bei CryptoMoE 8–15× | **Slot-MoE**: alle Experten dicht in den freien Slots (Breite ≤ 8.192 kostet eine Matvec, §4.1), weiches Gating im Reserve-Level (§4.5) | +1 Matvec, +0–1 ct×ct, **+0 Level**, kein Leakage | ✅ als dichtes GLU-Gate (§6.4: 100 % statt 99,75 %); MoE-Spezialfall (blockweise konstantes Gate) 📐 |
| **GLU / SwiGLU** (Shazeer 2020, LLaMA) | Swish/Sigmoid nicht polynomiell | bilineares GLU (xW₁)⊙(xW_g) bzw. Grad-4-Variante ((xW₁)⊙(xW_g))² im Reserve-Level | +64 PtMults, +3 Rotationen, +1 ct×ct; 0,342 s/Token | ✅ |
| **RoPE / ALiBi / NoPE** | RoPE braucht positionsabhängige Plaintexts (dynamische Encodes) | relativer Bias ρ_{t−j} im schiebenden Cache = ein statischer Plaintext; NoPE-Eingabe | 1 Addition | ✅ |
| **Position Coupling / Abacus** (Arithmetik) | Positions-IDs müssten vom Server berechnet werden | **Client-seitige Struktur-Annotation**: Der Client kennt den Klartext und schreibt Stellenwertigkeiten in freie Input-Dimensionen (Two-Hot) | 0 (Layer 0 ist linear in der Eingabe) | ✅ |
| **Umgekehrte Ziffern** (Lee et al. 2023) | – | Client-Tokenisierung, für Nutzer unsichtbar | 0 | ✅ |
| **Parallel Blocks / ReZero** (GPT-J, PaLM) | Tiefe pro Layer | Attention ∥ MLP teilen sich die 4 Level; Gains gefaltet | – | ✅ (aus v1) |
| **Speculative Decoding / MTP** | k Entwürfe = k Query-Positionen; akzeptierte Länge leakt (>90 % Query-Fingerprinting laut arXiv 2411.01076) | **Client-seitiges Drafting** (Client hat den Klartext, vgl. POST/ICML 2025 für 2PC); Server verifiziert k Positionen in blockdiagonalem Layout; Cache-Commit mit fester Rate gegen Leakage | ≈ k× Attention-Kosten, Matvecs teilen Baby-Steps | 📐 |
| **RL mit verifizierbaren Belohnungen** (GRPO, R1, L1/LCPO) | Training läuft im Klartext beim Modellbesitzer – kein FHE-Problem | **Token-kostenbewusstes RL**: Belohnung = korrekt − λ·Tokens (jedes Token kostet 0,36 s und 2,4 MB Upload); **FHE-in-the-loop**: Belohnung unter simuliertem CKKS-Rauschen und Wertebereichs-Strafe | 0 zur Laufzeit; kürzere Antworten sparen linear | 📐 |
| **Distillation** (MiniLLM, Rho et al.) | Lehrer-Architektur nicht FHE-tauglich | Lehrer-LLM → Schüler mit polynomieller Attention, bilinearen Scores, Slot-MLP | – | 📐 (Hauptpfad zur Chat-Fähigkeit) |
| **Quantisierung** (weniger Bits pro Gewicht) | In RNS-CKKS belegt jeder Limb ein 64-Bit-Wort, egal wie viele Bits das Modul hat | Gemessen: Skalierung 2²⁶ statt 2³³ spart **nichts** (gleiche 9 Limbs, 0,40 s, 225 MB), der Fehler steigt 150× (1,6·10⁻²); N = 2¹³ lehnt OpenFHE bei Tiefe 8 ab (HE-Standard). Was überträgt, ist **Tiefen-Quantisierung**: weniger Level pro Layer (Reserve-Level, Lookup-Layer 0, keine Masken-Level); dnum = 3 ist optimal (dnum = 2 erzwingt N = 2¹⁵ und 368 MB Keys, dnum = 9 → 614 MB Keys) | siehe §6.3 | ✅ |
| **SSMs / lineare Attention** (Mamba, RWKV, RetNet): konstante Kosten pro Token | Rekurrenz mit ct×ct-Updates verbraucht pro Token ein Level ⇒ Bootstrapping | Nicht nötig: Der in **einem** Ciphertext gepackte KV-Cache hat bereits konstante Kosten pro Token (bis 64 Tokens); additive lineare Attention (S_t = S_{t−1} + k_t v_tᵀ) wäre tiefenneutral, der Readout aber teurer | – | 📐 |

Drei verallgemeinerbare Lehren:

1. **Unter CKKS sind Parameter billig, Rotationen und Level teuer.** Die Motivation von MoE (Sparsity) verschwindet, solange die Breite in die Slots passt; was überträgt, ist das *multiplikative Mischen* (Gating) und die *Kapazität ohne Kosten*.
2. **Alles, was der Client aus dem Klartext ableiten kann, ist gratis und privat.** Tokenisierung, Struktur-Tags (Stellenwertigkeit, Satzgrenzen, Entitätstypen), Entwürfe für Speculative Decoding: Der Client ist ein vertrauenswürdiger Präprozessor, dessen Ausgabe verschlüsselt in Layer 0 einfließt, und Layer 0 ist linear.
3. **Reserve-Level sind eine Ressource.** Wenn parallele Pfade unterschiedlich tief sind, kann der kürzere Pfad Grad-4-Interaktionen (Gates, Experten-Mischung) ohne zusätzliche Tiefe aufnehmen.

## 8. Sicherheit und Leakage

- Das Bedrohungsmodell bleibt wie in v1 (semi-honest Server, 128-Bit-Sicherheit nach HE-Standard, N = 2¹⁴). Der Server sieht weiterhin nur Ciphertexts, Eval-Keys und Längen/Zeitpunkte.
- Der Signifikanz-Kanal (Position Coupling) liegt **innerhalb** des verschlüsselten Inputs; die Ziffernumkehr passiert lokal. Beides leakt nichts zusätzlich.
- Der schiebende Cache nutzt nur die öffentliche Position t; die Rotationen sind datenunabhängig.
- Unverändert offen: Längen-Leakage (Abhilfe: Padding/Dummy-Tokens), volle Logits beim Client (Modell-IP; Abhilfe: homomorphes Top-k kostet Tiefe), CKKS-IND-CPA^D (Li & Micciancio 2021) – der Client sendet nie entschlüsselte Werte zurück, nur neu verschlüsselte Tokens.
- Speculative Decoding würde ohne Gegenmaßnahme die Akzeptanzlänge offenlegen (§7).

## 9. Roadmap zur Chat-Fähigkeit („GPT-4-Klasse kommt später“)

Was skaliert aus v2 mit:

- **Breite.** Für d = 64·b (b Blöcke) wird jede Matvec zur b×b-Blockmatvec (PtMults ∝ d², Rotationen ∝ √d mit BSGS). Slot-Sättigung bleibt anwendbar, solange die Ausgaben in 8.192 (bzw. bei N = 2¹⁶: 32.768) Slots passen.
- **Tiefe.** 4 Level pro Layer; ab 3 Layern N = 2¹⁵/2¹⁶ oder Bootstrapping (CPU: Sekunden; GPU: ~0,1 s mit FIDESlib/Cheddar-Klasse). Die Reserve-Level-Idee reduziert den Bedarf, weil MLP-Pfade nicht verlängert werden müssen.
- **Vokabular.** Ausgabe: Logits über 32k Tokens = 4 Ciphertexts (eine Slot-gesättigte Matvec pro Ciphertext). Eingabe: **faktorisiertes Multi-Hot** (Token-ID = (hi, lo) mit je ≤ 181 Werten) passt in einen Ciphertext; Layer 0 bleibt linear.
- **Hardware.** FIDESlib/Phantom: ~1,1 ms pro Rotation bei N = 2¹⁶, 30 Limbs auf einer RTX 4090. Hochgerechnet auf unsere Parameter (N = 2¹⁴, 9 Limbs) **~10–20 ms pro Token** auf einer RTX-4090/5080-Klasse – eine Projektion, nicht gemessen.
- **Modell.** Distillation eines 10–50M-Parameter-Chatmodells (deutsch, TinyStories/Phi-artige Daten) in die v2-Architektur; Ziel: 4–6 Layer, d = 256–512, Bootstrapping auf GPU. Zum Vergleich: Llama-3-8B unter CKKS braucht heute 18 s/Token auf 8 GPUs (Park et al. 2026).

Nächste konkrete Schritte: (1) GPU-Backend für die bestehende Engine-Schnittstelle (`OpenFHEBackend` → FIDESlib), (2) mehrere Sessions gleichzeitig, (3) Batched Prefill über das blockdiagonale Layout, (4) Distillation eines größeren Chatmodells, (5) Speculative Decoding mit Leakage-freiem Commit.

## 10. Fazit

Beide Ziele dieser Runde sind erreicht: **0,365–0,40 s pro Token** auf einem 8-Kern-Desktop-CPU (Ziel ≤ 1 s; 0,75 s sogar auf einem einzigen Kern) und **fehlerfreie Addition auf 2.020 zurückgehaltenen Aufgaben** – unter echter Verschlüsselung identisch zur Klartext-Rechnung. Die Fortschritte kommen nicht aus mehr Rechenleistung, sondern aus Co-Design: Das Modell wird so gebaut, dass es die Struktur von CKKS nutzt (Slots, Level, statische Plaintexts), und die Aufgabe so tokenisiert, dass ein 2-Layer-1-Head-Modell sie algorithmisch lösen kann. Drei Ergebnisse halten wir für verallgemeinerbar: (1) Unter CKKS ist Modellbreite bis zur Slot-Grenze gratis – das ändert, welche LLM-Konzepte sich lohnen (Gating ja, Sparsity nein). (2) Der Client ist ein vertrauenswürdiger Präprozessor: Alles, was er aus dem Klartext ableitet, fließt verschlüsselt und kostenlos in Layer 0 ein. (3) Ungenutzte Level paralleler Pfade sind eine Ressource für höhere Polynomgrade ohne zusätzliche Tiefe. Der nächste Hebel ist Hardware (GPU, Projektion ~10–20 ms/Token) und ein per Distillation gewonnenes größeres Chatmodell.

## Literatur

FHE und private Inferenz
- Cheon, Kim, Kim, Song: Homomorphic Encryption for Arithmetic of Approximate Numbers (CKKS), ASIACRYPT 2017 – https://eprint.iacr.org/2016/421
- Halevi, Shoup: Algorithms in HElib, CRYPTO 2014 – https://eprint.iacr.org/2014/106 · Faster Homomorphic Linear Transformations in HElib, CRYPTO 2018 – https://eprint.iacr.org/2018/244
- Bossuat, Mouchet, Troncoso-Pastoriza, Hubaux: Efficient Bootstrapping for Approximate HE with Non-Sparse Keys (Double Hoisting), EUROCRYPT 2021 – https://eprint.iacr.org/2020/1203
- Juvekar, Vaikuntanathan, Chandrakasan: GAZELLE, USENIX Security 2018 – https://arxiv.org/abs/1801.05507
- Al Badawi et al.: OpenFHE, WAHC 2022 – https://eprint.iacr.org/2022/915
- Zhang et al.: NEXUS – Secure Transformer Inference Made Non-interactive, NDSS 2025 – https://eprint.iacr.org/2024/136
- Moon et al.: THOR, ACM CCS 2025 – https://dl.acm.org/doi/10.1145/3719027.3765150
- Zhang et al.: MOAI – https://eprint.iacr.org/2025/991
- Powerformer, ACL 2025 – https://aclanthology.org/2025.acl-long.543
- Rho et al.: Encryption-Friendly LLM Architecture, ICLR 2025 – https://arxiv.org/abs/2410.02486
- Yu et al.: Cachemir – FHE Inference of Generative LLMs with KV Cache, 2026 – https://arxiv.org/abs/2602.11470
- Park, Stehlé, Cheon et al.: Scaling up FHE-based PPML (Llama-3-8B), 2026 – https://arxiv.org/abs/2601.18511
- Al Badawi, Alexandru, Polyakov, Vaikuntanathan: SoK – Private LLM Inference using Approximate HE, 2026 – https://eprint.iacr.org/2026/935
- Agulló-Domingo et al.: FIDESlib – GPU CKKS, 2025 – https://arxiv.org/abs/2507.04775 · Phantom (Yang et al., IEEE TDSC) – https://eprint.iacr.org/2023/049
- Li, Micciancio: On the Security of HE on Approximate Numbers (IND-CPA^D), EUROCRYPT 2021 – https://eprint.iacr.org/2020/1533
- Zhou et al.: CryptoMoE – Private MoE Inference via Balanced Expert Routing, NeurIPS 2025 – https://arxiv.org/abs/2511.01197
- Nuriyev, Kulp: Routing leakage in MoE, 2026 – https://arxiv.org/abs/2602.04105
- POST – An Efficient Private GPT Never Autoregressively Decodes, ICML 2025 – https://arxiv.org/abs/2505.15252
- Privacy Risks of Speculative Decoding in LLMs, 2024 – https://arxiv.org/abs/2411.01076

LLM-Architektur, Decoding, Training
- Shazeer: GLU Variants Improve Transformer, 2020 – https://arxiv.org/abs/2002.05202
- So et al.: Primer (Squared ReLU), NeurIPS 2021 – https://arxiv.org/abs/2109.08668
- Bachlechner et al.: ReZero is All You Need, 2020 – https://arxiv.org/abs/2003.04887
- Chowdhery et al.: PaLM (Parallel Blocks), 2022 – https://arxiv.org/abs/2204.02311
- Press, Smith, Lewis: ALiBi, ICLR 2022 – https://arxiv.org/abs/2108.12409 · Kazemnejad et al.: NoPE / Positional Encoding and Length Generalization, NeurIPS 2023 – https://arxiv.org/abs/2305.19466
- Kacham, Mirrokni, Zhong: PolySketchFormer, ICML 2024 – https://arxiv.org/abs/2310.01655 · Katharopoulos et al.: Transformers are RNNs, ICML 2020 – https://arxiv.org/abs/2006.16236
- Fedus, Zoph, Shazeer: Switch Transformers, JMLR 2022 – https://arxiv.org/abs/2101.03961 · Jiang et al.: Mixtral, 2024 – https://arxiv.org/abs/2401.04088 · Dai et al.: DeepSeekMoE, 2024 – https://arxiv.org/abs/2401.06066 · Puigcerver et al.: Soft MoE, ICLR 2024 – https://arxiv.org/abs/2308.00951
- Leviathan, Kalman, Matias: Speculative Decoding, ICML 2023 – https://arxiv.org/abs/2211.17192 · Gloeckle et al.: Multi-Token Prediction, ICML 2024 – https://arxiv.org/abs/2404.19737 · Cai et al.: Medusa, 2024 – https://arxiv.org/abs/2401.10774
- Shao et al.: DeepSeekMath / GRPO, 2024 – https://arxiv.org/abs/2402.03300 · DeepSeek-AI: DeepSeek-R1, 2025 – https://arxiv.org/abs/2501.12948 · Aggarwal, Welleck: L1 / LCPO, COLM 2025 – https://arxiv.org/abs/2503.04697
- Gu et al.: MiniLLM, ICLR 2024 – https://arxiv.org/abs/2306.08543 · Eldan, Li: TinyStories, 2023 – https://arxiv.org/abs/2305.07759 · Gu, Dao: Mamba, 2023 – https://arxiv.org/abs/2312.00752

Arithmetik in kleinen Transformern
- Lee et al.: Teaching Arithmetic to Small Transformers, ICLR 2024 – https://arxiv.org/abs/2307.03381
- Cho et al.: Position Coupling, NeurIPS 2024 – https://arxiv.org/abs/2405.20671
- McLeish et al.: Transformers Can Do Arithmetic with the Right Embeddings (Abacus), NeurIPS 2024 – https://arxiv.org/abs/2405.17399
- Nanda et al.: Progress Measures for Grokking via Mechanistic Interpretability, ICLR 2023 – https://arxiv.org/abs/2301.05217
- Zhou et al.: What Algorithms Can Transformers Learn? (RASP-L), ICLR 2024 – https://arxiv.org/abs/2310.16028

# CipherChat v2: Slot-gesättigte CKKS-Inferenz und generalisierende Arithmetik für einen verschlüsselten Chatbot

Stand 20.11.2026 (Abschnitt 9 erweitert) · Code: `cipherchat/v2/`, Messlog: `docs/RESEARCH_LOG.md` · Messrechner: Intel Core Ultra 9 285K (8 P- + 16 E-Kerne), RTX 5080, OpenFHE 1.6.1

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

## 9. Der Weg zur GPT-4-Klasse: Was skaliert, was bricht, was hilft

**Leitthese.** Unter FHE skaliert der Rechenaufwand wie im Klartext – linear in den ausgewerteten Parametern –, nur mit einem Faktor von etwa 10³–10⁴. Daran allein scheitert der Weg zur GPT-4-Klasse nicht. Kritisch sind Dinge, die im Klartext keine oder eine kleine Rolle spielen: der Speicher der verschlüsselungsbereiten Gewichte, Tiefe und Bootstrapping, die Nichtlinearitäten vortrainierter LLMs, Sparsity (MoE), die unter FHE nicht ohne Leakage funktioniert, und die Kommunikation pro Token. Dieser Abschnitt quantifiziert jedes Problem, ordnet die vorhandene Forschung ein und schlägt eigene Lösungsideen vor. Er ist als Arbeitsprogramm für Folgepaper gedacht; alle Zahlen außer Stufe A sind Schätzungen aus einem Kostenmodell (§9.2), nicht Messungen.

### 9.1 Größenordnungen

| Stufe | Modellform | Parameter ausgewertet / gespeichert | PtMults pro Token | Rotationen pro Token | Bootstraps pro Token | Plaintext-Gewichte | Zeit pro Token |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A (gemessen) | CipherChat v2: d = 64, 2 Layer, Hidden 2.048, 59 Zeichen | 0,43 M | 576 | 99 | 0 | 0,35 GB | 0,34 s auf 8 CPU-Kernen; GPU-Projektion ~10–20 ms |
| B | Domänen-Assistent: d = 256, 6 Layer, BPE 8k | 8 M | ~250 | ~600 | 3 | unter 1 GB | ~0,6 s (1 GPU) |
| C | GPT-2-small-Klasse: d = 768, 12 Layer, 32k Vokabular | 0,14 B | ~4.200 | ~4.900 | 5 | ~70 GB | ~5 s (1 GPU) |
| D | 1B (Form von Llama-3.2-1B) | 1,2 B | ~38.000 | ~9.900 | 6 | ~0,6 TB | ~11–14 s (1 GPU) |
| E | 8B (Form von Llama-3-8B) | 7,5 B | ~230.000 | ~25.000 | 12 | ~3,7 TB | ~32–55 s (1 GPU) |
| F1 | Frontier-MoE (Form von DeepSeek-V3), nur aktive Experten | 33 B / 671 B | ~1,0 Mio. | ~200.000 | 23 | ~330 TB | ~4–6 min (1 GPU) |
| F2 | dieselbe MoE, alle Experten dicht (kein Routing-Leakage) | 700 B / 671 B | ~21 Mio. | ~300.000 | 23 | ~330 TB | ~19–55 min (1 GPU) |

Einordnung: Für 8B berichtet die Literatur 18 s pro Token auf 8 GPUs (Park et al. 2026) und 97 s auf einer GPU (Cachemir 2026) – beide mit Approximationen von Softmax und LayerNorm, also nicht FHE-nativ. Die Größe von GPT-4 ist nicht offengelegt; offene Modelle dieser Klasse haben 10¹¹–10¹² Parameter (Llama-3.1-405B dicht, DeepSeek-V3 mit 671B gesamt und 37B aktiv). Von Stufe A bis Stufe F liegen also fünf bis sechs Größenordnungen.

Annahmen (`scripts/scaling_estimate.py`): N = 2¹⁶ mit 32.768 Slots; volle Slot-Auslastung in den linearen Schichten (Packing wie bei Cachemir oder unser generalisiertes BSGS); FHE-native Layer wie in v2 mit ≈ 4,5 Level pro Layer und ~12 nutzbaren Leveln zwischen zwei Bootstraps; GPU-Kosten pro Operation auf RTX-4090/5090-Niveau: Rotation 0,8 ms, ct×ct 1,0 ms, PtMult 0,04 ms, Bootstrapping 20 ms (FIDESlib; Theodosian misst 12,8–22 ms auf einer RTX 5090), Encode eines Gewichts-Plaintexts 0,1 ms. Die Spanne der Zeiten reicht von „alle Gewichts-Plaintexts im Speicher“ bis „Gewichte werden pro Token neu encodiert“.

### 9.2 Kostenmodell

$$
T_{\text{Token}} \approx R\,c_{\text{rot}} + C\,c_{\text{ct}} + P\,c_{\text{pt}} + B\,c_{\text{bts}} \;\big(+\,P\,c_{\text{enc}}\ \text{bei gestreamten Gewichten}\big)
$$

mit P ≈ W/n (ausgewertete Gewichte geteilt durch die Slot-Zahl), R ≈ Σ über alle Matvecs 2√(W_i/n) plus 12·L·⌈T·d_kv/n⌉ für die Attention, C ≈ 2·L·⌈T·d_kv/n⌉ und B ≈ ⌈L·ℓ/ℓ_bts⌉ (L Layer, ℓ Level pro Layer, ℓ_bts nutzbare Level pro Bootstrap, T Kontextlänge). Daraus folgen fünf Skalierungsgesetze:

1. **Plaintext-Multiplikationen wachsen linear in den Parametern** – genau wie Klartext-FLOPs; nur die Konstante unterscheidet sich.
2. **Rotationen wachsen mit der Wurzel der Matrixgröße** (BSGS). Bei kleinen Modellen dominieren Rotationen (v2: nach Kostenmodell etwa zwei Drittel der Zeit), bei großen wieder die Plaintext-Multiplikationen und der Gewichtsspeicher.
3. **Kontext ist gratis, solange der KV-Cache in einen Ciphertext passt** (v2: 64 Tokens; bei N = 2¹⁶ und d_kv = 64 bis 512 Tokens). Danach wächst er linear in T·d_kv/n – wie im Klartext.
4. **Bootstraps hängen an der Tiefe pro Layer.** FHE-native Layer brauchen ℓ ≈ 4–5 Level, vortrainierte LLMs mit Approximationen von Softmax, Norm und SiLU nach unserer Abschätzung 10–20. Das ist ein Faktor 2–4 bei den Bootstraps.
5. **Der Plaintext-Speicher wächst linear in den Parametern mit ~500 Byte pro Gewicht** (16 MB pro 32.768 Gewichte bei vollem Level, 250-mal fp16). In v2 messen wir ~800 Byte pro Parameter (346 MB für 428k Parameter); Cachemir berichtet für Llama-3-8B „hunderte GB“ Gewichts-Plaintexts gegenüber 17 GB KV-Cache.

### 9.3 Was gut skaliert

- **Breite bis zur Slot-Grenze ist gratis.** Gemessen: Hidden 1.024 → 8.192 kostet 12 % mehr Zeit. Kleine und mittlere Modelle sollten deshalb breit und flach sein.
- **Lineare Algebra skaliert asymptotisch wie im Klartext** (Gesetz 1). Mit vollem Packing (MOAI, Cachemir, unser generalisiertes BSGS) geht kaum Slot-Kapazität verloren.
- **Der gepackte KV-Cache** hat konstante Kosten pro Token, solange T·d_kv ≤ n.
- **FHE-native Bausteine sind trainierbar.** Polynomielle Attention ab Grad 4 erreicht in Sprachmodell-Experimenten die Qualität von Softmax (PolySketchFormer); Softmax-Attention lässt sich selbst bei Llama-3.1-405B per Attention-Transfer und LoRA in eine andere Attention-Form überführen und schließt dabei den größten Teil der MMLU-Lücke (LoLCATs); norm-freies Training mit ReZero-Gains und gegatetem MLP funktioniert (diese Arbeit, kleine Skala).
- **Hardware.** GPUs sind pro Rotation rund 95× schneller als OpenFHE mit 24 Threads und 335× schneller als ein CPU-Kern (FIDESlib, N = 2¹⁶), Bootstrapping kostet auf einer RTX 5090 nur noch 13–22 ms (Theodosian), FHE-ASICs wie Intel Heracles versprechen bis 5.000× gegenüber einem Xeon.
- **Alles auf dem Client skaliert kostenlos mit:** Tokenisierung, Struktur-Annotationen, Retrieval über eigene Daten, Werkzeuge.

### 9.4 Was bricht – Probleme, Stand der Forschung, eigene Ideen

Überblick:

| # | Problem | Größenordnung | Stand der Forschung | Eigene Idee |
| --- | --- | --- | --- | --- |
| P1 | Plaintext-Explosion der Gewichte | 3,7 TB (8B), 330 TB (Frontier) | Encode auf der GPU, Gewichte auf niedrigem Level | fusionierte Encode-Mult-Kernel auf quantisierten Gewichten, Gewichts-Batching über Nutzer, strukturierte Layer |
| P2 | Tiefe und Bootstrapping | 6–23 Bootstraps pro Token | GPU-Bootstrapping in 13–22 ms; Platzierung (Orion, Cachemir, AutoFHE) | level-balancierte Architekturen, Client-gestütztes Refresh |
| P3 | Nichtlinearitäten vortrainierter LLMs | 10–20 Level pro Layer, Ausreißer bis 10³ | Approximationen (MOAI, Park et al.), Ersatzkerne (Powerformer, Rho et al.) | Distillations-Curriculum in FHE-native Schüler, Rotationen gegen Ausreißer |
| P4 | Lineare Schichten, wenn d die Slots füllt | ~230.000 PtMults pro Token (8B) | Packing (THOR, MOAI, Cachemir), block-zirkulante Gewichte (PrivCirNet) | Monarch-Layer als FHE-native Linearschicht |
| P5 | MoE und Sparsity | dicht ~20× teurer; Routing verrät bis 91 % der Tokens | CryptoMoE (balanciertes Routing, HE+MPC) | Slot-MoE (klein), Sitzungs-Routing mit Leakage-Budget (groß) |
| P6 | Langer Kontext | linear in T·d_kv/n | KV-Packing (Cachemir), Klartext-Kompression (StreamingLLM) | Client als Kontext-Manager, öffentliche Präfixe als Klartext-KV |
| P7 | Vokabular, Sampling, Rundreisen | 128k Logits = 4 Ciphertexts, 1 Rundreise pro Token | Client-Argmax, homomorphes Argmax (NEXUS), Drafts bei 2PC (POST) | faktorisierter Multi-Hot-Input, Client-Drafts, weiche Token-Rückkopplung |
| P8 | Kommunikation und Schlüssel | Keys im GB-Bereich, 10–20 MB Upload pro Token | hierarchische Rotations-Keys, Transciphering (HERA, Rubato, AES→CKKS) | Transciphering des Two-Hot-Inputs, Keys aus dem statischen Level-Plan |
| P9 | Qualität FHE-nativer Modelle | ab ~100M unerforscht | PolySketchFormer, Rho et al., LoLCATs | Skalierungsgesetze messen, Client-Werkzeuge entlasten das Modell |
| P10 | Betrieb und Kosten | eigene Keys pro Nutzer, kein Slot-Batching über Nutzer | kaum Literatur | Gewichts-Batching als Hebel für €/Token |

#### P1 · Plaintext-Explosion der Gewichte

**Problem.** Jede Plaintext-Multiplikation braucht die Gewichte in RNS-NTT-Form auf dem Level des Ciphertexts: N × Limbs × 8 Byte. Bei N = 2¹⁶ und ~30 Limbs sind das 16 MB für 32.768 Gewichte. Ein 8B-Modell braucht so ~3,7 TB, eine Frontier-MoE ~330 TB; GPUs haben 16–80 GB. Die Alternative, die Gewichte kompakt zu halten und pro Token neu zu encodieren, kostet bei 8B rund 230.000 Encodes pro Token, also ~23 s zusätzlich (Annahme 0,1 ms pro Encode). Dieses Problem existiert im Klartext nicht: Dort belegt ein Gewicht 1–2 Byte.

**Stand der Forschung.** Cachemir misst das Problem (hunderte GB für Llama-3-8B) und erwartet, dass es bei größeren Modellen dominiert; Encoding auf der GPU und Gewichte auf dem niedrigsten nötigen Level mildern es. Eine grundsätzliche Lösung ist uns nicht bekannt.

**Eigene Ideen.**

- **Fusionierte Encode-Mult-Kernel.** Gewichte bleiben als 8-Bit-Werte im Slot-Raum gespeichert (1 Byte pro Gewicht, 8 GB für 8B), FFT und NTT laufen direkt im Multiplikations-Kernel. Speicher wie ein int8-LLM, Mehraufwand ≈ eine NTT pro Limb und Plaintext.
- **Gewichts-Batching über Nutzer.** Die Plaintexts sind für alle Nutzer gleich – nur die Ciphertexts unterscheiden sich. Werden B Nutzer gleichzeitig bedient, amortisieren sich Encodieren und Streamen über B, genau wie Klartext-LLMs ihre Gewichte pro Batch nur einmal laden. Das verschiebt FHE-Inferenz vom Latenz- ins Durchsatz-Regime und ist vermutlich der wichtigste Hebel für die Kosten pro Token.
- **Strukturierte Gewichte** (P4) reduzieren die Zahl der Gewichte um ~√d.
- **Linearschichten spät in der Bootstrapping-Periode.** Auf niedrigen Leveln haben Plaintexts wenige Limbs (v2: Layer-1-Plaintexts sind ~2× kleiner als Layer-0). Platzierungs-Algorithmen wie Orion oder Cachemir optimieren bisher die Bootstraps; der Gewichtsspeicher sollte ein zweites Optimierungsziel werden.

#### P2 · Tiefe und Bootstrapping

**Problem.** 4–5 Level pro FHE-nativem Layer (geschätzt 10–20 bei vortrainierten LLMs) bedeuten bei 12–60 Layern 6–23 Bootstraps pro Token; auf einer CPU kostet jedes Bootstrapping Sekunden.

**Stand der Forschung.** GPU-Bootstrapping in 13–22 ms (Theodosian, RTX 5090); optimierte Platzierung (Orion; Cachemir mit Bootstrapping innerhalb nichtlinearer Schichten, 1,98× weniger Bootstrapping-Latenz; AutoFHE optimiert Polynomgrad und Platzierung gemeinsam).

**Eigene Ideen.**

- **Level-balancierte Architekturen.** Unser Ergebnis zum Reserve-Level verallgemeinert sich zu einer Entwurfsregel: Alle parallelen Pfade eines Blocks verbrauchen dieselbe Tiefe; was ein Pfad nicht braucht, wird für höhere Polynomgrade genutzt (Gates, Experten-Mischung). Eine Architektursuche mit dem Level-Budget als harter Nebenbedingung, wie AutoFHE sie für CNNs macht, fehlt für Transformer.
- **Client-gestütztes Refresh statt Bootstrapping.** Der Server addiert eine zufällige Maske r (gleichverteilt modulo q) zum Ciphertext, der Client entschlüsselt x + r, verschlüsselt neu auf vollem Level und schickt zurück; der Server zieht r ab. Das ist die Zwei-Parteien-Variante des kollektiven Bootstrappings aus der Multiparty-HE (Mouchet et al., PoPETs 2021). Kosten: eine Rundreise und ein Ciphertext pro Refresh (MB-Bereich) statt eines Bootstrappings. Anders als hybride HE/MPC-Protokolle (BOLT braucht 59,6 GB Kommunikation für eine BERT-base-Inferenz) bleiben die Nichtlinearitäten polynomiell; erneuert wird nur das Rauschbudget. Das lohnt sich, wo Server-Rechenzeit teuer ist (CPU-Server) oder viele Bootstraps anfallen. Offen sind die Sicherheitsanalyse (Maskenverteilung, Präzision, Bezug zu IND-CPA^D) und die Latenz bei vielen Refreshes pro Token.
- **Bootstrapping mit Nutzlast.** CKKS-Bootstrapping enthält selbst zwei lineare Transformationen (CoeffToSlot, SlotToCoeff). Lineare Modellschichten in diese Transformationen hineinzufalten, spart Matvecs und Level – zu prüfen.

#### P3 · Nichtlinearitäten vortrainierter LLMs

**Problem.** Softmax (Exponentialfunktion und Division), RMSNorm und LayerNorm (inverse Wurzel), SiLU und GELU brauchen hochgradige Polynome. Aktivierungsausreißer von 10³–10⁴ vergrößern den Approximationsbereich und damit Grad und Tiefe.

**Stand der Forschung.** Hochgradige Approximationen mit Bereichs-Tricks (MOAI, THOR; Park et al. mit Token-Prepending und orthogonalen Rotationen gegen Ausreißer); Ersatzkerne mit Nachtraining (Powerformer mit BRPmax, Rho et al. mit Gauß-Kernel und LoRA).

**Eigene Ideen.**

- **„FHE-ification“ als Distillations-Curriculum.** Komponenten schrittweise ersetzen: Softmax durch polynomielle Attention (Attention-Transfer wie bei LoLCATs), Norm durch ReZero-Gains und feste Skala, SiLU-GLU durch ((xW₁)⊙(xW_g))², relative Biases statt RoPE – und nach jedem Schritt mit LoRA nachtrainieren. LoLCATs zeigt, dass solche Umbauten der Attention bis 405B Parameter funktionieren.
- **Rotationen gegen Ausreißer** aus der Quantisierung (QuaRot, SpinQuant: Hadamard- bzw. gelernte Rotationen glätten die Aktivierungen) verkleinern den Wertebereich und damit Polynomgrad und Präzisionsbedarf. Rotationen sind linear und lassen sich in die Gewichte falten – unter FHE also gratis.
- **Wertebereichs-Regularisierung im Training**, wie unser Logit-Penalty: Kleine Aktivierungen (v2: max |x| = 19) senken den CKKS-Fehler und erlauben kleinere Skalierungen.

#### P4 · Lineare Schichten, wenn die Slots voll sind

**Problem.** Ab d ≈ n/2 ist Breite nicht mehr gratis. Ein 8B-Modell braucht ~230.000 Plaintext-Multiplikationen pro Token.

**Stand der Forschung.** Optimale Packings (THOR, MOAI, Cachemir); block-zirkulante Gewichte (PrivCirNet, NeurIPS 2024: die Matvec wird zur eindimensionalen Faltung, der HE-Aufwand sinkt proportional zur Blockgröße, 1,3–5× schneller als BOLT); SVD-Kompression (Euston – mit Leakage laut „Breaking Euston“).

**Eigene Idee: Monarch-Layer als FHE-native Linearschicht.** Eine Monarch-Matrix M = P L Pᵀ R besteht aus zwei blockdiagonalen Faktoren mit √d Blöcken der Größe √d × √d und einer festen Permutation P (Dao et al., ICML 2022). Blockdiagonale Faktoren passen ideal zu CKKS: Alle Blöcke laufen SIMD-parallel in einer generalisierten BSGS-Matvec (wie unsere Slot-Sättigung), die Permutation ist ein festes, datenunabhängiges Rotationsmuster. Erwartung: Plaintext-Multiplikationen und Gewichtsspeicher sinken von d² auf 2d√d – bei d = 4.096 um den Faktor 32 –, bei höchstens einem zusätzlichen Level. Offen ist, wie viel Qualität strukturierte Gewichte auf LLM-Skala kosten; Monarch und block-zirkulante Gewichte sind bisher vor allem bis GPT-2- und ViT-Größe untersucht.

#### P5 · Mixture-of-Experts und Sparsity

**Problem.** MoE spart nur, wenn wenige Experten ausgewertet werden, und die Auswahl ist eine Verzweigung auf verschlüsselten Daten. Dicht ausgewertet ist eine Frontier-MoE ~20× teurer (Stufe F2 gegenüber F1); ein offengelegtes Routing verrät bis zu 91 % der Tokens (Nuriyev & Kulp 2026).

**Stand der Forschung.** CryptoMoE (NeurIPS 2025) kombiniert HE und MPC und balanciert das Routing so, dass es weniger verrät; die dichte Referenz ist dort 8–15× teurer.

**Eigene Ideen.**

- **Slot-MoE** für kleine Modelle (diese Arbeit): Experten dicht in freien Slots, Gate im Reserve-Level – kein Leakage, bis zur Slot-Grenze keine Mehrkosten.
- **Sitzungs-Routing mit Leakage-Budget** für große Modelle. Der Client wählt pro Sitzung (nicht pro Token) einen von wenigen großen Domänen-Experten, z. B. einen von 8 – also 3 Bit pro Sitzung, vergleichbar mit der Server-Namensangabe (SNI) bei TLS. Das Leakage ist klein, explizit und prüfbar; innerhalb der Domäne wird dicht gerechnet. Offen: wie viel Qualität Sitzungs-Routing gegenüber Token-Routing kostet und wie man Domänen schneidet, sodass die Wahl wenig verrät (k-Anonymität).

#### P6 · Langer Kontext

**Problem.** Ab T·d_kv > n wächst der Aufwand linear. Ein 128k-Kontext eines 8B-Modells (d_kv = 1.024) wären 4.000 Ciphertexts pro Layer und Cache.

**Stand der Forschung.** KV-Cache-Packing mit variabler Länge (Cachemir); im Klartext Kompression wie Attention-Sinks (StreamingLLM) und Token-Auswahl (H2O).

**Eigene Ideen.**

- **Der Client als Kontext-Manager.** Der Client hat die ganze Konversation und seine Dokumente im Klartext. Statt eines riesigen verschlüsselten Caches schickt er pro Anfrage nur die relevanten Teile (eigenes Retrieval, eigene Zusammenfassung). Die Kontextlänge wird eine Client-Entscheidung statt einer Kostenexplosion beim Server.
- **Öffentliche Präfixe als Klartext-KV.** System-Prompt, Few-Shot-Beispiele oder öffentliche Dokumente kennt der Server ohnehin. Er kann ihre Keys und Values im Klartext berechnen und halten; Scores dagegen sind ct×pt-Produkte (ohne Relinearisierung, nach unseren Kostenannahmen ~25× billiger als ct×ct), und das Prefill dieser Tokens kostet kein FHE. Nur die privaten Tokens laufen verschlüsselt.
- **Hybride Langkontext-Layer:** Fenster-Attention über die letzten Tokens plus additive lineare Attention, deren Zustandsupdate S_t = S_{t−1} + φ(k_t)v_tᵀ tiefenneutral ist (nur Additionen pro Schritt).

#### P7 · Vokabular, Sampling und Rundreisen

**Problem.** Ein 128k-Vokabular passt als One-Hot nicht in einen Ciphertext (32.768 Slots), die Ausgabe sind 4 Logit-Ciphertexts, und pro Token ist eine Rundreise nötig (Client entschlüsselt, sampelt, verschlüsselt neu).

**Stand der Forschung.** Argmax beim Client (unser Protokoll), homomorphes Argmax (NEXUS), Speculative Decoding mit öffentlichem Draft-Modell bei 2PC (POST, ICML 2025).

**Eigene Ideen.**

- **Faktorisierter Multi-Hot-Input.** Die Token-ID wird als (hi, lo) mit je √V Werten kodiert (2 × 358 für 128k) und passt in einen Ciphertext; Layer 0 bleibt linear (E[tok] = E_hi[hi] + E_lo[lo]). Alternativ sendet der Client direkt die Embedding-Zeile, wenn das Embedding öffentlich sein darf.
- **Speculative Decoding mit Client-Drafts.** Der Client entwirft k Tokens mit einem kleinen lokalen Modell oder einem n-Gramm-Lookup – privat und gratis –, der Server prüft sie in einem Schritt. Gegen das Leakage der Akzeptanzlänge hilft eine feste Commit-Rate oder ein Cache-Update mit verschlüsselter Maske.
- **Weiche Token-Rückkopplung** (Forschungsfrage): Statt eines Argmax speist der Server eine polynomiell geschärfte Verteilung als „weiches“ Embedding zurück. Das spart die Rundreise, verlangt aber ein Training auf weiche Eingaben.

#### P8 · Kommunikation und Schlüssel

**Problem.** Mit Bootstrapping bei N = 2¹⁶ umfassen die Eval-Keys mehrere GB pro Nutzer (ein ResNet-18-Keyset hat 115,7 GB), jeder Token-Upload ist ein voller Ciphertext (~10–20 MB).

**Stand der Forschung.** Hierarchische Rotations-Keys (Lee et al., ASIACRYPT 2023: 115,7 GB → 2,91 GB, der Server leitet die übrigen Keys selbst ab); Transciphering mit HE-freundlichen Chiffren (HERA/RtF, ASIACRYPT 2021; Rubato, EUROCRYPT 2022) oder mit AES (Bae et al. 2025: unter 1 ms pro Block im Durchsatz).

**Eigene Ideen.**

- **Transciphering des Two-Hot-Inputs.** Der Input pro Token ist winzig (ein Token plus Struktur-Tag); mit Rubato verschlüsselt sind das Bytes statt Megabytes, der Server transcipheriert in CKKS.
- **Keys aus dem statischen Level-Plan ableiten.** Weil der Rechenplan in v2 statisch ist (alle Plaintexts sind fest), kennt der Server vorab jeden Rotationsschritt und jedes Level. Rotations-Keys werden nur für die Level ausgestellt, auf denen sie gebraucht werden; Keys auf niedrigen Leveln sind deutlich kleiner.

#### P9 · Qualität FHE-nativer Modelle

**Problem.** Polynomielle Attention, fehlendes LayerNorm und quadratische bzw. GLU²-Aktivierungen sind auf großer Skala kaum untersucht; unser Nachweis gilt für 0,4M Parameter.

**Stand der Forschung.** Polynomielle Attention ab Grad 4 erreicht in Sprachmodell-Experimenten die Qualität von Softmax (PolySketchFormer); FHE-freundliches Fine-Tuning mit Gauß-Kernel (Rho et al.); Attention-Umbau bis 405B (LoLCATs); kleine Modelle mit kuratierten Daten erreichen auf Benchmarks GPT-3.5-Niveau (Phi-3-mini, 3,8B Parameter).

**Eigene Ideen.**

- **Skalierungsgesetze FHE-nativer Architekturen messen**: 10M–1B Parameter, gleiche Daten wie ein Referenz-Transformer, Perplexity und Aufgaben-Benchmarks.
- **Client-Werkzeuge entlasten das Modell.** Exakte Fähigkeiten wie Rechnen mit langen Zahlen, Datumsrechnung oder Nachschlagen in eigenen Daten muss das Modell nicht selbst lernen. Es erzeugt verschlüsselt einen Werkzeugaufruf, den der Client lokal ausführt (Toolformer-Prinzip). Unter FHE ist das besonders natürlich, weil der Client ohnehin jedes Token sieht, und es verrät dem Server nichts.

#### P10 · Betrieb und Kosten

**Problem.** Jeder Nutzer hat eigene Schlüssel; Ciphertexts verschiedener Nutzer lassen sich nicht in denselben Slots mischen. Pro Sitzung belegen Keys (GB) und KV-Cache GPU-Speicher.

**Stand der Forschung.** Kaum Literatur zum Mehrnutzer-Betrieb; die SoK von Duality schlägt eine „Private LLM Card“ als Berichtsstandard vor.

**Eigene Ideen.** Gewichts-Batching (P1) als Hebel für die Kosten pro Token; Keys im Host-Speicher und Layer-weises Streaming auf die GPU; Kosten pro 1.000 Tokens als Standardgröße in Veröffentlichungen.

**Sicherheit im Großen.** Jede der Ideen oben, die mehr Interaktion bringt (Refresh, Speculative Decoding, Sitzungs-Routing), öffnet eine Leakage-Fläche und braucht eine eigene Analyse. IND-CPA^D bleibt relevant, sobald entschlüsselte Werte – auch maskiert – zurückfließen.

### 9.5 Gestufte Roadmap

| Stufe | Ziel | Kernschritte | Meilenstein bzw. Abbruchkriterium |
| --- | --- | --- | --- |
| B (ca. 3 Monate) | Domänen-Assistent, 5–20M Parameter | GPU-Backend (FIDESlib), Bootstrapping oder Client-Refresh, BPE mit 4–8k Tokens, Distillation aus einem kleinen LLM | höchstens 1 s pro Token auf einer GPU; Exact Match auf Zielaufgaben höchstens 2 Prozentpunkte unter dem Klartext-Modell |
| C (6–9 Monate) | GPT-2-small-Klasse | Monarch-Layer, Distillations-Curriculum, Client als Kontext-Manager | höchstens 5 s pro Token; Perplexity höchstens 10 % über der Klartext-Referenz |
| D (12–18 Monate) | 1–3B, „GPT-3.5-light“ | Gewichts-Streaming mit fusionierten Kerneln, Gewichts-Batching über Nutzer, mehrere GPUs | höchstens 15 s pro Token; Kosten pro 1.000 Tokens beziffert |
| E (18–36 Monate) | 8B und mehr | FHE-Beschleuniger (Heracles-Klasse), Speculative Decoding mit Client-Drafts | Niveau von Park et al. (18 s pro Token auf 8 GPUs) auf einer GPU |
| F (mehrjährig) | GPT-4-Klasse | ~10³-mal schnellere Hardware, √d-Gewinne durch Struktur, Sitzungs-Routing mit Leakage-Budget | – |

Bis Stufe E erreicht ist, bleibt der Hybrid aus der SSOT pragmatisch: TEE für große Modelle, FHE für die sensibelsten Aufgaben. Hinzu kommt Client-seitige Vorverarbeitung (Schwärzen von Personendaten, Retrieval) vor jedem großen Modell.

### 9.6 Offene Forschungsfragen für Folgepaper

1. **Skalierungsgesetze FHE-nativer Transformer** (polynomielle Attention, bilineare Scores, norm-frei, GLU²) von 10M bis 1B Parametern gegenüber Standard-Transformern bei gleichen Daten.
2. **Distillation vortrainierter LLMs in FHE-native Schüler:** Welche Komponente lässt sich mit welchem Verlust ersetzen, und wie viel LoRA-Nachtraining ist nötig?
3. **Strukturierte Linearschichten unter CKKS** (Monarch, block-zirkulant, Kronecker, low-rank plus sparse): gemessene Pareto-Front aus Kosten und Qualität bei 100M–1B Parametern.
4. **Client-gestütztes Refresh:** formale Sicherheitsanalyse (Maskenverteilung, Präzision, IND-CPA^D) und Latenz-/Kostenmodell gegenüber GPU-Bootstrapping.
5. **Plaintext-Speicher:** fusionierte Encode-Mult-Kernel, quantisierte Slot-Speicherung und Gewichts-Batching – Durchsatz pro GPU in Tokens pro Sekunde über alle Nutzer.
6. **MoE unter FHE:** informationstheoretische Leakage-Schranken für Sitzungs-Routing und Qualität von Domänen-Experten.
7. **Öffentliche Präfixe als Klartext-KV:** Wie viel FHE-Arbeit spart das bei typischen Chat-Workloads mit System-Prompts und RAG?
8. **Speculative Decoding und Multi-Token-Prediction** mit leakage-freiem Commit.
9. **Weiche Token-Rückkopplung** für eine Generierung ohne Rundreisen.
10. **Benchmark- und Berichtsstandard** mit Latenz, Durchsatz, Speicher, Kommunikation, Kosten pro 1.000 Tokens und Leakage – anknüpfend an die „Private LLM Card“ der SoK.

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

Skalierung und Weg zur GPT-4-Klasse (Abschnitt 9)
- Theodosian – GPU-Bootstrapping in 12,8–22 ms auf RTX 5090, 2025 – https://arxiv.org/abs/2512.18345
- Cheddar – Swift FHE Library for CUDA GPUs, 2024/25 – https://arxiv.org/abs/2407.13055
- Ebel, Garimella, Reagen: Orion – FHE Framework for Deep Learning, ASPLOS 2025 – https://arxiv.org/abs/2311.03470
- Ao, Boddeti: AutoFHE, USENIX Security 2024 – https://arxiv.org/abs/2310.08012
- Mouchet, Troncoso-Pastoriza, Bossuat, Hubaux: Multiparty HE from RLWE (kollektives Bootstrapping), PoPETs 2021 – https://eprint.iacr.org/2020/304
- Pang et al.: BOLT, IEEE S&P 2024 – https://eprint.iacr.org/2023/1893 · BLB – Breaking the Layer Barrier, 2025 – https://arxiv.org/abs/2508.19525
- Xu, Wu, Wang, Li: PrivCirNet – Block Circulant Transformation, NeurIPS 2024 – https://arxiv.org/abs/2405.14569
- Dao et al.: Monarch – Expressive Structured Matrices, ICML 2022 – https://arxiv.org/abs/2204.00595
- Lee, Lee, Kim, No: Hierarchical Rotation Keys, ASIACRYPT 2023 – https://eprint.iacr.org/2022/532
- Cho et al.: Transciphering Framework (RtF/HERA), ASIACRYPT 2021 – https://eprint.iacr.org/2020/1335 · Ha et al.: Rubato, EUROCRYPT 2022 – https://iacr.org/archive/eurocrypt2022/132760256/132760256.pdf
- Bae, Cheon, Kang, Kim: High-Throughput AES Transciphering using CKKS, 2025 – https://eprint.iacr.org/2025/1865
- Ashkboos et al.: QuaRot, NeurIPS 2024 – https://arxiv.org/abs/2404.00456 · Liu et al.: SpinQuant, ICLR 2025 – https://arxiv.org/abs/2405.16406
- Zhang et al.: LoLCATs – Low-Rank Linearizing of LLMs, ICLR 2025 – https://arxiv.org/abs/2410.10254
- Xiao et al.: StreamingLLM (Attention Sinks), ICLR 2024 – https://arxiv.org/abs/2309.17453
- Schick et al.: Toolformer, NeurIPS 2023 – https://arxiv.org/abs/2302.04761
- OpenAI: GPT-4 Technical Report, 2023 – https://arxiv.org/abs/2303.08774 · Llama Team: The Llama 3 Herd of Models, 2024 – https://arxiv.org/abs/2407.21783 · DeepSeek-AI: DeepSeek-V3, 2024 – https://arxiv.org/abs/2412.19437 · Abdin et al.: Phi-3, 2024 – https://arxiv.org/abs/2404.14219
- Gao et al.: Euston, IEEE S&P 2026 – https://eprint.iacr.org/2026/046 · Breaking Euston – https://arxiv.org/abs/2604.17238
- IEEE Spectrum: Intel Heracles FHE-Chip, 2026 – https://spectrum.ieee.org/fhe-intel

# CipherChat – Handover für den lokalen Agenten

> Übergabe-Dokument (Stand 2026-11-18). Single Source of Truth für Vision, Research und Roadmap ist die
> Notion-Seite „HTTPS für LLMs – Privacy-Preserving LLM Start-up (Research & MVP)“.
> Dieses Dokument beschreibt den **Code-Stand des MVP** (`cipherchat/fhe_engine.py`, v1).
> Weiterentwicklungen (v2-Engine, neues Modell, Benchmarks) stehen in `docs/RESEARCH_LOG.md`.

## 1. Umgebung

- Der FHE-Teil (`server/`, `client/`, `cipherchat/fhe_*`, `scripts/`) läuft nur mit **Python 3.12 auf Linux x86_64**.
- Das Wheel `openfhe==1.6.1.0.24.4` ist als `py3-none-any` getaggt, enthält aber eine Binärdatei für CPython 3.12 (Ubuntu 24.04).
  Unter Python 3.13 kommt deshalb `ModuleNotFoundError: openfhe.openfhe`. Unter Windows WSL2 oder Docker nehmen.
- Training läuft mit beliebigem Python und torch (getestet: Python 3.13, torch 2.15, CPU). Training braucht kein openfhe, der FHE-Teil kein torch.
- Nicht getestet: Docker-Build, `run_demo.sh` als Skript (die gleichen uvicorn-Befehle liefen manuell), Colab-Notebook, `--device cuda`.

## 2. Invarianten (nicht brechen)

Blockgröße **B = 64 Slots**:

- `d = 64`
- Vokabular ≤ 64 (aktuell 59)
- `d_ff` ein Vielfaches von 64
- genau 1 Head

Kontext: N = 16384 ergibt 8192 Slots. Ein Block im KV-Cache hat 128 Slots, also passen 64 Tokens Kontext hinein.

Tiefe = 4·L.

- L = 2 → Tiefe 8 → N = 16384 (Skalierung 2³³, erstes Modulus 45 Bit).
- L = 3 → OpenFHE wählt automatisch N = 32768, das ist 3–4× langsamer.

Training muss exakt die FHE-Operationen nutzen:

- parallele Blöcke `x += α_a·Attn + α_m·MLP`
- Attention `w = (q·k + c)²/(t+1)`
- Aktivierung `m = (xW1 + b1)²`
- kein LayerNorm

Wer `model.py` ändert, muss auch `reference.py` und `fhe_engine.py` anpassen. Danach in dieser Reihenfolge prüfen:

1. torch gegen `reference.py`: Abweichung < 1e-6
2. `scripts/test_fhe_vs_plain.py`: Logit-Fehler etwa 1e-3
3. `scripts/benchmark_fhe.py`

## 3. Weight-Folding (`reference.FoldedModel`, Konvention `y = x @ W + b`)

Beim Export:

- `Wvo = α_a·Wv·Wo`
- `bvo = α_a·bv·Wo`
- `bo' = α_a·bo`
- `W2' = α_m·W2`
- `b2' = α_m·b2`

Layer 0 bekommt den One-Hot-Vektor, das Embedding E ist eingefaltet:

- `Mq = E·Wq`, analog `Mk`, `Mv = E·Wvo`, `Mz_b = E·W1[:, Block b]`
- Positions-Biases über `β = P[t]` (`pos_biases()`)

Layer ≥ 1: `A = I`, der Input ist der verschlüsselte Residual-Stream.

Letzte Schicht:

- Der Value-Cache speichert `u_j = v'_j·Wu`.
- `Mxu = A·Wu`, `W2_b ← W2_b·Wu`, `bfinal = bu + (bo + b2)·Wu`.

Andere Schichten: `x_{l+1} = x + a + Σ m_b·W2_b + bo + b2`.

## 4. Slot-Layouts und Operationen (`fhe_engine.py`)

- **rep**: Ein 64-Vektor wird 128× hintereinander kopiert (Periode 64). So bleibt jede Rotation um weniger als 64 innerhalb jedes Blocks zyklisch.
- **K-Cache** (ein Ciphertext pro Layer): Block j umfasst die Slots `[128j, 128j+128)` und enthält `[k_j, k_j]`.
- **V-Cache**: Block j enthält `[v_j, 0]`.

Matrix-Vektor-Produkt (BSGS, verbraucht 1 Level):

- `d_k[c] = M[(c+k)%64, c]`
- `y = Σ_{i<4} rot(Σ_{j<16} rot(x,j) ⊙ P[i][j], 16i)`
- `P[i][j] = roll(tile(d_{16i+j}) [⊙ Maske], +16i)`
- Die Baby-Steps laufen gehoistet.
- `EvalRotate(ct, k)` ergibt `res[i] = x[i+k]`.

`k_t` und `v_t` in Block t einfügen: über maskierte Diagonalen (`kmask` bzw. `vmask`), die pro Token neu encodiert werden.
Das spart ein Level, kostet aber Zeit.

q wird mit `sc = 1/√(t+1)` über neu encodierte Diagonalen skaliert. Der Shift ist `c·sc`.

Scores:

- `S = q ⊙ K`, dann Fenstersumme mit Rotationen `[1, 2, 4, 8, 16, 32]`
- danach `w = (S + c·sc)²`
- Müll in den zweiten Blockhälften und in leeren Blöcken j > t wird durch die Nullen in V eliminiert.

Ausgabe:

- `av = w ⊙ V`, dann Blocksumme mit Rotationen `[128 … 4096]`
- anschließend `av + rot(av, −64)`, das ergibt wieder das rep-Layout
- Für die Logits braucht es diesen letzten Schritt nicht, der Client liest die Slots 0..63.

25 Rotations-Keys: 1..15, 16, 32, 48, 128..4096, −64.

Level-Ablauf (L = 2):

- Kritischer Pfad: One-Hot(0) → Projektion(1) → S(2) → w(3) → x1(4) → Projektion(5) → S(6) → w(7) → Logits(8)
- Der MLP-Pfad z(1) → m(2) → m·W2(3) liegt daneben.

Prefill: Mit `need_logits=False` fügt die letzte Schicht nur k und u in den Cache ein.

## 5. OpenFHE-Fehlerquellen (teuer gelernt)

FLEXIBLEAUTO und Rescaling:

- `cc.Rescale()` tut nichts.
- Ein `EvalMult(ct, pt)` auf einem Input mit `noiseScaleDeg 2` rescalet jedes Mal eine Kopie.
- Fix: `EvalMultMutable(ct, ones@level+1)` in-place vor den Baby-Rotationen. Bei ct×ct mit den Caches ebenfalls `EvalMultMutable` nutzen.

Plaintext-Level beim Encodieren:

- Multiplikation: `ct.GetLevel() + (deg − 1)`
- Addition: `ct.GetLevel()`
- Encodieren auf Level == Tiefe schlägt fehl.

`EvalAddMany` braucht `ADVANCEDSHE`.

Deserialisierte Ciphertexts sind bei `EvalMult(ct, pt)` etwa 5× langsamer.

- Fix: `EvalAdd(zero, ct)` mit einer serverseitig erzeugten Verschlüsselung von 0. `zero` muss der erste Operand sein.
- Dafür schickt der Client den Public Key als 4. Frame mit.
- Das brachte 110 s → 69 s pro Antwort.

Keys im selben Prozess erneut laden schlägt fehl. Vorher `ofhe.ClearEvalMultKeys()` und `CryptoContext.ClearEvalAutomorphismKeys()` aufrufen.
Der Server macht das bei jeder neuen Session, deshalb geht nur ein Client gleichzeitig.

Serialisierung:

- `Serialize(obj, BINARY)` gibt `bytes` zurück.
- Die Key-Deserialisierer geben `None` zurück und füllen eine statische Map.
- `EnablePrecomputeCRTTablesAfterDeserializaton` bringt nichts.

cProfile sieht die C++-Zeit nicht. Zum Profilen die Engine-Methoden mit Timern wrappen.

## 6. Performance (2 vCPU)

Zeit pro Token:

| | im selben Prozess | über HTTP |
|---|---|---|
| Prompt-Token | 2,1 s | 2,2 s |
| generiertes Token | 2,6 s | 3,1 s |

Operationen pro generiertem Token:

- ~163 Rotationen
- 1024 Plaintext-Multiplikationen
- 10 ct×ct-Multiplikationen
- ~420 Encodes, davon 384 dynamisch (q sowie K/V beider Layer)

Datenmengen:

- Eval-Keys: 225 MB
- pro Token hoch: 2,36 MB
- Logits zurück: 0,53 MB

Kosten der Grundoperationen:

- Rotation: 10–14 ms, gehoistet 7 ms
- Plaintext-Multiplikation: 0,6 ms
- Encode: ~2 ms

## 7. Training

Datensatz:

- `--holdout` hält standardmäßig 15 % der Rechen- und Vergleichspaare zurück.
- Loss nur auf Antwort + EOS.

Hyperparameter:

- AdamW, lr 3e-3, betas (0,9, 0,98), wd 0,01
- 300 Warmup-Schritte, dann Cosine-Abfall
- Batch 128, 8000 Schritte, Grad-Clip 1,0
- Zusatzterm `1e-3·mean(logits²)`

Initialisierung:

- Wq/Wk mit Standardabweichung `0,3/√d`
- Embedding 0,5, Positionen 0,3
- b1 = 0,5, α = 0,1, c = 1,0

Beobachtete Beträge:

- max|s| ≈ 100–140
- max|w| ≈ 300–1000
- max|x| ≈ 30–70
- Grenze auf dem letzten CKKS-Level: etwa ±2000

Gewichte:

- `cipherchat_demo`: holdout 0, Default im Server
- `cipherformer_l2`: holdout 0,15, damit wurde ausgewertet (Vergleiche 82 %, Rechnen 39 %)

## 8. Protokoll und APIs

Server:

- `GET /info`
- `POST /session`: Body sind 4 Frames im Format `<u64 LE Länge><bytes>`, nämlich cryptocontext, mult_key, rot_keys, public_key
- `POST /session/{sid}/reset`
- `POST /session/{sid}/token?need_logits=0|1`: liefert 204 beim Prefill, sonst einen Logit-Ciphertext
- `GET /server_view`
- Es gibt einen globalen Lock, und die FHE-Rechnung blockiert den Event-Loop.

Client:

- Beim Start: Schlüsselerzeugung und Handshake.
- `POST /api/chat` liefert SSE-Events (`status`, `upload`, `decrypt`, `token`, `done`).
- Pro Nachricht wird der Kontext zurückgesetzt, es gibt also keinen Chatverlauf.
- Gewählt wird per Greedy-Argmax, höchstens 20 neue Tokens.
- Umgebungsvariablen: `CIPHERCHAT_SERVER`, `CIPHERCHAT_WEIGHTS`, `CIPHERCHAT_MAX_NEW`.

## 9. Bekannte Lücken

- Nur ein Client pro Server.
- Kein Auth und kein TLS zwischen Client und Server.
- Kein Padding, die Länge der Nachricht ist sichtbar.
- Der Client sieht die vollen Logits, das ist ein Risiko für das Modell-IP.
- `on_event("startup")` ist deprecated → auf `lifespan` umstellen.
- `--act_limit` wird nicht benutzt.

## 10. Nächste Aufgaben (priorisiert)

1. Lokal mit mehr Kernen messen.
2. Giant-Step-Rotationen bei Matvecs zusammenlegen, deren Ergebnisse ohnehin addiert werden. Spart etwa 12 Rotationen pro Token.
3. Dynamische Encodes reduzieren:
   - q-Plaintexts pro Position cachen, oder
   - mit fester Normierung statt `1/(t+1)` neu trainieren.
4. Mehrere Sessions gleichzeitig ermöglichen (Key-Tags pro Session verwalten).
5. Batched Prefill: alle Prompt-Tokens in einem Ciphertext. Braucht ein Layout mit 192-Slot-Blöcken plus Masken.
6. GPU-Backend evaluieren (FIDESlib, Phantom).
7. Größeres Modell:
   - L = 3 mit N = 32768 oder Bootstrapping
   - Vokabular über 64 (BPE) braucht mehrere 64er-Blöcke.
8. Homomorphes Top-k und Padding.
9. Transciphering-Prototyp, zuerst mit einer HE-freundlichen Chiffre statt AES.

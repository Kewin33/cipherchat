# CipherChat – "HTTPS für LLMs" (FHE-MVP)

Ein Chatbot, bei dem der **Server das Sprachmodell nur auf verschlüsselten Daten rechnet**.
Der Client verschlüsselt jedes Token mit **CKKS (Fully Homomorphic Encryption, OpenFHE)**, der Server
wertet einen kleinen Transformer homomorph aus und schickt **verschlüsselte Logits** zurück –
nur der Client kann sie entschlüsseln. Der Secret Key verlässt nie das Gerät des Nutzers.

```
 Browser-UI ──(localhost)──► Client (Secret Key)                 Server (nur Eval-Keys)
                              │ 1. Handshake: CKKS-Parameter + Eval-Keys ──────────────►
                              │ 2. Enc(one-hot(Token))  ───────────────────────────────► FHE-Transformer
                              │                                                        (verschl. KV-Cache)
                              │ 3. ◄──────────────────────────────── Enc(Logits)
                              │ 4. Dec → argmax → nächstes Token → zurück zu 2.
```

## Was genau verschlüsselt ist
| | Server sieht | Server sieht nicht |
|---|---|---|
| Prompt & Antwort | Anzahl Tokens (Länge), Zeitpunkte | Inhalt (jedes Zeichen ist ein CKKS-Ciphertext) |
| Attention / Zwischenwerte | – | alles (Scores, Aktivierungen, KV-Cache sind verschlüsselt) |
| Ausgabe | – | Logits / Wahrscheinlichkeiten (nur Client entschlüsselt) |
| Modell | Gewichte (gehören dem Server) | – |

Sicherheitsmodell: *semi-honest* Server, 128-Bit-Sicherheit nach HE-Standard (N = 16384).
Nicht abgedeckt im MVP: ein aktiv bösartiger Server (Integrität/Verifizierbarkeit), Längen-Leakage (→ Padding),
Seitenkanäle, und CKKS-Decryption-Leakage (IND-CPA^D) falls entschlüsselte Werte an den Server zurückfließen.
Im MVP fließen nur gesampelte Tokens zurück – und zwar wieder verschlüsselt.

## Modell: CipherFormer (FHE-freundlich)
* 2 parallele Transformer-Blöcke (Attention ∥ MLP, wie GPT-J/PaLM) → nur 4 multiplikative Level pro Block
* **Polynomielle Attention** statt Softmax: `w_tj = (q_t·k_j + c)² / (t+1)` – kein exp, kein max, keine Division durch verschlüsselte Werte
* **Quadrat-Aktivierung** im MLP, **kein LayerNorm** (ReZero-Gains werden beim Export in die Gewichte gefaltet)
* Embedding wird in die erste Projektion, Unembedding in den letzten Value-Cache gefaltet → Gesamttiefe 8
* d = 64, d_ff = 128, Vokabular = 59 Zeichen, Kontext = 64 Tokens, ~78k Parameter
* `weights/cipherchat_demo.*` = Demo-Modell (auf allen Paaren trainiert); `weights/cipherformer_l2.*` = Evaluationsmodell mit
  15 % zurückgehaltenen Rechen-/Vergleichsaufgaben (Generalisierung: Vergleiche 82 %, Rechnen 39 % → das Modell memoriert vor allem)
* Datensatz: synthetischer deutscher Mini-Chat (Smalltalk, Hauptstädte, Farben, Gegenteile, + − Rechnen, Parität, Vergleiche, Zählen)

## FHE-Engine (Server)
* CKKS, Ringdimension 16384 (8192 Slots), Skalierung 2^33, FLEXIBLEAUTO, HYBRID Key Switching
* Vektor-Matrix-Produkte mit **BSGS-Diagonalmethode** (16 gehoistete Baby-Steps × 4 Giant-Steps)
* **Verschlüsselter KV-Cache**: alle Keys einer Schicht in *einem* Ciphertext (Block j = [k_j, k_j]),
  alle Values in einem Ciphertext (Block j = [v_j, 0]) → Scores gegen alle Positionen mit 1 Multiplikation + 6 Rotationen
* Prompt-Prefill überspringt den Logit-Pfad der letzten Schicht

Gemessen (2 vCPU, Sandbox, `scripts/benchmark_fhe.py`): **~2,7 s pro generiertem Token, ~2,1 s pro Prompt-Token**,
Logit-Fehler < 3·10⁻³, 8/8 Test-Prompts mit identischer Antwort wie im Klartext.
Ende-zu-Ende über HTTP (Client und Server als getrennte Prozesse): ~2,2 s pro Prompt-Token, ~3,1 s pro generiertem Token
(z. B. „hauptstadt von italien?“ → „rom.“ in 69 s). Handshake: ~225 MB Evaluation-Keys (einmal pro Session), 2,3 MB pro Token hoch, 0,5 MB zurück.

## Schnellstart

### Variante A: Linux / WSL2 (Ubuntu 24.04, Python 3.12)
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-fhe.txt          # OpenFHE-Wheel ist für Python 3.12 / Ubuntu 24.04 gebaut
./run_demo.sh                                 # startet Server (:8000) und Client (:8501)
# Browser: http://localhost:8501
```
Server und Client lassen sich auch auf getrennten Rechnern starten:
```bash
# Rechner 1 (Server):
uvicorn server.app:app --host 0.0.0.0 --port 8000
# Rechner 2 (Client, hier liegt der Secret Key):
CIPHERCHAT_SERVER=http://<server-ip>:8000 uvicorn client.app:app --port 8501
```

### Variante B: Docker (Windows/macOS/Linux)
```bash
docker compose up --build      # danach http://localhost:8501
```

### Tests
```bash
python scripts/test_fhe_vs_plain.py                      # FHE vs. Klartext, Token für Token
python scripts/test_fhe_vs_plain.py --prompts "ist 4 gerade?" "hauptstadt von polen?"
```

## Training (CPU, RTX 4070 oder Colab)
```bash
pip install -r requirements-train.txt
python train.py --steps 8000                                   # CPU, ~6 min
python train.py --device cuda --steps 30000 --batch 512        # RTX 4070 / Colab-GPU
```
Ausgabe: `weights/<name>.npz/.json/.pt`. Den Server mit anderem Modell starten:
`CIPHERCHAT_WEIGHTS=weights/<name>.npz uvicorn server.app:app`.
Regeln, damit das Modell FHE-tauglich bleibt: `d = 64`, `d_ff` Vielfaches von 64, Vokabular ≤ 64, Kontext ≤ 64.
Mehr Layer → Tiefe 4·L → ab 3 Layern wählt OpenFHE N = 32768 (langsamer, aber funktioniert).
Notebook: `notebooks/train_colab.ipynb`.

## Experiment zu Ansatz 2 (AES-Ciphertext → nächster Ciphertext)
`python experiments/aes_learnability.py` trainiert dasselbe Netz auf Klartext, AES-CTR (korrekt, zufällige Nonce),
AES-ECB pro Zeichen und AES-CTR mit fester Nonce und greift die lernbaren Varianten an. Ergebnis in
`docs/aes_experiment_results.json`. Ergebnis (2-Layer-GRU, 20.000 Nachrichten):

| Variante | Val.-CE (nats) | Next-Token-Acc. | Angriff nur auf Ciphertexts |
|---|---|---|---|
| Klartext | 0,337 | 86,6 % | – |
| AES-CTR, zufällige Nonce | 5,674 (Zufall 5,545) | 0,4 % (= Zufall) | – (nicht lernbar, sicher) |
| AES-ECB pro Zeichen | 0,338 | 86,8 % | 96,3 % der Zeichen entschlüsselt (`experiments/aes_attacks.py`) |
| AES-CTR, feste Nonce | 0,486 | 83,2 % | 100 % der Bytes entschlüsselt |

→ Entweder sicher und nicht lernbar, oder lernbar und gebrochen. Der korrekte Weg ist Transciphering (AES → CKKS).

## v2: Slot-gesättigte Engine + generalisierende Addition (Forschungsstand)

Details, Formeln und alle Messungen: `docs/RESEARCH_LOG.md` und `docs/PAPER_v2.md` (Paper auch in Notion).

* **0,365 s pro Token** auf 8 Kernen (v1: 1,20 s), 0,75 s auf einem Kern (v1: 2,13 s); 0 statt 384 dynamische Encodes,
  96 statt ~163 Rotationen, 512 statt 1.024 Plaintext-Multiplikationen; Plaintext-Cache ~300 MB statt ~800 MB.
* Generalisiertes BSGS (eine Matvec → beliebiges Slot-Layout), MLP bis 8.192 Hidden-Units ohne Mehrkosten,
  schiebender KV-Cache mit relativen Positions-Biases, bilineare Scores ohne Query-Projektion, Gated MLP im Reserve-Level.
* Addition 0–99 auf **zurückgehaltenen** Operandenpaaren über Client-seitiges Position Coupling (Two-Hot-Input) und
  Little-Endian-Zahlen – beides ändert an der Engine nichts.

```bash
python scripts/v2_check.py                                   # torch = Referenz = Slot-Simulation (ohne OpenFHE)
python train_v2.py --device cuda --rev --coupling --mlp sqglu --hidden 2048 --name cf2_add
python scripts/v2_fhe_test.py --weights weights/cf2_add.npz --heldout 40     # verschlüsselt, mit Referenzvergleich
CIPHERCHAT_WEIGHTS=weights/cf2_add.npz ./run_demo.sh         # Server/Client erkennen v2 automatisch
```

## Projektstruktur
```
cipherchat/vocab.py          Zeichen-Vokabular (≤ 64)
cipherchat/dataset.py        synthetischer deutscher Mini-Chat
cipherchat/model.py          CipherFormer (PyTorch) + Export/Falten der Gewichte
cipherchat/reference.py      Klartext-Referenz, die exakt die FHE-Rechnung nachbildet
cipherchat/fhe_common.py     CKKS-Parameter, Slot-Layouts, BSGS-Vektoren, Serialisierung
cipherchat/fhe_engine.py     homomorphe Inferenz (Server)
cipherchat/client_crypto.py  Schlüssel, Verschlüsselung, Entschlüsselung (Client)
server/app.py                FastAPI-Server: bekommt nur Ciphertexts + Eval-Keys
client/app.py, client/static Client-Proxy + Chat-UI ("Was der Server sieht")
train.py                     Training
experiments/                 AES-Lernbarkeits-Experiment
```

## Roadmap (Kurzfassung)
1. Batched Prefill (ganzer Prompt in einem Ciphertext), GPU-FHE (Phantom / FIDESlib / Cheddar / HEonGPU)
2. Größeres Modell (BPE-Vokabular, mehr Layer mit Bootstrapping), Distillation von einem Open-Source-LLM
3. Client im Browser (WASM) + Transciphering (AES/HE-freundliche Chiffre → CKKS) gegen die Upload-Größe
4. Hybrid-Tier: TEE (Confidential GPU) für große Modelle + FHE für die sensibelsten Teile

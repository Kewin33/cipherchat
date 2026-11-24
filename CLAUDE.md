# CipherChat – Agent-Leitfaden

FHE-Chatbot („HTTPS für LLMs“): Der Client verschlüsselt Tokens mit CKKS (OpenFHE), der Server rechnet einen
kleinen Transformer nur auf Ciphertexts. Single Source of Truth für Vision und Roadmap ist die Notion-Seite
„HTTPS für LLMs – Privacy-Preserving LLM Start-up (Research & MVP)“.

## Erst lesen
- `docs/HANDOVER.md` – Übergabe des MVP (v1): Invarianten, Weight-Folding, Slot-Layouts, OpenFHE-Fallen, Protokoll.
- `docs/RESEARCH_LOG.md` – Stand der v2-Forschung (schnellere Engine, Additions-Modell), Messwerte, offene Punkte.
- `docs/V3.md` – v3: Qwen2.5-0.5B nicht-interaktiv unter CKKS mit Bootstrapping auf der GPU (Design, Messwerte, Fallen).
- `graphify-out/GRAPH_REPORT.md` – Architektur-Überblick aus dem Knowledge Graph (siehe unten).

## Drei Generationen im Code
- **v1** (MVP, unverändert lauffähig): `cipherchat/{model,reference,fhe_common,fhe_engine,client_crypto}.py`, `server/`, `client/`, `train.py`.
- **v2** (Forschung): `cipherchat/v2/` – `layout.py` (generalisiertes BSGS, Slot-Layouts), `engine.py`
  (gleiche Engine auf `NumpyBackend` = exakte Slot-Simulation und `OpenFHEBackend`), `model.py` (CipherFormer2),
  `reference.py` (Token-für-Token-Referenz), `dataset.py`, `crypto.py`; Training `train_v2.py`, Tests `scripts/v2_*.py`.
- **v3** (Qwen2.5-0.5B, nicht-interaktiv, GPU): `cipherchat/v3/` – `config.json` (einzige Quelle für Modell- und
  CKKS-Parameter, auch für C++), `approx.py` (Newton-Näherungen, Aktivierungsfit), `model.py` (Umbau + Kalibrierung),
  `fold.py` (Gewichte für die Engine), `engine.py` (Engine3 auf `NumpyBackend` / `PhantomBackend`), `service.py`,
  `fhe/` (C++/CUDA: `bench_m1.cu`, pybind-Modul `v3fhe` über PhantomFHE + NEXUS-Bootstrapping); Training `train_v3.py`,
  Server `server/app_v3.py`, Tests `scripts/v3_*.py`. `client/app.py` schaltet zwischen v1/v2/v3 um (`run_demo_all.sh`).

## Umgebung
- FHE nur mit Python 3.12 auf Linux x86_64 (OpenFHE-Wheel). Windows-Rechner: nur numpy-Simulation und Doku.
- Rechenmaschine `ssh at-10-1-2` (Core Ultra 9 285K: P-Kerne 0–7, E-Kerne 8–23; RTX 5080 16 GB; geteilt mit anderen Nutzern).
  Projektkopie `~/cipherchat`, venv `~/cipherchat/.venv` (openfhe, numpy, fastapi, torch cu128).
  Sync: `tar czf - --exclude=graphify-out --exclude=__pycache__ . | ssh at-10-1-2 'tar xzf - -C ~/cipherchat'`.
- Benchmarks mit `OMP_NUM_THREADS=8 taskset -c 0-7` (P-Kerne); mehr Threads bringen in OpenFHE-Python nichts.
- v3: `at-10-1-1` … `at-10-1-4` haben je eine RTX 5080 und **dasselbe NFS-Home** (ein Sync reicht für alle).
  Sync nur der v3-Teile: `bash scripts/v3_sync.sh <host>`; Toolchain (NTL, NEXUS, Build): `scripts/v3_setup_remote.sh`.
  Lange Jobs mit `nohup … < /dev/null &` starten und Logs lesen; SSH über den Sprungrechner bricht bei langen Läufen ab.
  Training (~14 GB) und FHE-Engine (~8,5 GB) passen nicht gleichzeitig auf eine Karte.

## Invarianten (Details: HANDOVER §2)
- Blockgröße 64: d = 64, Vokabular ≤ 64, genau 1 Head, Kontext ≤ 64; Tiefe 4·L, L = 2 → N = 16384.
- Training muss exakt die FHE-Operationen nutzen. Wer `model.py` ändert, passt `reference.py` und die Engine an und prüft:
  v1: torch vs. reference < 1e-6 → `scripts/test_fhe_vs_plain.py` (Logit-Fehler ~1e-3) → `scripts/benchmark_fhe.py`.
  v2: `scripts/v2_check.py` (torch vs. Referenz vs. numpy-Slot-Simulation, Fehler < 1e-9) → `scripts/v2_fhe_test.py` (OpenFHE).
- v2 zusätzlich: Konstanter Kanal x[63] = 1 (trägt alle Biases), Attention-Normierung fest (kappa), relative Positions-Biases,
  der KV-Cache schiebt pro Token um einen 128-Slot-Block (neuester Eintrag immer im letzten Block).
- v3: N = 2¹⁶, log QP 1480 ≤ 1546 (h = 192), 10 Rechenebenen pro Bootstrap; öffentliches Präfix `<|endoftext|>` + Zeilenumbruch
  (exakt im Klartext, verhindert massive activations); Bootstrapping nur auf PER-Ciphertexts (1024-periodisch) und mit
  |x| ≤ 1 (Grenzen pro Schicht in `weights/v3/<name>_bounds.json`). Prüfkette: `scripts/v3_check.py --backend numpy`
  (Fehler ~1e-8 gegen float64-Referenz) → `--backend phantom` (Kosinus > 0.99) → `scripts/v3_range_diag.py` (Perplexität ohne Bereichsschutz).

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` (on Windows: `python -m graphify query "<question>"`) before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` (Windows: `python -m graphify update .`) to keep the graph current (AST-only, no API cost).
- A committed graphify-out/graph.json is rewritten whole on every build, so a large diff to it in a pull request is expected. Do not flag it as churn or suggest skipping the graph rebuild or its hooks.

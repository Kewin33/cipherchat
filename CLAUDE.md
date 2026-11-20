# CipherChat – Agent-Leitfaden

FHE-Chatbot („HTTPS für LLMs“): Der Client verschlüsselt Tokens mit CKKS (OpenFHE), der Server rechnet einen
kleinen Transformer nur auf Ciphertexts. Single Source of Truth für Vision und Roadmap ist die Notion-Seite
„HTTPS für LLMs – Privacy-Preserving LLM Start-up (Research & MVP)“.

## Erst lesen
- `docs/HANDOVER.md` – Übergabe des MVP (v1): Invarianten, Weight-Folding, Slot-Layouts, OpenFHE-Fallen, Protokoll.
- `docs/RESEARCH_LOG.md` – Stand der v2-Forschung (schnellere Engine, Additions-Modell), Messwerte, offene Punkte.
- `graphify-out/GRAPH_REPORT.md` – Architektur-Überblick aus dem Knowledge Graph (siehe unten).

## Zwei Generationen im Code
- **v1** (MVP, unverändert lauffähig): `cipherchat/{model,reference,fhe_common,fhe_engine,client_crypto}.py`, `server/`, `client/`, `train.py`.
- **v2** (Forschung): `cipherchat/v2/` – `layout.py` (generalisiertes BSGS, Slot-Layouts), `engine.py`
  (gleiche Engine auf `NumpyBackend` = exakte Slot-Simulation und `OpenFHEBackend`), `model.py` (CipherFormer2),
  `reference.py` (Token-für-Token-Referenz), `dataset.py`, `crypto.py`; Training `train_v2.py`, Tests `scripts/v2_*.py`.

## Umgebung
- FHE nur mit Python 3.12 auf Linux x86_64 (OpenFHE-Wheel). Windows-Rechner: nur numpy-Simulation und Doku.
- Rechenmaschine `ssh at-10-1-2` (Core Ultra 9 285K: P-Kerne 0–7, E-Kerne 8–23; RTX 5080 16 GB; geteilt mit anderen Nutzern).
  Projektkopie `~/cipherchat`, venv `~/cipherchat/.venv` (openfhe, numpy, fastapi, torch cu128).
  Sync: `tar czf - --exclude=graphify-out --exclude=__pycache__ . | ssh at-10-1-2 'tar xzf - -C ~/cipherchat'`.
- Benchmarks mit `OMP_NUM_THREADS=8 taskset -c 0-7` (P-Kerne); mehr Threads bringen in OpenFHE-Python nichts.

## Invarianten (Details: HANDOVER §2)
- Blockgröße 64: d = 64, Vokabular ≤ 64, genau 1 Head, Kontext ≤ 64; Tiefe 4·L, L = 2 → N = 16384.
- Training muss exakt die FHE-Operationen nutzen. Wer `model.py` ändert, passt `reference.py` und die Engine an und prüft:
  v1: torch vs. reference < 1e-6 → `scripts/test_fhe_vs_plain.py` (Logit-Fehler ~1e-3) → `scripts/benchmark_fhe.py`.
  v2: `scripts/v2_check.py` (torch vs. Referenz vs. numpy-Slot-Simulation, Fehler < 1e-9) → `scripts/v2_fhe_test.py` (OpenFHE).
- v2 zusätzlich: Konstanter Kanal x[63] = 1 (trägt alle Biases), Attention-Normierung fest (kappa), relative Positions-Biases,
  der KV-Cache schiebt pro Token um einen 128-Slot-Block (neuester Eintrag immer im letzten Block).

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` (on Windows: `python -m graphify query "<question>"`) before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` (Windows: `python -m graphify update .`) to keep the graph current (AST-only, no API cost).
- A committed graphify-out/graph.json is rewritten whole on every build, so a large diff to it in a pull request is expected. Do not flag it as churn or suggest skipping the graph rebuild or its hooks.

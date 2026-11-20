# Graph Report - cipherchat  (2026-11-20)

## Corpus Check
- 49 files · ~29,256 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 13 file(s) not represented in the graph (top: .log 9, (none) 3, .ipynb 1)

## Summary
- 427 nodes · 907 edges · 18 communities (11 shown, 7 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 35 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8755f085`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- FHE Server & CKKS Parameters
- v2_fhe_test.py
- Client Crypto & Chat Client
- AES Learnability Experiment
- Slot Layouts (Generalized BSGS)
- CipherFormer2
- v2 Engine Decoding Step
- v1 FHE Engine
- make_architecture_figure.py
- v2_check.py
- v2 Numpy Simulator & Research Log
- v1 Reference & Model
- v2 Paper (docs)
- Demo Launcher

## God Nodes (most connected - your core abstractions)
1. `OpenFHEBackend` - 20 edges
2. `EngineV2` - 19 edges
3. `FHEServerModel` - 18 edges
4. `NumpyBackend` - 16 edges
5. `Task` - 15 edges
6. `create_session()` - 14 edges
7. `CipherChat v2: Slot-gesättigte CKKS-Inferenz und generalisierende Arithmetik für einen verschlüsselten Chatbot` - 13 edges
8. `CipherFormer2` - 12 edges
9. `FoldedModel` - 12 edges
10. `ClientCrypto` - 12 edges

## Surprising Connections (you probably didn't know these)
- `from_rep_plaintexts()` --semantically_similar_to--> `BSGS diagonal matrix-vector product`  [INFERRED] [semantically similar]
  cipherchat/v2/layout.py → docs/HANDOVER.md
- `2. v2-Engine: Ideen und warum sie funktionieren` --references--> `NumpyBackend`  [INFERRED]
  docs/RESEARCH_LOG.md → cipherchat/v2/engine.py
- `OpenFHEBackend` --references--> `Zwei Generationen im Code`  [INFERRED]
  cipherchat/v2/engine.py → CLAUDE.md
- `9. Roadmap zur Chat-Fähigkeit („GPT-4-Klasse kommt später“)` --references--> `OpenFHEBackend`  [INFERRED]
  docs/PAPER_v2.md → cipherchat/v2/engine.py
- `create_session()` --rationale_for--> `Deserialized ciphertexts 5x slower (zero-add fix)`  [INFERRED]
  server/app.py → docs/HANDOVER.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **FHE-exact model chain: torch model -> folded reference -> FHE engine** — cipherchat_model_cipherformer, cipherchat_reference_foldedmodel, cipherchat_reference_plainreference, cipherchat_fhe_engine_fheservermodel [EXTRACTED 1.00]
- **Encrypted token step: encrypt -> matvec/attention -> logits -> decrypt** — cipherchat_client_crypto_clientcrypto_encrypt_token, cipherchat_fhe_engine_fheservermodel_step, cipherchat_client_crypto_clientcrypto_decrypt_logits, server_app_token, client_app_chat [INFERRED 0.95]

## Communities (18 total, 7 thin omitted)

### Community 0 - "FHE Server & CKKS Parameters"
Cohesion: 0.07
Nodes (32): bsgs_vectors(), de_cc(), de_ct(), depth_for_layers(), diagonals(), kmask(), make_context(), max_context() (+24 more)

### Community 1 - "v2_fhe_test.py"
Cohesion: 0.08
Nodes (17): _q(), sample_batch(), to_ids(), other_items(), reverse_digits(), significance(), split_pairs(), Task (+9 more)

### Community 2 - "Client Crypto & Chat Client"
Cohesion: 0.05
Nodes (19): ClientCrypto, ClientCryptoV2, ChatIn, connect(), frames(), index(), server_view(), _softmax() (+11 more)

### Community 3 - "AES Learnability Experiment"
Cohesion: 0.05
Nodes (23): build_pairs(), eval_prompts(), Block, CipherFormer, masked_ce(), masked_ce(), Polynomial attention w=(q.k+c)^2/(t+1), Training recipe (AdamW 3e-3, 8000 steps, logit penalty) (+15 more)

### Community 4 - "Slot Layouts (Generalized BSGS)"
Cohesion: 0.17
Nodes (12): cache_add_vector(), cross_steps(), from_rep_plaintexts(), full_block0(), full_cache_k(), full_cache_v(), full_from_hidden(), full_hidden() (+4 more)

### Community 6 - "v2 Engine Decoding Step"
Cohesion: 0.16
Nodes (4): EngineV2, Known gaps (no TLS, length leakage, full logits to client), Prioritized next tasks (giant-step merge, static encodes, GPU, batched prefill), Roadmap (batched prefill, GPU FHE, BPE, WASM client, TEE hybrid)

### Community 8 - "make_architecture_figure.py"
Cohesion: 0.40
Nodes (6): add(), arrow(), block(), box(), line(), text()

### Community 9 - "v2_check.py"
Cohesion: 0.21
Nodes (3): Folded2, mlp_act(), PlainReference2

### Community 10 - "v2 Numpy Simulator & Research Log"
Cohesion: 0.07
Nodes (14): NumpyBackend, CipherChat – Agent-Leitfaden, Erst lesen, graphify, Invarianten (Details: HANDOVER §2), Umgebung, Zwei Generationen im Code, 1. Baseline (v1-Engine, unverändert) (+6 more)

### Community 11 - "v1 Reference & Model"
Cohesion: 0.14
Nodes (3): FoldedModel, PlainReference, Performance on 2 vCPU (2.6 s/generated token)

### Community 13 - "v2 Paper (docs)"
Cohesion: 0.05
Nodes (25): OpenFHEBackend, 10. Fazit, 1. Motivation und Ziele, 2. Ausgangslage: wohin die Zeit in v1 geht, 3. Verwandte Arbeiten, 4.1 Generalisiertes BSGS: eine Matvec, beliebiges Ausgabe-Layout, 4.2 Schiebender KV-Cache und relative Positionen, 4.3 Bilineare Scores ohne Query-Projektion (+17 more)

## Knowledge Gaps
- **10 isolated node(s):** `CKKS via OpenFHE (N=16384, scale 2^33, HYBRID key switching)`, `openfhe>=1.6.1 (Python 3.12 / Linux wheel)`, `Performance on 2 vCPU (2.6 s/generated token)`, `Training recipe (AdamW 3e-3, 8000 steps, logit penalty)`, `Local panel (secret key, decrypted top-3)` (+5 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 176 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `OpenFHEBackend` connect `v2 Paper (docs)` to `FHE Server & CKKS Parameters`, `v2_fhe_test.py`, `v2 Numpy Simulator & Research Log`, `Slot Layouts (Generalized BSGS)`?**
  _High betweenness centrality (0.193) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `OpenFHEBackend` (e.g. with `Zwei Generationen im Code` and `9. Roadmap zur Chat-Fähigkeit („GPT-4-Klasse kommt später“)`) actually correct?**
  _`OpenFHEBackend` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `CKKS via OpenFHE (N=16384, scale 2^33, HYBRID key switching)`, `openfhe>=1.6.1 (Python 3.12 / Linux wheel)`, `Performance on 2 vCPU (2.6 s/generated token)` to the rest of the system?**
  _10 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `FHE Server & CKKS Parameters` be split into smaller, more focused modules?**
  _Cohesion score 0.06586538461538462 - nodes in this community are weakly interconnected._
- **Should `v2_fhe_test.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07922077922077922 - nodes in this community are weakly interconnected._
- **Should `Client Crypto & Chat Client` be split into smaller, more focused modules?**
  _Cohesion score 0.05061224489795919 - nodes in this community are weakly interconnected._
- **Should `AES Learnability Experiment` be split into smaller, more focused modules?**
  _Cohesion score 0.05026300409117475 - nodes in this community are weakly interconnected._
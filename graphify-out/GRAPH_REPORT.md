# Graph Report - cipherchat  (2026-11-20)

## Corpus Check
- Corpus is ~27,264 words - fits in a single context window. You may not need a graph.

## Summary
- 416 nodes · 883 edges · 17 communities (12 shown, 5 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 35 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Datasets & v1 Model Training
- v2 Numpy Simulator & Research Log
- v1 Reference & Model
- v2 Client Keys
- Client Crypto & Chat Client
- AES Learnability Experiment
- v2 Addition Task & Tokenization
- v2 Engine Decoding Step
- v1 FHE Engine
- v2 OpenFHE Backend
- CipherFormer2 Model & Reference (v2)
- FHE Server & CKKS Parameters
- Slot Layouts (Generalized BSGS)
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
8. `FoldedModel` - 12 edges
9. `ClientCrypto` - 12 edges
10. `CipherFormer2` - 12 edges

## Surprising Connections (you probably didn't know these)
- `from_rep_plaintexts()` --semantically_similar_to--> `BSGS diagonal matrix-vector product`  [INFERRED] [semantically similar]
  cipherchat/v2/layout.py → docs/HANDOVER.md
- `OpenFHEBackend` --references--> `9. Roadmap zur Chat-Fähigkeit („GPT-4-Klasse kommt später“)`  [INFERRED]
  cipherchat/v2/engine.py → docs/PAPER_v2.md
- `CipherFormer` --references--> `CipherFormer FHE-friendly transformer`  [EXTRACTED]
  cipherchat/model.py → README.md
- `NumpyBackend` --references--> `2. v2-Engine: Ideen und warum sie funktionieren`  [INFERRED]
  cipherchat/v2/engine.py → docs/RESEARCH_LOG.md
- `OpenFHEBackend` --references--> `Zwei Generationen im Code`  [INFERRED]
  cipherchat/v2/engine.py → CLAUDE.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **FHE-exact model chain: torch model -> folded reference -> FHE engine** — cipherchat_model_cipherformer, cipherchat_reference_foldedmodel, cipherchat_reference_plainreference, cipherchat_fhe_engine_fheservermodel [EXTRACTED 1.00]
- **Encrypted token step: encrypt -> matvec/attention -> logits -> decrypt** — cipherchat_client_crypto_clientcrypto_encrypt_token, cipherchat_fhe_engine_fheservermodel_step, cipherchat_client_crypto_clientcrypto_decrypt_logits, server_app_token, client_app_chat [INFERRED 0.95]

## Communities (17 total, 5 thin omitted)

### Community 1 - "Datasets & v1 Model Training"
Cohesion: 0.08
Nodes (18): EncryptedSession, CipherFormer, build_pairs(), eval_prompts(), _q(), sample_batch(), to_ids(), masked_ce() (+10 more)

### Community 10 - "v2 Numpy Simulator & Research Log"
Cohesion: 0.07
Nodes (14): NumpyBackend, CipherChat – Agent-Leitfaden, Erst lesen, graphify, Invarianten (Details: HANDOVER §2), Umgebung, Zwei Generationen im Code, 1. Baseline (v1-Engine, unverändert) (+6 more)

### Community 11 - "v1 Reference & Model"
Cohesion: 0.12
Nodes (5): Block, FoldedModel, PlainReference, Polynomial attention w=(q.k+c)^2/(t+1), CipherFormer FHE-friendly transformer

### Community 2 - "Client Crypto & Chat Client"
Cohesion: 0.07
Nodes (18): ClientCrypto, ChatIn, significance(), chat(), gen(), send(), connect(), frames() (+10 more)

### Community 3 - "AES Learnability Experiment"
Cohesion: 0.08
Nodes (12): LM, attack_ctr_fixed(), attack_ecb(), attack_ctr_fixed_nonce(), attack_ecb_frequency(), batches(), evaluate(), make_messages() (+4 more)

### Community 5 - "v2 Addition Task & Tokenization"
Cohesion: 0.27
Nodes (3): Task, reverse_digits(), main()

### Community 6 - "v2 Engine Decoding Step"
Cohesion: 0.16
Nodes (4): EngineV2, Known gaps (no TLS, length leakage, full logits to client), Prioritized next tasks (giant-step merge, static encodes, GPU, batched prefill), Roadmap (batched prefill, GPU FHE, BPE, WASM client, TEE hybrid)

### Community 7 - "v1 FHE Engine"
Cohesion: 0.19
Nodes (4): FHEServerModel, kmask(), vmask(), Encrypted KV-cache block layout

### Community 9 - "CipherFormer2 Model & Reference (v2)"
Cohesion: 0.08
Nodes (8): Session2, CipherFormer2, Folded2, PlainReference2, masked_ce(), input_vector(), mlp_act(), evaluate()

### Community 0 - "FHE Server & CKKS Parameters"
Cohesion: 0.07
Nodes (24): de_cc(), de_ct(), depth_for_layers(), make_context(), max_context(), n_slots(), rotation_indices(), ser() (+16 more)

### Community 4 - "Slot Layouts (Generalized BSGS)"
Cohesion: 0.10
Nodes (19): bsgs_vectors(), diagonals(), onehot_rep(), tile(), cache_add_vector(), cross_steps(), from_rep_plaintexts(), full_block0() (+11 more)

### Community 13 - "v2 Paper (docs)"
Cohesion: 0.08
Nodes (24): 10. Fazit, 1. Motivation und Ziele, 2. Ausgangslage: wohin die Zeit in v1 geht, 3. Verwandte Arbeiten, 4.1 Generalisiertes BSGS: eine Matvec, beliebiges Ausgabe-Layout, 4.2 Schiebender KV-Cache und relative Positionen, 4.3 Bilineare Scores ohne Query-Projektion, 4.4 Layer 0 als Lookup und Fusion der Summen (+16 more)

## Knowledge Gaps
- **10 isolated node(s):** `CKKS via OpenFHE (N=16384, scale 2^33, HYBRID key switching)`, `openfhe>=1.6.1 (Python 3.12 / Linux wheel)`, `Performance on 2 vCPU (2.6 s/generated token)`, `Training recipe (AdamW 3e-3, 8000 steps, logit penalty)`, `Local panel (secret key, decrypted top-3)` (+5 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 172 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `OpenFHEBackend` connect `v2 OpenFHE Backend` to `FHE Server & CKKS Parameters`, `CipherFormer2 Model & Reference (v2)`, `v2 Numpy Simulator & Research Log`, `v2 Paper (docs)`?**
  _High betweenness centrality (0.198) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `OpenFHEBackend` (e.g. with `Zwei Generationen im Code` and `9. Roadmap zur Chat-Fähigkeit („GPT-4-Klasse kommt später“)`) actually correct?**
  _`OpenFHEBackend` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `CKKS via OpenFHE (N=16384, scale 2^33, HYBRID key switching)`, `openfhe>=1.6.1 (Python 3.12 / Linux wheel)`, `Performance on 2 vCPU (2.6 s/generated token)` to the rest of the system?**
  _10 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Datasets & v1 Model Training` be split into smaller, more focused modules?**
  _Cohesion score 0.07607843137254902 - nodes in this community are weakly interconnected._
- **Why does `9. Roadmap zur Chat-Fähigkeit („GPT-4-Klasse kommt später“)` connect `v2 Paper (docs)` to `v2 OpenFHE Backend`?**
  _High betweenness centrality (0.108) - this node is a cross-community bridge._
- **Should `v2 Numpy Simulator & Research Log` be split into smaller, more focused modules?**
  _Cohesion score 0.07142857142857142 - nodes in this community are weakly interconnected._
- **Should `v1 Reference & Model` be split into smaller, more focused modules?**
  _Cohesion score 0.12105263157894737 - nodes in this community are weakly interconnected._
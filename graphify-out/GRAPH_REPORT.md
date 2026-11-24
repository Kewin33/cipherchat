# Graph Report - cipherchat  (2026-11-24)

## Corpus Check
- 82 files · ~79,432 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 13 file(s) not represented in the graph (top: .log 9, (none) 3, .ipynb 1)

## Summary
- 820 nodes · 1663 edges · 53 communities (36 shown, 17 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 74 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `27557bb6`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- server/app.py
- train_v2.py
- client/app.py
- aes_learnability.py
- Slot Layouts (Generalized BSGS)
- Task
- v2 Engine Decoding Step
- Ctx
- make_architecture_figure.py
- v2_fhe_test.py
- NumpyBackend
- v1 Reference & Model
- ser
- OpenFHEBackend
- Demo Launcher
- json
- Engine3
- Env
- v3/model.py
- NumpyBackend
- CipherChat v3 – nicht-interaktive FHE-Inferenz für Qwen2.5-0.5B auf einer RTX 5080
- fold.py
- NewtonApprox
- train_v3.py
- v3common.hpp
- PhantomBackend
- numpy
- Task
- train.py
- client_crypto.py
- Prof
- torch
- Chain
- CipherChat v2: Slot-gesättigte CKKS-Inferenz und generalisierende Arithmetik für einen verschlüsselten Chatbot
- bench_m1.cu
- ClientCrypto
- convert
- CipherChat v2 – Forschungslog
- app_v3.py
- v2_keyswitch_sweep.py
- main
- .start
- CipherChat – Agent-Leitfaden
- 4. Engine v2
- 6. Experimente
- report_pos.md
- v3_report.py
- generate
- v3_setup_remote.sh
- run_demo_all.sh
- v3_sync.sh

## God Nodes (most connected - your core abstractions)
1. `Ctx` - 55 edges
2. `Engine3` - 24 edges
3. `OpenFHEBackend` - 21 edges
4. `EngineV2` - 19 edges
5. `NumpyBackend` - 19 edges
6. `FHEServerModel` - 18 edges
7. `PhantomBackend` - 18 edges
8. `main()` - 18 edges
9. `Env` - 17 edges
10. `main()` - 16 edges

## Surprising Connections (you probably didn't know these)
- `from_rep_plaintexts()` --semantically_similar_to--> `BSGS diagonal matrix-vector product`  [INFERRED] [semantically similar]
  cipherchat/v2/layout.py → docs/HANDOVER.md
- `9. Roadmap zur Chat-Fähigkeit („GPT-4-Klasse kommt später“)` --references--> `OpenFHEBackend`  [INFERRED]
  docs/PAPER_v2.md → cipherchat/v2/engine.py
- `GPU (RTX 5080) – Projektion, nicht gemessen` --references--> `OpenFHEBackend`  [INFERRED]
  docs/RESEARCH_LOG.md → cipherchat/v2/engine.py
- `make_context()` --references--> `CKKS via OpenFHE (N=16384, scale 2^33, HYBRID key switching)`  [INFERRED]
  cipherchat/fhe_common.py → README.md
- `tile()` --references--> `rep slot layout (64-vector tiled 128x)`  [INFERRED]
  cipherchat/fhe_common.py → docs/HANDOVER.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **FHE-exact model chain: torch model -> folded reference -> FHE engine** — cipherchat_model_cipherformer, cipherchat_reference_foldedmodel, cipherchat_reference_plainreference, cipherchat_fhe_engine_fheservermodel [EXTRACTED 1.00]
- **Encrypted token step: encrypt -> matvec/attention -> logits -> decrypt** — cipherchat_client_crypto_clientcrypto_encrypt_token, cipherchat_fhe_engine_fheservermodel_step, cipherchat_client_crypto_clientcrypto_decrypt_logits, server_app_token, client_app_chat [INFERRED 0.95]

## Communities (53 total, 17 thin omitted)

### Community 0 - "server/app.py"
Cohesion: 0.08
Nodes (17): EncryptedSession, FHEServerModel, docker-compose: server + client services, Known gaps (no TLS, length leakage, full logits to client), Prioritized next tasks (giant-step merge, static encodes, GPU, batched prefill), CipherChat - HTTPS for LLMs (FHE MVP), Roadmap (batched prefill, GPU FHE, BPE, WASM client, TEE hybrid), create_session() (+9 more)

### Community 1 - "train_v2.py"
Cohesion: 0.14
Nodes (8): build_pairs(), _q(), sample_batch(), to_ids(), other_items(), split_pairs(), encode_text(), normalize()

### Community 2 - "client/app.py"
Cohesion: 0.13
Nodes (16): chat(), chat_v12(), chat_v3(), ChatIn, connect(), frames(), index(), probe() (+8 more)

### Community 3 - "aes_learnability.py"
Cohesion: 0.09
Nodes (11): attack_ctr_fixed(), attack_ecb(), attack_ctr_fixed_nonce(), attack_ecb_frequency(), batches(), evaluate(), LM, make_messages() (+3 more)

### Community 4 - "Slot Layouts (Generalized BSGS)"
Cohesion: 0.12
Nodes (14): cache_add_vector(), cross_steps(), from_rep_plaintexts(), full_block0(), full_cache_k(), full_cache_v(), full_from_hidden(), full_hidden() (+6 more)

### Community 7 - "Ctx"
Cohesion: 0.06
Nodes (18): Ctx, boot, boot0, boot_phases, boot_steps, cfg, ch, ckks (+10 more)

### Community 8 - "make_architecture_figure.py"
Cohesion: 0.40
Nodes (6): add(), arrow(), block(), box(), line(), text()

### Community 9 - "v2_fhe_test.py"
Cohesion: 0.19
Nodes (4): Folded2, input_vector(), mlp_act(), PlainReference2

### Community 11 - "v1 Reference & Model"
Cohesion: 0.12
Nodes (5): Block, FoldedModel, PlainReference, Polynomial attention w=(q.k+c)^2/(t+1), CipherFormer FHE-friendly transformer

### Community 18 - "Engine3"
Cohesion: 0.07
Nodes (8): Engine3, engine_rotation_steps(), RotPlan, Session3, 5. Engine (`cipherchat/v3/engine.py`), calib_from_wikitext(), main(), stats()

### Community 19 - "Env"
Cohesion: 0.14
Nodes (19): decrypt(), encode_for(), encrypt_at(), Env, ckks, ctx, glk, n (+11 more)

### Community 20 - "v3/model.py"
Cohesion: 0.15
Nodes (8): masked_ce(), acc_dtype(), allowed_mask(), ApproxRMSNorm, power_attention(), range_excess(), register(), torch>=2.2 (training only)

### Community 22 - "CipherChat v3 – nicht-interaktive FHE-Inferenz für Qwen2.5-0.5B auf einer RTX 5080"
Cohesion: 0.09
Nodes (13): ClientV3, V3Service, 10. Reproduktion, 1. Kurzfassung, 2. Architektur, 3. Modell-Umbau (Klartext, `cipherchat/v3/model.py`, `approx.py`), 4. CKKS-Parameter und Mikrobenchmarks (Meilenstein 1, `results/v3/m1_*.json`), 6. Qualität (Meilenstein 2) (+5 more)

### Community 23 - "fold.py"
Cohesion: 0.14
Nodes (7): build(), FoldedModel, pad(), perm_q(), rope_matrix(), wikitext_batches(), fhe_state()

### Community 24 - "NewtonApprox"
Cohesion: 0.14
Nodes (6): fit_start(), NewtonApprox, poly_eval(), search_newton(), silu(), PolyAct

### Community 25 - "train_v3.py"
Cohesion: 0.17
Nodes (7): load_config(), main(), kl_div(), main(), prefix_ids(), token_stream(), wikitext_ppl()

### Community 26 - "v3common.hpp"
Cohesion: 0.15
Nodes (5): cfg_get(), load_config(), Profiler, ms, random_vec()

### Community 28 - "numpy"
Cohesion: 0.23
Nodes (9): bsgs_vectors(), diagonals(), kmask(), onehot_rep(), tile(), vmask(), Encrypted KV-cache block layout, rep slot layout (64-vector tiled 128x) (+1 more)

### Community 29 - "Task"
Cohesion: 0.18
Nodes (6): reverse_digits(), significance(), Task, send(), evaluate(), main()

### Community 30 - "train.py"
Cohesion: 0.17
Nodes (8): eval_prompts(), CipherFormer, masked_ce(), decode(), encode_prompt(), Training recipe (AdamW 3e-3, 8000 steps, logit penalty), evaluate(), main()

### Community 31 - "client_crypto.py"
Cohesion: 0.20
Nodes (7): depth_for_layers(), make_context(), max_context(), n_slots(), rotation_indices(), make_context_v2(), CKKS via OpenFHE (N=16384, scale 2^33, HYBRID key switching)

### Community 32 - "Prof"
Cohesion: 0.16
Nodes (9): OpTimer, key, Prof, cnt, e0, e1, ms, on (+1 more)

### Community 33 - "torch"
Cohesion: 0.29
Nodes (7): fit_poly_act(), calibrate(), calibrate_inverse(), _dn_cat(), qrange(), range_penalty(), run_calib()

### Community 34 - "Chain"
Cohesion: 0.18
Nodes (10): Chain, bits, boot_levels, compute_levels, log_qp, scale, special_count, total_level (+2 more)

### Community 35 - "CipherChat v2: Slot-gesättigte CKKS-Inferenz und generalisierende Arithmetik für einen verschlüsselten Chatbot"
Cohesion: 0.17
Nodes (11): 10. Fazit, 1. Motivation und Ziele, 2. Ausgangslage: wohin die Zeit in v1 geht, 3. Verwandte Arbeiten, 5. Addition, die generalisiert, 7. LLM-Konzepte auf FHE übertragen – Analyse, 8. Sicherheit und Leakage, 9. Roadmap zur Chat-Fähigkeit („GPT-4-Klasse kommt später“) (+3 more)

### Community 37 - "ClientCrypto"
Cohesion: 0.24
Nodes (3): ClientCrypto, Local panel (secret key, decrypted top-3), Session protocol (4 length-prefixed frames)

### Community 38 - "convert"
Cohesion: 0.24
Nodes (7): AttnState, convert(), load_fhe_state(), main(), h(), h2(), rec()

### Community 39 - "CipherChat v2 – Forschungslog"
Cohesion: 0.22
Nodes (8): 1. Baseline (v1-Engine, unverändert), 2. v2-Engine: Ideen und warum sie funktionieren, 3. v2 unter echtem CKKS (OpenFHE), 4. Addition, die generalisiert, 5. Offene Punkte, CipherChat v2 – Forschungslog, GPU (RTX 5080) – Projektion, nicht gemessen, Unter echter Verschlüsselung (`cf2_rev_cpl_sqglu_h2048`)

### Community 41 - "v2_keyswitch_sweep.py"
Cohesion: 0.29
Nodes (4): de_cc(), de_ct(), Session2, context()

### Community 42 - "main"
Cohesion: 0.33
Nodes (6): main(), max_abs_err(), mb(), gpu_used_bytes(), max_logqp_128(), pool_reserved_high()

### Community 43 - ".start"
Cohesion: 0.38
Nodes (4): now_ms(), count, SyncTimer, t0

### Community 44 - "CipherChat – Agent-Leitfaden"
Cohesion: 0.29
Nodes (6): CipherChat – Agent-Leitfaden, Drei Generationen im Code, Erst lesen, graphify, Invarianten (Details: HANDOVER §2), Umgebung

### Community 45 - "4. Engine v2"
Cohesion: 0.29
Nodes (7): 4.1 Generalisiertes BSGS: eine Matvec, beliebiges Ausgabe-Layout, 4.2 Schiebender KV-Cache und relative Positionen, 4.3 Bilineare Scores ohne Query-Projektion, 4.4 Layer 0 als Lookup und Fusion der Summen, 4.5 Gated MLP im Reserve-Level (GLU / Soft-MoE), 4.6 Operationsbilanz pro generiertem Token (2 Layer, gezählt), 4. Engine v2

### Community 46 - "6. Experimente"
Cohesion: 0.33
Nodes (6): 6.1 Exaktheit, 6.2 Laufzeit und Skalierung (v2, Hidden 8.192), 6.3 Speicher, 6.4 Addition, 6.5 Ende-zu-Ende unter FHE, 6. Experimente

### Community 47 - "report_pos.md"
Cohesion: 0.40
Nodes (4): Meilenstein 3: ein Block (pos) – ms pro Schicht und Token, Meilenstein 4: ganzes Modell (pos) – pro Token, numpy-Simulation (pos, 6 Token, 24 Schichten), Qualität (pos)

### Community 48 - "v3_report.py"
Cohesion: 0.60
Nodes (3): hot_rows(), load(), main()

## Knowledge Gaps
- **50 isolated node(s):** `ctx`, `ckks`, `glk`, `rlk`, `n` (+45 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 349 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **17 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `OpenFHEBackend` connect `OpenFHEBackend` to `server/app.py`, `CipherChat v2: Slot-gesättigte CKKS-Inferenz und generalisierende Arithmetik für einen verschlüsselten Chatbot`, `Slot Layouts (Generalized BSGS)`, `CipherChat v2 – Forschungslog`, `v2_keyswitch_sweep.py`, `CipherChat – Agent-Leitfaden`, `json`?**
  _High betweenness centrality (0.123) - this node is a cross-community bridge._
- **What connects `ctx`, `ckks`, `glk` to the rest of the system?**
  _50 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `server/app.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07729468599033816 - nodes in this community are weakly interconnected._
- **Why does `Ctx` connect `Ctx` to `Prof`, `Chain`, `bench_m1.cu`, `main`?**
  _High betweenness centrality (0.121) - this node is a cross-community bridge._
- **Should `train_v2.py` be split into smaller, more focused modules?**
  _Cohesion score 0.13538461538461538 - nodes in this community are weakly interconnected._
- **Why does `9. Roadmap zur Chat-Fähigkeit („GPT-4-Klasse kommt später“)` connect `CipherChat v2: Slot-gesättigte CKKS-Inferenz und generalisierende Arithmetik für einen verschlüsselten Chatbot` to `OpenFHEBackend`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **Should `client/app.py` be split into smaller, more focused modules?**
  _Cohesion score 0.13 - nodes in this community are weakly interconnected._
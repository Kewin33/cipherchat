# Graph Report - cipherchat-v5  (2026-11-26)

## Corpus Check
- 115 files · ~111,252 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 17 file(s) not represented in the graph (top: .log 12, (none) 4, .ipynb 1)

## Summary
- 1064 nodes · 2139 edges · 73 communities (46 shown, 27 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 92 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `b7b641ac`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- server/app.py
- train.py
- client/app.py
- aes_learnability.py
- Slot Layouts (Generalized BSGS)
- CipherFormer2
- v2 Engine Decoding Step
- Ctx
- make_architecture_figure.py
- v2_fhe_test.py
- NumpyBackend
- v1 Reference & Model
- ClientCryptoV2
- OpenFHEBackend
- Demo Launcher
- main
- Engine3
- bench_m1.cu
- v3/model.py
- NumpyBackend
- CipherChat v3 – nicht-interaktive FHE-Inferenz für Qwen2.5-0.5B auf einer RTX 5080
- fold.py
- NewtonApprox
- train_v3.py
- json
- PhantomBackend
- numpy
- Engine5
- Ctx5
- fhe_common.py
- Prof
- torch
- Chain
- CipherChat v2: Slot-gesättigte CKKS-Inferenz und generalisierende Arithmetik für einen verschlüsselten Chatbot
- PYBIND11_MODULE
- ClientCrypto
- main
- CipherChat v2 – Forschungslog
- app_v3.py
- math
- PhantomCiphertext
- main
- CipherChat – Agent-Leitfaden
- Prof
- FHEServerModel
- report_pos.md
- v3_report.py
- PhantomBackend5
- v3_setup_remote.sh
- run_demo_all.sh
- v3_sync.sh
- os
- v5/engine.py
- NumpyBackendScaleDev
- service.py
- make_chain5
- connect
- Chain5
- chat
- server_view
- Layout
- patch_phantom.py
- v5_runs.sh
- PYBIND11_MODULE
- v5_setup_remote.sh
- v5_sync.sh

## God Nodes (most connected - your core abstractions)
1. `Ctx5` - 59 edges
2. `Ctx` - 55 edges
3. `Engine3` - 24 edges
4. `NumpyBackend` - 22 edges
5. `Engine5` - 21 edges
6. `OpenFHEBackend` - 20 edges
7. `EngineV2` - 19 edges
8. `FHEServerModel` - 18 edges
9. `PhantomBackend` - 18 edges
10. `main()` - 18 edges

## Surprising Connections (you probably didn't know these)
- `from_rep_plaintexts()` --semantically_similar_to--> `BSGS diagonal matrix-vector product`  [INFERRED] [semantically similar]
  cipherchat/v2/layout.py → docs/HANDOVER.md
- `GPU (RTX 5080) – Projektion, nicht gemessen` --references--> `OpenFHEBackend`  [INFERRED]
  docs/RESEARCH_LOG.md → cipherchat/v2/engine.py
- `make_context()` --references--> `CKKS via OpenFHE (N=16384, scale 2^33, HYBRID key switching)`  [INFERRED]
  cipherchat/fhe_common.py → README.md
- `tile()` --references--> `rep slot layout (64-vector tiled 128x)`  [INFERRED]
  cipherchat/fhe_common.py → docs/HANDOVER.md
- `bsgs_vectors()` --references--> `BSGS diagonal matrix-vector product`  [INFERRED]
  cipherchat/fhe_common.py → docs/HANDOVER.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **FHE-exact model chain: torch model -> folded reference -> FHE engine** — cipherchat_model_cipherformer, cipherchat_reference_foldedmodel, cipherchat_reference_plainreference, cipherchat_fhe_engine_fheservermodel [EXTRACTED 1.00]
- **Encrypted token step: encrypt -> matvec/attention -> logits -> decrypt** — cipherchat_client_crypto_clientcrypto_encrypt_token, cipherchat_fhe_engine_fheservermodel_step, cipherchat_client_crypto_clientcrypto_decrypt_logits, server_app_token, client_app_chat [INFERRED 0.95]

## Communities (73 total, 27 thin omitted)

### Community 0 - "server/app.py"
Cohesion: 0.20
Nodes (9): EncryptedSession, Session2, create_session(), log(), new_model(), new_session(), reset(), set_cc() (+1 more)

### Community 1 - "train.py"
Cohesion: 0.07
Nodes (19): build_pairs(), eval_prompts(), _q(), sample_batch(), to_ids(), CipherFormer, masked_ce(), other_items() (+11 more)

### Community 2 - "client/app.py"
Cohesion: 0.16
Nodes (12): significance(), decode(), chat_v12(), send(), index(), probe(), resolve_auto(), _softmax() (+4 more)

### Community 3 - "aes_learnability.py"
Cohesion: 0.09
Nodes (11): attack_ctr_fixed(), attack_ecb(), attack_ctr_fixed_nonce(), attack_ecb_frequency(), batches(), evaluate(), LM, make_messages() (+3 more)

### Community 4 - "Slot Layouts (Generalized BSGS)"
Cohesion: 0.12
Nodes (14): cache_add_vector(), cross_steps(), from_rep_plaintexts(), full_block0(), full_cache_k(), full_cache_v(), full_from_hidden(), full_hidden() (+6 more)

### Community 6 - "v2 Engine Decoding Step"
Cohesion: 0.16
Nodes (4): EngineV2, Known gaps (no TLS, length leakage, full logits to client), Prioritized next tasks (giant-step merge, static encodes, GPU, batched prefill), Roadmap (batched prefill, GPU FHE, BPE, WASM client, TEE hybrid)

### Community 7 - "Ctx"
Cohesion: 0.06
Nodes (18): Ctx, boot, boot0, boot_phases, boot_steps, cfg, ch, ckks (+10 more)

### Community 8 - "make_architecture_figure.py"
Cohesion: 0.40
Nodes (6): add(), arrow(), block(), box(), line(), text()

### Community 9 - "v2_fhe_test.py"
Cohesion: 0.11
Nodes (5): Folded2, input_vector(), mlp_act(), PlainReference2, context()

### Community 11 - "v1 Reference & Model"
Cohesion: 0.21
Nodes (4): Block, PlainReference, Polynomial attention w=(q.k+c)^2/(t+1), CipherFormer FHE-friendly transformer

### Community 17 - "main"
Cohesion: 0.31
Nodes (4): hotspot_table(), calib_from_wikitext(), main(), stats()

### Community 18 - "Engine3"
Cohesion: 0.08
Nodes (3): Engine3, engine_rotation_steps(), Session3

### Community 19 - "bench_m1.cu"
Cohesion: 0.12
Nodes (20): decrypt(), encode_for(), encrypt_at(), Env, ckks, ctx, glk, n (+12 more)

### Community 20 - "v3/model.py"
Cohesion: 0.11
Nodes (11): acc_dtype(), allowed_mask(), ApproxRMSNorm, AttnState, convert(), fhe_state(), PolyAct, power_attention() (+3 more)

### Community 22 - "CipherChat v3 – nicht-interaktive FHE-Inferenz für Qwen2.5-0.5B auf einer RTX 5080"
Cohesion: 0.20
Nodes (9): 10. Reproduktion, 1. Kurzfassung, 3. Modell-Umbau (Klartext, `cipherchat/v3/model.py`, `approx.py`), 4. CKKS-Parameter und Mikrobenchmarks (Meilenstein 1, `results/v3/m1_*.json`), 6. Qualität (Meilenstein 2), 7. Ein Block verschlüsselt (Meilenstein 3, `results/v3/m3_block_pos.json`), 8. Ganzes Modell (Meilenstein 4, `results/v3/m4_full_pos.json`), 9. Grenzen und offene Punkte (+1 more)

### Community 23 - "fold.py"
Cohesion: 0.16
Nodes (6): build(), FoldedModel, pad(), perm_q(), rope_matrix(), wikitext_batches()

### Community 24 - "NewtonApprox"
Cohesion: 0.21
Nodes (3): fit_start(), NewtonApprox, search_newton()

### Community 25 - "train_v3.py"
Cohesion: 0.19
Nodes (7): load_config(), main(), kl_div(), main(), prefix_ids(), token_stream(), wikitext_ppl()

### Community 26 - "json"
Cohesion: 0.13
Nodes (7): cfg_get(), load_config(), Profiler, ms, random_vec(), gpu_used_bytes(), pool_reserved_high()

### Community 28 - "numpy"
Cohesion: 0.14
Nodes (10): bsgs_vectors(), diagonals(), kmask(), onehot_rep(), tile(), vmask(), FoldedModel, Encrypted KV-cache block layout (+2 more)

### Community 29 - "Engine5"
Cohesion: 0.06
Nodes (18): RotPlan, Engine5, 5. Engine (`cipherchat/v3/engine.py`), 10. Reproduktion, 1. Kurzfassung, 2. Was v5 ändert, 4. Laufzeit: wohin die Zeit geht (pro Token, GPU-ms, letztes Token), 5. Qualität (+10 more)

### Community 30 - "Ctx5"
Cohesion: 0.07
Nodes (14): Ctx5, cfg, ch, context, encoder, fused, glk, mac_chunk (+6 more)

### Community 31 - "fhe_common.py"
Cohesion: 0.15
Nodes (12): de_cc(), de_ct(), depth_for_layers(), make_context(), max_context(), n_slots(), rotation_indices(), ser() (+4 more)

### Community 32 - "Prof"
Cohesion: 0.16
Nodes (9): OpTimer, key, Prof, cnt, e0, e1, ms, on (+1 more)

### Community 33 - "torch"
Cohesion: 0.23
Nodes (8): fit_poly_act(), poly_eval(), silu(), calibrate(), calibrate_inverse(), _dn_cat(), qrange(), run_calib()

### Community 34 - "Chain"
Cohesion: 0.18
Nodes (10): Chain, bits, boot_levels, compute_levels, log_qp, scale, special_count, total_level (+2 more)

### Community 35 - "CipherChat v2: Slot-gesättigte CKKS-Inferenz und generalisierende Arithmetik für einen verschlüsselten Chatbot"
Cohesion: 0.05
Nodes (40): 10. Fazit, 1. Motivation und Ziele, 2. Ausgangslage: wohin die Zeit in v1 geht, 3. Verwandte Arbeiten, 4.1 Generalisiertes BSGS: eine Matvec, beliebiges Ausgabe-Layout, 4.2 Schiebender KV-Cache und relative Positionen, 4.3 Bilineare Scores ohne Query-Projektion, 4.4 Layer 0 als Lookup und Fusion der Summen (+32 more)

### Community 38 - "main"
Cohesion: 0.60
Nodes (5): load_fhe_state(), main(), h(), h2(), rec()

### Community 39 - "CipherChat v2 – Forschungslog"
Cohesion: 0.20
Nodes (9): 1. Baseline (v1-Engine, unverändert), 2. v2-Engine: Ideen und warum sie funktionieren, 3. v2 unter echtem CKKS (OpenFHE), 4. Addition, die generalisiert, 5. Offene Punkte, 6. v5: Obergrenze ohne Fehlerterm (26.11.2026, NICHT SICHER), CipherChat v2 – Forschungslog, GPU (RTX 5080) – Projektion, nicht gemessen (+1 more)

### Community 40 - "app_v3.py"
Cohesion: 0.14
Nodes (5): generate(), GenIn, info(), server_view(), _startup()

### Community 41 - "math"
Cohesion: 0.17
Nodes (3): masked_ce(), torch>=2.2 (training only), model()

### Community 43 - "main"
Cohesion: 0.21
Nodes (9): main(), mb(), gpu_used_bytes(), max_logqp_128(), now_ms(), pool_reserved_high(), count, SyncTimer (+1 more)

### Community 44 - "CipherChat – Agent-Leitfaden"
Cohesion: 0.29
Nodes (6): CipherChat – Agent-Leitfaden, Erst lesen, Generationen im Code, graphify, Invarianten (Details: HANDOVER §2), Umgebung

### Community 45 - "Prof"
Cohesion: 0.13
Nodes (10): now_ms(), OpTimer, key, Prof, cnt, e0, e1, ms (+2 more)

### Community 47 - "report_pos.md"
Cohesion: 0.40
Nodes (4): Meilenstein 3: ein Block (pos) – ms pro Schicht und Token, Meilenstein 4: ganzes Modell (pos) – pro Token, numpy-Simulation (pos, 6 Token, 24 Schichten), Qualität (pos)

### Community 48 - "v3_report.py"
Cohesion: 0.60
Nodes (3): hot_rows(), load(), main()

### Community 53 - "os"
Cohesion: 0.22
Nodes (4): main(), main(), tok_stats(), main()

### Community 54 - "v5/engine.py"
Cohesion: 0.20
Nodes (4): NumpyBackendLPP, rotation_keys(), Session5, used_rotation_steps()

### Community 55 - "NumpyBackendScaleDev"
Cohesion: 0.20
Nodes (3): chain_primes(), NumpyBackendScaleDev, 6. Was der Rescale-Rundungsfehler noch ausmacht

### Community 56 - "service.py"
Cohesion: 0.22
Nodes (3): ClientV3, V3Service, 2. Architektur

### Community 57 - "make_chain5"
Cohesion: 0.22
Nodes (7): MacPtrs, ct, pt, make_chain5(), v5_mac_kernel(), v5_scalar_mul_kernel(), 3. PhantomFHE-Patches (`cipherchat/v5/fhe/patch_phantom.py`)

### Community 59 - "connect"
Cohesion: 0.25
Nodes (4): connect(), frames(), Session protocol (4 length-prefixed frames), unpack_frames()

### Community 60 - "Chain5"
Cohesion: 0.25
Nodes (8): Chain5, bits, levels, log_q, log_qp, lpp, nscale, special_count

### Community 61 - "chat"
Cohesion: 0.29
Nodes (5): chat(), chat_v3(), ChatIn, status(), Chat UI (SSE streaming)

### Community 62 - "server_view"
Cohesion: 0.40
Nodes (3): server_view(), Server-view panel (what the server sees), server_view()

### Community 67 - "v5_runs.sh"
Cohesion: 0.67
Nodes (3): LD_LIBRARY_PATH, v5_runs.sh script, wait_gpu()

## Knowledge Gaps
- **80 isolated node(s):** `ctx`, `ckks`, `glk`, `rlk`, `n` (+75 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 457 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **27 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Ctx5` connect `Ctx5` to `.lift`, `.mul_pt_sum`, `PhantomCiphertext`, `Prof`, `make_chain5`, `json`, `Chain5`?**
  _High betweenness centrality (0.086) - this node is a cross-community bridge._
- **What connects `ctx`, `ckks`, `glk` to the rest of the system?**
  _80 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `train.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06623376623376623 - nodes in this community are weakly interconnected._
- **Why does `Ctx` connect `Ctx` to `Prof`, `main`, `Chain`, `bench_m1.cu`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Should `aes_learnability.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09230769230769231 - nodes in this community are weakly interconnected._
- **Why does `OpenFHEBackend` connect `OpenFHEBackend` to `server/app.py`, `Slot Layouts (Generalized BSGS)`, `CipherChat v2 – Forschungslog`, `v2_fhe_test.py`, `CipherChat – Agent-Leitfaden`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **Should `Slot Layouts (Generalized BSGS)` be split into smaller, more focused modules?**
  _Cohesion score 0.1206896551724138 - nodes in this community are weakly interconnected._
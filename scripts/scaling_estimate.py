"""Back-of-the-envelope cost model for FHE-native decoder models (Section 9 of docs/PAPER_v2.md).

All numbers are ESTIMATES. Assumptions (documented in the paper):
  * CKKS N = 2^16 (n = 32,768 slots), linear layers with full slot utilisation (interleaved/replicated
    packing a la Cachemir): plaintext mults ~= weights / n, rotations ~= 2*sqrt(weights/n) per matvec.
  * FHE-native layer (this work): polynomial attention, no LayerNorm, GLU^2 MLP -> ~4.5 levels/layer,
    ~12 usable levels between bootstraps; one residual ciphertext per bootstrap while d <= n.
  * Attention over a packed KV cache: 2 * ceil(ctx * d_kv / n) ct*ct mults + 6 rotations each.
  * GPU per-op costs (RTX 4090/5090 class, N=2^16): rotation 0.8 ms (avg over levels; FIDESlib: 1.1 ms at
    top level), ct*ct 1.0 ms, pt mult 0.04 ms, bootstrapping 20 ms (Theodosian: 12.8-22 ms on RTX 5090).
  * Plaintext weights: ~16 MB per encoded plaintext at full level (65,536 coeffs x ~30 limbs x 8 B);
    when they do not fit in GPU memory they must be re-encoded on the fly (~0.1 ms each, assumption).
"""
import math

N_SLOTS = 32768
GPU = dict(rot=0.8e-3, ctct=1.0e-3, pt=0.04e-3, bts=20e-3, encode=0.1e-3)
PT_MB = 16.0


def model(name, d, layers, d_ff, vocab, ctx, kv_frac=1.0, glu=True, stored_params=None):
    """d_ff = MLP width that is actually evaluated per token (for MoE: active experts only, or all experts when
    evaluated densely); stored_params = all weights the server must hold as plaintexts (MoE: all experts)."""
    d_kv = int(d * kv_frac)
    lin = d * d * 2 + d * d_kv * 2 + d_ff * d * (3 if glu else 2)       # q,o + k,v + MLP per layer
    matvecs = [d * d, d * d, d * d_kv, d * d_kv] + [d_ff * d] * (3 if glu else 2)
    pt = layers * lin / N_SLOTS + vocab * d / N_SLOTS
    rot = layers * sum(2 * math.sqrt(max(1.0, m / N_SLOTS)) + math.log2(N_SLOTS / min(d, N_SLOTS)) for m in matvecs)
    kv_ct = math.ceil(ctx * d_kv / N_SLOTS)
    rot += layers * 2 * kv_ct * 6
    ctct = layers * (2 * kv_ct + 2 * math.ceil(d_ff / N_SLOTS) + 1)
    bts = math.ceil(layers * 4.5 / 12) * math.ceil(d / N_SLOTS)
    params = layers * lin + vocab * d
    stored = stored_params or params
    pt_mem_tb = stored / N_SLOTS * PT_MB / 1e6
    t_compute = rot * GPU["rot"] + ctct * GPU["ctct"] + pt * GPU["pt"] + bts * GPU["bts"]
    t_encode = pt * GPU["encode"]
    return dict(name=name, params=params, stored=stored, pt=pt, rot=rot, ctct=ctct, bts=bts, pt_mem_tb=pt_mem_tb,
                t_gpu_cached=t_compute, t_gpu_stream=t_compute + t_encode)


rows = [
    model("B: Domaenen-Assistent", d=256, layers=6, d_ff=1024, vocab=8000, ctx=256),
    model("C: GPT-2-small-Klasse", d=768, layers=12, d_ff=3072, vocab=32000, ctx=1024),
    model("D: 1B (Llama-3.2-1B-Form)", d=2048, layers=16, d_ff=8192, vocab=128256, ctx=2048, kv_frac=0.25),
    model("E: 8B (Llama-3-8B-Form)", d=4096, layers=32, d_ff=14336, vocab=128256, ctx=1024, kv_frac=0.25),
    # DeepSeek-V3 shape: d=7168, 61 layers, 256 routed + 1 shared expert of width 2048, 8 routed active
    model("F1: Frontier-MoE, nur aktive Experten (Routing leakt)", d=7168, layers=61, d_ff=9 * 2048, vocab=129280,
          ctx=4096, kv_frac=0.25, stored_params=671e9),
    model("F2: Frontier-MoE, alle Experten dicht (kein Leakage)", d=7168, layers=61, d_ff=257 * 2048, vocab=129280,
          ctx=4096, kv_frac=0.25, stored_params=671e9),
]
for r in rows:
    print(f"{r['name']:55s} eval {r['params']/1e9:7.2f} B / stored {r['stored']/1e9:7.1f} B | ptmul {r['pt']:10.0f} | rot {r['rot']:8.0f} | "
          f"ct*ct {r['ctct']:6.0f} | bts {r['bts']:3d} | pt-mem {r['pt_mem_tb']:9.2f} TB | "
          f"GPU cached {r['t_gpu_cached']:7.2f} s | GPU stream {r['t_gpu_stream']:7.2f} s")

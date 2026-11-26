// CipherChat v5 – Python binding of the GPU CKKS backend WITHOUT bootstrapping, built against a patched PhantomFHE
// (cipherchat/v5/fhe/patch_phantom.py).  With PHANTOM_ZERO_ERROR the RLWE error term is 0 everywhere:
//
//      *** NOT SECURE. Upper-bound experiment only. The secret key is computable from the public key. ***
//
// Differences to v3fhe (cipherchat/v3/fhe/src/pyv3fhe.cu):
//   - no bootstrapper: the modulus chain is long enough for the whole model (hundreds of primes)
//   - one special prime and single-digit key switching (PHANTOM_SINGLE_DIGIT_KS): every key is one ciphertext big
//   - decrypt() first drops the ciphertext to <= 2 primes (decoding big chains is not supported by the patched build)
//   - fused plaintext multiply-accumulate for the BSGS inner products (v5_mac_kernel)
//   - optionally two levels per 60-bit prime (levels_per_prime = 2, see Chain5)
//
// Level convention as in v3: level = number of multiplications left; with one level per prime that is #primes - 1.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "phantom.h"

namespace py = pybind11;
using json = nlohmann::json;
using namespace phantom;

static size_t gpu_used_bytes() {
  cudaDeviceSynchronize();
  size_t f, t;
  cudaMemGetInfo(&f, &t);
  return t - f;
}

static size_t pool_reserved_high() {
  cudaMemPool_t pool;
  cudaDeviceGetDefaultMemPool(&pool, 0);
  uint64_t v = 0;
  cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &v);
  return v;
}

static double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Prof {
  bool on = false;
  std::string section = "-";
  std::map<std::string, double> ms;
  std::map<std::string, long> cnt;
  cudaEvent_t e0, e1;
  Prof() {
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
  }
};

struct OpTimer {
  Prof &p;
  std::string key;
  OpTimer(Prof &p_, const char *op) : p(p_) {
    if (!p.on) return;
    key = p.section + "|" + op;
    cudaDeviceSynchronize();
    cudaEventRecord(p.e0, 0);
  }
  ~OpTimer() {
    if (!p.on) return;
    cudaDeviceSynchronize();
    cudaEventRecord(p.e1, 0);
    cudaEventSynchronize(p.e1);
    float t = 0;
    cudaEventElapsedTime(&t, p.e0, p.e1);
    p.ms[key] += t;
    p.cnt[key] += 1;
  }
};

// ---------------------------------------------------------------- fused sum_b ct_b * pt_b (NTT domain)
// One pass over memory instead of k (multiply_plain + add): products accumulate lazily in 128 bit and are
// Barrett-reduced every `red_every` terms (k q^2 < q 2^62 keeps the single-subtraction Barrett exact).
constexpr int MAC_MAX = 16;
struct MacPtrs {
  const uint64_t *ct[MAC_MAX];
  const uint64_t *pt[MAC_MAX];
};

__global__ void v5_mac_kernel(uint64_t *out, MacPtrs p, int k, bool accumulate, const DModulus *mod,
                              size_t n, size_t nprimes, int red_every) {
  const size_t per_poly = nprimes * n;
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < 2 * per_poly; tid += blockDim.x * gridDim.x) {
    const size_t idx = tid % per_poly;
    const size_t prime = idx / n;
    const uint64_t q = mod[prime].value();
    const uint64_t *mu = mod[prime].const_ratio();
    arith::uint128_t acc;
    acc.hi = 0;
    acc.lo = accumulate ? out[tid] : 0;
    int c = accumulate ? 1 : 0;
    for (int b = 0; b < k; b++) {
      arith::uint128_t prod = arith::multiply_uint64_uint64(p.ct[b][tid], p.pt[b][idx]);
      arith::add_uint128_uint128(acc, prod, acc);
      if (++c >= red_every) {
        acc.lo = arith::barrett_reduce_uint128_uint64(acc, q, mu);
        acc.hi = 0;
        c = 1;
      }
    }
    out[tid] = arith::barrett_reduce_uint128_uint64(acc, q, mu);
  }
}

__global__ void v5_scalar_mul_kernel(uint64_t *data, uint64_t c, const DModulus *mod, size_t n, size_t nprimes) {
  const size_t per_poly = nprimes * n;
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < 2 * per_poly; tid += blockDim.x * gridDim.x) {
    const size_t prime = (tid % per_poly) / n;
    data[tid] = arith::multiply_and_barrett_reduce_uint64(data[tid], c, mod[prime].value(), mod[prime].const_ratio());
  }
}

// Modulus chain. levels_per_prime (lpp) = 1: one scale prime of `scale_bits` per level (as v3).
// lpp = 2: primes of `prime_bits` (~2^60) with scale Delta = 2^scale_bits (~2^30); a ciphertext is either "full"
// (scale Delta) or "half" (scale Delta^2, one multiplication done but not rescaled). A rescale by q ~ Delta^2 takes
// Delta^3 -> Delta and Delta^4 -> Delta^2, so one 64-bit word per coefficient carries two levels.
// Virtual level v = lpp == 1 ? l : 2 l + (full ? 1 : 0) with l = #primes - 1. Every multiplication costs one virtual
// level, except a product of two half ciphertexts at the same prime level (two).
struct Chain5 {
  std::vector<int> bits;  // q0, scale primes, special primes
  int levels, lpp, nscale, special_count, log_q, log_qp;
  double scale;
};

static Chain5 make_chain5(const json &ck) {
  Chain5 ch;
  ch.levels = ck["levels"];
  ch.lpp = ck.value("levels_per_prime", 1);
  if (ch.lpp != 1 && ch.lpp != 2) throw std::invalid_argument("levels_per_prime must be 1 or 2");
  ch.special_count = ck.value("special_count", 1);
  ch.nscale = ch.lpp == 1 ? ch.levels : (ch.levels + 1) / 2;
  int pb = ch.lpp == 1 ? (int)ck["scale_bits"] : (int)ck.value("prime_bits", 2 * (int)ck["scale_bits"]);
  ch.bits.push_back(ck["base_bits"]);
  for (int i = 0; i < ch.nscale; i++) ch.bits.push_back(pb);
  for (int i = 0; i < ch.special_count; i++) ch.bits.push_back(ck["special_bits"]);
  ch.log_q = 0;
  for (size_t i = 0; i + ch.special_count < ch.bits.size(); i++) ch.log_q += ch.bits[i];
  ch.log_qp = ch.log_q + ch.special_count * (int)ck["special_bits"];
  ch.scale = std::pow(2.0, (int)ck["scale_bits"]);
  // single-digit key switching converts all of Q_l to P in one 128-bit accumulation (Phantom's bconv_matmul):
  // #primes * 2^(prime bits) * 2^(special bits) must stay below 2^128
  double acc_bits = std::log2(std::ldexp(1.0, (int)ck["base_bits"]) + ch.nscale * std::ldexp(1.0, pb)) +
                    (int)ck["special_bits"];
  if (acc_bits >= 127.5)
    throw std::invalid_argument("base conversion would overflow 128 bit (" + std::to_string(acc_bits) +
                                " bits): use a smaller special prime (special_bits)");
  return ch;
}

class Ctx5 {
 public:
  json cfg;
  Chain5 ch;
  std::unique_ptr<PhantomContext> context;
  std::unique_ptr<PhantomSecretKey> sk;
  std::unique_ptr<PhantomPublicKey> pk;
  std::unique_ptr<PhantomRelinKey> rlk;
  PhantomGaloisKey glk;
  std::unique_ptr<PhantomCKKSEncoder> encoder;
  size_t n;
  Prof prof;
  json timing = json::object();
  bool fused = true;
  int mac_chunk = 4;  // plaintexts alive at once in the fused product (each one ciphertext polynomial big)

  explicit Ctx5(const std::string &cfg_json) {
    cfg = json::parse(cfg_json);
    const json &ck = cfg["ckks"];
    ch = make_chain5(ck);
    size_t N = size_t(1) << (int)ck["logN"];
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    double t0 = now_ms();
    parms.set_coeff_modulus(arith::CoeffModulus::Create(N, ch.bits));
    parms.set_special_modulus_size(ch.special_count);
    int h = ck.value("hamming_weight", 0);
    if (h > 0) parms.set_secret_key_hamming_weight(h);
    timing["primes_ms"] = now_ms() - t0;
    t0 = now_ms();
    context.reset(new PhantomContext(parms));
    timing["context_ms"] = now_ms() - t0;
    t0 = now_ms();
    sk.reset(new PhantomSecretKey(*context));
    pk.reset(new PhantomPublicKey(sk->gen_publickey(*context)));
    rlk.reset(new PhantomRelinKey(sk->gen_relinkey(*context)));
    cudaDeviceSynchronize();
    timing["sk_pk_relin_ms"] = now_ms() - t0;
    encoder.reset(new PhantomCKKSEncoder(*context));
    n = encoder->slot_count();
  }

  int gen_rotation_keys(std::vector<int> steps) {
    for (auto &s : steps) s = ((s % (int)n) + (int)n) % (int)n;
    std::sort(steps.begin(), steps.end());
    steps.erase(std::unique(steps.begin(), steps.end()), steps.end());
    steps.erase(std::remove(steps.begin(), steps.end(), 0), steps.end());
    double t0 = now_ms();
    glk = sk->create_galois_keys_from_steps(*context, steps);
    cudaDeviceSynchronize();
    timing["galois_ms"] = now_ms() - t0;
    return (int)steps.size();
  }

  // ---------------- level / scale bookkeeping
  double D() const { return ch.scale; }
  double D2() const { return ch.scale * ch.scale; }
  int plevel(const PhantomCiphertext &ct) const { return (int)ct.coeff_modulus_size() - 1; }
  bool is_full(const PhantomCiphertext &ct) const { return ch.lpp == 1 || ct.scale() < std::pow(ch.scale, 1.5); }
  int level(const PhantomCiphertext &ct) const {
    return ch.lpp == 1 ? plevel(ct) : 2 * plevel(ct) + (is_full(ct) ? 1 : 0);
  }
  size_t chain_for_plevel(int l) const { return (size_t)(1 + ch.nscale - l); }

  uint64_t q_last(const PhantomCiphertext &ct) const {
    return context->get_context_data(ct.chain_index()).parms().coeff_modulus().back().value();
  }

  const DModulus *dmod() const { return context->gpu_rns_tables().modulus(); }
  cudaStream_t stream() const { return phantom::util::global_variables::default_stream->get_stream(); }

  void scalar_mul(PhantomCiphertext &ct, uint64_t c) {
    size_t total = 2 * ct.coeff_modulus_size() * 2 * n;
    int threads = 256;
    int blocks = (int)std::min<size_t>((total + threads - 1) / threads, 65535);
    v5_scalar_mul_kernel<<<blocks, threads, 0, stream()>>>(ct.data(), c, dmod(), 2 * n, ct.coeff_modulus_size());
  }

  void switch_plevel(PhantomCiphertext &ct, int l) {
    size_t ci = chain_for_plevel(l);
    if (ci > ct.chain_index()) ::mod_switch_to_inplace(*context, ct, ci);
  }

  // half (Delta^2) at prime level l -> full (Delta) at l - 1: multiply by round(q_last / Delta), rescale
  void lift(PhantomCiphertext &ct) {
    OpTimer t(prof, "lift");
    uint64_t c = (uint64_t)std::llround((double)q_last(ct) / D());
    scalar_mul(ct, c);
    ::rescale_to_next_inplace(*context, ct);
    ct.scale() = D();
  }

  // full (Delta) -> half (Delta^2) at the same prime level: multiply by the integer Delta (exact)
  void to_half(PhantomCiphertext &ct) {
    OpTimer t(prof, "to_half");
    scalar_mul(ct, (uint64_t)D());
    ct.scale() = D2();
  }

  // bring ct down to virtual level v (no-op if it is not above v)
  void set_level(PhantomCiphertext &ct, int v) {
    if (level(ct) <= v) return;
    if (ch.lpp == 1) {
      switch_plevel(ct, v);
      return;
    }
    int lt = v / 2;
    bool want_full = v % 2 == 1;
    if (want_full) {
      if (!is_full(ct)) {  // half at l > lt
        switch_plevel(ct, lt + 1);
        lift(ct);
      } else {
        switch_plevel(ct, lt);
      }
    } else {
      switch_plevel(ct, lt);
      if (is_full(ct)) to_half(ct);
    }
  }

  void align(PhantomCiphertext &a, PhantomCiphertext &b) {
    int v = std::min(level(a), level(b));
    set_level(a, v);
    set_level(b, v);
    b.scale() = a.scale();  // nominal scales; true scales differ by ~|q - Delta^lpp| / q
  }

  std::vector<double> vec_of(py::array_t<double, py::array::c_style | py::array::forcecast> a) const {
    if ((size_t)a.size() > n) throw std::invalid_argument("vector longer than slot count");
    std::vector<double> v(a.data(), a.data() + a.size());
    v.resize(n, 0.0);
    return v;
  }

  // ---------------- operations
  PhantomCiphertext encrypt(py::array_t<double, py::array::c_style | py::array::forcecast> a, int lvl) {
    OpTimer t(prof, "encrypt");
    PhantomPlaintext pt;
    auto v = vec_of(a);
    encoder->encode(*context, v, D(), pt, 1);
    PhantomCiphertext ct;
    pk->encrypt_asymmetric(*context, pt, ct);
    ct.scale() = D();
    set_level(ct, lvl);
    return ct;
  }

  py::array_t<double> decrypt(const PhantomCiphertext &ct) {
    OpTimer t(prof, "decrypt");
    PhantomCiphertext c = ct;
    set_level(c, ch.lpp == 1 ? 1 : 3);  // q0 + q1: decodable (patched build decodes <= 64 primes)
    PhantomPlaintext pt;
    sk->decrypt(*context, c, pt);
    std::vector<double> v;
    encoder->decode(*context, pt, v);
    py::array_t<double> out(v.size());
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
  }

  PhantomCiphertext mod_down(const PhantomCiphertext &ct, int lvl) {
    OpTimer t(prof, "mod_down");
    PhantomCiphertext c = ct;
    set_level(c, lvl);
    return c;
  }

  PhantomCiphertext add(const PhantomCiphertext &a, const PhantomCiphertext &b) {
    PhantomCiphertext x = a, y = b;
    align(x, y);
    OpTimer t(prof, "add");
    ::add_inplace(*context, x, y);
    return x;
  }

  PhantomCiphertext sub(const PhantomCiphertext &a, const PhantomCiphertext &b) {
    PhantomCiphertext x = a, y = b;
    align(x, y);
    OpTimer t(prof, "add");
    ::sub_inplace(*context, x, y);
    return x;
  }

  PhantomCiphertext add_pt(const PhantomCiphertext &a, py::array_t<double, py::array::c_style | py::array::forcecast> v) {
    std::vector<double> vv = vec_of(v);
    PhantomPlaintext pt;
    {
      OpTimer t(prof, "encode");
      encoder->encode(*context, vv, a.scale(), pt, a.chain_index());
    }
    OpTimer t(prof, "add");
    PhantomCiphertext x = a;
    ::add_plain_inplace(*context, x, pt);
    return x;
  }

  PhantomCiphertext mul_int(const PhantomCiphertext &a, long k) {
    OpTimer t(prof, "add");
    if (k == 0) throw std::invalid_argument("mul_int by 0");
    PhantomCiphertext acc = a;
    bool neg = k < 0;
    k = std::labs(k);
    for (long i = 1; i < k; i++) ::add_inplace(*context, acc, a);
    if (neg) ::negate_inplace(*context, acc);
    return acc;
  }

  // sum_i ct_i * v_i. lpp = 1 (or half inputs): plaintexts at scale q_last * Delta / s, one rescale -> Delta.
  // lpp = 2 with full inputs: plaintexts at scale Delta, no rescale -> half (Delta^2).
  PhantomCiphertext mul_pt_sum(std::vector<PhantomCiphertext> cts,
                               std::vector<py::array_t<double, py::array::c_style | py::array::forcecast>> vs) {
    if (cts.size() != vs.size() || cts.empty()) throw std::invalid_argument("mul_pt_sum: size mismatch");
    int v = level(cts[0]);
    for (auto &c : cts) v = std::min(v, level(c));
    for (auto &c : cts) set_level(c, v);
    const bool rescale = ch.lpp == 1 || !is_full(cts[0]);
    const double s_in = ch.lpp == 1 ? D() : (is_full(cts[0]) ? D() : D2());
    const double q = (double)q_last(cts[0]);
    const double pt_scale = rescale ? q * D() / s_in : D();
    PhantomCiphertext acc = fused ? mac_fused(cts, vs, pt_scale) : mac_plain(cts, vs, pt_scale);
    if (rescale) {
      OpTimer t(prof, "rescale");
      ::rescale_to_next_inplace(*context, acc);
      acc.scale() = D();
    } else {
      acc.scale() = D2();
    }
    return acc;
  }

  PhantomCiphertext mac_plain(std::vector<PhantomCiphertext> &cts,
                              std::vector<py::array_t<double, py::array::c_style | py::array::forcecast>> &vs,
                              double pt_scale) {
    PhantomCiphertext acc;
    for (size_t i = 0; i < cts.size(); i++) {
      std::vector<double> v = vec_of(vs[i]);
      PhantomPlaintext pt;
      {
        OpTimer t(prof, "encode");
        encoder->encode(*context, v, pt_scale, pt, cts[i].chain_index());
      }
      OpTimer t(prof, "ptmul");
      auto prod = ::multiply_plain(*context, cts[i], pt);
      if (i == 0)
        acc = prod;
      else
        ::add_inplace(*context, acc, prod);
    }
    return acc;
  }

  // one pass over memory instead of k (multiply_plain + add)
  PhantomCiphertext mac_fused(std::vector<PhantomCiphertext> &cts,
                              std::vector<py::array_t<double, py::array::c_style | py::array::forcecast>> &vs,
                              double pt_scale) {
    const size_t ci = cts[0].chain_index();
    const size_t nprimes = cts[0].coeff_modulus_size();
    const size_t N = 2 * n;
    uint64_t qmax = 0;
    for (auto &m : context->get_context_data(ci).parms().coeff_modulus()) qmax = std::max(qmax, m.value());
    const int red_every = (int)std::max<double>(1.0, std::floor(std::ldexp(1.0, 62) / (double)qmax));
    PhantomCiphertext acc = cts[0];  // allocation with the right shape; overwritten
    for (size_t c0 = 0; c0 < cts.size(); c0 += mac_chunk) {
      size_t k = std::min(cts.size() - c0, (size_t)mac_chunk);
      std::vector<PhantomPlaintext> pts(k);
      {
        OpTimer t(prof, "encode");
        for (size_t b = 0; b < k; b++) {
          std::vector<double> v = vec_of(vs[c0 + b]);
          encoder->encode(*context, v, pt_scale, pts[b], ci);
        }
      }
      OpTimer t(prof, "ptmac");
      MacPtrs p{};
      for (size_t b = 0; b < k; b++) {
        p.ct[b] = cts[c0 + b].data();
        p.pt[b] = pts[b].data();
      }
      size_t total = 2 * nprimes * N;
      int threads = 256;
      int blocks = (int)std::min<size_t>((total + threads - 1) / threads, 65535);
      v5_mac_kernel<<<blocks, threads, 0, stream()>>>(acc.data(), p, (int)k, c0 > 0, dmod(), N, nprimes, red_every);
    }  // plaintext buffers are released stream-ordered on the same stream
    return acc;
  }

  PhantomCiphertext mul(const PhantomCiphertext &a, const PhantomCiphertext &b) {
    PhantomCiphertext x = a, y = b;
    if (ch.lpp == 1) {
      align(x, y);
    } else {
      // common prime level; an operand that has a spare virtual level enters full (lift / mod-switch)
      int l = std::min(plevel(x), plevel(y));
      set_level(x, level(x) >= 2 * l + 1 ? 2 * l + 1 : 2 * l);
      set_level(y, level(y) >= 2 * l + 1 ? 2 * l + 1 : 2 * l);
    }
    OpTimer t(prof, "ctmul");
    int nfull = ch.lpp == 1 ? 2 : (int)is_full(x) + (int)is_full(y);
    ::multiply_and_relin_inplace(*context, x, y, *rlk);
    if (ch.lpp == 1) {
      ::rescale_to_next_inplace(*context, x);
      x.scale() = D();
    } else if (nfull == 2) {
      x.scale() = D2();  // Delta^2: no rescale
    } else {
      ::rescale_to_next_inplace(*context, x);
      x.scale() = nfull == 1 ? D() : D2();  // Delta^3 / q ~ Delta, Delta^4 / q ~ Delta^2
    }
    return x;
  }

  PhantomCiphertext rot(const PhantomCiphertext &a, int k) {
    OpTimer t(prof, "rot");
    PhantomCiphertext x = a;
    k = ((k % (int)n) + (int)n) % (int)n;
    if (k) ::rotate_vector_inplace(*context, x, k, glk);
    return x;
  }

  py::dict profile() {
    py::dict d;
    for (auto &kv : prof.ms) d[py::str(kv.first)] = py::make_tuple(kv.second, prof.cnt[kv.first]);
    return d;
  }

  void reset_profile() {
    prof.ms.clear();
    prof.cnt.clear();
  }

  py::dict mem_info() {
    py::dict d;
    d["used_mb"] = gpu_used_bytes() / 1048576.0;
    d["pool_high_mb"] = pool_reserved_high() / 1048576.0;
    return d;
  }

  py::dict info() {
    py::dict d;
    d["slots"] = n;
    d["levels"] = ch.lpp == 1 ? ch.nscale : 2 * ch.nscale + 1;  // virtual level of a fresh ciphertext
    d["compute_levels"] = d["levels"];
    d["levels_per_prime"] = ch.lpp;
    d["scale_primes"] = ch.nscale;
    d["log_q"] = ch.log_q;
    d["log_qp"] = ch.log_qp;
    d["n_primes"] = (int)ch.bits.size();
#ifdef PHANTOM_ZERO_ERROR
    d["zero_error"] = true;
#else
    d["zero_error"] = false;
#endif
#ifdef PHANTOM_SINGLE_DIGIT_KS
    d["single_digit_ks"] = true;
#else
    d["single_digit_ks"] = false;
#endif
    d["timing"] = py::str(timing.dump());
    return d;
  }

  size_t ct_bytes(const PhantomCiphertext &ct) const { return 2 * 2 * n * ct.coeff_modulus_size() * 8; }
};

PYBIND11_MODULE(v5fhe, m) {
  m.doc() = "CipherChat v5 GPU CKKS backend without bootstrapping (patched PhantomFHE; e = 0 -> NOT SECURE)";
  py::class_<PhantomCiphertext>(m, "Ct");
  py::class_<Ctx5>(m, "Ctx")
      .def(py::init<const std::string &>(), py::arg("config_json"))
      .def("gen_rotation_keys", &Ctx5::gen_rotation_keys)
      .def("level", &Ctx5::level)
      .def("encrypt", &Ctx5::encrypt)
      .def("decrypt", &Ctx5::decrypt)
      .def("mod_down", &Ctx5::mod_down)
      .def("add", &Ctx5::add)
      .def("sub", &Ctx5::sub)
      .def("add_pt", &Ctx5::add_pt)
      .def("mul_int", &Ctx5::mul_int)
      .def("mul_pt_sum", &Ctx5::mul_pt_sum)
      .def("mul", &Ctx5::mul)
      .def("rot", &Ctx5::rot)
      .def("profile", &Ctx5::profile)
      .def("reset_profile", &Ctx5::reset_profile)
      .def("set_profiling", [](Ctx5 &c, bool on) { c.prof.on = on; })
      .def("set_section", [](Ctx5 &c, const std::string &s) { c.prof.section = s; })
      .def("set_fused", [](Ctx5 &c, bool on, int chunk) {
        if (chunk < 1 || chunk > MAC_MAX) throw std::invalid_argument("chunk must be in [1, 16]");
        c.fused = on;
        c.mac_chunk = chunk;
      }, py::arg("on"), py::arg("chunk") = 4)
      .def("mem_info", &Ctx5::mem_info)
      .def("trim", [](Ctx5 &c) {
        cudaDeviceSynchronize();
        cudaMemPool_t pool;
        cudaDeviceGetDefaultMemPool(&pool, 0);
        cudaMemPoolTrimTo(pool, 0);
      })
      .def("info", &Ctx5::info)
      .def("ct_bytes", &Ctx5::ct_bytes);
}

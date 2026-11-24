// CipherChat v3 – Python binding of the GPU CKKS backend (PhantomFHE + NEXUS-CUDA bootstrapping).
//
// The Python engine (cipherchat/v3/engine.py) orchestrates; every homomorphic operation runs here on the GPU.
// Level convention: level = number of rescales left = (#primes of the ciphertext) - 1; after bootstrapping
// a ciphertext has `compute_levels` levels.
//
// Profiling: when enabled, every op is timed with CUDA events (after a device sync, Phantom uses several
// streams) and accumulated under (section, op); `section` is set from Python ("attn.scores", "ffn.down", ...).
// cudaMemGetInfo / pool high-water marks are reported by mem_info().
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <regex>
#include <sstream>

#include "bootstrapping/Bootstrapper.cuh"
#include "ckks_evaluator.cuh"
#include "v3common.hpp"

namespace py = pybind11;
using namespace phantom;
using namespace nexus;

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

class Ctx {
 public:
  json cfg;
  Chain ch;
  std::unique_ptr<PhantomContext> context;
  std::unique_ptr<PhantomSecretKey> sk;
  std::unique_ptr<PhantomPublicKey> pk;
  std::unique_ptr<PhantomRelinKey> rlk;
  PhantomGaloisKey glk;
  std::unique_ptr<PhantomCKKSEncoder> encoder;
  std::unique_ptr<CKKSEvaluator> ckks;
  std::unique_ptr<Bootstrapper> boot;
  std::unique_ptr<Bootstrapper> boot0;   // logn = 0: replicated scalars (no CoeffToSlot / SlotToCoeff)
  std::vector<int> boot_steps;
  size_t n;
  int logn_boot;
  Prof prof;
  json boot_phases = json::object();
  long n_boot = 0;

  Ctx(const std::string &cfg_json, int logn_boot_) : logn_boot(logn_boot_) {
    cfg = json::parse(cfg_json);
    ch = make_chain(cfg);
    auto parms = make_params(cfg, ch);
    context.reset(new PhantomContext(parms));
    sk.reset(new PhantomSecretKey(*context));
    pk.reset(new PhantomPublicKey(sk->gen_publickey(*context)));
    rlk.reset(new PhantomRelinKey(sk->gen_relinkey(*context)));
    encoder.reset(new PhantomCKKSEncoder(*context));
    ckks.reset(new CKKSEvaluator(context.get(), pk.get(), sk.get(), encoder.get(), rlk.get(), &glk, ch.scale));
    n = encoder->slot_count();
    const json &bp = cfg["ckks"]["boot"];
    int logN = cfg["ckks"]["logN"];
    boot.reset(new Bootstrapper(bp["loge"], logn_boot, logN - 1, ch.total_level, ch.scale, bp["K"], bp["deg"],
                                bp["scale_factor"], bp["inverse_deg"], ckks.get()));
    {
      std::stringstream buf;
      auto *old = std::cout.rdbuf(buf.rdbuf());
      boot->prepare_mod_polynomial();
      boot_steps = {0};
      for (int i = 0; i < logN - 1; i++) boot_steps.push_back(1 << i);
      boot->addLeftRotKeys_Linear_to_vector_3(boot_steps);
      boot0.reset(new Bootstrapper(bp["loge"], 0, logN - 1, ch.total_level, ch.scale, bp["K"], bp["deg"],
                                   bp["scale_factor"], bp["inverse_deg"], ckks.get()));
      boot0->prepare_mod_polynomial();
      std::cout.rdbuf(old);
    }
  }

  // rotation keys: bootstrapping steps + the engine's steps (call once)
  int gen_rotation_keys(std::vector<int> steps) {
    for (int s : boot_steps) steps.push_back(s);
    for (auto &s : steps) s = ((s % (int)n) + (int)n) % (int)n;
    std::sort(steps.begin(), steps.end());
    steps.erase(std::unique(steps.begin(), steps.end()), steps.end());
    ckks->decryptor.create_galois_keys_from_steps(steps, glk);
    boot->slot_vec.push_back(logn_boot);
    boot0->slot_vec.push_back(0);
    std::stringstream buf;
    auto *old = std::cout.rdbuf(buf.rdbuf());
    boot->generate_LT_coefficient_3();
    std::cout.rdbuf(old);
    return (int)steps.size();
  }

  int level(const PhantomCiphertext &ct) const { return (int)ct.coeff_modulus_size() - 1; }

  size_t chain_for_level(int lvl) const {  // chain index of a ciphertext with `lvl` levels left
    // fresh ciphertexts are at chain index 1 with total_level levels
    return (size_t)(1 + ch.total_level - lvl);
  }

  uint64_t q_last(const PhantomCiphertext &ct) const {
    return context->get_context_data(ct.chain_index()).parms().coeff_modulus().back().value();
  }

  std::vector<double> vec_of(py::array_t<double, py::array::c_style | py::array::forcecast> a) const {
    if ((size_t)a.size() > n) throw std::invalid_argument("vector longer than slot count");
    std::vector<double> v(a.data(), a.data() + a.size());
    v.resize(n, 0.0);
    return v;
  }

  PhantomCiphertext encrypt(py::array_t<double, py::array::c_style | py::array::forcecast> a, int lvl) {
    OpTimer t(prof, "encrypt");
    PhantomPlaintext pt;
    auto v = vec_of(a);
    encoder->encode(*context, v, ch.scale, pt, 1);
    PhantomCiphertext ct;
    pk->encrypt_asymmetric(*context, pt, ct);
    size_t ci = chain_for_level(lvl);
    if (ci > ct.chain_index()) ::mod_switch_to_inplace(*context, ct, ci);
    return ct;
  }

  py::array_t<double> decrypt(const PhantomCiphertext &ct) {
    OpTimer t(prof, "decrypt");
    PhantomPlaintext pt;
    PhantomCiphertext c = ct;
    sk->decrypt(*context, c, pt);
    std::vector<double> v;
    encoder->decode(*context, pt, v);
    py::array_t<double> out(v.size());
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
  }

  void align(PhantomCiphertext &a, PhantomCiphertext &b) {
    if (a.chain_index() < b.chain_index())
      ::mod_switch_to_inplace(*context, a, b.chain_index());
    else if (b.chain_index() < a.chain_index())
      ::mod_switch_to_inplace(*context, b, a.chain_index());
    b.scale() = a.scale();
  }

  PhantomCiphertext mod_down(const PhantomCiphertext &ct, int lvl) {
    OpTimer t(prof, "mod_down");
    PhantomCiphertext c = ct;
    size_t ci = chain_for_level(lvl);
    if (ci > c.chain_index()) ::mod_switch_to_inplace(*context, c, ci);
    return c;
  }

  PhantomCiphertext add(const PhantomCiphertext &a, const PhantomCiphertext &b) {
    OpTimer t(prof, "add");
    PhantomCiphertext x = a, y = b;
    align(x, y);
    ::add_inplace(*context, x, y);
    return x;
  }

  PhantomCiphertext sub(const PhantomCiphertext &a, const PhantomCiphertext &b) {
    OpTimer t(prof, "add");
    PhantomCiphertext x = a, y = b;
    align(x, y);
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
    PhantomCiphertext x = a;
    if (k == 0) throw std::invalid_argument("mul_int by 0");
    PhantomCiphertext acc = a;
    bool neg = k < 0;
    k = std::labs(k);
    for (long i = 1; i < k; i++) ::add_inplace(*context, acc, a);
    if (neg) ::negate_inplace(*context, acc);
    return acc;
  }

  // sum_i ct_i * v_i with one rescale at the end (lazy rescale); plaintexts encoded at scale q_last
  PhantomCiphertext mul_pt_sum(std::vector<PhantomCiphertext> cts,
                               std::vector<py::array_t<double, py::array::c_style | py::array::forcecast>> vs) {
    if (cts.size() != vs.size() || cts.empty()) throw std::invalid_argument("mul_pt_sum: size mismatch");
    size_t ci = 0;
    for (auto &c : cts) ci = std::max(ci, (size_t)c.chain_index());
    for (auto &c : cts)
      if (c.chain_index() < ci) ::mod_switch_to_inplace(*context, c, ci);
    PhantomCiphertext acc;
    for (size_t i = 0; i < cts.size(); i++) {
      std::vector<double> v = vec_of(vs[i]);
      PhantomPlaintext pt;
      {
        OpTimer t(prof, "encode");
        encoder->encode(*context, v, (double)q_last(cts[i]), pt, cts[i].chain_index());
      }
      OpTimer t(prof, "ptmul");
      auto prod = ::multiply_plain(*context, cts[i], pt);
      if (i == 0)
        acc = prod;
      else
        ::add_inplace(*context, acc, prod);
    }
    OpTimer t(prof, "rescale");
    double s = cts[0].scale();
    ::rescale_to_next_inplace(*context, acc);
    acc.scale() = s;
    return acc;
  }

  PhantomCiphertext mul(const PhantomCiphertext &a, const PhantomCiphertext &b) {
    OpTimer t(prof, "ctmul");
    PhantomCiphertext x = a, y = b;
    align(x, y);
    double s = x.scale();
    ::multiply_and_relin_inplace(*context, x, y, *rlk);
    ::rescale_to_next_inplace(*context, x);
    x.scale() = s;
    return x;
  }

  PhantomCiphertext rot(const PhantomCiphertext &a, int k) {
    OpTimer t(prof, "rot");
    PhantomCiphertext x = a;
    k = ((k % (int)n) + (int)n) % (int)n;
    if (k) ::rotate_vector_inplace(*context, x, k, glk);
    return x;
  }

  PhantomCiphertext bootstrap(const PhantomCiphertext &a) {
    PhantomCiphertext x = a;
    if (x.chain_index() < (size_t)ch.total_level + 1)
      ::mod_switch_to_inplace(*context, x, (size_t)ch.total_level + 1);
    PhantomCiphertext out;
    std::stringstream buf;
    auto *old = std::cout.rdbuf(buf.rdbuf());
    {
      OpTimer t(prof, "bootstrap");
      boot->bootstrap_3(out, x);
    }
    std::cout.rdbuf(old);
    static const std::regex re(
        "(Modulus Raising|Subsum|Coeff-to-slot|Modular reduction|Slot-to-coeff)\\.\\.\\. ([0-9.]+)ms");
    std::string s = buf.str();
    for (std::sregex_iterator it(s.begin(), s.end(), re), end; it != end; ++it) {
      std::string k = (*it)[1].str();
      double v = std::stod((*it)[2].str());
      boot_phases[k] = boot_phases.value(k, 0.0) + v;
    }
    n_boot++;
    out.scale() = ch.scale;
    return out;
  }

  // bootstrapping of a ciphertext whose slots all hold the same real value (logn = 0 sparse packing)
  PhantomCiphertext bootstrap_scalar(const PhantomCiphertext &a) {
    PhantomCiphertext x = a;
    if (x.chain_index() < (size_t)ch.total_level + 1)
      ::mod_switch_to_inplace(*context, x, (size_t)ch.total_level + 1);
    PhantomCiphertext out;
    std::stringstream buf;
    auto *old = std::cout.rdbuf(buf.rdbuf());
    {
      OpTimer t(prof, "bootstrap");
      boot0->bootstrap_3(out, x);
    }
    std::cout.rdbuf(old);
    n_boot++;
    out.scale() = ch.scale;
    return out;
  }

  py::dict profile() {
    py::dict d;
    for (auto &kv : prof.ms) d[py::str(kv.first)] = py::make_tuple(kv.second, prof.cnt[kv.first]);
    return d;
  }

  void reset_profile() {
    prof.ms.clear();
    prof.cnt.clear();
    boot_phases = json::object();
    n_boot = 0;
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
    d["compute_levels"] = ch.compute_levels;
    d["boot_levels"] = ch.boot_levels;
    d["total_level"] = ch.total_level;
    d["log_qp"] = ch.log_qp;
    d["logn_boot"] = logn_boot;
    d["bits"] = ch.bits;
    return d;
  }

  size_t ct_bytes(const PhantomCiphertext &ct) const { return 2 * 2 * n * ct.coeff_modulus_size() * 8; }
};

PYBIND11_MODULE(v3fhe, m) {
  m.doc() = "CipherChat v3 GPU CKKS backend (PhantomFHE + NEXUS bootstrapping)";
  py::class_<PhantomCiphertext>(m, "Ct");
  py::class_<Ctx>(m, "Ctx")
      .def(py::init<const std::string &, int>(), py::arg("config_json"), py::arg("logn_boot") = 10)
      .def("gen_rotation_keys", &Ctx::gen_rotation_keys)
      .def("level", &Ctx::level)
      .def("encrypt", &Ctx::encrypt)
      .def("decrypt", &Ctx::decrypt)
      .def("mod_down", &Ctx::mod_down)
      .def("add", &Ctx::add)
      .def("sub", &Ctx::sub)
      .def("add_pt", &Ctx::add_pt)
      .def("mul_int", &Ctx::mul_int)
      .def("mul_pt_sum", &Ctx::mul_pt_sum)
      .def("mul", &Ctx::mul)
      .def("rot", &Ctx::rot)
      .def("bootstrap", &Ctx::bootstrap)
      .def("bootstrap_scalar", &Ctx::bootstrap_scalar)
      .def("profile", &Ctx::profile)
      .def("reset_profile", &Ctx::reset_profile)
      .def("set_profiling", [](Ctx &c, bool on) { c.prof.on = on; })
      .def("set_section", [](Ctx &c, const std::string &s) { c.prof.section = s; })
      .def("boot_phases", [](Ctx &c) { return c.boot_phases.dump(); })
      .def("n_boot", [](Ctx &c) { return c.n_boot; })
      .def("mem_info", &Ctx::mem_info)
      .def("boot_steps", [](Ctx &c) { return c.boot_steps; })
      .def("info", &Ctx::info)
      .def("ct_bytes", &Ctx::ct_bytes);
}

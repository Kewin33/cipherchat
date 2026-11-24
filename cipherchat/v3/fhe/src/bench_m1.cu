// CipherChat v3 – Milestone 1: CKKS parameters from config.json on the RTX 5080.
// Measures: key / VRAM budget, sparse-slot bootstrapping (per phase), a 1024x1024 ct x pt matvec in the
// replicated "offset -> periodic" layout (BSGS), the periodic -> offset layout switch and per-op costs.
//
// usage: v3_bench_m1 config.json [ckks.special_count=6] [bench.logn_boot=[10,13]] [bench.out=m1.json]
#include <regex>
#include <sstream>

#include "bootstrapping/Bootstrapper.cuh"
#include "ckks_evaluator.cuh"
#include "v3common.hpp"

using namespace phantom;
using namespace nexus;

static double mb(size_t b) { return b / 1048576.0; }

struct Env {
  PhantomContext *ctx;
  CKKSEvaluator *ckks;
  PhantomGaloisKey *glk;
  PhantomRelinKey *rlk;
  size_t n;  // slots
};

static uint64_t q_last(const Env &e, const PhantomCiphertext &ct) {
  return e.ctx->get_context_data(ct.chain_index()).parms().coeff_modulus().back().value();
}

// encode v (length n) at the ciphertext's level with scale = the prime that the next rescale drops,
// so ct(scale s) x pt -> rescale -> scale s exactly
static PhantomPlaintext encode_for(const Env &e, const std::vector<double> &v, const PhantomCiphertext &ct) {
  PhantomPlaintext pt;
  e.ckks->encoder.encode(v, ct.chain_index(), (double)q_last(e, ct), pt);
  return pt;
}

static PhantomCiphertext rot(const Env &e, const PhantomCiphertext &ct, int step) {
  PhantomCiphertext out = ct;
  if (step % (int)e.n == 0) return out;
  ::rotate_vector_inplace(*e.ctx, out, step, *e.glk);
  return out;
}

static std::vector<double> rot_plain(const std::vector<double> &v, long s) {  // out[i] = v[i + s]
  long n = v.size();
  std::vector<double> o(n);
  for (long i = 0; i < n; i++) o[i] = v[((i + s) % n + n) % n];
  return o;
}

// ---- matvec: X offset layout (slot r*D+j = x[(j + r*S) mod D]) -> periodic y = M x (slot r*D+j = y[j])
struct MatvecTimes {
  double encode = 0, mult = 0, rot = 0, rescale = 0;
};

static PhantomCiphertext matvec(const Env &e, const PhantomCiphertext &X, const std::vector<double> &M, int D, int B,
                                MatvecTimes &tm) {
  int n = e.n, R = n / D, S = D / R, G = S / B;
  SyncTimer t;
  std::vector<PhantomCiphertext> xb(B);
  for (int b = 0; b < B; b++) xb[b] = rot(e, X, b);
  tm.rot += t.ms();
  PhantomCiphertext acc;
  bool have_acc = false;
  for (int g = 0; g < G; g++) {
    PhantomCiphertext inner;
    for (int b = 0; b < B; b++) {
      int tt = g * B + b;
      t.start();
      std::vector<double> P(n);
      for (int r = 0; r < R; r++)
        for (int j = 0; j < D; j++) {
          int rp = (r + (j + tt >= D ? 1 : 0)) % R;
          int k = (rp * S + tt) % D;
          P[r * D + j] = M[(size_t)j * D + (j + k) % D];
        }
      auto Q = rot_plain(P, -(long)g * B);
      auto pt = encode_for(e, Q, xb[b]);
      tm.encode += t.ms();
      t.start();
      auto prod = ::multiply_plain(*e.ctx, xb[b], pt);
      if (b == 0)
        inner = prod;
      else
        ::add_inplace(*e.ctx, inner, prod);
      tm.mult += t.ms();
    }
    t.start();
    ::rescale_to_next_inplace(*e.ctx, inner);
    inner.scale() = X.scale();
    tm.rescale += t.ms();
    t.start();
    auto r = rot(e, inner, g * B);
    if (!have_acc) {
      acc = r;
      have_acc = true;
    } else
      ::add_inplace(*e.ctx, acc, r);
    tm.rot += t.ms();
  }
  t.start();
  for (int s = D; s < n; s *= 2) ::add_inplace(*e.ctx, acc, rot(e, acc, s));
  tm.rot += t.ms();
  return acc;
}

// periodic (period D) -> offset layout: out = sum_r mask_r * rot(Y, r*S), BSGS over r
static PhantomCiphertext periodic_to_offset(const Env &e, const PhantomCiphertext &Y, int D, int Bc, MatvecTimes &tm) {
  int n = e.n, R = n / D, S = D / R, Gc = R / Bc;
  SyncTimer t;
  std::vector<PhantomCiphertext> yb(Bc);
  for (int b = 0; b < Bc; b++) yb[b] = rot(e, Y, b * S);
  tm.rot += t.ms();
  PhantomCiphertext acc;
  for (int g = 0; g < Gc; g++) {
    PhantomCiphertext inner;
    for (int b = 0; b < Bc; b++) {
      int r = g * Bc + b;
      t.start();
      std::vector<double> mask(n, 0.0);
      for (int j = 0; j < D; j++) mask[r * D + j] = 1.0;
      auto pt = encode_for(e, rot_plain(mask, -(long)g * Bc * S), yb[b]);
      tm.encode += t.ms();
      t.start();
      auto prod = ::multiply_plain(*e.ctx, yb[b], pt);
      if (b == 0)
        inner = prod;
      else
        ::add_inplace(*e.ctx, inner, prod);
      tm.mult += t.ms();
    }
    t.start();
    ::rescale_to_next_inplace(*e.ctx, inner);
    inner.scale() = Y.scale();
    tm.rescale += t.ms();
    t.start();
    auto r = rot(e, inner, g * Bc * S);
    if (g == 0)
      acc = r;
    else
      ::add_inplace(*e.ctx, acc, r);
    tm.rot += t.ms();
  }
  return acc;
}

static std::vector<double> decrypt(const Env &e, PhantomCiphertext &ct) {
  PhantomPlaintext pt;
  e.ckks->decryptor.decrypt(ct, pt);
  std::vector<double> v;
  e.ckks->encoder.decode(pt, v);
  return v;
}

static PhantomCiphertext encrypt_at(const Env &e, const std::vector<double> &v, double scale, int drop_levels) {
  PhantomPlaintext pt;
  e.ckks->encoder.encode(v, scale, pt);
  PhantomCiphertext ct;
  e.ckks->encryptor.encrypt(pt, ct);
  for (int i = 0; i < drop_levels; i++) ::mod_switch_to_next_inplace(*e.ctx, ct);
  return ct;
}

static double max_abs_err(const std::vector<double> &a, const std::vector<double> &b, size_t n) {
  double m = 0;
  for (size_t i = 0; i < n; i++) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

int main(int argc, char **argv) {
  std::string cfg_path = argc > 1 ? argv[1] : "config.json";
  json cfg = load_config(cfg_path, argc, argv, 2);
  json out;
  Chain ch = make_chain(cfg);
  const json &ck = cfg["ckks"];
  int logN = ck["logN"], h = ck["hamming_weight"], D = ck["block"];
  int bound = max_logqp_128(logN, h);
  std::cout << "chain bits:";
  for (int b : ch.bits) std::cout << " " << b;
  std::cout << "\nlog QP = " << ch.log_qp << " (128-bit bound " << bound << ", h=" << h << ")"
            << " compute levels " << ch.compute_levels << ", boot levels " << ch.boot_levels << ", special "
            << ch.special_count << std::endl;
  out["params"] = {{"logN", logN}, {"bits", ch.bits}, {"log_qp", ch.log_qp}, {"bound_128", bound},
                   {"secure_128", ch.log_qp <= bound}, {"hamming_weight", h}};
  out["vram_mb"]["baseline"] = mb(gpu_used_bytes());

  SyncTimer T;
  auto parms = make_params(cfg, ch);
  PhantomContext context(parms);
  out["vram_mb"]["context"] = mb(gpu_used_bytes());
  PhantomSecretKey sk(context);
  PhantomPublicKey pk = sk.gen_publickey(context);
  PhantomRelinKey rlk = sk.gen_relinkey(context);
  out["vram_mb"]["sk_pk_relin"] = mb(gpu_used_bytes());
  PhantomGaloisKey glk;
  PhantomCKKSEncoder encoder(context);
  CKKSEvaluator ckks(&context, &pk, &sk, &encoder, &rlk, &glk, ch.scale);
  Env e{&context, &ckks, &glk, &rlk, encoder.slot_count()};
  std::cout << "slots " << e.n << ", context+keys " << T.ms() / 1e3 << " s" << std::endl;

  // ---- bootstrappers (one per sparse slot count) + rotation keys
  auto logns = cfg_get<std::vector<int>>(cfg, "bench.logn_boot", {10});
  const json &bp = ck["boot"];
  std::vector<std::unique_ptr<Bootstrapper>> boots;
  std::vector<int> steps = {0};
  for (int i = 0; i < logN - 1; i++) steps.push_back(1 << i);
  for (int logn : logns) {
    boots.emplace_back(new Bootstrapper(bp["loge"], logn, logN - 1, ch.total_level, ch.scale, bp["K"], bp["deg"],
                                        bp["scale_factor"], bp["inverse_deg"], &ckks));
    boots.back()->prepare_mod_polynomial();
    boots.back()->addLeftRotKeys_Linear_to_vector_3(steps);
  }
  size_t n_boot_steps = 0;
  {
    std::sort(steps.begin(), steps.end());
    steps.erase(std::unique(steps.begin(), steps.end()), steps.end());
    n_boot_steps = steps.size();
  }
  // matvec + layout switch steps (D=1024, R=32, S=32): babies 1..B-1, giants B,2B,..; folds D*2^i; S*b; Bc*S*g
  int n = e.n, R = n / D, S = D / R;
  int B = cfg_get<int>(cfg, "bench.baby", 8), Bc = cfg_get<int>(cfg, "bench.baby_conv", 8);
  for (int b = 1; b < B; b++) steps.push_back(b);       // babies
  for (int g = B; g < S; g += B) steps.push_back(g);    // giants
  for (int s = D; s < n; s *= 2) steps.push_back(s);    // folds
  for (int b = 1; b < Bc; b++) steps.push_back(b * S);  // layout switch babies
  for (int g = 1; g < R / Bc; g++) steps.push_back(g * Bc * S);
  std::sort(steps.begin(), steps.end());
  steps.erase(std::unique(steps.begin(), steps.end()), steps.end());
  std::cout << "rotation keys: " << steps.size() << " (" << n_boot_steps << " for bootstrapping)" << std::endl;
  size_t before_keys = gpu_used_bytes();
  T.start();
  ckks.decryptor.create_galois_keys_from_steps(steps, glk);
  double t_keys = T.ms();
  size_t after_keys = gpu_used_bytes();
  out["keys"] = {{"rotation_keys", steps.size()}, {"boot_keys", n_boot_steps},
                 {"galois_mb", mb(after_keys - before_keys)}, {"gen_s", t_keys / 1e3},
                 {"mb_per_key", mb(after_keys - before_keys) / steps.size()}};
  out["vram_mb"]["galois_keys"] = mb(after_keys);
  std::cout << "galois keys: " << mb(after_keys - before_keys) << " MB (" << t_keys / 1e3 << " s)" << std::endl;

  for (size_t i = 0; i < boots.size(); i++) {
    boots[i]->slot_vec.push_back(logns[i]);
    boots[i]->generate_LT_coefficient_3();
  }
  out["vram_mb"]["boot_lt_coeffs"] = mb(gpu_used_bytes());
  std::cout << "VRAM after bootstrapping setup: " << mb(gpu_used_bytes()) << " MB" << std::endl;

  // ---- bootstrapping per sparse slot count
  std::regex re("(Modulus Raising|Subsum|Coeff-to-slot|Modular reduction|Slot-to-coeff)\\.\\.\\. ([0-9.]+)ms");
  int reps = cfg_get<int>(cfg, "bench.reps", 3);
  for (size_t i = 0; i < boots.size(); i++) {
    int logn = logns[i];
    size_t nn = size_t(1) << logn;
    auto base = random_vec(nn, -1, 1, 7 + logn);
    std::vector<double> v(n);
    for (int k = 0; k < n; k++) v[k] = base[k % nn];
    json jb;
    std::vector<double> totals;
    for (int r = 0; r < reps; r++) {
      auto ct = encrypt_at(e, v, ch.scale, ch.total_level);
      auto before = decrypt(e, ct);
      PhantomCiphertext res;
      std::stringstream buf;
      auto *old = std::cout.rdbuf(buf.rdbuf());
      SyncTimer tb;
      boots[i]->bootstrap_3(res, ct);
      double tt = tb.ms();
      std::cout.rdbuf(old);
      totals.push_back(tt);
      std::string s = buf.str();
      json phases;
      for (std::sregex_iterator it(s.begin(), s.end(), re), end; it != end; ++it)
        phases[(*it)[1].str()] = std::stod((*it)[2].str());
      auto after = decrypt(e, res);
      double err = max_abs_err(before, after, n);
      int lvl = (int)res.coeff_modulus_size() - 1;
      std::cout << "bootstrap logn=" << logn << ": " << tt << " ms, max err " << err << ", levels left " << lvl
                << " | " << phases.dump() << std::endl;
      jb["runs"].push_back({{"ms", tt}, {"max_err", err}, {"levels_left", lvl}, {"phases", phases}});
    }
    std::sort(totals.begin(), totals.end());
    jb["median_ms"] = totals[totals.size() / 2];
    out["bootstrap"]["logn" + std::to_string(logn)] = jb;
  }
  out["vram_mb"]["peak_pool_after_boot"] = mb(pool_reserved_high());

  // ---- matvec 1024x1024 at the top compute level (= level after bootstrapping)
  auto x = random_vec(D, -1, 1, 11);
  auto M = random_vec((size_t)D * D, -1.0 / 32, 1.0 / 32, 12);
  std::vector<double> xo(n), yref(D, 0.0);
  for (int r = 0; r < R; r++)
    for (int j = 0; j < D; j++) xo[r * D + j] = x[(j + r * S) % D];
  for (int i = 0; i < D; i++)
    for (int j = 0; j < D; j++) yref[i] += M[(size_t)i * D + j] * x[j];
  json jm;
  for (int r = 0; r < reps; r++) {
    auto X = encrypt_at(e, xo, ch.scale, ch.boot_levels);
    MatvecTimes tm;
    SyncTimer tt;
    auto Y = matvec(e, X, M, D, B, tm);
    double t_mv = tt.ms();
    MatvecTimes tc;
    tt.start();
    auto Z = periodic_to_offset(e, Y, D, Bc, tc);
    double t_cv = tt.ms();
    auto y = decrypt(e, Y), z = decrypt(e, Z);
    double err_y = 0, err_z = 0;
    for (int k = 0; k < n; k++) {
      err_y = std::max(err_y, std::fabs(y[k] - yref[k % D]));
      int rr = k / D, j = k % D;
      err_z = std::max(err_z, std::fabs(z[k] - yref[(j + rr * S) % D]));
    }
    std::cout << "matvec 1024x1024: " << t_mv << " ms (encode " << tm.encode << ", mult " << tm.mult << ", rot "
              << tm.rot << ", rescale " << tm.rescale << ") err " << err_y << " | periodic->offset " << t_cv
              << " ms (encode " << tc.encode << ", rot " << tc.rot << ") err " << err_z << std::endl;
    jm["runs"].push_back({{"matvec_ms", t_mv},
                          {"matvec_split", {{"encode", tm.encode}, {"mult", tm.mult}, {"rot", tm.rot}, {"rescale", tm.rescale}}},
                          {"matvec_err", err_y},
                          {"to_offset_ms", t_cv},
                          {"to_offset_split", {{"encode", tc.encode}, {"mult", tc.mult}, {"rot", tc.rot}, {"rescale", tc.rescale}}},
                          {"to_offset_err", err_z}});
  }
  out["matvec_1024"] = jm;

  // ---- per-op costs at the top compute level and at the bottom
  json ops;
  for (int drop : {ch.boot_levels, ch.total_level - 2}) {
    auto a = encrypt_at(e, random_vec(n, -1, 1, 1), ch.scale, drop);
    auto b = encrypt_at(e, random_vec(n, -1, 1, 2), ch.scale, drop);
    int lv = (int)a.coeff_modulus_size() - 1;
    const int K = 20;
    SyncTimer t;
    for (int k = 0; k < K; k++) {
      auto r = rot(e, a, 1 + (k % 7));
    }
    double t_rot = t.ms() / K;
    t.start();
    for (int k = 0; k < K; k++) {
      auto c = a;
      ::multiply_and_relin_inplace(*e.ctx, c, b, rlk);
      ::rescale_to_next_inplace(*e.ctx, c);
    }
    double t_mul = t.ms() / K;
    auto pv = random_vec(n, -1, 1, 3);
    t.start();
    PhantomPlaintext pt;
    for (int k = 0; k < K; k++) pt = encode_for(e, pv, a);
    double t_enc = t.ms() / K;
    t.start();
    for (int k = 0; k < K; k++) {
      auto c = ::multiply_plain(*e.ctx, a, pt);
    }
    double t_pm = t.ms() / K;
    std::cout << "ops @ level " << lv << ": rot " << t_rot << " ms, ct*ct+relin+rescale " << t_mul << " ms, encode "
              << t_enc << " ms, ct*pt " << t_pm << " ms" << std::endl;
    ops["level" + std::to_string(lv)] = {{"rot_ms", t_rot}, {"ctct_relin_rescale_ms", t_mul}, {"encode_ms", t_enc},
                                         {"ctpt_ms", t_pm},
                                         {"ct_mb", mb((size_t)2 * (2 * n) * a.coeff_modulus_size() * 8)}};
  }
  out["ops"] = ops;
  out["vram_mb"]["peak_pool"] = mb(pool_reserved_high());
  out["vram_mb"]["end"] = mb(gpu_used_bytes());
  std::string path = cfg_get<std::string>(cfg, "bench.out", "m1.json");
  std::ofstream(path) << out.dump(1);
  std::cout << "wrote " << path << std::endl;
  return 0;
}

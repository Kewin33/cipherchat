// CipherChat v3 – shared helpers: config.json loading, CKKS context from config, timers, VRAM tracking.
#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "phantom.h"

using json = nlohmann::json;

// ---------------------------------------------------------------- config
// load config.json and apply "a.b.c=<json value>" overrides from argv[first..]
inline json load_config(const std::string &path, int argc, char **argv, int first) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open config " + path);
  json c = json::parse(f);
  for (int i = first; i < argc; i++) {
    std::string a = argv[i];
    auto eq = a.find('=');
    if (eq == std::string::npos) continue;
    std::string key = a.substr(0, eq), val = a.substr(eq + 1);
    std::string ptr = "/" + key;
    for (auto &ch : ptr)
      if (ch == '.') ch = '/';
    json v;
    try {
      v = json::parse(val);
    } catch (...) {
      v = val;  // plain string
    }
    c[json::json_pointer(ptr)] = v;
  }
  return c;
}

template <class T>
inline T cfg_get(const json &c, const std::string &dotted, T dflt) {
  std::string ptr = "/" + dotted;
  for (auto &ch : ptr)
    if (ch == '.') ch = '/';
  auto p = json::json_pointer(ptr);
  return c.contains(p) ? c[p].get<T>() : dflt;
}

// ---------------------------------------------------------------- CKKS modulus chain
struct Chain {
  std::vector<int> bits;  // q0, compute primes, bootstrapping primes, special primes
  int compute_levels, boot_levels, special_count, total_level, log_qp;
  double scale;
};

inline Chain make_chain(const json &cfg) {
  const json &ck = cfg["ckks"];
  Chain ch;
  ch.compute_levels = ck["compute_levels"];
  ch.boot_levels = ck["boot_levels"];
  ch.special_count = ck["special_count"];
  ch.bits.push_back(ck["base_bits"]);
  for (int i = 0; i < ch.compute_levels; i++) ch.bits.push_back(ck["scale_bits"]);
  for (int i = 0; i < ch.boot_levels; i++) ch.bits.push_back(ck["boot_bits"]);
  for (int i = 0; i < ch.special_count; i++) ch.bits.push_back(ck["special_bits"]);
  ch.total_level = ch.compute_levels + ch.boot_levels;
  ch.log_qp = 0;
  for (int b : ch.bits) ch.log_qp += b;
  ch.scale = std::pow(2.0, (int)ck["scale_bits"]);
  return ch;
}

// 128-bit bound on log QP for N = 2^logN (HE standard, ternary secret) and the sparse-secret (h = 192) bound
// used by Lattigo's N16QP1546H192H32 bootstrapping parameters.
inline int max_logqp_128(int logN, int h) {
  static const std::map<int, int> dense = {{13, 218}, {14, 438}, {15, 881}, {16, 1772}};
  if (h > 0 && logN == 16) return 1546;
  if (h > 0 && logN == 15) return 768;
  return dense.at(logN);
}

inline phantom::EncryptionParameters make_params(const json &cfg, const Chain &ch) {
  const json &ck = cfg["ckks"];
  phantom::EncryptionParameters parms(phantom::scheme_type::ckks);
  size_t N = size_t(1) << (int)ck["logN"];
  parms.set_poly_modulus_degree(N);
  parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(N, ch.bits));
  parms.set_special_modulus_size(ch.special_count);
  int h = ck["hamming_weight"];
  if (h > 0) parms.set_secret_key_hamming_weight(h);
  return parms;
}

// ---------------------------------------------------------------- timing + memory
inline double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// device-synchronised wall clock (Phantom uses several streams; a global sync makes the split exact)
struct SyncTimer {
  double t0;
  SyncTimer() { start(); }
  void start() {
    cudaDeviceSynchronize();
    t0 = now_ms();
  }
  double ms() {
    cudaDeviceSynchronize();
    return now_ms() - t0;
  }
};

inline size_t gpu_used_bytes() {
  cudaDeviceSynchronize();
  size_t f, t;
  cudaMemGetInfo(&f, &t);
  return t - f;
}

inline size_t pool_reserved_high() {
  cudaMemPool_t pool;
  cudaDeviceGetDefaultMemPool(&pool, 0);
  uint64_t v = 0;
  cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &v);
  return v;
}

// accumulates time per named hotspot
struct Profiler {
  std::map<std::string, double> ms;
  std::map<std::string, long> count;
  void add(const std::string &k, double t) {
    ms[k] += t;
    count[k] += 1;
  }
  json to_json() const {
    json j;
    for (auto &kv : ms) j[kv.first] = {{"ms", kv.second}, {"count", count.at(kv.first)}};
    return j;
  }
};

inline std::vector<double> random_vec(size_t n, double lo, double hi, uint64_t seed) {
  std::mt19937_64 g(seed);
  std::uniform_real_distribution<double> d(lo, hi);
  std::vector<double> v(n);
  for (auto &x : v) x = d(g);
  return v;
}

"""CipherChat v5 – make a patched copy of PhantomFHE (from the NEXUS checkout) for the INSECURE e = 0 experiments.

    python cipherchat/v5/fhe/patch_phantom.py <phantom-fhe source dir> <destination dir>

The source tree is never modified (v3 keeps building against it). Every change is guarded by a preprocessor macro,
so the copy compiles to stock Phantom unless the macros are set (cipherchat/v5/fhe/CMakeLists.txt sets them per
target):

  PHANTOM_ZERO_ERROR      sample_error_poly() returns 0: no RLWE error in the public key, the relinearisation and
                          Galois (key-switching) keys, public-key and symmetric encryption.  b = -a*s + m exactly.
                          ==> NOT SECURE: s = -b * a^-1 mod q can be read off the public key.
  PHANTOM_V5_LONG_CHAIN   modulus chains with hundreds of primes (no bootstrapping):
                            - COEFF_MOD_COUNT_MAX 64 -> 4096
                            - NTT tables on the host only for the key level (they are only read there; the per-level
                              copies cost O(L^2 N) memory)
                            - the RNSBase of the full Q is built once instead of once per level (O(L^3) each)
                            - big_qiHat (L x L words, needed only to decode) is uploaded only for <= 64 primes;
                              decoding a longer ciphertext throws (mod-switch it down first)
                            - RNSBase: no pairwise gcd check (O(L^2)), punctured products Q / q_i by one-word
                              division (O(L^2) instead of O(L^3) per base; there are a few bases per level)
  PHANTOM_V5_CENTERED     the rescale and the single-prime key-switching / encryption mod-down use the centered
                          representative (round instead of floor). Phantom floors there; the bias of 1/2 per
                          coefficient becomes up to ~N/(2 pi) / Delta per operation in the slots: invisible at
                          Delta = 2^46..2^50, ~1e-5 at Delta = 2^30 (two levels per prime).
  PHANTOM_SINGLE_DIGIT_KS hybrid key switching with ONE digit (the whole Q_l) and one special prime P:
                          key = (-a s + P s', a) over QP, i.e. the size of one ciphertext, and the base conversions are
                          O(L N) instead of O(L^2 N).  Only correct because the keys carry no error: with e != 0 the
                          key-switching noise would be ~ Q_l * e / P (garbage).  Requires PHANTOM_ZERO_ERROR.
"""
import os
import shutil
import sys

PATCHES = [
    # ---------------------------------------------------------------- e = 0
    ("src/prng.cu",
     "        out[tid] = static_cast<uint64_t>(cbd) + (flag & mod_value);\n",
     "        out[tid] = static_cast<uint64_t>(cbd) + (flag & mod_value);\n"
     "#ifdef PHANTOM_ZERO_ERROR\n"
     "        out[tid] = 0;  // CipherChat v5: RLWE error term e = 0 (INSECURE upper-bound experiment)\n"
     "#endif\n"),
    # ---------------------------------------------------------------- long chains
    ("include/util/defines.h",
     "constexpr int COEFF_MOD_COUNT_MAX = 64;\n",
     "#ifdef PHANTOM_V5_LONG_CHAIN\n"
     "constexpr int COEFF_MOD_COUNT_MAX = 4096;  // CipherChat v5: chains without bootstrapping\n"
     "#else\n"
     "constexpr int COEFF_MOD_COUNT_MAX = 64;\n"
     "#endif\n"),
    ("src/context.cu",
     "        small_ntt_tables_ = make_shared<RNSNTT>(coeff_count_power, coeff_modulus);\n",
     "#ifdef PHANTOM_V5_LONG_CHAIN\n"
     "        // CipherChat v5: only the key level's host NTT tables are ever read (PhantomContext ctor)\n"
     "        if (coeff_modulus_size == key_modulus.size())\n"
     "#endif\n"
     "        small_ntt_tables_ = make_shared<RNSNTT>(coeff_count_power, coeff_modulus);\n"),
    ("src/rns.cu",
     "        RNSBase base_Q(modulus_Q);\n        base_Q_.init(base_Q, stream);\n",
     "#ifdef PHANTOM_V5_LONG_CHAIN\n"
     "        // CipherChat v5: the full-Q base is identical on every level; build it once (O(L^3) host work)\n"
     "        static std::vector<uint64_t> v5_q_values;\n"
     "        static std::unique_ptr<RNSBase> v5_base_Q;\n"
     "        {\n"
     "            std::vector<uint64_t> vals(size_Q);\n"
     "            for (size_t i = 0; i < size_Q; i++) vals[i] = modulus_Q[i].value();\n"
     "            if (!v5_base_Q || vals != v5_q_values) {\n"
     "                v5_base_Q.reset(new RNSBase(modulus_Q));\n"
     "                v5_q_values = vals;\n"
     "            }\n"
     "        }\n"
     "        const RNSBase &base_Q = *v5_base_Q;\n"
     "#else\n"
     "        RNSBase base_Q(modulus_Q);\n"
     "#endif\n"
     "        base_Q_.init(base_Q, stream);\n"),
    ("src/rns_base.cu",
     "        big_qiHat_ = phantom::util::make_cuda_auto_ptr<uint64_t>(size_ * size_, stream);\n"
     "        cudaMemcpyAsync(big_qiHat(), cpu_rns_base.big_qiHat(), size_ * size_ * sizeof(std::uint64_t),\n"
     "                        cudaMemcpyHostToDevice, stream);\n",
     "#ifdef PHANTOM_V5_LONG_CHAIN\n"
     "        if (size_ <= 64) {  // CipherChat v5: L x L words per base and level; only decoding reads it\n"
     "#endif\n"
     "        big_qiHat_ = phantom::util::make_cuda_auto_ptr<uint64_t>(size_ * size_, stream);\n"
     "        cudaMemcpyAsync(big_qiHat(), cpu_rns_base.big_qiHat(), size_ * size_ * sizeof(std::uint64_t),\n"
     "                        cudaMemcpyHostToDevice, stream);\n"
     "#ifdef PHANTOM_V5_LONG_CHAIN\n"
     "        }\n"
     "#endif\n"),
    ("src/rns_base.cu",
     "        uint32_t rns_poly_uint64_count = sparse_coeff_count * size();\n",
     "#ifdef PHANTOM_V5_LONG_CHAIN\n"
     "        if (!big_qiHat_.get())\n"
     "            throw invalid_argument(\"v5: decoding needs <= 64 primes, mod-switch the ciphertext down first\");\n"
     "#endif\n"
     "        uint32_t rns_poly_uint64_count = sparse_coeff_count * size();\n",
     2),
    ("src/util/rns.cu",
     "            for (size_t j = 0; j < i; j++) {\n"
     "                // The base must be coprime\n"
     "                if (!are_coprime(rnsbase[i].value(), rnsbase[j].value())) {\n"
     "                    throw invalid_argument(\"rnsbase is invalid\");\n"
     "                }\n"
     "            }\n",
     "#ifndef PHANTOM_V5_LONG_CHAIN  // CipherChat v5: O(L^2) gcds per base and level; the chain holds distinct primes\n"
     "            for (size_t j = 0; j < i; j++) {\n"
     "                // The base must be coprime\n"
     "                if (!are_coprime(rnsbase[i].value(), rnsbase[j].value())) {\n"
     "                    throw invalid_argument(\"rnsbase is invalid\");\n"
     "                }\n"
     "            }\n"
     "#endif\n",
     2),
    ("src/util/rns.cu",
     "            // Create punctured products\n"
     "            for (size_t i = 0; i < size_; i++) {\n"
     "                multiply_many_uint64_except(rnsbase_values.data(), size_, i, prod_hat_.data() + i * size_);\n"
     "            }\n"
     "\n"
     "            // Compute the full product, i.e., qiHat[0] * Q_[0]\n"
     "            auto temp_mpi = std::vector<uint64_t>(size_);\n"
     "            multiply_uint(prod_hat_.data(), size_, mod_[0].value(), size_, temp_mpi.data());\n"
     "            set_uint(temp_mpi.data(), size_, prod_mod_.data());\n",
     "#ifdef PHANTOM_V5_LONG_CHAIN\n"
     "            // CipherChat v5: Q once (O(L^2)), punctured products Q / q_i by one-word division (O(L) each)\n"
     "            // instead of L products of L - 1 words (O(L^3) per base and level)\n"
     "            multiply_many_uint64(rnsbase_values.data(), size_, prod_mod_.data());\n"
     "            for (size_t i = 0; i < size_; i++) {\n"
     "                unsigned __int128 rem = 0;\n"
     "                uint64_t *dst = prod_hat_.data() + i * size_;\n"
     "                for (size_t w = size_; w-- > 0;) {\n"
     "                    rem = (rem << 64) | prod_mod_[w];\n"
     "                    dst[w] = static_cast<uint64_t>(rem / rnsbase_values[i]);\n"
     "                    rem %= rnsbase_values[i];\n"
     "                }\n"
     "            }\n"
     "#else\n"
     "            // Create punctured products\n"
     "            for (size_t i = 0; i < size_; i++) {\n"
     "                multiply_many_uint64_except(rnsbase_values.data(), size_, i, prod_hat_.data() + i * size_);\n"
     "            }\n"
     "\n"
     "            // Compute the full product, i.e., qiHat[0] * Q_[0]\n"
     "            auto temp_mpi = std::vector<uint64_t>(size_);\n"
     "            multiply_uint(prod_hat_.data(), size_, mod_[0].value(), size_, temp_mpi.data());\n"
     "            set_uint(temp_mpi.data(), size_, prod_mod_.data());\n"
     "#endif\n"),
    # ---------------------------------------------------------------- centered rounding
    ("src/rns.cu",
     "            // ci[last] mod qj\n"
     "            dst[tid] = barrett_reduce_uint64_uint64(c_last_coeff, mod.value(), mod.const_ratio()[1]);\n",
     "            // ci[last] mod qj\n"
     "#ifdef PHANTOM_V5_CENTERED\n"
     "            // CipherChat v5: centered representative of ci[last] -> the rescale rounds instead of flooring\n"
     "            const uint64_t q_last_v5 = base_q[next_base_q_size].value();\n"
     "            if (c_last_coeff > (q_last_v5 >> 1)) {\n"
     "                uint64_t r = barrett_reduce_uint64_uint64(q_last_v5 - c_last_coeff, mod.value(),\n"
     "                                                          mod.const_ratio()[1]);\n"
     "                dst[tid] = r ? mod.value() - r : 0;\n"
     "            } else\n"
     "#endif\n"
     "            dst[tid] = barrett_reduce_uint64_uint64(c_last_coeff, mod.value(), mod.const_ratio()[1]);\n"),
    ("src/rns_bconv.cu",
     "    const uint64_t coeff = src[coeff_idx];\n"
     "    uint64_t result;\n"
     "    if (in_prime > out_prime)\n"
     "        result = barrett_reduce_uint64_uint64(coeff, out_prime, barret_ratio);\n"
     "    else\n"
     "        result = coeff;\n"
     "    dst[tid] = result;\n",
     "    const uint64_t coeff = src[coeff_idx];\n"
     "    uint64_t result;\n"
     "#ifdef PHANTOM_V5_CENTERED\n"
     "    // CipherChat v5: centered representative of [c]_P -> the key-switching mod-down rounds instead of flooring\n"
     "    if (coeff > (in_prime >> 1)) {\n"
     "        uint64_t r = barrett_reduce_uint64_uint64(in_prime - coeff, out_prime, barret_ratio);\n"
     "        result = r ? out_prime - r : 0;\n"
     "    } else\n"
     "        result = barrett_reduce_uint64_uint64(coeff, out_prime, barret_ratio);\n"
     "#else\n"
     "    if (in_prime > out_prime)\n"
     "        result = barrett_reduce_uint64_uint64(coeff, out_prime, barret_ratio);\n"
     "    else\n"
     "        result = coeff;\n"
     "#endif\n"
     "    dst[tid] = result;\n"),
    ("src/rns_bconv.cu",
     "        // BFV and CKKS\n"
     "        base_P_to_Ql_conv_.bConv_BEHZ(delta.get(), cx_i + size_Ql_n, n, stream);\n",
     "        // BFV and CKKS\n"
     "#ifdef PHANTOM_V5_CENTERED\n"
     "        if (scheme == scheme_type::ckks && size_P_ == 1)  // CipherChat v5: centered (encryption mod-down)\n"
     "            moddown_bconv_single_p_kernel<<<size_Ql_n / blockDimGlb.x, blockDimGlb, 0, stream>>>(\n"
     "                    delta.get(), cx_i + size_Ql_n, n, base_QlP_.base(), size_QlP);\n"
     "        else\n"
     "#endif\n"
     "        base_P_to_Ql_conv_.bConv_BEHZ(delta.get(), cx_i + size_Ql_n, n, stream);\n"),
    # ---------------------------------------------------------------- single-digit key switching
    ("src/rns.cu",
     "            size_t alpha = size_P;\n",
     "#ifdef PHANTOM_SINGLE_DIGIT_KS\n"
     "            size_t alpha = size_Q;  // CipherChat v5: one digit = all of Q_l (needs e = 0)\n"
     "#else\n"
     "            size_t alpha = size_P;\n"
     "#endif\n"),
    ("src/rns_bconv.cu",
     "    size_t alpha = size_P;\n    size_t beta = v_base_part_Ql_to_compl_part_QlP_conv_.size();\n",
     "#ifdef PHANTOM_SINGLE_DIGIT_KS\n"
     "    size_t alpha = size_Ql;  // CipherChat v5: one digit = all of Q_l\n"
     "#else\n"
     "    size_t alpha = size_P;\n"
     "#endif\n"
     "    size_t beta = v_base_part_Ql_to_compl_part_QlP_conv_.size();\n"),
    ("src/secretkey.cu",
     "    size_t dnum = size_Q / size_P;\n    size_t alpha = size_P;\n",
     "#ifdef PHANTOM_SINGLE_DIGIT_KS\n"
     "    size_t dnum = 1;          // CipherChat v5: key = (-a s + P s', a), valid only without error\n"
     "    size_t alpha = size_Q;\n"
     "#else\n"
     "    size_t dnum = size_Q / size_P;\n"
     "    size_t alpha = size_P;\n"
     "#endif\n"),
]

GUARD = """
#if defined(PHANTOM_SINGLE_DIGIT_KS) && !defined(PHANTOM_ZERO_ERROR)
#error "PHANTOM_SINGLE_DIGIT_KS is only correct with PHANTOM_ZERO_ERROR (key-switching noise ~ Q e / P otherwise)"
#endif
"""


def main(src, dst):
    if os.path.abspath(src) == os.path.abspath(dst):
        raise SystemExit("destination must differ from the source tree")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("build*", ".git"))
    for item in PATCHES:
        rel, old, new = item[:3]
        count = item[3] if len(item) > 3 else 1
        path = os.path.join(dst, rel)
        text = open(path).read()
        found = text.count(old)
        if found != count:
            raise SystemExit(f"patch failed: {rel}: expected {count} match(es), found {found}:\n{old}")
        open(path, "w").write(text.replace(old, new))
    path = os.path.join(dst, "include/util/defines.h")
    open(path, "a").write(GUARD)
    print(f"patched PhantomFHE copy in {dst} ({len(PATCHES)} patches)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])

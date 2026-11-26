#!/usr/bin/env bash
# CipherChat v5 – the GPU measurement series of docs/V5.md, one job after the other (the card is shared; the engines
# do not fit next to each other). Run from the synced checkout on at-10-1-x after scripts/v5_setup_remote.sh:
#   nohup bash scripts/v5_runs.sh [step ...] > logs/v5_runs.log 2>&1 < /dev/null &
# Steps (default: all): v5_lpp1 v5_lpp2 v3 v3e0 rounding
# v3 uses the unpatched module of the regular v3 build (V3_BUILD, default ~/cipherchat/build_v3).
cd "$(dirname "$0")/.."
source .venv/bin/activate
B=${BUILD:-build_v5}
V3_BUILD=${V3_BUILD:-$HOME/cipherchat/build_v3}
CK=weights/v3/q05_pos.pt
TOK=${TOKENS:-3}
mkdir -p results/v5 logs
export LD_LIBRARY_PATH=$PWD/$B:$LD_LIBRARY_PATH
steps=${*:-v5_lpp1 v5_lpp2 v3 v3e0 rounding}
wait_gpu() {   # the card is shared: wait until (almost) nothing else is on it
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt "${GPU_FREE_BELOW_MB:-3500}" ]; do
    sleep 30
  done
}
for s in $steps; do
  wait_gpu
  echo "=== $s $(date) | GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  case $s in
    v5_lpp1)   # e = 0, one level per 50-bit prime, 866 levels, N = 2^13, fused plaintext products
      PYTHONPATH=$B python scripts/v5_check.py --device cpu --ckpt $CK --backend phantom --tokens $TOK --levels 866 \
        --out results/v5/check_phantom_lpp1.json ;;
    v5_lpp2)   # e = 0, two levels per 60-bit prime (Delta = 2^30)
      PYTHONPATH=$B python scripts/v5_check.py --device cpu --ckpt $CK --backend phantom --tokens $TOK --lpp 2 --scale_bits 30 \
        --prime_bits 60 --special_bits 50 --levels ${LPP2_LEVELS:-1151} --out results/v5/check_phantom_lpp2.json ;;
    v3)        # reference: v3 as it is (e != 0, 128 bit, bootstrapping), same machine and prompt
      PYTHONPATH=$V3_BUILD python scripts/v3_check.py --ckpt $CK --backend phantom --tokens $TOK \
        --bounds weights/v3/q05_pos_bounds.json --out results/v5/v3_check_phantom_pos.json ;;
    v3e0)      # v3 parameters and bootstrapping, only the error term removed (patched Phantom, PHANTOM_ZERO_ERROR)
      PYTHONPATH=${V3E0_BUILD:-$B/v3e0} python scripts/v3_check.py --ckpt $CK --backend phantom --tokens $TOK \
        --bounds weights/v3/q05_pos_bounds.json --out results/v5/v3e0_check_phantom_pos.json ;;
    rounding)  # rescale rounding: first 4 layers, decrypted after every layer. Two levels per prime with Delta =
               # 2^20 / 2^25 / 2^30 (primes ~ Delta^2, so the prime-vs-scale deviation stays small), and one level per
               # 50-bit prime. (One level per small prime is no test of rounding: primes = 1 mod 2N spread over
               # several % around 2^20..2^30 and the nominal scale bookkeeping would dominate.)
      for sb in 20 25 30; do
        PYTHONPATH=$B python scripts/v5_check.py --device cpu --ckpt $CK --backend phantom --tokens 1 --layers 4 --levels 201 \
          --lpp 2 --scale_bits $sb --prime_bits $((2 * sb)) --special_bits 50 --per_layer --out results/v5/rounding_lpp2_sb$sb.json
      done
      PYTHONPATH=$B python scripts/v5_check.py --device cpu --ckpt $CK --backend phantom --tokens 1 --layers 4 --levels 146 \
        --scale_bits 50 --per_layer --out results/v5/rounding_lpp1_sb50.json ;;
    *) echo "unknown step $s"; exit 1 ;;
  esac || echo "!!! step $s failed"
done
echo "=== done $(date)"
